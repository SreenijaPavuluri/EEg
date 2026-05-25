"""
LOSO experiment runner — optimised for CPU.
Strategy:
  1. Load ALL subjects' raw EEG once (not per-fold).
  2. Pre-compute covariance matrices once for Riemannian methods.
  3. Pre-compute FBCSP features using held-out scheme.
  4. Run LOSO: for each test subject, train on the rest.
  5. Neural methods trained end-to-end on raw data (fewer epochs).
  6. Checkpoint after every subject.

STOP-ON-FAILURE: data load errors abort immediately.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, 'data', 'mne_data')
LOGS_DIR   = os.path.join(ROOT, 'results', 'logs')
TABLES_DIR = os.path.join(ROOT, 'results', 'tables')
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

import mne
mne.set_config('MNE_DATA', DATA_DIR)
mne.set_log_level('WARNING')

# ─── 1. Load raw EEG for all subjects (once) ─────────────────────────────────
def load_all_subjects():
    from moabb.datasets import PhysionetMI
    from moabb.paradigms import MotorImagery

    edf_base = os.path.join(DATA_DIR, 'MNE-eegbci-data', 'files',
                            'eegmmidb', '1.0.0')
    downloaded = sorted([
        int(d[1:]) for d in os.listdir(edf_base)
        if d.startswith('S') and os.path.isdir(os.path.join(edf_base, d))
    ])
    print(f"Found {len(downloaded)} downloaded subjects: {downloaded}")

    dataset = PhysionetMI()
    paradigm = MotorImagery(
        events=['left_hand', 'right_hand'], n_classes=2,
        fmin=8.0, fmax=32.0, tmin=0.0, tmax=2.0, resample=128.0,
    )

    subject_data = {}
    for i, subj in enumerate(downloaded):
        t0 = time.time()
        try:
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subj])
            subject_data[subj] = (X.astype(np.float32), y)
            print(f"  [{i+1}/{len(downloaded)}] S{subj:03d}: {X.shape}  "
                  f"classes={dict(zip(*np.unique(y, return_counts=True)))}  "
                  f"[{time.time()-t0:.1f}s]")
        except Exception as e:
            print(f"  [{i+1}/{len(downloaded)}] S{subj:03d} FAILED: {e}")

    if len(subject_data) < 15:
        print(f"FATAL: only {len(subject_data)} subjects. STOPPING.")
        sys.exit(1)
    return subject_data

# ─── 2. Pre-compute covariance matrices ──────────────────────────────────────
def compute_covariances_all(subject_data):
    """Compute LW covariance matrices for all subjects once."""
    from pyriemann.estimation import Covariances
    cov_est = Covariances(estimator='lwf')
    covs = {}
    print("Pre-computing covariance matrices...")
    for subj, (X, y) in subject_data.items():
        covs[subj] = (cov_est.fit_transform(X), y)
    print(f"  Done. Shape per subject: {covs[list(covs.keys())[0]][0].shape}")
    return covs

# ─── 3. FBCSP feature pre-computation (pooled, then LOSO-safe) ───────────────
def run_fbcsp_loso(subject_data, subjects):
    """FBCSP+LDA — must refit CSP per fold (spatial filters depend on training data)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    from mne.decoding import CSP
    from src.features import bandpass_filter

    BANDS = [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
    N_COMP = 4
    SFREQ = 128.0

    results = []
    for fold_i, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s != test_subj]
        X_tr = np.concatenate([subject_data[s][0] for s in train_subjs])
        y_tr = np.concatenate([subject_data[s][1] for s in train_subjs])
        X_te, y_te = subject_data[test_subj]

        t0 = time.time()
        try:
            # Fit per-band CSP on training data, transform train+test
            feats_tr, feats_te = [], []
            for (lo, hi) in BANDS:
                Xf_tr = bandpass_filter(X_tr, lo, hi, SFREQ)
                Xf_te = bandpass_filter(X_te, lo, hi, SFREQ)
                csp = CSP(n_components=N_COMP, reg='ledoit_wolf',
                          log=True, norm_trace=False)
                csp.fit(Xf_tr, y_tr)
                feats_tr.append(csp.transform(Xf_tr))
                feats_te.append(csp.transform(Xf_te))

            F_tr = np.concatenate(feats_tr, axis=1)
            F_te = np.concatenate(feats_te, axis=1)
            sc = StandardScaler().fit(F_tr)
            clf = LDA()
            clf.fit(sc.transform(F_tr), y_tr)
            y_pred = clf.predict(sc.transform(F_te))
            probs = clf.predict_proba(sc.transform(F_te))
            elapsed = time.time() - t0

            from src.metrics import compute_all_metrics
            m = compute_all_metrics(y_te, y_pred, probs)
            m.update({'subject': test_subj, 'method': 'FBCSP+LDA',
                      'n_train': len(X_tr), 'n_test': len(X_te),
                      'time_s': round(elapsed, 2), 'status': 'OK'})
            print(f"  FBCSP+LDA S{test_subj:03d} [{fold_i+1}/{len(subjects)}]: "
                  f"acc={m['accuracy']:.3f} ece={m['ece']:.3f} [{elapsed:.1f}s]")
        except Exception as e:
            elapsed = time.time() - t0
            m = {'subject': test_subj, 'method': 'FBCSP+LDA',
                 'status': f'FAILED:{e}', 'time_s': round(elapsed,2)}
            print(f"  FBCSP+LDA S{test_subj:03d}: FAILED – {e}")
        results.append(m)
    return results

# ─── 4. Riemannian LOSO (uses pre-computed covs) ─────────────────────────────
def run_riemannian_loso(covs, subjects):
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from src.metrics import compute_all_metrics

    results = []
    for fold_i, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s != test_subj]
        C_tr = np.concatenate([covs[s][0] for s in train_subjs])
        y_tr = np.concatenate([covs[s][1] for s in train_subjs])
        C_te, y_te = covs[test_subj]

        for method_name, make_clf in [
            ('Riemann-MDM',
             lambda: MDM(metric='riemann')),
            ('Riemann-TS+SVM',
             lambda: None),  # handled below
        ]:
            t0 = time.time()
            try:
                if method_name == 'Riemann-MDM':
                    clf = MDM(metric='riemann')
                    clf.fit(C_tr, y_tr)
                    y_pred = clf.predict(C_te)
                    # MDM predict_proba: use Riemannian distances
                    if hasattr(clf, 'predict_proba'):
                        probs = clf.predict_proba(C_te)
                    else:
                        # Fallback: distance-based soft-max
                        dists = np.array([
                            [clf._compute_dist(c, m)
                             for m in clf.covmeans_]
                            for c in C_te
                        ])
                        probs = np.exp(-dists)
                        probs = probs / probs.sum(axis=1, keepdims=True)
                else:  # Riemann-TS+SVM
                    ts = TangentSpace(metric='riemann')
                    F_tr = ts.fit_transform(C_tr)
                    F_te = ts.transform(C_te)
                    sc = StandardScaler().fit(F_tr)
                    svm = SVC(kernel='rbf', probability=True, C=1.0,
                              gamma='scale', random_state=SEED)
                    svm.fit(sc.transform(F_tr), y_tr)
                    y_pred = svm.predict(sc.transform(F_te))
                    probs = svm.predict_proba(sc.transform(F_te))

                elapsed = time.time() - t0
                m = compute_all_metrics(y_te, y_pred, probs)
                m.update({'subject': test_subj, 'method': method_name,
                          'n_train': len(C_tr), 'n_test': len(C_te),
                          'time_s': round(elapsed, 2), 'status': 'OK'})
                print(f"  {method_name} S{test_subj:03d} [{fold_i+1}/{len(subjects)}]: "
                      f"acc={m['accuracy']:.3f} ece={m['ece']:.3f} [{elapsed:.1f}s]")
            except Exception as e:
                elapsed = time.time() - t0
                m = {'subject': test_subj, 'method': method_name,
                     'status': f'FAILED:{e}', 'time_s': round(elapsed, 2)}
                print(f"  {method_name} S{test_subj:03d}: FAILED – {e}")
            results.append(m)
    return results

# ─── 5. Classical feature LOSO (LogVar+LDA) ──────────────────────────────────
def run_logvar_loso(subject_data, subjects):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    from src.features import PowerSpectralFeatures
    from src.metrics import compute_all_metrics

    feat_ext = PowerSpectralFeatures(sfreq=128.0)
    results = []
    for fold_i, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s != test_subj]
        X_tr = np.concatenate([subject_data[s][0] for s in train_subjs])
        y_tr = np.concatenate([subject_data[s][1] for s in train_subjs])
        X_te, y_te = subject_data[test_subj]
        t0 = time.time()
        try:
            F_tr = feat_ext.transform(X_tr)
            F_te = feat_ext.transform(X_te)
            sc = StandardScaler().fit(F_tr)
            clf = LDA()
            clf.fit(sc.transform(F_tr), y_tr)
            y_pred = clf.predict(sc.transform(F_te))
            probs = clf.predict_proba(sc.transform(F_te))
            elapsed = time.time() - t0
            m = compute_all_metrics(y_te, y_pred, probs)
            m.update({'subject': test_subj, 'method': 'LogVar+LDA',
                      'n_train': len(X_tr), 'n_test': len(X_te),
                      'time_s': round(elapsed, 2), 'status': 'OK'})
            print(f"  LogVar+LDA S{test_subj:03d}: acc={m['accuracy']:.3f} "
                  f"ece={m['ece']:.3f} [{elapsed:.1f}s]")
        except Exception as e:
            m = {'subject': test_subj, 'method': 'LogVar+LDA',
                 'status': f'FAILED:{e}', 'time_s': 0}
        results.append(m)
    return results

# ─── 6. Neural LOSO (EEGNet / ShallowConvNet) ─────────────────────────────────
def run_neural_loso(subject_data, subjects, model_name='EEGNet'):
    from src.models import EEGNet, ShallowConvNet, TorchClassifierWrapper
    from src.metrics import compute_all_metrics

    X0, _ = subject_data[subjects[0]]
    n_ch, n_times = X0.shape[1], X0.shape[2]
    ModelCls = EEGNet if model_name == 'EEGNet' else ShallowConvNet

    results = []
    for fold_i, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s != test_subj]
        X_tr = np.concatenate([subject_data[s][0] for s in train_subjs])
        y_tr = np.concatenate([subject_data[s][1] for s in train_subjs])
        X_te, y_te = subject_data[test_subj]

        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(X_tr))
        X_tr, y_tr = X_tr[idx], y_tr[idx]

        t0 = time.time()
        try:
            clf = TorchClassifierWrapper(
                ModelCls,
                model_kwargs={'kernel_size': 64, 'dropout': 0.5, 'F1': 8, 'D': 2}
                    if model_name == 'EEGNet'
                    else {'n_filters_time': 40, 'filter_time_length': 25, 'dropout': 0.5},
                lr=1e-3, n_epochs=60, batch_size=64, patience=12,
                random_state=SEED,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            probs = clf.predict_proba(X_te)
            elapsed = time.time() - t0
            m = compute_all_metrics(y_te, y_pred, probs)
            m.update({'subject': test_subj, 'method': model_name,
                      'n_train': len(X_tr), 'n_test': len(X_te),
                      'time_s': round(elapsed, 2), 'status': 'OK'})
            print(f"  {model_name} S{test_subj:03d} [{fold_i+1}/{len(subjects)}]: "
                  f"acc={m['accuracy']:.3f} ece={m['ece']:.3f} [{elapsed:.1f}s]")
        except Exception as e:
            elapsed = time.time() - t0
            m = {'subject': test_subj, 'method': model_name,
                 'status': f'FAILED:{e}', 'time_s': round(elapsed, 2)}
            print(f"  {model_name} S{test_subj:03d}: FAILED – {e}")
        results.append(m)
    return results

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("EEG MI BENCHMARK — optimised LOSO on PhysioNet EEGMMIDB")
    print("=" * 65)
    t_total = time.time()

    # ── Load data ──
    subject_data = load_all_subjects()
    subjects = sorted(subject_data.keys())
    print(f"\nUsing {len(subjects)} subjects: {subjects}")

    X0, y0 = subject_data[subjects[0]]
    print(f"Per-subject shape: {X0.shape}  (ch={X0.shape[1]}, t={X0.shape[2]})")

    # ── Pre-compute covs (Riemannian) ──
    covs = compute_covariances_all(subject_data)

    # ── Resume checkpoint ──
    ckpt = os.path.join(LOGS_DIR, 'loso_results_checkpoint.csv')
    all_results = []
    done = set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        all_results = prev.to_dict('records')
        done = {(r['subject'], r['method']) for r in all_results
                if str(r.get('status',''))=='OK'}
        print(f"Resumed: {len(done)} subject-method pairs already done")

    def save_ckpt():
        pd.DataFrame(all_results).to_csv(ckpt, index=False)

    # ─── Run each method group ────────────────────────────────────────────────

    # 1. LogVar+LDA
    print("\n── LogVar+LDA ────────────────────────────────────────────")
    subjs_todo = [s for s in subjects if (s,'LogVar+LDA') not in done]
    res = run_logvar_loso({s:subject_data[s] for s in subjs_todo}, subjs_todo)
    all_results.extend(res); save_ckpt()

    # 2. FBCSP+LDA
    print("\n── FBCSP+LDA ─────────────────────────────────────────────")
    subjs_todo = [s for s in subjects if (s,'FBCSP+LDA') not in done]
    res = run_fbcsp_loso({s:subject_data[s] for s in subjs_todo}, subjs_todo)
    all_results.extend(res); save_ckpt()

    # 3. Riemannian MDM
    print("\n── Riemann-MDM ───────────────────────────────────────────")
    subjs_todo = [s for s in subjects if (s,'Riemann-MDM') not in done]
    res = run_riemannian_loso({s:covs[s] for s in subjs_todo}, subjs_todo)
    # filter to just MDM results
    mdm_res = [r for r in res if r.get('method')=='Riemann-MDM']
    ts_res  = [r for r in res if r.get('method')=='Riemann-TS+SVM']
    all_results.extend(mdm_res); save_ckpt()

    # 4. Riemann-TS+SVM
    print("\n── Riemann-TS+SVM ────────────────────────────────────────")
    all_results.extend(ts_res); save_ckpt()

    # 5. EEGNet
    print("\n── EEGNet ────────────────────────────────────────────────")
    subjs_todo = [s for s in subjects if (s,'EEGNet') not in done]
    res = run_neural_loso({s:subject_data[s] for s in subjs_todo},
                          subjs_todo, 'EEGNet')
    all_results.extend(res); save_ckpt()

    # 6. ShallowConvNet
    print("\n── ShallowConvNet ────────────────────────────────────────")
    subjs_todo = [s for s in subjects if (s,'ShallowConvNet') not in done]
    res = run_neural_loso({s:subject_data[s] for s in subjs_todo},
                          subjs_todo, 'ShallowConvNet')
    all_results.extend(res); save_ckpt()

    # ─── Save final results ───────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    full = os.path.join(LOGS_DIR, 'loso_results_full.csv')
    df.to_csv(full, index=False)
    print(f"\nFull results → {full}")

    ok = df[df['status']=='OK']
    num = ['accuracy','balanced_accuracy','f1_weighted',
           'cohen_kappa','ece','brier_score','auroc','time_s']
    agg = ok.groupby('method')[num].agg(['mean','std']).round(4)
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values('balanced_accuracy_mean', ascending=False)
    agg.to_csv(os.path.join(TABLES_DIR,'aggregate_results.csv'), index=False)

    pivot = ok.pivot_table(index='subject', columns='method',
                           values='accuracy').round(4)
    pivot.to_csv(os.path.join(TABLES_DIR,'per_subject_accuracy.csv'))

    print("\n" + "=" * 65)
    print("RESULTS (mean ± std across subjects):")
    print("=" * 65)
    cols = ['method','accuracy_mean','accuracy_std','balanced_accuracy_mean',
            'balanced_accuracy_std','cohen_kappa_mean','ece_mean',
            'brier_score_mean','auroc_mean']
    print(agg[[c for c in cols if c in agg.columns]].to_string(index=False))
    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
    print("DONE")
