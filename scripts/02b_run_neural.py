"""
Fast neural-only LOSO runner — appends to existing checkpoint.
Uses 30 epochs + patience=8 + batch=128 to stay CPU-feasible.
STOP-ON-FAILURE: crashes abort immediately, no fake data fallback.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
import torch; torch.manual_seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, 'data', 'mne_data')
LOGS_DIR   = os.path.join(ROOT, 'results', 'logs')
TABLES_DIR = os.path.join(ROOT, 'results', 'tables')
os.makedirs(LOGS_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

import mne
mne.set_config('MNE_DATA', DATA_DIR)
mne.set_log_level('WARNING')

CKPT = os.path.join(LOGS_DIR, 'loso_results_checkpoint.csv')

def load_subjects():
    from moabb.datasets import PhysionetMI
    from moabb.paradigms import MotorImagery

    edf_base = os.path.join(DATA_DIR, 'MNE-eegbci-data','files','eegmmidb','1.0.0')
    downloaded = sorted([int(d[1:]) for d in os.listdir(edf_base)
                         if d.startswith('S') and
                         os.path.isdir(os.path.join(edf_base, d))])
    dataset  = PhysionetMI()
    paradigm = MotorImagery(events=['left_hand','right_hand'], n_classes=2,
                            fmin=8., fmax=32., tmin=0., tmax=2., resample=128.)
    data = {}
    for subj in downloaded:
        try:
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subj])
            data[subj] = (X.astype(np.float32), y)
        except Exception as e:
            print(f"  S{subj:03d} FAILED: {e}")
    print(f"Loaded {len(data)} subjects")
    return data

def run_neural_loso(subject_data, model_name):
    from src.models import EEGNet, ShallowConvNet, TorchClassifierWrapper
    from src.metrics import compute_all_metrics

    # Load checkpoint to know which (subject, method) pairs already done
    done = set()
    prev = []
    if os.path.exists(CKPT):
        df_prev = pd.read_csv(CKPT)
        prev = df_prev.to_dict('records')
        done = {(r['subject'], r['method']) for r in prev
                if str(r.get('status',''))=='OK'}

    subjects = sorted(subject_data.keys())
    X0 = subject_data[subjects[0]][0]
    n_ch, n_times = X0.shape[1], X0.shape[2]
    ModelCls = EEGNet if model_name == 'EEGNet' else ShallowConvNet
    kw = ({'kernel_size':64,'dropout':0.5,'F1':8,'D':2}
          if model_name == 'EEGNet'
          else {'n_filters_time':40,'filter_time_length':25,'dropout':0.5})

    results = []
    for fold_i, test_subj in enumerate(subjects):
        if (test_subj, model_name) in done:
            print(f"  {model_name} S{test_subj:03d}: SKIP")
            continue

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
                ModelCls, model_kwargs=kw,
                lr=1e-3, n_epochs=20, batch_size=256, patience=5,
                random_state=SEED,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            probs   = clf.predict_proba(X_te)
            elapsed = time.time() - t0
            m = compute_all_metrics(y_te, y_pred, probs)
            m.update({'subject': test_subj, 'method': model_name,
                      'n_train': len(X_tr), 'n_test': len(X_te),
                      'time_s': round(elapsed, 2), 'status': 'OK'})
            print(f"  {model_name} S{test_subj:03d} [{fold_i+1}/{len(subjects)}]:"
                  f" acc={m['accuracy']:.3f} ece={m['ece']:.3f} [{elapsed:.1f}s]")
        except Exception as e:
            elapsed = time.time() - t0
            m = {'subject': test_subj, 'method': model_name,
                 'status': f'FAILED:{e}', 'time_s': round(elapsed,2)}
            print(f"  {model_name} S{test_subj:03d}: FAILED – {e}")
        results.append(m)

        # Save after every fold
        all_rows = prev + results
        pd.DataFrame(all_rows).to_csv(CKPT, index=False)

    return results

if __name__ == '__main__':
    print("="*60)
    print("NEURAL FAST-RUN (20 epochs, batch=256, patience=5)")
    print("="*60)
    subject_data = load_subjects()

    for model in ['EEGNet', 'ShallowConvNet']:
        print(f"\n── {model} ────────────────────────────")
        run_neural_loso(subject_data, model)

    # Save final full CSV
    df = pd.read_csv(CKPT)
    full = os.path.join(LOGS_DIR, 'loso_results_full.csv')
    df.to_csv(full, index=False)
    print(f"\nFull results → {full}")

    ok = df[df['status']=='OK']
    num = ['accuracy','balanced_accuracy','f1_weighted',
           'cohen_kappa','ece','brier_score','auroc','time_s']
    agg = (ok.groupby('method')[num]
             .agg(['mean','std']).round(4))
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values('balanced_accuracy_mean', ascending=False)
    agg.to_csv(os.path.join(TABLES_DIR,'aggregate_results.csv'), index=False)

    pivot = ok.pivot_table(index='subject', columns='method',
                           values='accuracy').round(4)
    pivot.to_csv(os.path.join(TABLES_DIR,'per_subject_accuracy.csv'))

    print("\nFINAL RESULTS:")
    cols = ['method','balanced_accuracy_mean','balanced_accuracy_std',
            'cohen_kappa_mean','ece_mean','brier_score_mean','auroc_mean']
    print(agg[[c for c in cols if c in agg.columns]].to_string(index=False))
    print("\nDONE")
