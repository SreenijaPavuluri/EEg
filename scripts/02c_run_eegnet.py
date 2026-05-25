"""EEGNet-only LOSO runner. Saves each fold to its own partial CSV."""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
SEED=42; np.random.seed(SEED)
import torch; torch.manual_seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT,'data','mne_data')
LOGS_DIR = os.path.join(ROOT,'results','logs')
os.makedirs(LOGS_DIR, exist_ok=True)

import mne; mne.set_config('MNE_DATA',DATA_DIR); mne.set_log_level('WARNING')
OUT = os.path.join(LOGS_DIR, 'eegnet_results.csv')

from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
from src.models import EEGNet, TorchClassifierWrapper
from src.metrics import compute_all_metrics

edf_base = os.path.join(DATA_DIR,'MNE-eegbci-data','files','eegmmidb','1.0.0')
downloaded = sorted([int(d[1:]) for d in os.listdir(edf_base)
                     if d.startswith('S') and os.path.isdir(os.path.join(edf_base,d))])
dataset  = PhysionetMI()
paradigm = MotorImagery(events=['left_hand','right_hand'], n_classes=2,
                        fmin=8., fmax=32., tmin=0., tmax=2., resample=128.)
data = {}
for s in downloaded:
    try:
        X, y, _ = paradigm.get_data(dataset=dataset, subjects=[s])
        data[s] = (X.astype(np.float32), y)
    except: pass
print(f"Loaded {len(data)} subjects")

subjects = sorted(data.keys())
done = set()
rows = []
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    rows = prev.to_dict('records')
    done = {r['subject'] for r in rows if r.get('status')=='OK'}

for i, test in enumerate(subjects):
    if test in done:
        print(f"  EEGNet S{test:03d}: SKIP"); continue
    train = [s for s in subjects if s != test]
    X_tr = np.concatenate([data[s][0] for s in train])
    y_tr = np.concatenate([data[s][1] for s in train])
    X_te, y_te = data[test]
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]
    t0 = time.time()
    try:
        clf = TorchClassifierWrapper(EEGNet,
              model_kwargs={'kernel_size':64,'dropout':0.5,'F1':8,'D':2},
              lr=1e-3, n_epochs=10, batch_size=256, patience=10, random_state=SEED)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        probs  = clf.predict_proba(X_te)
        el = time.time()-t0
        m = compute_all_metrics(y_te, y_pred, probs)
        m.update({'subject':test,'method':'EEGNet','n_train':len(X_tr),
                  'n_test':len(X_te),'time_s':round(el,2),'status':'OK'})
        print(f"  EEGNet S{test:03d} [{i+1}/{len(subjects)}]: acc={m['accuracy']:.3f} ece={m['ece']:.3f} [{el:.0f}s]")
    except Exception as e:
        el = time.time()-t0
        m = {'subject':test,'method':'EEGNet','status':f'FAILED:{e}','time_s':round(el,2)}
        print(f"  EEGNet S{test:03d}: FAILED – {e}")
    rows.append(m)
    pd.DataFrame(rows).to_csv(OUT, index=False)

print(f"EEGNet done → {OUT}")
