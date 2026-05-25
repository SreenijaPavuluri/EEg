"""
Quick verification: load first 5 subjects, confirm data shape and class balance.
Uses only already-downloaded subjects (no new downloads needed).
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data', 'mne_data')
sys.path.insert(0, ROOT)

import mne
mne.set_config('MNE_DATA', DATA_DIR)
mne.set_log_level('WARNING')

from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

dataset = PhysionetMI()
paradigm = MotorImagery(
    events=['left_hand', 'right_hand'], n_classes=2,
    fmin=8.0, fmax=32.0, tmin=0.0, tmax=2.0, resample=128.0,
)

# Only try subjects that are actually downloaded
edf_base = os.path.join(DATA_DIR, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0')
downloaded = sorted([
    int(d[1:]) for d in os.listdir(edf_base)
    if d.startswith('S') and os.path.isdir(os.path.join(edf_base, d))
])
print(f"Subjects with downloaded data: {len(downloaded)} → {downloaded[:10]}...")

ok_subjects = []
rows = []
for subj in downloaded[:5]:
    try:
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
        classes, counts = np.unique(y, return_counts=True)
        rows.append({'subject': subj, 'n_trials': len(y), 'shape': X.shape,
                     'classes': dict(zip(classes.tolist(), counts.tolist()))})
        ok_subjects.append(subj)
        print(f"  S{subj:03d}: shape={X.shape}, classes={dict(zip(classes.tolist(), counts.tolist()))}")
    except Exception as e:
        print(f"  S{subj:03d}: FAILED – {e}")

if not ok_subjects:
    print("FATAL: No subjects loadable. STOPPING.")
    sys.exit(1)

# Save report
import json, pandas as pd
report = {
    'n_downloaded_subjects': len(downloaded),
    'downloaded_subject_ids': downloaded,
    'sample_shape': list(rows[0]['shape']) if rows else None,
    'sfreq': 128,
    'classes': ['left_hand', 'right_hand'],
    'probe_ok': len(ok_subjects),
}
os.makedirs(os.path.join(ROOT, 'results', 'tables'), exist_ok=True)
with open(os.path.join(ROOT, 'results', 'tables', 'dataset_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nVerification PASSED – {len(downloaded)} subjects available, shape={rows[0]['shape']}")
print(json.dumps(report, indent=2))
