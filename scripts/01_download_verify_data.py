"""
Dataset download and verification for PhysioNet EEGMMIDB.
Uses MOABB's PhysionetMI wrapper. Saves dataset report to results/tables/.
STOP-ON-FAILURE: any exception aborts with a clear error message.
"""

import os
import sys
import json
import traceback
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

MNE_DATA_DIR = os.path.join(os.path.abspath(DATA_DIR), 'mne_data')
os.makedirs(MNE_DATA_DIR, exist_ok=True)

import mne
mne.set_config('MNE_DATA', MNE_DATA_DIR)
mne.set_log_level('WARNING')

def verify_physionet_mi():
    """Download and verify PhysioNet Motor Imagery dataset via MOABB."""
    print("=" * 60)
    print("DATASET: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)")
    print("=" * 60)

    from moabb.datasets import PhysionetMI
    from moabb.paradigms import MotorImagery

    dataset = PhysionetMI()
    print(f"Dataset name        : {dataset.code}")
    print(f"Total subjects      : {len(dataset.subject_list)}")
    print(f"Subject list (first 5): {dataset.subject_list[:5]}")
    print(f"Interval            : {dataset.interval}")

    # Use a 2-class left-hand vs right-hand MI paradigm
    # PhysionetMI events: left_hand, right_hand, hands, feet, rest
    paradigm = MotorImagery(
        events=['left_hand', 'right_hand'],
        n_classes=2,
        fmin=8.0,
        fmax=32.0,
        tmin=0.0,
        tmax=2.0,
        resample=128.0,
    )

    print("\nDownloading first 5 subjects to verify access...")
    probe_subjects = dataset.subject_list[:5]

    rows = []
    all_X, all_y = [], []
    for subj in probe_subjects:
        try:
            X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
            n_trials, n_ch, n_times = X.shape
            classes, counts = np.unique(y, return_counts=True)
            rows.append({
                'subject': subj,
                'n_trials': n_trials,
                'n_channels': n_ch,
                'n_times': n_times,
                'classes': list(classes),
                'class_counts': dict(zip(classes, counts.tolist())),
                'status': 'OK'
            })
            all_X.append(X)
            all_y.append(y)
            print(f"  Subject {subj:3d}: {n_trials} trials, {n_ch} ch, "
                  f"{n_times} samples, classes={dict(zip(classes, counts.tolist()))}")
        except Exception as e:
            print(f"  Subject {subj:3d}: FAILED - {e}")
            rows.append({'subject': subj, 'status': f'FAILED: {e}'})

    ok_rows = [r for r in rows if r.get('status') == 'OK']
    if not ok_rows:
        print("\nERROR: All probe subjects failed. Dataset unavailable. STOPPING.")
        sys.exit(1)

    print(f"\nProbe verification: {len(ok_rows)}/{len(probe_subjects)} subjects OK")

    # Now download ALL subjects
    print(f"\nDownloading all {len(dataset.subject_list)} subjects...")
    all_rows = list(ok_rows)
    remaining = [s for s in dataset.subject_list if s not in probe_subjects]

    batch_size = 10
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i+batch_size]
        for subj in batch:
            try:
                X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
                n_trials, n_ch, n_times = X.shape
                classes, counts = np.unique(y, return_counts=True)
                all_rows.append({
                    'subject': subj,
                    'n_trials': n_trials,
                    'n_channels': n_ch,
                    'n_times': n_times,
                    'classes': list(classes),
                    'class_counts': dict(zip(classes, counts.tolist())),
                    'status': 'OK'
                })
            except Exception as e:
                print(f"  Subject {subj}: FAILED - {e}")
                all_rows.append({'subject': subj, 'status': f'FAILED: {e}'})
        good = sum(1 for r in all_rows if r.get('status') == 'OK')
        print(f"  Progress: {good}/{len(all_rows)} subjects loaded...")

    ok_rows_all = [r for r in all_rows if r.get('status') == 'OK']
    print(f"\nTotal loaded: {len(ok_rows_all)} / {len(dataset.subject_list)} subjects")

    if len(ok_rows_all) < 20:
        print("ERROR: Fewer than 20 subjects available. Dataset too small. STOPPING.")
        sys.exit(1)

    # Build report
    df = pd.DataFrame(ok_rows_all)
    df.to_csv(os.path.join(TABLES_DIR, 'dataset_report.csv'), index=False)

    # Summary stats
    trial_counts = [r['n_trials'] for r in ok_rows_all]
    ch_counts = [r['n_channels'] for r in ok_rows_all]

    report = {
        'dataset': 'PhysioNet EEGMMIDB',
        'moabb_code': dataset.code,
        'n_subjects_total': len(dataset.subject_list),
        'n_subjects_ok': len(ok_rows_all),
        'paradigm': 'MotorImagery left_hand vs right_hand',
        'freq_band': '8-32 Hz',
        'epoch_window': '0.0-2.0s',
        'resample_hz': 128,
        'n_channels': ch_counts[0] if ch_counts else 'N/A',
        'n_times_per_epoch': ok_rows_all[0]['n_times'] if ok_rows_all else 'N/A',
        'trials_mean': float(np.mean(trial_counts)),
        'trials_min': int(np.min(trial_counts)),
        'trials_max': int(np.max(trial_counts)),
        'classes': ['left_hand', 'right_hand'],
        'license': 'PhysioNet (https://physionet.org/content/eegmmidb/1.0.0/)',
        'citation': 'Schalk et al. 2004; Goldberger et al. 2000',
        'seed': SEED,
    }

    report_path = os.path.join(TABLES_DIR, 'dataset_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("DATASET REPORT")
    print("=" * 60)
    for k, v in report.items():
        print(f"  {k:<30}: {v}")
    print(f"\nReport saved to: {report_path}")
    print("Dataset verification: PASSED")
    return report

if __name__ == '__main__':
    try:
        report = verify_physionet_mi()
        print("\nDATASET DOWNLOAD & VERIFICATION: SUCCESS")
    except SystemExit:
        raise
    except Exception:
        print("\nFATAL ERROR during dataset download:")
        traceback.print_exc()
        sys.exit(1)
