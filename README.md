# Riemannian Geometry vs Neural Baselines for Cross-Subject MI EEG
## A Calibration-Aware Benchmark — IEEE Conference Paper

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: PhysioNet EEGMMIDB](https://img.shields.io/badge/Dataset-PhysioNet%20EEGMMIDB-orange.svg)](https://physionet.org/content/eegmmidb/1.0.0/)

---

## Overview

This repository contains the full reproducible codebase for our IEEE paper:

> **"Riemannian Geometry vs. Neural Baselines for Cross-Subject Motor Imagery EEG: A Calibration-Aware Benchmark"**

### Key Contributions
1. **First calibration-aware cross-subject MI benchmark** — reporting ECE and Brier score alongside standard MI metrics (accuracy, balanced accuracy, Cohen's κ, AUROC).
2. **Direct comparison of Riemannian vs. neural methods** under strict LOSO (leave-one-subject-out) evaluation — a gap in prior EEG foundation model papers (MIRepNet, LEAD, LaBraM, EEGPT).
3. **CPU-feasible, fully reproducible pipeline** — no GPU required; runs end-to-end on a laptop.
4. **Finding**: Riemannian methods (MDM, Tangent-Space SVM) match neural baselines (EEGNet, ShallowConvNet) in cross-subject accuracy while producing better-calibrated probability outputs.

---

## Dataset

| Property | Value |
|---|---|
| Name | PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) |
| Subjects used | 30 (subjects 1–30) |
| Channels | 64 |
| Sampling rate | 128 Hz (resampled) |
| Epoch window | 0–2 s post-cue |
| Frequency band | 8–32 Hz |
| Classes | left_hand vs right_hand imagery |
| Trials/subject | ~45 |
| Access | Free, public: https://physionet.org/content/eegmmidb/1.0.0/ |
| Citation | Schalk et al. 2004; Goldberger et al. 2000 |

Data is downloaded automatically via [MNE-Python](https://mne.tools/) on first run.

---

## Methods Evaluated

| Method | Family | CPU Time/Fold |
|---|---|---|
| LogVar + LDA | Classical | < 1 s |
| FBCSP + LDA | Classical | < 5 s |
| Riemann-MDM | Riemannian | < 5 s |
| Riemann-TS + SVM | Riemannian | < 10 s |
| EEGNet | Neural (compact) | 1–3 min |
| ShallowConvNet | Neural (compact) | 1–3 min |

---

## Repository Structure

```
EEg/
├── notebooks/          # Jupyter notebooks for exploration (optional)
├── src/
│   ├── features.py     # Feature extraction (FBCSP, covariance, log-power)
│   ├── models.py       # EEGNet, ShallowConvNet, Riemannian wrappers
│   └── metrics.py      # Accuracy, ECE, Brier, AUROC
├── configs/
│   └── experiment_config.json
├── data/               # Auto-populated by download (gitignored)
├── results/
│   ├── figures/        # All PDF + PNG figures
│   ├── tables/         # CSV + LaTeX tables
│   └── logs/           # Per-fold results, checkpoints
├── paper/
│   └── manuscript.tex  # IEEE LaTeX manuscript
├── scripts/
│   ├── 00_quick_verify.py      # Verify downloaded data
│   ├── 01_download_verify_data.py  # Full dataset download
│   ├── 02_run_experiments.py   # Main LOSO experiment runner
│   └── 03_generate_figures.py  # Figure and table generation
├── README.md
├── requirements.txt
└── run_all.sh
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/SreenijaPavuluri/EEg.git
cd EEg

# 2. Create virtual environment (Python 3.9+)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Reproducing All Results

```bash
source venv/bin/activate

# Step 1: Verify / download data (downloads ~300 MB from PhysioNet)
python3 scripts/00_quick_verify.py

# Step 2: Run all LOSO experiments (saves checkpoint after each fold)
python3 scripts/02_run_experiments.py

# Step 3: Generate all figures and LaTeX tables
python3 scripts/03_generate_figures.py
```

Or run everything at once:
```bash
bash run_all.sh
```

Results are saved to:
- `results/logs/loso_results_full.csv` — per-subject per-method metrics
- `results/tables/aggregate_results.csv` — mean ± std summary
- `results/tables/per_subject_accuracy.csv` — pivot table
- `results/tables/table_results.tex` — LaTeX table for paper
- `results/figures/` — all PDF and PNG figures

---

## Experiment Configuration

See `configs/experiment_config.json` for all hyperparameters and settings.
All experiments use seed 42.

---

## Citation

If you use this code or results, please cite:

```bibtex
@inproceedings{eeg_riemannian_calibration_2026,
  title   = {Riemannian Geometry vs. Neural Baselines for Cross-Subject
             Motor Imagery EEG: A Calibration-Aware Benchmark},
  author  = {Anonymous},
  booktitle = {IEEE International Conference on Neural Engineering},
  year    = {2026},
}
```

And the dataset:
```bibtex
@article{schalk2004bci2000,
  title   = {BCI2000: A General-Purpose Brain-Computer Interface (BCI) System},
  author  = {Schalk, Gerwin and McFarland, Dennis J and Hinterberger, Thilo
             and Birbaumer, Niels and Wolpaw, Jonathan R},
  journal = {IEEE Trans. Biomed. Eng.},
  volume  = {51}, number = {6}, pages = {1034--1043}, year = {2004}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Inspiration

This benchmark is inspired by and positioned relative to:
- [MIRepNet](https://arxiv.org/abs/2507.20254) — MI EEG foundation model
- [LEAD](https://arxiv.org/abs/2502.01678) — AD EEG foundation model
- [EEG Foundation Models Review](https://github.com/sagnikde03/EEG-Foundation-Models-Review)
