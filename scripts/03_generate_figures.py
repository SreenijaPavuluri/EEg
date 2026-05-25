"""
Figure generation for IEEE paper.
Reads results from results/logs/loso_results_full.csv
Produces all paper figures in results/figures/
Run AFTER 02_run_experiments.py completes.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR  = os.path.join(ROOT, 'results', 'logs')
TABLES_DIR = os.path.join(ROOT, 'results', 'tables')
FIGS_DIR  = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

PALETTE = {
    'LogVar+LDA':       '#9ecae1',
    'FBCSP+LDA':        '#3182bd',
    'Riemann-MDM':      '#74c476',
    'Riemann-TS+SVM':   '#238b45',
    'EEGNet':           '#fd8d3c',
    'ShallowConvNet':   '#e6550d',
}
METHOD_ORDER = list(PALETTE.keys())

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

def load_results():
    path = os.path.join(LOGS_DIR, 'loso_results_full.csv')
    if not os.path.exists(path):
        path = os.path.join(LOGS_DIR, 'loso_results_checkpoint.csv')
    df = pd.read_csv(path)
    ok = df[df['status'] == 'OK'].copy()
    print(f"Loaded {len(ok)} OK rows from {len(df['subject'].unique())} subjects × "
          f"{len(df['method'].unique())} methods")
    return ok

# ─── Fig 1: Balanced accuracy box-plot ──────────────────────────────────────
def fig_balanced_accuracy(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    data_list = [df[df['method'] == m]['balanced_accuracy'].values
                 for m in METHOD_ORDER if m in df['method'].values]
    labels = [m for m in METHOD_ORDER if m in df['method'].values]
    colors = [PALETTE[m] for m in labels]

    bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance (0.5)')
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Cross-Subject MI Classification: Balanced Accuracy (LOSO)')
    ax.set_ylim(0.35, 1.0)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig1_balanced_accuracy.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ─── Fig 2: ECE (calibration) bar chart ─────────────────────────────────────
def fig_calibration_ece(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: ECE bar
    ax = axes[0]
    methods = [m for m in METHOD_ORDER if m in df['method'].values]
    ece_means = [df[df['method']==m]['ece'].mean() for m in methods]
    ece_stds  = [df[df['method']==m]['ece'].std()  for m in methods]
    colors = [PALETTE[m] for m in methods]
    bars = ax.bar(range(len(methods)), ece_means, yerr=ece_stds,
                  color=colors, alpha=0.8, capsize=4, error_kw={'linewidth':1.2})
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Expected Calibration Error (ECE) ↓')
    ax.set_title('Calibration: Mean ECE per Method')
    ax.yaxis.grid(True, linestyle=':', alpha=0.7)

    # Right: Brier score bar
    ax2 = axes[1]
    brier_means = [df[df['method']==m]['brier_score'].mean() for m in methods]
    brier_stds  = [df[df['method']==m]['brier_score'].std()  for m in methods]
    ax2.bar(range(len(methods)), brier_means, yerr=brier_stds,
            color=colors, alpha=0.8, capsize=4, error_kw={'linewidth':1.2})
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylabel('Brier Score ↓')
    ax2.set_title('Calibration: Mean Brier Score per Method')
    ax2.yaxis.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig2_calibration.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ─── Fig 3: Reliability diagrams (calibration curves) ───────────────────────
def fig_reliability_diagrams(df_raw):
    """Re-compute reliability diagrams from raw subject-level results.
    This requires probs columns — not stored in current CSV.
    We compute ECE-curve from stored ece values and create illustrative per-method plots.
    """
    # Since we only have aggregate metrics (not raw probabilities) in the CSV,
    # we plot ECE vs balanced accuracy scatter to show the calibration-accuracy tradeoff.
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = [m for m in METHOD_ORDER if m in df_raw['method'].values]
    for m in methods:
        sub = df_raw[df_raw['method'] == m]
        x = sub['balanced_accuracy'].values
        y = sub['ece'].values
        ax.scatter(x, y, color=PALETTE[m], label=m, alpha=0.6, s=40)
        ax.annotate(m, (x.mean(), y.mean()),
                    fontsize=7.5, ha='center', va='bottom',
                    color=PALETTE[m], fontweight='bold')

    ax.set_xlabel('Balanced Accuracy ↑')
    ax.set_ylabel('Expected Calibration Error ↓')
    ax.set_title('Accuracy–Calibration Tradeoff (per subject, LOSO)')
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax.xaxis.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig3_acc_ece_scatter.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ─── Fig 4: Per-subject heatmap ─────────────────────────────────────────────
def fig_per_subject_heatmap():
    path = os.path.join(TABLES_DIR, 'per_subject_accuracy.csv')
    if not os.path.exists(path):
        print("Skipping heatmap: per_subject_accuracy.csv not found")
        return
    pivot = pd.read_csv(path, index_col=0)
    # Reorder columns
    cols = [m for m in METHOD_ORDER if m in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.3, vmax=1.0, linewidths=0.3,
                cbar_kws={'label': 'Accuracy'}, ax=ax)
    ax.set_title('Per-Subject Accuracy Across Methods (LOSO)', pad=12)
    ax.set_xlabel('Method')
    ax.set_ylabel('Subject ID')
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig4_per_subject_heatmap.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")

# ─── Fig 5: Cohen's kappa comparison ────────────────────────────────────────
def fig_kappa(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = [m for m in METHOD_ORDER if m in df['method'].values]
    kappa_means = [df[df['method']==m]['cohen_kappa'].mean() for m in methods]
    kappa_stds  = [df[df['method']==m]['cohen_kappa'].std()  for m in methods]
    colors = [PALETTE[m] for m in methods]

    x = np.arange(len(methods))
    ax.bar(x, kappa_means, yerr=kappa_stds, color=colors, alpha=0.85,
           capsize=4, error_kw={'linewidth':1.2})
    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel("Cohen's κ ↑")
    ax.set_title("Cross-Subject MI: Cohen's Kappa (LOSO)")
    ax.yaxis.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig5_kappa.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")

# ─── Fig 6: Inference time vs accuracy ──────────────────────────────────────
def fig_efficiency(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = [m for m in METHOD_ORDER if m in df['method'].values]
    for m in methods:
        sub = df[df['method'] == m]
        t = sub['time_s'].mean()
        acc = sub['balanced_accuracy'].mean()
        ax.scatter(t, acc, s=120, color=PALETTE[m], label=m, zorder=5)
        ax.annotate(m, (t, acc), textcoords='offset points',
                    xytext=(5, 2), fontsize=8, color=PALETTE[m])
    ax.set_xlabel('Mean Training+Inference Time per Fold (s) →')
    ax.set_ylabel('Mean Balanced Accuracy ↑')
    ax.set_title('Efficiency vs Performance')
    ax.set_xscale('log')
    ax.xaxis.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig6_efficiency.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")

# ─── LaTeX table ────────────────────────────────────────────────────────────
def make_latex_table(df):
    methods = [m for m in METHOD_ORDER if m in df['method'].values]
    rows = []
    for m in methods:
        sub = df[df['method'] == m]
        rows.append({
            'Method': m,
            'Acc': f"{sub['accuracy'].mean():.3f}±{sub['accuracy'].std():.3f}",
            'Bal. Acc': f"{sub['balanced_accuracy'].mean():.3f}±{sub['balanced_accuracy'].std():.3f}",
            'κ': f"{sub['cohen_kappa'].mean():.3f}±{sub['cohen_kappa'].std():.3f}",
            'AUROC': f"{sub['auroc'].mean():.3f}±{sub['auroc'].std():.3f}",
            'ECE ↓': f"{sub['ece'].mean():.3f}±{sub['ece'].std():.3f}",
            'Brier ↓': f"{sub['brier_score'].mean():.3f}±{sub['brier_score'].std():.3f}",
        })
    tab = pd.DataFrame(rows)
    latex = tab.to_latex(index=False, escape=False,
                         caption='LOSO cross-subject MI classification results on '
                                 'PhysioNet EEGMMIDB (30 subjects). '
                                 'Mean±std across subjects.',
                         label='tab:results')
    out = os.path.join(TABLES_DIR, 'table_results.tex')
    with open(out, 'w') as f:
        f.write(latex)
    tab.to_csv(os.path.join(TABLES_DIR, 'table_results.csv'), index=False)
    print(f"Saved LaTeX table: {out}")
    return tab

if __name__ == '__main__':
    print("Generating figures...")
    df = load_results()

    fig_balanced_accuracy(df)
    fig_calibration_ece(df)
    fig_reliability_diagrams(df)
    fig_per_subject_heatmap()
    fig_kappa(df)
    fig_efficiency(df)
    tab = make_latex_table(df)

    print("\nMain results table:")
    print(tab.to_string(index=False))
    print("\nAll figures saved to results/figures/")
