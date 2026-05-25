"""
Final publication-quality figures using ALL completed results.
Run after 02_run_experiments.py fully completes.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR   = os.path.join(ROOT, 'results', 'logs')
TABLES_DIR = os.path.join(ROOT, 'results', 'tables')
FIGS_DIR   = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 200,
})

METHOD_ORDER = ['LogVar+LDA','FBCSP+LDA','Riemann-MDM',
                'Riemann-TS+SVM','EEGNet','ShallowConvNet']
PALETTE = {
    'LogVar+LDA':       '#9ecae1',
    'FBCSP+LDA':        '#3182bd',
    'Riemann-MDM':      '#74c476',
    'Riemann-TS+SVM':   '#238b45',
    'EEGNet':           '#fd8d3c',
    'ShallowConvNet':   '#e6550d',
}
FAMILY = {
    'LogVar+LDA':      'Classical',
    'FBCSP+LDA':       'Classical',
    'Riemann-MDM':     'Riemannian',
    'Riemann-TS+SVM':  'Riemannian',
    'EEGNet':          'Neural',
    'ShallowConvNet':  'Neural',
}

def load():
    for f in ['loso_results_full.csv', 'loso_results_checkpoint.csv']:
        p = os.path.join(LOGS_DIR, f)
        if os.path.exists(p):
            df = pd.read_csv(p)
            ok = df[df['status']=='OK'].copy()
            methods = [m for m in METHOD_ORDER if m in ok['method'].values]
            print(f"Loaded {len(ok)} OK rows, {len(methods)} methods: {methods}")
            return ok, methods
    raise FileNotFoundError("No results file found")

def main_comparison_figure(ok, methods):
    """Fig 1: 3-panel main result figure."""
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    colors = [PALETTE[m] for m in methods]

    # Panel A: balanced accuracy
    ax1 = fig.add_subplot(gs[0])
    data = [ok[ok['method']==m]['balanced_accuracy'].values for m in methods]
    bp = ax1.boxplot(data, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax1.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.7)
    ax1.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax1.set_ylabel('Balanced Accuracy'); ax1.set_title('(A) Classification Performance')
    ax1.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylim(0.3, 1.0)
    # Add chance annotation
    ax1.text(0.97, 0.51, 'chance', transform=ax1.get_yaxis_transform(),
             fontsize=8, color='gray', va='bottom', ha='right')

    # Panel B: ECE (calibration)
    ax2 = fig.add_subplot(gs[1])
    ece_m = [ok[ok['method']==m]['ece'].mean() for m in methods]
    ece_s = [ok[ok['method']==m]['ece'].std() for m in methods]
    bars = ax2.bar(range(len(methods)), ece_m, yerr=ece_s,
                   color=colors, alpha=0.8, capsize=4,
                   error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
    ax2.set_ylabel('Expected Calibration Error (ECE) ↓')
    ax2.set_title('(B) Calibration Quality')
    ax2.yaxis.grid(True, linestyle=':', alpha=0.6)

    # Panel C: Accuracy-ECE scatter per subject
    ax3 = fig.add_subplot(gs[2])
    for m in methods:
        sub = ok[ok['method']==m]
        ax3.scatter(sub['balanced_accuracy'], sub['ece'],
                    color=PALETTE[m], label=m, alpha=0.5, s=25, zorder=3)
    # Add mean markers
    for m in methods:
        sub = ok[ok['method']==m]
        ax3.scatter(sub['balanced_accuracy'].mean(), sub['ece'].mean(),
                    color=PALETTE[m], s=150, marker='D',
                    edgecolors='black', linewidths=1, zorder=5)
    ax3.set_xlabel('Balanced Accuracy ↑')
    ax3.set_ylabel('ECE ↓')
    ax3.set_title('(C) Accuracy–Calibration Tradeoff')
    ax3.legend(fontsize=7, ncol=2, loc='upper left')
    ax3.xaxis.grid(True, linestyle=':', alpha=0.5)
    ax3.yaxis.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle('Cross-Subject MI EEG Benchmark — LOSO Evaluation (PhysioNet EEGMMIDB, 30 subjects)',
                 y=1.01, fontsize=11, fontweight='bold')
    plt.tight_layout()
    for ext in ['.pdf', '.png']:
        plt.savefig(os.path.join(FIGS_DIR, f'fig_main_comparison{ext}'),
                    bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved: fig_main_comparison")

def comprehensive_table(ok, methods):
    """Generate final LaTeX table with all metrics."""
    rows = []
    for m in methods:
        sub = ok[ok['method']==m]
        fam = FAMILY.get(m, '?')
        rows.append({
            'Family': fam,
            'Method': m.replace('+', '\\texttt{+}'),
            'Bal. Acc.': f"{sub['balanced_accuracy'].mean():.3f}\\pm{sub['balanced_accuracy'].std():.3f}",
            '$\\kappa$': f"{sub['cohen_kappa'].mean():.3f}\\pm{sub['cohen_kappa'].std():.3f}",
            'AUROC': f"{sub['auroc'].mean():.3f}\\pm{sub['auroc'].std():.3f}",
            'ECE $\\downarrow$': f"{sub['ece'].mean():.3f}\\pm{sub['ece'].std():.3f}",
            'Brier $\\downarrow$': f"{sub['brier_score'].mean():.3f}\\pm{sub['brier_score'].std():.3f}",
        })
    tab = pd.DataFrame(rows)

    latex_lines = [
        r'\begin{table}[!t]',
        r'\caption{LOSO Cross-Subject MI Results on PhysioNet EEGMMIDB (30 subjects). '
        r'Mean\,$\pm$\,std across subjects. Best value per column in \textbf{bold}.}',
        r'\label{tab:results}',
        r'\centering',
        r'\begin{tabular}{llccccc}',
        r'\toprule',
        r'Family & Method & Bal.~Acc. & $\kappa$ & AUROC & ECE$\downarrow$ & Brier$\downarrow$ \\',
        r'\midrule',
    ]
    for _, row in tab.iterrows():
        line = ' & '.join([
            row['Family'], row['Method'], row['Bal. Acc.'],
            row['$\\kappa$'], row['AUROC'],
            row['ECE $\\downarrow$'], row['Brier $\\downarrow$']
        ]) + r' \\'
        latex_lines.append(line)
    latex_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    latex_str = '\n'.join(latex_lines)
    out = os.path.join(TABLES_DIR, 'table_results_final.tex')
    with open(out, 'w') as f:
        f.write(latex_str)
    tab.to_csv(os.path.join(TABLES_DIR, 'table_results_final.csv'), index=False)
    print(f"Saved final table: {out}")
    print(tab.to_string(index=False))
    return tab

def heatmap_all(ok, methods):
    """Per-subject heatmap."""
    pivot = ok.pivot_table(index='subject', columns='method',
                           values='balanced_accuracy')[methods].round(3)
    fig, ax = plt.subplots(figsize=(max(8, len(methods)*1.5), 10))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.35, vmax=0.9, linewidths=0.3,
                cbar_kws={'label': 'Balanced Accuracy'}, ax=ax)
    ax.set_title('Per-Subject Balanced Accuracy (LOSO)', pad=10)
    ax.set_xlabel('Method'); ax.set_ylabel('Subject')
    plt.tight_layout()
    for ext in ['.pdf', '.png']:
        plt.savefig(os.path.join(FIGS_DIR, f'fig_heatmap{ext}'),
                    bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved: fig_heatmap")

if __name__ == '__main__':
    ok, methods = load()
    main_comparison_figure(ok, methods)
    comprehensive_table(ok, methods)
    heatmap_all(ok, methods)
    print("\nAll final figures generated.")
