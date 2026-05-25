"""
Statistical analysis of LOSO results.
Wilcoxon signed-rank tests between all method pairs.
Generates significance table and cross-subject variance analysis.
Run after 02_run_experiments.py completes.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR   = os.path.join(ROOT, 'results', 'logs')
TABLES_DIR = os.path.join(ROOT, 'results', 'tables')
FIGS_DIR   = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# Load
path = os.path.join(LOGS_DIR, 'loso_results_full.csv')
if not os.path.exists(path):
    path = os.path.join(LOGS_DIR, 'loso_results_checkpoint.csv')
df = pd.read_csv(path)
ok = df[df['status'] == 'OK'].copy()
methods = sorted(ok['method'].unique())
print(f"Loaded {len(ok)} results across {len(methods)} methods")

# ─── Wilcoxon signed-rank tests (balanced accuracy) ──────────────────────────
print("\n── Wilcoxon signed-rank tests (balanced accuracy) ──")
pval_matrix = pd.DataFrame(np.nan, index=methods, columns=methods)
stat_matrix = pd.DataFrame(np.nan, index=methods, columns=methods)

for m1 in methods:
    for m2 in methods:
        if m1 == m2:
            continue
        a = ok[ok['method']==m1].sort_values('subject')['balanced_accuracy'].values
        b = ok[ok['method']==m2].sort_values('subject')['balanced_accuracy'].values
        common_subjects = (set(ok[ok['method']==m1]['subject'].values) &
                          set(ok[ok['method']==m2]['subject'].values))
        a = ok[(ok['method']==m1) & (ok['subject'].isin(common_subjects))]\
            .sort_values('subject')['balanced_accuracy'].values
        b = ok[(ok['method']==m2) & (ok['subject'].isin(common_subjects))]\
            .sort_values('subject')['balanced_accuracy'].values
        if len(a) > 1 and len(b) > 1:
            try:
                stat, pval = stats.wilcoxon(a, b, alternative='two-sided')
                pval_matrix.loc[m1, m2] = round(pval, 4)
                stat_matrix.loc[m1, m2] = round(stat, 2)
            except Exception as e:
                pass

pval_matrix.to_csv(os.path.join(TABLES_DIR, 'wilcoxon_pvalues.csv'))
print(pval_matrix.to_string())

# ─── Per-subject variance analysis ───────────────────────────────────────────
print("\n── Per-subject variance (balanced accuracy) ──")
subj_stats = ok.groupby('method')['balanced_accuracy'].agg(
    ['mean', 'std', lambda x: x.max()-x.min()]
).round(3)
subj_stats.columns = ['mean', 'std', 'range']
print(subj_stats.to_string())
subj_stats.to_csv(os.path.join(TABLES_DIR, 'subject_variance.csv'))

# ─── Calibration vs accuracy tradeoff per subject ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

PALETTE = {
    'LogVar+LDA':       '#9ecae1',
    'FBCSP+LDA':        '#3182bd',
    'Riemann-MDM':      '#74c476',
    'Riemann-TS+SVM':   '#238b45',
    'EEGNet':           '#fd8d3c',
    'ShallowConvNet':   '#e6550d',
}

# Left: per-method balanced accuracy distribution
ax = axes[0]
data_list = [ok[ok['method']==m]['balanced_accuracy'].values for m in methods]
colors = [PALETTE.get(m, '#888888') for m in methods]
bp = ax.boxplot(data_list, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xticks(range(1, len(methods)+1))
ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Balanced Accuracy')
ax.set_title('Cross-Subject MI Performance (LOSO)')
ax.yaxis.grid(True, linestyle=':', alpha=0.6)

# Right: p-value heatmap
ax2 = axes[1]
p_plot = pval_matrix.astype(float).fillna(1.0)
mask = np.eye(len(methods), dtype=bool)
sns.heatmap(p_plot, annot=True, fmt='.3f', cmap='RdYlGn_r',
            vmin=0, vmax=0.1, ax=ax2, mask=mask, linewidths=0.5,
            cbar_kws={'label': 'p-value (Wilcoxon)'})
ax2.set_title('Wilcoxon Signed-Rank p-values\n(balanced accuracy)')

plt.tight_layout()
out = os.path.join(FIGS_DIR, 'fig7_stats_analysis.pdf')
plt.savefig(out, bbox_inches='tight')
plt.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=150)
plt.close()
print(f"\nSaved: {out}")

# ─── Summary stats for manuscript ────────────────────────────────────────────
print("\n── Summary for manuscript ──")
for m in methods:
    sub = ok[ok['method']==m]
    print(f"{m:20s}: bacc={sub['balanced_accuracy'].mean():.3f}±{sub['balanced_accuracy'].std():.3f}  "
          f"ece={sub['ece'].mean():.3f}±{sub['ece'].std():.3f}  "
          f"auroc={sub['auroc'].mean():.3f}±{sub['auroc'].std():.3f}  "
          f"brier={sub['brier_score'].mean():.3f}±{sub['brier_score'].std():.3f}")
