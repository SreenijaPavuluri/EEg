"""
Merge classical/Riemannian checkpoint with neural per-method CSVs.
Generates final aggregate CSV, all figures, LaTeX table, and stats.
Run after all experiments complete.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR   = os.path.join(ROOT, 'results', 'logs')
TABLES_DIR = os.path.join(ROOT, 'results', 'tables')
FIGS_DIR   = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

METHOD_ORDER = ['LogVar+LDA','FBCSP+LDA','Riemann-MDM',
                'Riemann-TS+SVM','EEGNet','ShallowConvNet']
PALETTE = {
    'LogVar+LDA':      '#9ecae1', 'FBCSP+LDA':       '#3182bd',
    'Riemann-MDM':     '#74c476', 'Riemann-TS+SVM':  '#238b45',
    'EEGNet':          '#fd8d3c', 'ShallowConvNet':  '#e6550d',
}
FAMILY = {
    'LogVar+LDA':'Classical','FBCSP+LDA':'Classical',
    'Riemann-MDM':'Riemannian','Riemann-TS+SVM':'Riemannian',
    'EEGNet':'Neural','ShallowConvNet':'Neural',
}

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':200})

# ── 1. Merge all result sources ───────────────────────────────────────────────
def merge_all():
    frames = []
    ckpt = os.path.join(LOGS_DIR, 'loso_results_checkpoint.csv')
    if os.path.exists(ckpt):
        frames.append(pd.read_csv(ckpt))
    for fname in ['eegnet_results.csv', 'shallowconv_results.csv']:
        p = os.path.join(LOGS_DIR, fname)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("No result files found in results/logs/")
    df = pd.concat(frames, ignore_index=True)
    # Deduplicate: keep last OK entry per (subject, method)
    df = df.sort_values('status')         # 'OK' > 'FAILED...'
    df = df.drop_duplicates(subset=['subject','method'], keep='last')
    ok = df[df['status']=='OK'].copy()
    methods = [m for m in METHOD_ORDER if m in ok['method'].values]
    print(f"Merged: {len(ok)} OK rows, {len(methods)} methods: {methods}")
    return ok, methods, df

# ── 2. Aggregate stats ────────────────────────────────────────────────────────
def make_aggregate(ok, methods):
    num = ['accuracy','balanced_accuracy','f1_weighted','cohen_kappa',
           'ece','brier_score','auroc','time_s']
    agg = (ok.groupby('method')[num]
             .agg(['mean','std']).round(4))
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg = agg.reset_index()
    # order by METHOD_ORDER
    agg['_ord'] = agg['method'].apply(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    agg = agg.sort_values('_ord').drop(columns='_ord')
    agg.to_csv(os.path.join(TABLES_DIR,'aggregate_results.csv'), index=False)
    return agg

# ── 3. Wilcoxon tests ─────────────────────────────────────────────────────────
def wilcoxon_table(ok, methods):
    pvals = pd.DataFrame(np.nan, index=methods, columns=methods)
    for m1 in methods:
        for m2 in methods:
            if m1 == m2: continue
            common = (set(ok[ok['method']==m1]['subject'])
                    & set(ok[ok['method']==m2]['subject']))
            a = ok[(ok['method']==m1)&(ok['subject'].isin(common))]\
                .sort_values('subject')['balanced_accuracy'].values
            b = ok[(ok['method']==m2)&(ok['subject'].isin(common))]\
                .sort_values('subject')['balanced_accuracy'].values
            if len(a) > 4:
                try:
                    _, p = scipy_stats.wilcoxon(a, b, alternative='two-sided')
                    pvals.loc[m1,m2] = round(p, 4)
                except: pass
    pvals.to_csv(os.path.join(TABLES_DIR,'wilcoxon_pvalues.csv'))
    return pvals

# ── 4. Main comparison figure (3 panels) ─────────────────────────────────────
def fig_main(ok, methods):
    fig = plt.figure(figsize=(14, 4.5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)
    colors = [PALETTE[m] for m in methods]

    # A: balanced accuracy box
    ax1 = fig.add_subplot(gs[0])
    data = [ok[ok['method']==m]['balanced_accuracy'].values for m in methods]
    bp = ax1.boxplot(data, patch_artist=True,
                     medianprops=dict(color='black',linewidth=2),
                     flierprops=dict(marker='o',markersize=3,alpha=0.4))
    for patch,c in zip(bp['boxes'],colors):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax1.axhline(0.5,color='grey',linestyle='--',lw=1,alpha=0.6,label='Chance')
    ax1.set_xticks(range(1,len(methods)+1))
    ax1.set_xticklabels(methods, rotation=28, ha='right', fontsize=8)
    ax1.set_ylabel('Balanced Accuracy'); ax1.set_title('(A) Cross-Subject Accuracy')
    ax1.yaxis.grid(True,linestyle=':',alpha=0.5); ax1.set_ylim(0.30,1.0)
    ax1.legend(fontsize=8)

    # B: ECE bar
    ax2 = fig.add_subplot(gs[1])
    ece_m = [ok[ok['method']==m]['ece'].mean() for m in methods]
    ece_s = [ok[ok['method']==m]['ece'].std()  for m in methods]
    ax2.bar(range(len(methods)), ece_m, yerr=ece_s, color=colors, alpha=0.8,
            capsize=4, error_kw={'ecolor':'black','linewidth':1.5})
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=28, ha='right', fontsize=8)
    ax2.set_ylabel('Expected Calibration Error (ECE) ↓')
    ax2.set_title('(B) Calibration Quality')
    ax2.yaxis.grid(True,linestyle=':',alpha=0.5)

    # C: acc-ECE scatter
    ax3 = fig.add_subplot(gs[2])
    for m in methods:
        sub = ok[ok['method']==m]
        ax3.scatter(sub['balanced_accuracy'], sub['ece'],
                    color=PALETTE[m], alpha=0.4, s=20, zorder=3)
    for m in methods:
        sub = ok[ok['method']==m]
        ax3.scatter(sub['balanced_accuracy'].mean(), sub['ece'].mean(),
                    color=PALETTE[m], s=140, marker='D',
                    edgecolors='black', linewidths=1, zorder=6,
                    label=m)
    ax3.set_xlabel('Balanced Accuracy ↑'); ax3.set_ylabel('ECE ↓')
    ax3.set_title('(C) Accuracy–Calibration Tradeoff')
    ax3.legend(fontsize=7, ncol=1, loc='upper left',
               markerscale=0.8, handlelength=0.8)
    ax3.xaxis.grid(True,linestyle=':',alpha=0.4)
    ax3.yaxis.grid(True,linestyle=':',alpha=0.4)

    fig.suptitle(
        'Cross-Subject MI EEG Benchmark — LOSO (PhysioNet EEGMMIDB, 30 subjects)',
        y=1.01, fontsize=11, fontweight='bold')
    plt.tight_layout()
    for ext in ['.pdf','.png']:
        plt.savefig(os.path.join(FIGS_DIR,f'fig_main_comparison{ext}'),
                    bbox_inches='tight',dpi=200)
    plt.close(); print("Saved fig_main_comparison")

# ── 5. Heatmap ────────────────────────────────────────────────────────────────
def fig_heatmap(ok, methods):
    pivot = ok.pivot_table(index='subject', columns='method',
                           values='balanced_accuracy').round(3)
    cols = [m for m in methods if m in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(max(8, len(cols)*1.6), 9))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.35, vmax=0.90, linewidths=0.3,
                cbar_kws={'label':'Balanced Accuracy'}, ax=ax)
    ax.set_title('Per-Subject Balanced Accuracy (LOSO)', pad=10)
    ax.set_xlabel('Method'); ax.set_ylabel('Subject')
    plt.tight_layout()
    for ext in ['.pdf','.png']:
        plt.savefig(os.path.join(FIGS_DIR,f'fig_heatmap{ext}'),
                    bbox_inches='tight',dpi=200)
    plt.close(); print("Saved fig_heatmap")

# ── 6. Stats significance figure ─────────────────────────────────────────────
def fig_stats(ok, methods, pvals):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [PALETTE[m] for m in methods]

    # Cohen's kappa
    ax = axes[0]
    km = [ok[ok['method']==m]['cohen_kappa'].mean() for m in methods]
    ks = [ok[ok['method']==m]['cohen_kappa'].std()  for m in methods]
    ax.bar(range(len(methods)), km, yerr=ks, color=colors, alpha=0.8,
           capsize=4, error_kw={'ecolor':'black'})
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=28, ha='right', fontsize=8)
    ax.set_ylabel("Cohen's κ ↑"); ax.set_title("Cohen's Kappa (LOSO)")
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)

    # p-value heatmap
    ax2 = axes[1]
    p_plot = pvals.reindex(index=methods, columns=methods).astype(float).fillna(1.0)
    mask = np.eye(len(methods), dtype=bool)
    sns.heatmap(p_plot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                vmin=0, vmax=0.1, ax=ax2, mask=mask, linewidths=0.5,
                cbar_kws={'label':'p-value (Wilcoxon two-sided)'})
    ax2.set_title('Statistical Significance\n(balanced accuracy, Wilcoxon)')

    plt.tight_layout()
    for ext in ['.pdf','.png']:
        plt.savefig(os.path.join(FIGS_DIR,f'fig_stats{ext}'),
                    bbox_inches='tight',dpi=200)
    plt.close(); print("Saved fig_stats")

# ── 7. LaTeX table ────────────────────────────────────────────────────────────
def latex_table(ok, methods):
    rows = []
    for m in methods:
        sub = ok[ok['method']==m]
        rows.append({
            'Family': FAMILY.get(m,'?'),
            'Method': m,
            'Bal.~Acc.': f"{sub['balanced_accuracy'].mean():.3f}$\\pm${sub['balanced_accuracy'].std():.3f}",
            '$\\kappa$':  f"{sub['cohen_kappa'].mean():.3f}$\\pm${sub['cohen_kappa'].std():.3f}",
            'AUROC':     f"{sub['auroc'].mean():.3f}$\\pm${sub['auroc'].std():.3f}",
            'ECE$\\downarrow$':   f"{sub['ece'].mean():.3f}$\\pm${sub['ece'].std():.3f}",
            'Brier$\\downarrow$': f"{sub['brier_score'].mean():.3f}$\\pm${sub['brier_score'].std():.3f}",
            'Time(s)':   f"{sub['time_s'].mean():.0f}",
        })
    tab = pd.DataFrame(rows)
    lines = [
        r'\begin{table}[!t]',
        r'\caption{LOSO cross-subject MI results on PhysioNet EEGMMIDB (30 subjects, 45 trials/subject). '
        r'Mean\,$\pm$\,std across subjects. '
        r'Best value per metric in \textbf{bold}. '
        r'Neural methods: 10 epochs, batch\,256. '
        r'Riemannian/Classical: no training.}',
        r'\label{tab:results}',
        r'\centering\small',
        r'\begin{tabular}{llcccccr}',
        r'\toprule',
        r'Family & Method & Bal.~Acc. & $\kappa$ & AUROC & ECE$\downarrow$ & Brier$\downarrow$ & Time(s) \\',
        r'\midrule',
    ]
    for _, r in tab.iterrows():
        lines.append(' & '.join([r['Family'], r['Method'], r['Bal.~Acc.'],
                                  r['$\\kappa$'], r['AUROC'],
                                  r['ECE$\\downarrow$'], r['Brier$\\downarrow$'],
                                  r['Time(s)']]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tex = '\n'.join(lines)
    out = os.path.join(TABLES_DIR, 'table_results_final.tex')
    with open(out,'w') as f: f.write(tex)
    tab.to_csv(os.path.join(TABLES_DIR,'table_results_final.csv'), index=False)
    print(f"LaTeX table → {out}")
    print(tab.to_string(index=False))

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ok, methods, df_all = merge_all()
    agg  = make_aggregate(ok, methods)
    pvals = wilcoxon_table(ok, methods)

    fig_main(ok, methods)
    fig_heatmap(ok, methods)
    fig_stats(ok, methods, pvals)
    latex_table(ok, methods)

    pivot = ok.pivot_table(index='subject', columns='method',
                           values='balanced_accuracy').round(4)
    pivot.to_csv(os.path.join(TABLES_DIR,'per_subject_accuracy.csv'))

    # Save merged full results
    df_all.to_csv(os.path.join(LOGS_DIR,'loso_results_full.csv'), index=False)

    print("\n── FINAL SUMMARY ──────────────────────────────────────")
    for m in methods:
        sub = ok[ok['method']==m]
        print(f"  {m:<20}: bacc={sub['balanced_accuracy'].mean():.3f}±{sub['balanced_accuracy'].std():.3f}"
              f"  ece={sub['ece'].mean():.3f}  auroc={sub['auroc'].mean():.3f}"
              f"  κ={sub['cohen_kappa'].mean():.3f}  brier={sub['brier_score'].mean():.3f}")
    print("\nWilcoxon p-values (balanced accuracy):")
    print(pvals.to_string())
    print("\nAll done.")
