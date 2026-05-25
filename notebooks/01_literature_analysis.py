"""
Literature Analysis — EEG Foundation Models Review (2023-2026)
Produces structured taxonomy table and gap analysis.
Run as a script: python3 notebooks/01_literature_analysis.py
"""
import os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(ROOT, 'results', 'tables')
FIGS_DIR   = os.path.join(ROOT, 'results', 'figures')
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# ─── Load literature table ────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(TABLES_DIR, 'literature_table.csv'))
print(f"Loaded {len(df)} papers from review repository.")
print(df[['Paper','Year','Venue','Task','Architecture']].to_string(index=False))

# ─── Fig: Year distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

year_counts = df['Year'].value_counts().sort_index()
axes[0].bar(year_counts.index.astype(str), year_counts.values,
            color=['#9ecae1','#3182bd','#238b45','#fd8d3c'],
            alpha=0.85)
axes[0].set_title('EEG Foundation Model Papers by Year')
axes[0].set_ylabel('Count')
axes[0].yaxis.grid(True, linestyle=':', alpha=0.6)

# Task distribution
task_counts = df['Task'].value_counts()
axes[1].barh(task_counts.index[:10], task_counts.values[:10],
             color='#3182bd', alpha=0.8)
axes[1].set_title('Task Distribution')
axes[1].set_xlabel('Count')
axes[1].xaxis.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
out = os.path.join(FIGS_DIR, 'lit_year_task_distribution.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")

# ─── Summary stats ────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────")
print(f"Total papers: {len(df)}")
print(f"Papers with code: {(df['Code Available']=='Yes').sum()}")
print(f"Channel-agnostic: {(df['Channel-Agnostic']=='Yes').sum()}")
print(f"Subject-independent eval: {(df['Subject-Indep']=='Yes').sum()}")
print(f"Calibration reported: {(df['Calibration']=='Yes').sum()}")
print(f"\nGap: {(df['Calibration']=='No').sum()} / {len(df)} papers report NO calibration metrics!")

# Save enriched table
df.to_csv(os.path.join(TABLES_DIR, 'literature_table_final.csv'), index=False)
print("\nLiterature analysis complete.")
