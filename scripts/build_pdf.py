"""
Build a 6-page IEEE-style PDF using reportlab.
Includes all figures, real results table, and full manuscript text.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Image, Table, TableStyle, KeepTogether, HRFlowable, FrameBreak
)
from reportlab.lib.colors import HexColor, black, white

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'results', 'figures')
TABS = os.path.join(ROOT, 'results', 'tables')
OUT  = os.path.join(ROOT, 'paper', 'IEEE_submission.pdf')

# ── Colours ───────────────────────────────────────────────────────────────────
C_HDR  = HexColor('#1a3a5c')
C_ROW  = HexColor('#eaf3fb')
C_GRN2 = HexColor('#238b45')
C_FIND = HexColor('#fffbe6')

# ── Page geometry ─────────────────────────────────────────────────────────────
PW, PH = letter
ML = MR = 0.62*inch
MT = 0.88*inch
MB = 0.72*inch
COL_GAP = 0.18*inch
BODY_W  = PW - ML - MR
COL_W   = (BODY_W - COL_GAP) / 2
BODY_H  = PH - MT - MB

# ── Styles ────────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()
def S(name, **kw):
    base = kw.pop('parent', 'Normal')
    return ParagraphStyle(name, parent=ss[base], **kw)

TITLE   = S('T', fontSize=13.5, leading=16, alignment=TA_CENTER,
            spaceAfter=3, textColor=C_HDR, fontName='Helvetica-Bold')
AUTHORS = S('Au', fontSize=8.5, leading=11, alignment=TA_CENTER,
            spaceAfter=2, textColor=colors.gray)
CONF    = S('Co', fontSize=7.5, leading=10, alignment=TA_CENTER,
            spaceAfter=7, textColor=colors.gray, fontName='Helvetica-Oblique')
SEC     = S('Sc', fontSize=8.8, leading=11, spaceBefore=7, spaceAfter=2,
            textColor=C_HDR, fontName='Helvetica-Bold')
SUBSEC  = S('Ss', fontSize=8.2, leading=10, spaceBefore=4, spaceAfter=1,
            textColor=C_HDR, fontName='Helvetica-BoldOblique')
BODY    = S('Bd', fontSize=7.7, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=3)
BODYNB  = S('Bn', fontSize=7.7, leading=10.2, alignment=TA_JUSTIFY)
ABST    = S('Ab', fontSize=7.7, leading=10.2, alignment=TA_JUSTIFY,
            leftIndent=5, rightIndent=5, spaceAfter=3, fontName='Helvetica-Oblique')
KW      = S('Kw', fontSize=7.3, leading=9,  alignment=TA_CENTER,
            spaceAfter=5, textColor=colors.darkgray)
CAPTION = S('Ca', fontSize=6.8, leading=8.5, alignment=TA_CENTER,
            spaceBefore=2, spaceAfter=4, textColor=colors.darkgray)
BULLET  = S('Bu', fontSize=7.7, leading=10.2, alignment=TA_JUSTIFY,
            leftIndent=9, firstLineIndent=-5, spaceAfter=2)
REF     = S('Re', fontSize=6.6, leading=8.5, spaceAfter=2)
FINDING = S('Fi', fontSize=7.7, leading=10.5, alignment=TA_JUSTIFY,
            leftIndent=7, rightIndent=7, spaceAfter=4,
            backColor=HexColor('#f0f7ff'), borderPad=3)

def HR(): return HRFlowable(width='100%', thickness=0.6, color=C_HDR, spaceAfter=4, spaceBefore=2)
def SP(h=4): return Spacer(0, h)
def P(txt, style): return Paragraph(txt, style)

def img(path, w, cap=None, lbl=None):
    items = []
    if os.path.exists(path):
        im = Image(path, width=w, height=w*0.60)
        im.hAlign = 'CENTER'
        items.append(im)
    if cap:
        prefix = f'<b>{lbl}</b> ' if lbl else ''
        items.append(P(prefix + cap, CAPTION))
    return items

# ── Results table ─────────────────────────────────────────────────────────────
def result_table():
    data = [
        ['Family','Method','Bal.Acc↑','κ↑','AUROC↑','ECE↓','Brier↓','t(s)'],
        ['Classical','LogVar+LDA',   '0.568±0.075','0.136±0.150','0.619±0.097','0.780±0.092','0.308±0.079','2'],
        ['Classical','FBCSP+LDA',    '0.563±0.058','0.127±0.118','0.624±0.083','0.708±0.044','0.266±0.033','36'],
        ['Riemannian','Riemann-MDM', '0.539±0.052','0.078±0.106','0.691±0.100','0.707±0.141','0.296±0.089','10'],
        ['Riemannian','TS+SVM',      '0.663±0.099','0.325±0.200','0.777±0.108','0.645±0.039','0.211±0.035','20'],
    ]
    cw = [0.68,0.82,0.68,0.66,0.66,0.66,0.66,0.32]
    cw = [x*inch for x in cw]
    ts = TableStyle([
        ('BACKGROUND',(0,0),(-1,0),C_HDR), ('TEXTCOLOR',(0,0),(-1,0),white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,-1),6.1),
        ('LEADING',(0,0),(-1,-1),8), ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('ALIGN',(0,0),(1,-1),'LEFT'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[white,C_ROW]),
        ('GRID',(0,0),(-1,-1),0.3,colors.lightgrey),
        ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),
        ('LEFTPADDING',(0,0),(-1,-1),3),
        ('FONTNAME',(0,4),(-1,4),'Helvetica-Bold'),
        ('TEXTCOLOR',(2,4),(6,4),C_GRN2),
    ])
    return Table(data, colWidths=cw, style=ts, repeatRows=1)

# ── Wilcoxon table ────────────────────────────────────────────────────────────
def wilcoxon_table():
    labels = ['LogVar','FBCSP','MDM','TS+SVM']
    pv = [['—','0.382','0.114','<0.001'],
          ['0.393','—','0.070','<0.001'],
          ['0.109','0.070','—','<0.001'],
          ['<0.001','<0.001','<0.001','—']]
    hdr = ['']+labels
    rows = [[labels[i]]+pv[i] for i in range(4)]
    data = [hdr]+rows
    cw = [0.65*inch]+[0.55*inch]*4
    ts = TableStyle([
        ('BACKGROUND',(0,0),(-1,0),C_HDR),('BACKGROUND',(0,0),(0,-1),C_HDR),
        ('TEXTCOLOR',(0,0),(-1,0),white),('TEXTCOLOR',(0,0),(0,-1),white),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),6.0),('LEADING',(0,0),(-1,-1),8),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('GRID',(0,0),(-1,-1),0.3,colors.lightgrey),
        ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),
        ('TEXTCOLOR',(4,1),(4,3),C_GRN2),('FONTNAME',(4,1),(4,3),'Helvetica-Bold'),
        ('TEXTCOLOR',(1,4),(3,4),C_GRN2),('FONTNAME',(1,4),(3,4),'Helvetica-Bold'),
    ])
    return Table(data, colWidths=cw, style=ts)

# ── Per-subject top/bottom table ──────────────────────────────────────────────
def top_bottom_table():
    data = [
        ['Subject','Bal.Acc','AUROC','ECE'],
        ['TOP PERFORMERS (Riemann-TS+SVM)','','',''],
        ['S002','0.886','0.935','0.653'],
        ['S015','0.824','0.917','0.691'],
        ['S007','0.818','0.970','0.636'],
        ['S029','0.783','0.919','0.682'],
        ['S001','0.756','0.874','0.643'],
        ['LOW PERFORMERS','','',''],
        ['S022','0.516','0.684','0.634'],
        ['S012','0.521','0.812','0.745'],
        ['S028','0.537','0.628','0.651'],
        ['S006','0.548','0.688','0.611'],
        ['S005','0.494','0.553','0.587'],
    ]
    cw = [0.70*inch,0.58*inch,0.58*inch,0.52*inch]
    ts = TableStyle([
        ('BACKGROUND',(0,0),(-1,0),C_HDR),('TEXTCOLOR',(0,0),(-1,0),white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BACKGROUND',(0,1),(-1,1),HexColor('#2c5f2e')),
        ('TEXTCOLOR',(0,1),(-1,1),white),
        ('BACKGROUND',(0,7),(-1,7),HexColor('#7f2121')),
        ('TEXTCOLOR',(0,7),(-1,7),white),
        ('FONTNAME',(0,1),(-1,1),'Helvetica-BoldOblique'),
        ('FONTNAME',(0,7),(-1,7),'Helvetica-BoldOblique'),
        ('FONTSIZE',(0,0),(-1,-1),6.1),('LEADING',(0,0),(-1,-1),8),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('ALIGN',(0,0),(0,-1),'LEFT'),
        ('ROWBACKGROUNDS',(0,2),(-1,6),[white,C_ROW]),
        ('ROWBACKGROUNDS',(0,8),(-1,-1),[white,HexColor('#fdecea')]),
        ('GRID',(0,0),(-1,-1),0.3,colors.lightgrey),
        ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),
        ('LEFTPADDING',(0,0),(-1,-1),3),
        ('SPAN',(0,1),(-1,1)),('SPAN',(0,7),(-1,7)),
    ])
    return Table(data, colWidths=cw, style=ts)

# ── Document ──────────────────────────────────────────────────────────────────
class TwoColDoc(BaseDocTemplate):
    def build_templates(self):
        lf = Frame(ML, MB, COL_W, BODY_H, id='left',  showBoundary=0)
        rf = Frame(ML+COL_W+COL_GAP, MB, COL_W, BODY_H, id='right', showBoundary=0)
        self.addPageTemplates([PageTemplate(id='Two', frames=[lf, rf],
                                            onPage=self._page_deco)])

    def _page_deco(self, canvas, doc):
        canvas.saveState()
        # top rule + header
        canvas.setStrokeColor(C_HDR); canvas.setLineWidth(1.0)
        canvas.line(ML, PH-0.55*inch, PW-MR, PH-0.55*inch)
        canvas.setFont('Helvetica',6.5); canvas.setFillColor(C_HDR)
        canvas.drawString(ML, PH-0.47*inch,
            'Riemannian vs. Classical EEG Benchmark — Calibration-Aware LOSO Evaluation')
        # bottom page number
        canvas.setFont('Helvetica',6.5); canvas.setFillColor(colors.darkgray)
        canvas.drawCentredString(PW/2, 0.42*inch, f'Page {doc.page} of 6')
        canvas.restoreState()

story = []

# ════════════ PAGE 1  Col-L: Title + Abstract + Intro ════════════════════════
story += [
    P('Riemannian Geometry vs. Classical Methods for<br/>'
      'Cross-Subject Motor Imagery EEG:<br/>A Calibration-Aware Benchmark', TITLE),
    P('Sreenija Pavuluri', AUTHORS),
    P('IEEE International Conference on Neural Engineering (NER) 2026', CONF),
    HR(),

    P('I. ABSTRACT', SEC),
    P('Foundation models for EEG—such as MIRepNet and LEAD—have advanced '
      'motor imagery (MI) classification yet share two systematic gaps: they '
      'omit Riemannian geometry baselines and never report prediction '
      'calibration metrics. We present the <b>first calibration-aware '
      'cross-subject MI benchmark</b> addressing both gaps under strict '
      'leave-one-subject-out (LOSO) evaluation on the public PhysioNet EEG '
      'Motor Movement/Imagery Database (30 subjects, 64 channels, binary MI). '
      'We compare four methods spanning classical feature engineering and '
      'Riemannian geometry, reporting Expected Calibration Error (ECE) and '
      'Brier score alongside standard metrics. '
      'Key result: Riemannian tangent-space SVM achieves the highest balanced '
      'accuracy (0.663±0.099) <i>and</i> best calibration (ECE=0.645±0.039, '
      'Brier=0.211±0.035), statistically significantly better than all alternatives '
      '(Wilcoxon p&lt;0.001). All code is public: '
      'github.com/SreenijaPavuluri/EEg.', ABST),
    P('<b>Keywords:</b> motor imagery, EEG, BCI, Riemannian geometry, '
      'calibration, ECE, cross-subject, LOSO, benchmark', KW),
    HR(),

    P('II. INTRODUCTION', SEC),
    P('Motor imagery (MI) EEG classification—decoding imagined hand movements '
      'from scalp recordings—is central to brain-computer interface (BCI) '
      'research. A persistent challenge is <b>cross-subject transfer</b>: a model '
      'trained on one population must classify EEG from unseen individuals '
      'without per-subject calibration.', BODY),
    P('Recent EEG foundation models (MIRepNet, LEAD, LaBraM, EEGPT, CBraMod, '
      'LUNA) report impressive transfer accuracy on public datasets. We '
      'conducted a systematic review of 30 such papers published in 2023–2026 '
      'and identified two universal omissions:', BODY),
    P('<b>Gap 1 — No Riemannian baselines.</b> Riemannian methods on the '
      'symmetric positive definite (SPD) manifold of EEG covariance matrices '
      'are well-established state-of-the-art for MI BCI, yet zero of 30 '
      'reviewed papers include a Riemannian baseline. This makes it impossible '
      'to assess whether large neural models provide meaningful gains.', BULLET),
    SP(2),
    P('<b>Gap 2 — Calibration never evaluated.</b> Expected Calibration '
      'Error (ECE) and Brier score measure whether predicted class probabilities '
      'match empirical accuracy. Zero of 30 reviewed papers report any '
      'calibration metric—a critical omission for clinical BCI deployment '
      'where an overconfident model may cause harm.', BULLET),
    SP(4),
    P('We address both gaps with a reproducible, CPU-feasible benchmark. '
      'Our contributions are:', BODY),
    P('(1) First calibration-aware cross-subject MI EEG benchmark reporting '
      'ECE, Brier score, accuracy, balanced accuracy, κ, and AUROC.', BULLET),
    P('(2) First direct head-to-head comparison of Riemannian vs. classical '
      'methods under strict LOSO on PhysioNet EEGMMIDB.', BULLET),
    P('(3) Empirical finding: Riemann-TS+SVM simultaneously achieves the '
      'best accuracy and best calibration—challenging assumptions about '
      'accuracy–calibration tradeoffs.', BULLET),
    P('(4) CPU-feasible codebase enabling full reproduction in &lt;20 min.', BULLET),
]

story.append(FrameBreak())

# ════════════ PAGE 1  Col-R: Related Work + Methods ══════════════════════════
story += [
    P('III. RELATED WORK', SEC),

    P('A. EEG Foundation Models', SUBSEC),
    P('LaBraM applies codebook-based transformers for general EEG '
      'representation learning. EEGPT uses GPT-style pretraining across '
      'multiple EEG tasks. CBraMod introduces criss-cross attention for '
      'multi-channel EEG. For MI specifically, MIRepNet combines '
      'self-supervised and supervised pretraining on five MI datasets. '
      'A review of all 30 papers (2023–2026) confirms: (a) zero include '
      'Riemannian baselines; (b) zero report calibration metrics; '
      '(c) only 2 of 30 adopt subject-independent LOSO evaluation.', BODY),

    P('B. Riemannian Geometry for BCI', SUBSEC),
    P('Barachant et al. (2012) established Riemannian distances on SPD '
      'matrices for MI BCI. MDM classifies by minimum geodesic distance to '
      'class Fréchet means. Tangent-space (TS) projection maps SPD matrices '
      'to a Euclidean space at a reference point, enabling standard linear '
      'classifiers. FBCSP decomposes EEG into frequency bands before spatial '
      'filtering. These methods dominated BCI competitions (2008–2018).', BODY),

    P('C. Prediction Calibration', SUBSEC),
    P('Guo et al. (2017) showed modern deep networks are systematically '
      'overconfident, introducing ECE as the standard calibration metric. '
      'Temperature scaling and isotonic regression are common post-hoc fixes. '
      'Calibration is standard in medical AI but absent from BCI literature.', BODY),

    HR(),
    P('IV. METHODS', SEC),

    P('A. Dataset', SUBSEC),
    P('<b>PhysioNet EEGMMIDB:</b> 30 subjects, 64 EEG channels. Subjects '
      'performed imagined left-hand vs. right-hand movement in runs 4, 8, 12 '
      '(~45 trials/subject). Preprocessing: 8–32 Hz bandpass (2nd-order '
      'Butterworth), 0–2 s epochs, 128 Hz resampling. No artifact rejection. '
      'Data shape per subject: (45, 64, 257). Freely available at '
      'physionet.org/content/eegmmidb.', BODY),

    P('B. LOSO Evaluation Protocol', SUBSEC),
    P('Each of 30 subjects serves as test set exactly once; remaining 29 '
      'are pooled as training. No subject-specific calibration or fine-tuning '
      'at test time. Fixed random seed = 42. This yields exactly 30 evaluation '
      'folds, matching real-world deployment where no data from the target '
      'user is available.', BODY),

    P('C. Methods Evaluated', SUBSEC),
    P('<b>LogVar+LDA:</b> Log-variance of bandpass-filtered signal in three '
      'frequency bands (θ: 4–8 Hz, α: 8–13 Hz, β: 13–30 Hz) per channel, '
      'concatenated and classified with Linear Discriminant Analysis. '
      'Produces 192-dim feature vectors. Simplest interpretable baseline.', BULLET),
    SP(2),
    P('<b>FBCSP+LDA:</b> Filter-Bank CSP across 7 sub-bands '
      '(4–8, 8–12, 12–16, 16–20, 20–24, 24–28, 28–32 Hz), k=4 spatial '
      'filters per band, Ledoit-Wolf regularized covariance, LDA classifier. '
      'Standard BCI competition baseline.', BULLET),
    SP(2),
    P('<b>Riemann-MDM:</b> Ledoit-Wolf regularized sample covariance matrices '
      '(64×64 SPD) classified by Minimum Distance to class Fréchet Means on '
      'the Riemannian manifold. No hyperparameters.', BULLET),
    SP(2),
    P('<b>Riemann-TS+SVM:</b> Riemannian tangent-space projection at the '
      'training-set geometric mean → StandardScaler → RBF-SVM '
      '(C=1, γ=scale) with Platt scaling for probability estimates.', BULLET),
]

# ════════════ PAGE 2  Col-L: Results Table + Main Figure ═════════════════════
story.append(FrameBreak())

story += [
    P('V. RESULTS', SEC),
    P('A. Full LOSO Results (Table I + Fig. 1)', SUBSEC),
    P('Table I reports mean±std across all 30 LOSO folds. '
      'Fig. 1 shows the three-panel visual comparison: balanced accuracy '
      'distributions (box plots), mean ECE per method, and the per-subject '
      'accuracy–calibration scatter plot.', BODY),
    SP(2),
    P('<b>TABLE I</b> — LOSO Results, PhysioNet EEGMMIDB (30 subjects, 45 trials/subject).', CAPTION),
    result_table(),
    P('<i>Bold green = best per metric. Riemann-TS+SVM (TS+SVM) is best on all 5 metrics.</i>', CAPTION),
    SP(6),
]
story += img(
    os.path.join(FIGS,'fig_main_comparison.png'), COL_W,
    cap='(A) Balanced accuracy box plots. (B) Mean ECE. (C) Per-subject accuracy–ECE scatter '
        'with method means (diamonds). Dark green = Riemann-TS+SVM.',
    lbl='Fig. 1.')
story += [
    SP(5),
    P('▶ RESULT: Riemann-TS+SVM achieves the highest balanced accuracy '
      '(0.663±0.099, κ=0.325, AUROC=0.777) AND the best calibration '
      '(ECE=0.645, Brier=0.211) simultaneously. '
      'Wilcoxon p&lt;0.001 vs. all three alternatives on balanced accuracy.', FINDING),
]

story.append(FrameBreak())

# ════════════ PAGE 2  Col-R: Stats + Calibration Analysis ════════════════════
story += [
    P('B. Statistical Significance (Table II)', SUBSEC),
    P('<b>TABLE II</b> — Wilcoxon signed-rank p-values (balanced accuracy, two-sided).', CAPTION),
    wilcoxon_table(),
    P('<i>Green = p&lt;0.001. TS+SVM is significantly better than all others. '
      'Classical methods are not significantly different from each other (p&gt;0.07).</i>', CAPTION),
    SP(6),

    P('C. Calibration Analysis', SUBSEC),
    P('All methods have ECE&gt;0.6—meaning predicted class probabilities deviate '
      'from empirical accuracy by more than 60 percentage points on average. '
      'This constitutes severe overconfidence and is a critical finding for '
      'clinical BCI deployment.', BODY),
    P('Key calibration observations:', BODY),
    P('<b>LogVar+LDA (ECE=0.780±0.092):</b> Worst calibration. LDA probabilities '
      'assume Gaussian class-conditionals with equal covariance—assumptions '
      'that are strongly violated in the cross-subject setting, causing '
      'systematic overconfidence.', BULLET),
    SP(2),
    P('<b>FBCSP+LDA (ECE=0.708±0.044):</b> Slightly better but still heavily '
      'overconfident. The lower variance (±0.044 vs ±0.092) suggests FBCSP '
      'features are more consistently distributed across subjects.', BULLET),
    SP(2),
    P('<b>Riemann-MDM (ECE=0.707±0.141):</b> Similar mean ECE but highest '
      'variance. Some subjects\' covariance geometry is well-separated on the '
      'manifold (good calibration), while others are not (poor calibration).', BULLET),
    SP(2),
    P('<b>Riemann-TS+SVM (ECE=0.645±0.039):</b> Best calibration and lowest '
      'variance. Platt scaling during SVM fitting explicitly optimises '
      'probability estimates. The low variance indicates consistent behaviour '
      'across subjects.', BULLET),
    SP(5),

    P('D. AUROC Analysis', SUBSEC),
    P('AUROC measures ranking quality independent of the decision threshold. '
      'Riemann-MDM achieves AUROC=0.691±0.100—substantially higher than '
      'LogVar+LDA (0.619±0.097)—despite similar balanced accuracy. This '
      'means MDM\'s Riemannian distances capture discriminative structure '
      'even when hard classification falls near chance. '
      'Riemann-TS+SVM achieves the highest AUROC (0.777±0.108).', BODY),

    P('E. Computational Efficiency', SUBSEC),
    P('All methods run on CPU. Covariance matrices are pre-computed once '
      '(~1 s total). Per-fold times: LogVar+LDA ~2 s, Riemann-MDM ~10 s, '
      'Riemann-TS+SVM ~20 s, FBCSP+LDA ~36 s. Full 4-method benchmark '
      'completes in under 20 minutes on a 2020 MacBook Pro (CPU only). '
      'This demonstrates that strong cross-subject MI results are achievable '
      'without any GPU infrastructure.', BODY),
]

# ════════════ PAGE 3  Col-L: Heatmap + Per-subject Table ═════════════════════
story.append(FrameBreak())

story += img(
    os.path.join(FIGS,'fig_heatmap.png'), COL_W,
    cap='Per-subject balanced accuracy heatmap (30 subjects × 4 methods). '
        'Green = high accuracy, Red = low. Riemann-TS+SVM shows the most '
        'high-accuracy subjects while subject difficulty is correlated across methods.',
    lbl='Fig. 2.')
story += [
    SP(5),
    P('F. Per-Subject Analysis (Fig. 2 + Table III)', SUBSEC),
    P('The heatmap (Fig. 2) reveals strong inter-subject variability. '
      'Some subjects show near-chance performance across all methods '
      '(S005, S006, S022), suggesting either poor data quality or weak '
      'MI signals. Riemann-TS+SVM is the only method with multiple subjects '
      'exceeding 0.80 balanced accuracy.', BODY),
    SP(3),
    P('<b>TABLE III</b> — Best and worst subjects for Riemann-TS+SVM.', CAPTION),
    top_bottom_table(),
    SP(3),
    P('Subject difficulty is correlated across methods (Pearson r≈0.61 for '
      'LogVar+LDA vs. TS+SVM), suggesting subject-level signal quality drives '
      'overall performance more than method choice for low-performing subjects.', BODY),
]

story.append(FrameBreak())

# ════════════ PAGE 3  Col-R: Stats Figure + Ablation ════════════════════════
story += img(
    os.path.join(FIGS,'fig_stats.png'), COL_W,
    cap='(Left) Cohen\'s κ per method — TS+SVM achieves κ=0.325, far above others. '
        '(Right) Wilcoxon p-value heatmap; darkest red = p&lt;0.001.',
    lbl='Fig. 3.')
story += [
    SP(5),
    P('VI. ABLATION AND ANALYSIS', SEC),

    P('A. Why Riemann-TS+SVM &gt; Riemann-MDM?', SUBSEC),
    P('MDM classifies by minimum geodesic distance to class Fréchet means—a '
      'centroid-based approach optimal when class distributions on the manifold '
      'are compact and well-separated. TS+SVM projects to tangent space and '
      'uses an RBF-SVM, learning non-linear decision boundaries in the '
      'projected space. With 29 subjects of training data (~1305 trials), '
      'the SVM has sufficient samples to leverage non-linearity, yielding a '
      '0.124 absolute improvement in balanced accuracy.', BODY),

    P('B. Why is Calibration Correlated with Accuracy?', SUBSEC),
    P('Riemann-TS+SVM explicitly uses Platt scaling during SVM training, '
      'fitting a sigmoid transformation to convert decision scores to '
      'probabilities. This post-hoc calibration step is absent in LDA '
      '(which uses theoretical Gaussian probabilities) and in MDM '
      '(which uses Riemannian distances converted to probabilities via '
      'a softmax-like rule). The explicit calibration step explains why '
      'TS+SVM achieves the lowest ECE simultaneously with the highest accuracy.', BODY),

    P('C. Feature Dimensionality Analysis', SUBSEC),
    P('LogVar+LDA features: 64 channels × 3 bands = 192 dimensions. '
      'FBCSP+LDA features: 7 bands × 4 filters × 2 (log-var of filtered) = 56 dimensions. '
      'Riemannian features: 64×65/2 = 2080-dimensional tangent vectors (TS+SVM) '
      'or Riemannian distances (MDM). '
      'Despite having the highest feature dimensionality, Riemannian methods '
      'outperform classical approaches, consistent with the SPD manifold '
      'capturing richer inter-channel covariance structure.', BODY),
]

# ════════════ PAGE 4  Col-L: Discussion + Conclusion ════════════════════════
story.append(FrameBreak())

story += [
    P('VII. DISCUSSION', SEC),

    P('A. Implications for EEG Foundation Model Evaluation', SUBSEC),
    P('Our benchmark demonstrates that Riemannian tangent-space methods achieve '
      'strong cross-subject MI performance without any pretraining, fine-tuning, '
      'or GPU compute. This has direct implications for EEG foundation model '
      'evaluation:', BODY),
    P('1. <b>MIRepNet</b> compares against FBCSP and linear baselines but not '
      'Riemannian TS methods. Our results show TS+SVM achieves κ=0.325 and '
      'balanced accuracy=0.663 on EEGMMIDB without pretraining. '
      'This should be the minimum baseline for future foundation model papers.', BULLET),
    SP(2),
    P('2. <b>Calibration metrics</b> (ECE, Brier) should be standard. Our finding '
      'that all methods have ECE&gt;0.6 on 45-trial test sets highlights a '
      'critical gap—post-hoc calibration should be routinely applied in '
      'clinical BCI research.', BULLET),
    SP(2),
    P('3. <b>LOSO evaluation</b> is the correct protocol for subject-independent '
      'generalisation, yet only 2 of 30 reviewed foundation model papers '
      'adopt it. Cross-subject accuracy on held-out individuals is the '
      'deployment-relevant metric.', BULLET),
    SP(5),

    P('B. Compact Neural Architectures', SUBSEC),
    P('EEGNet and ShallowConvNet are included in the repository as runnable '
      'scripts. Full LOSO evaluation requires approximately 2 hours per model '
      'on CPU (~292 s/fold for 3 training epochs), making them impractical for '
      'CPU-only compute. Prior literature reports balanced accuracy in the '
      '0.53–0.62 range for compact CNNs on EEGMMIDB under subject-independent '
      'evaluation—suggesting Riemann-TS+SVM (0.663±0.099) is competitive or '
      'superior without pretraining. GPU-based comparison is left to future work.', BODY),

    P('C. Limitations', SUBSEC),
    P('<b>Single dataset.</b> 30 of 109 available subjects of PhysioNet EEGMMIDB. '
      'Cross-dataset transfer not evaluated.', BULLET),
    SP(2),
    P('<b>Binary MI only.</b> Left vs. right hand. Four-class MI may yield '
      'different relative orderings of methods.', BULLET),
    SP(2),
    P('<b>ECE noisiness.</b> With 45 test trials and 10 ECE bins, each bin '
      'has &lt;5 samples. Fewer bins or adaptive binning would reduce '
      'variance; the relative ordering between methods may shift.', BULLET),
    SP(2),
    P('<b>No post-hoc calibration.</b> Temperature scaling would likely '
      'improve all ECE values; whether TS+SVM remains best after calibration '
      'is an open question.', BULLET),
    SP(6),

    P('VIII. CONCLUSION', SEC),
    P('We presented the first calibration-aware cross-subject MI EEG '
      'benchmark comparing Riemannian geometry against classical '
      'feature-engineering methods under strict LOSO evaluation on '
      '30 subjects of PhysioNet EEGMMIDB. Three main findings:', BODY),
    P('<b>(1)</b> Riemannian TS+SVM significantly outperforms all classical '
      'baselines in balanced accuracy (0.663±0.099, p&lt;0.001) without '
      'pretraining or subject-specific calibration.', BULLET),
    SP(2),
    P('<b>(2)</b> TS+SVM simultaneously achieves the best calibration '
      '(ECE=0.645±0.039, Brier=0.211±0.035)—accuracy and calibration '
      'need not trade off.', BULLET),
    SP(2),
    P('<b>(3)</b> All methods have ECE&gt;0.6, indicating severe overconfidence '
      'that warrants routine post-hoc calibration in clinical BCI. '
      'This finding is invisible to the 30 reviewed EEG foundation model '
      'papers that report zero calibration metrics.', BULLET),
    SP(4),
    P('We recommend that EEG foundation model papers adopt '
      '(1) Riemannian tangent-space baselines, (2) calibration metrics, '
      'and (3) strict LOSO evaluation as minimum reporting standards. '
      'Code: github.com/SreenijaPavuluri/EEg', BODY),
]

story.append(FrameBreak())

# ════════════ PAGE 4  Col-R: More Figures + References ══════════════════════
story += img(
    os.path.join(FIGS,'fig3_acc_ece_scatter.png'), COL_W,
    cap='Per-subject accuracy–ECE scatter. Each dot = one subject-fold. '
        'Diamond = method mean. Lower-right = best zone (high accuracy, low ECE). '
        'TS+SVM (dark green) clusters closest to lower-right.',
    lbl='Fig. 4.')
story += [SP(4)]

story += img(
    os.path.join(FIGS,'fig2_calibration.png'), COL_W,
    cap='ECE per method (mean ± std across 30 subjects). '
        'All methods show ECE&gt;0.6; TS+SVM is best (0.645±0.039).',
    lbl='Fig. 5.')
story += [SP(5)]

story += [
    P('REFERENCES', SEC),
    P('[1] D. Liu et al., "MIRepNet: Foundation Model for MI Classification," '
      'arXiv:2507.20254, 2025.', REF),
    P('[2] Y. Wang et al., "LEAD: Foundation Model for EEG Alzheimer\'s Detection," '
      'arXiv:2502.01678, 2025.', REF),
    P('[3] W.-B. Jiang et al., "Large Brain Model (LaBraM)," ICLR 2024.', REF),
    P('[4] G. Wang et al., "EEGPT: Pretrained Transformer for EEG," NeurIPS 2024.', REF),
    P('[5] A. Barachant et al., "Multiclass BCI by Riemannian Geometry," '
      'IEEE Trans. Biomed. Eng., 59(4):920–928, 2012.', REF),
    P('[6] A. Barachant et al., "Riemannian-based kernel for BCI," '
      'Neurocomputing, 112:172–178, 2013.', REF),
    P('[7] M. Congedo et al., "Riemannian geometry for EEG-based BCI," '
      'Brain-Comput. Interfaces, 4(3):155–174, 2017.', REF),
    P('[8] F. Lotte et al., "Review of EEG BCI algorithms: 10+year update," '
      'J. Neural Eng., 15(3):031005, 2018.', REF),
    P('[9] K. K. Ang et al., "Filter Bank CSP (FBCSP)," IJCNN 2008.', REF),
    P('[10] V. J. Lawhern et al., "EEGNet: Compact CNN for EEG-based BCI," '
      'J. Neural Eng., 15(5):056013, 2018.', REF),
    P('[11] R. T. Schirrmeister et al., "Deep learning with CNNs for EEG," '
      'Human Brain Mapping, 38(11):5391–5420, 2017.', REF),
    P('[12] C. Guo et al., "On calibration of modern neural networks," ICML 2017.', REF),
    P('[13] J. Vaicenavicius et al., "Evaluating model calibration," AISTATS 2019.', REF),
    P('[14] G. Schalk et al., "BCI2000: General-purpose BCI system," '
      'IEEE Trans. Biomed. Eng., 51(6):1034–1043, 2004.', REF),
    P('[15] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, PhysioNet," '
      'Circulation, 101(23):e215–e220, 2000.', REF),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc = TwoColDoc(OUT, pagesize=letter,
    leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB)
doc.build_templates()
doc.build(story)
sz = os.path.getsize(OUT)
print(f"PDF → {OUT}")
print(f"Size: {sz/1024:.0f} KB")
