# EEG Foundation Model Gap Analysis

## Taxonomy of Directions
1. General-purpose EEG representation (LaBraM, EEGPT, CBraMod, LUNA, FEMBA)
2. Task-specific foundation models (MIRepNet→MI, LEAD→AD, M4CEA→Epilepsy)
3. Multimodal/LLM-aligned (NeuroLM, EEG Emotion Copilot)
4. Architecture variants (Mamba-based: EEGMamba, FEMBA, Mentality; Graph: GEFM)
5. Channel-topology-aware (LUNA, DIVER-0, GEFM)

## Identified Gaps

### Gap 1: Riemannian geometry excluded from foundation model comparisons
- ALL reviewed papers use transformer/Mamba/CNN architectures
- Riemannian geometry (SPD manifold, covariance-based) is the SOTA classical baseline for MI
- Zero papers compare Riemannian baselines fairly vs. neural foundation models
- Riemannian methods are CPU-feasible and highly interpretable
- Feasibility: HIGH (pyRiemann, MNE, MOABB all support this)

### Gap 2: Calibration of EEG foundation models never studied
- Zero reviewed papers report Expected Calibration Error (ECE) or reliability diagrams
- Critical for BCI clinical deployment
- MIRepNet and LEAD both focus solely on accuracy/F1/kappa
- Feasibility: HIGH (only needs softmax outputs + calibration metrics)

### Gap 3: Cross-subject transfer degradation under channel topology mismatch
- MIRepNet uses fixed 64-ch; LEAD uses fixed montages
- Real-world deployment requires adapting to different headsets
- LUNA claims topology-agnostic but no ablation on MI tasks
- DIVER-0 is equivariant but no MI evaluation
- Feasibility: MEDIUM (needs datasets with different channel counts)

### Gap 4: Parameter-efficient fine-tuning (PEFT) for MI EEG foundation models
- No papers apply LoRA/adapters to EEG FMs for MI adaptation
- Direct port from NLP literature
- Feasibility: MEDIUM (needs a pretrained model to attach adapters to)

## Selected Paper Direction

**"Riemannian Geometry Meets Neural Foundation Models: A Cross-Subject Motor Imagery Benchmark with Calibration Analysis"**

### Rationale
- Addresses Gaps 1 and 2 simultaneously
- Fully CPU-feasible: Riemannian operations are O(C^3) not O(N^2)
- Uses only PhysioNet EEGMMIDB (109 subjects, 64 channels, free download via MNE)
- Novel contribution: first systematic comparison of Riemannian vs. neural foundation model features for MI with proper calibration analysis under strict subject-independent evaluation
- Positioned relative to MIRepNet: extends their evaluation by (a) adding Riemannian baselines they omit, (b) measuring calibration they ignore, (c) analyzing what kinds of subjects/sessions transfer worst

### Novelty Claims (honest)
1. First to benchmark Riemannian + neural methods on the same MI datasets with calibration metrics
2. First calibration analysis of MI EEG classification methods under cross-subject conditions
3. Systematic analysis of cross-subject transfer degradation factors

### What we CAN'T claim
- We are not proposing a new foundation model architecture
- We are not training on 100M parameters
- Results are on a single dataset (PhysioNet EEGMMIDB) with optional BCIC-IV-2a

### Revised Scope (CPU-safe, honest)
Given compute constraints, we build a rigorous benchmark with:
- Riemannian CSP + MDM/FgMDM (classical, CPU-safe)
- EEGNet (compact neural, CPU-trainable per subject)
- ShallowConvNet (compact neural baseline)  
- FBCSP + LDA (frequency-band CSP, strong baseline)
- Covariance + Riemannian SVM (tangent space)
- Subject-independent LOSO-CV evaluation
- Calibration metrics: ECE, reliability diagrams, Brier score
- Transfer analysis: performance by session, by subject variance, by data size
