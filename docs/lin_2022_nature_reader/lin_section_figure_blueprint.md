# Lin 2022 Section And Figure Blueprint For Our Manuscript

This file translates the Lin 2022 article architecture into a working blueprint for the residual-aware EEG/tACS manuscript.

## Lin Structure

| Lin section | What it does | Main figure/table |
|---|---|---|
| Abstract | Defines the prognosis problem, transfer question, cohort, modalities, model and headline performance. | None |
| Introduction | Moves from stroke recovery heterogeneity to prediction variables and transferability gap. | None |
| Participants | Provides ethics, registration, consent, recruitment, inclusion criteria, safety and patient table. | Table I |
| Clinical/Biomechanical measurements | Defines score ranges, measurement timing, and model input variables. | Table I |
| Neurophysiological measurements | Defines EEG recording, preprocessing, PSD and connectivity features. | None |
| Treatments | Defines manual and robot-assisted stretching protocols. | Fig. 2 |
| Labeling | Converts recovery residuals into proportional/poor-recovery classes. | None |
| Deep learning model | Shows multi-input CNN architecture and binary recovery output. | Fig. 1 |
| SHAP/statistics | Defines explainability and support statistics. | None |
| Results A | Reports source-treatment model ROC and confusion matrix. | Fig. 3, Fig. 4 |
| Results B | Tests transfer to second treatment. | Table II, Fig. 5 |
| Results C/D | Reports key factors and group characteristics. | Fig. 6 |
| Discussion/Conclusion | Interprets transferability, key features, limitations and future work. | None |

## Recommended Structure For Our Paper

| Our section | Lin analogue | Required content | Figure/table placement |
|---|---|---|---|
| Abstract | Abstract | Clinical problem; EEG prognosis gap; baseline EO/EC PSD+WPLI; residual-aware SSL-CNN; n=19; LOSO; locked metrics; exploratory boundary. | None |
| Introduction | Introduction | Upper-limb recovery variability; proportional recovery; EEG as scalable marker; WPLI/PSD rationale; small-cohort and leakage risks; residual-aware SSL contribution. | No figure in Introduction |
| Study cohort | Participants | Site/ethics/consent placeholders; n=29 source records, n=28 EEG indexed, n=19 supervised; supervised labels 10/9; non-supervised EEG pool; missing author fields. | Table 1 |
| Outcome definition | Labeling | FMA-UE expected improvement, observed improvement, residual, median threshold, signed residual distance; training-only residual heads; no test-time residual input. | Formula block, no figure |
| tACS protocol | Treatments | Target, C3/C4 rule, 20 Hz, 1000 microampere, 20 min, daily x14, impedance; evaluator/blinding and safety placeholders. | If using revised initial format, Figure 1 after Study Design |
| EEG acquisition/preprocessing | Neurophysiological measurements | 64-channel source, 62 retained channels, EO/EC, EEGLAB files, affected-side alignment, PSD/WPLI details, MNE visualization protocol. | No standalone main figure unless target journal wants methods figure |
| Model architecture | Deep learning model | PSD/WPLI branches, EO/EC gated fusion, patient-level Barlow pretraining, residual/ranking/soft-label auxiliary heads, classification inference. | Figure 1 for Nature compact version; Figures 1-3 for expanded revised-initial version |
| Validation/statistics | Statistics | Patient-level LOSO, ten seeds, bootstrap, permutation, McNemar, paired comparisons, Brier/calibration, no segment-level independence. | Supplementary precision figure |
| Main performance | Results A | Locked metrics, confusion matrix, ROC/PR/calibration, comparison with EEG logistic baseline and no-SSL CNN. | Figure 2, Table 2 |
| Robustness/ablation | Results B | No-SSL CNN, SSL without residual heads, residual-aware without SSL, feature/state/band ablations, threshold sensitivity. | Figure 3, Table 3 |
| Clinical incremental caveat | No direct Lin analogue | Clinical-only baseline is strong; EEG incremental value is not proven; present as exploratory. | Supplementary Figure 4 |
| Explainability | Results C | IG/SmoothGrad, occlusion, MNE topomap, WPLI connectivity, top features, stability and biomarker caveats. | Figure 4, Table 4, Supplementary Figure 2 |
| Discussion | Discussion | Feasibility; residual-aware supervision; SSL value and limits; leakage control; clinical-only caveat; neurophysiology; limitations. | None |
| Data/code availability | Stronger than Lin | Derived-data deposit, controlled raw EEG access, source-data workbook, code archive, commit/version placeholders. | Source-data workbook |

## Figure Contract For Our Main Figures

### Figure 1

Core conclusion: Baseline EEG can be converted into a leakage-controlled residual-aware SSL-CNN prognosis workflow.

Figure archetype: schematic-led composite.

Backend: Python for generated manuscript figure; MNE-Python for topomap/connectivity subpanels where used.

Panel map: participant flow; tACS/study design; PSD/WPLI feature extraction; patient-level Barlow SSL; residual-aware supervised inference.

Reviewer risk: Must clearly show no post-treatment data enter model inputs and no segment-level validation unit is used.

### Figure 2

Core conclusion: The final model shows promising patient-level discrimination and calibration, with wide uncertainty.

Figure archetype: quantitative grid.

Panel map: model comparison; ROC; PR; confusion matrix; calibration/Brier; bootstrap intervals.

Reviewer risk: Avoid claiming superiority when paired intervals include no effect.

### Figure 3

Core conclusion: Residual-aware supervision is the strongest ablation-supported training signal, while SSL alone is not independently proven.

Figure archetype: quantitative grid.

Panel map: seed stability; model ablation; feature family; state/band ablation; threshold/precision boundary.

Reviewer risk: Multiple ablations in n=19 should be framed as robustness and hypothesis support.

### Figure 4

Core conclusion: Model attribution localizes prediction information to state-dependent PSD patterns and beta-band connectivity.

Figure archetype: asymmetric mixed-modality figure.

Panel map: branch/state occlusion; PSD heatmap; MNE topomap; WPLI top edges; WPLI connectivity map; attribution stability.

Reviewer risk: Label all panels as attribution/explainability outputs, not causal biomarkers.

## What To Avoid When Imitating Lin

1. Do not copy Lin's relatively optimistic transferability framing. Our study has no external validation and should be more conservative.
2. Do not place treatment images as main evidence unless the target journal specifically needs a clinical-protocol visual. Our main evidence is prediction and EEG explanation.
3. Do not use only ROC and confusion matrix. Include calibration, bootstrap intervals and paired tests because prediction-model reviewers expect uncertainty.
4. Do not treat feature attributions as validated physiology. They are hypothesis-generating.
5. Do not hide clinical-only model strength. It is a critical limitation and protects the paper from overclaiming.

