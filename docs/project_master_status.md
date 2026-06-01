# Project Master Status

## Objective

Predict whether 19 supervised stroke patients achieve proportional upper-limb recovery after tACS using pre-treatment resting-state EEG. Validation is patient-level LOSO-CV with repeated seeds for neural models.

## Data And Labels

- Supervised cohort: 19 patients with baseline EEG and pre/post clinical scores.
- Outcome: proportional recovery label from residual threshold `Residual <= 1.5`.
- Residual: predicted change from proportional recovery rule minus observed FMA change.
- Main inference threshold: fixed `y_score >= 0.5`.
- No final test labels are used to retune thresholds or select a new model in the current manuscript-support phase.

## EEG Features

- Main CNN input: PSD EO/EC with shape `62 x 90`.
- Main CNN input: WPLI EO/EC with shape `1891 x 6`.
- iCOH features exist in the project but are not part of the final model.
- Features use the canonical 62-channel order and affected-side alignment before modeling.

## Baselines

- ML EEG baselines include Logistic L1, Logistic L2, and SVM RBF.
- Clinical LOSO baselines now include FMA_pre-only, MBI_pre-only, age/sex/duration, and baseline clinical-only logistic models.
- qEEG-only logistic is retained as biomarker/supporting analysis, not as the main deep model.
- no-SSL CNN remains an important reference with the same PSD+WPLI EO/EC gated CNN input structure.

## SSL Attempts

Explored SSL families included feature-level masked/VICReg variants, Segment Barlow, Patient-level Barlow, qEEG-guided SSL, hard-negative fine-tuning, Mixup, SAM/SWA stabilized fine-tuning, and residual-aware auxiliary fine-tuning. Negative or mixed pilots were documented rather than promoted.

## Patient-Level Barlow

Patient-level Barlow is the main SSL backbone family. Earlier Patient-level Barlow runs improved seedmean accuracy but did not reliably improve ROC/PR/Brier. The final selected model keeps the Patient-level Barlow encoder and changes only supervised fine-tuning.

## Final Model

`residualaware_highrank_swa_clsalpha1` is the current final model:

- Patient-level Barlow SSL-CNN.
- PSD+WPLI EO/EC gated CNN, `embedding_dim=32`, `dropout=0.0`.
- Residual-aware auxiliary supervised heads during training.
- SWA.
- Final prediction uses only the binary classification head.
- No qEEG branch, no clinical branch, no MIL, no Dual encoder, no raw EEG CNN-BLSTM.

Locked 10-seed mean metrics: accuracy `0.8368`, balanced accuracy `0.8306`, ROC AUC `0.8600`, PR AUC `0.8585`, Brier `0.1416`. Seedmean10 accuracy is `0.8421`.

## Explainability

The explanation target is the classification logit. The committed attribution summaries use SmoothGrad-smoothed Integrated Gradients, plus branch/state/feature occlusion and stability checks. Real scalp topomaps have now been generated from the provided electrode coordinates under `results/figures/explainability/topomaps/`.

## Negative Pilots

- Hard-negative focal fine-tuning: mixed/negative seed0 pilot; not continued.
- Manifold Mixup/modality dropout: not promoted as final.
- qEEG-guided SSL-CNN: useful supplementary biomarker analysis but did not exceed no-SSL reference mean accuracy.
- SAM-only/staged SAM+SWA: weaker than SWA-only in Patient-level Barlow stabilization pilot.
- Segment-level SSL and dual/MIL directions were not promoted as the final model.

## Pending Manuscript Tasks

- Write manuscript text around the locked final model.
- Decide how to present strong clinical-only baseline performance without overclaiming EEG superiority.
- Keep SSL data scope wording conservative because `all-patient` SSL pretraining is historical unlabeled pretraining, not a fully prospective baseline-only setting.
- Emphasize pilot/exploratory status and need for external validation.
