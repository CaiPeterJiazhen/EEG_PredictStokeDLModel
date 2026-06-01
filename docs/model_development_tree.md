# Model Development Tree

This document records the main modeling path so the manuscript does not look like it only reports a single successful run.

## Baseline Layer

- ML EEG baseline: Logistic L1, Logistic L2, SVM RBF.
- Clinical baseline: FMA_pre-only, MBI_pre-only, age/sex/duration, full baseline clinical logistic.
- no-SSL CNN: PSD+WPLI EO/EC gated CNN without SSL.

## SSL Layer

- Feature-level SSL: masked/VICReg/Barlow-style exploratory runs.
- Segment-level SSL: PSD/WPLI Segment Barlow and ensembles; useful but not final.
- Patient-level Barlow: closest SSL family to the final target.

## Negative Or Supplementary Branches

- Hard-negative focal fine-tuning: seed0 mixed/negative; stopped.
- Manifold Mixup and modality dropout: not promoted.
- MIL, Dual encoder, raw EEG CNN-BLSTM: not continued.
- qEEG-guided branch: kept as supplementary biomarker/residual analysis, not the final model.
- SAM-only/staged SAM+SWA: not selected after pilot.

## Final Branch

- Patient-level Barlow SSL-CNN.
- Residual-aware auxiliary supervised fine-tuning.
- SWA.
- Classification-head inference.
- Fixed threshold 0.5.

Final model group: `residualaware_highrank_swa_clsalpha1`.

## Decision Rule

Model selection stopped at the locked residual-aware Patient-level Barlow SSL-CNN. Subsequent work only adds clinical baselines, sensitivity checks, explainability, paper tables, and reporting documentation.
