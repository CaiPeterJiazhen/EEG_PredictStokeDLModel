# Prediction Model Reporting Checklist

## Study Objective

Predict proportional upper-limb recovery after tACS in stroke using baseline resting-state EEG.

## Participants

Report the 19-patient supervised cohort, inclusion basis, baseline EEG availability, and pre/post clinical scores.

## Outcome Definition

Define residual, the locked threshold `1.5`, binary labels, and fixed inference threshold `0.5`.

## Predictors

Main predictors are baseline PSD EO/EC and WPLI EO/EC. Clinical variables and qEEG are reported as baselines/supporting analyses, not part of the final deep model.

## EEG Preprocessing

Describe resting-state EO/EC preprocessing, canonical 62-channel ordering, affected-side alignment, and feature extraction.

## Feature Extraction

Report PSD shape `62 x 90`, WPLI shape `1891 x 6`, bands, channel order, and fold-local scaling.

## Model Architecture

Report Patient-level Barlow SSL-CNN with gated PSD/WPLI branches, residual-aware auxiliary training heads, SWA, and classification-head inference.

## SSL Data Scope

State that fold-specific SSL excludes the held-out test subject. If `all-patient` unlabeled records are used, call it historical unlabeled pretraining rather than fully prospective baseline-only SSL.

## LOSO Validation

Use patient-level LOSO. Explain that seeds are repeated model fits, not independent patients.

## Multi-Seed Stability

Report mean, std, min, max metrics over 10 seeds and seedmean patient-level metrics.

## Calibration

Report Brier score and reliability/calibration where available. Do not tune the threshold on final LOSO labels.

## Statistical Analysis

Report bootstrap CIs over 19 patients, paired score comparisons, paired correctness tests, and random-label permutation where applicable.

## Explainability

Report SmoothGrad-smoothed Integrated Gradients, occlusion, stability, sanity checks, topomaps, and network-level biomarker validation.

## Code/Data Availability

Commit scripts, summary CSVs, figures, and docs. Do not commit raw EEG, feature caches, or checkpoints unless separately approved.

## Limitations

Small n, single cohort, no external validation, exploratory explainability, and possible historical SSL data scope limitations.

## External Validation Status

No external validation is currently available; frame conclusions as exploratory and hypothesis-generating.
