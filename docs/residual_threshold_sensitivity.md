# Residual Threshold Sensitivity

This analysis keeps the locked final model unchanged:
`residualaware_highrank_swa_clsalpha1`, fixed classification-head inference, and fixed threshold 0.5.
It does not reselect the outcome threshold or the model.

## Locked Outcome Definition

The primary binary label remains the current cohort residual threshold:
`Residual <= 1.5` is proportional recovery and `Residual > 1.5` is poor recovery.
At this locked threshold the cohort has 10 positive and 9 negative subjects.

The final seed-mean model at this threshold has accuracy 0.842,
balanced accuracy 0.833, ROC AUC 0.844,
PR AUC 0.836, and Brier score 0.126.

## Sensitivity Analysis

Alternative residual thresholds are reported only as sensitivity checks in
`results/metrics/residual_threshold_sensitivity.csv`. They must not be used to
replace the locked primary threshold without a new pre-specified analysis plan.

## Continuous Residual Association

The final model score has Spearman correlation with residual:
rho = -0.464, p = 0.0452.
Because higher residual indicates poorer proportional-recovery fit, a negative
association is directionally expected.

Using signed distance (`1.5 - residual`), Spearman rho = 0.464,
p = 0.0452. This is the continuous target behind the
residual-aware auxiliary training heads.

## Rationale For Residual-Aware Auxiliary Objectives

The binary outcome is a thresholded version of a continuous residual. The
regression, pairwise-ranking, and soft-label heads use continuous residual
structure during training, while final inference remains the binary
classification head with threshold 0.5. This preserves the locked clinical
decision rule while reducing information loss from binarization.

## Cohort Note

This is a 19-patient LOSO pilot cohort. Threshold sensitivity is descriptive
and should be treated as manuscript support, not as evidence for a newly
optimized label definition.
