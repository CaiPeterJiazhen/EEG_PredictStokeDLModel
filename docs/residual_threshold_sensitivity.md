# Residual Label Threshold Sensitivity

Primary labels in the current project are derived from the residual threshold of 1.5: `Residual <= 1.5` is the proportional-recovery group.

The supervised cohort median residual is 1.500. If this median was selected after viewing the full cohort, it is a task-definition choice that can introduce optimistic bias at the label-definition level. The model training and LOSO prediction files are unchanged here; this script only re-scores existing subject-level predictions under alternative label definitions.

## Locked Current Definition

| model                                         |   n_subjects |   n_positive |   n_negative |   accuracy |   roc_auc |   pr_auc |   brier_score |
|:----------------------------------------------|-------------:|-------------:|-------------:|-----------:|----------:|---------:|--------------:|
| ML_EEG_updated_no_selector_logistic_l1        |           19 |           10 |            9 |   0.736842 |  0.711111 | 0.775406 |      0.208001 |
| no_SSL_CNN_updated_sub05_sub28_seedensemble10 |           19 |           10 |            9 |   0.842105 |  0.811111 | 0.807984 |      0.177034 |
| residual_aware_SSL_CNN_seedmean10             |           19 |           10 |            9 |   0.842105 |  0.844444 | 0.836025 |      0.125887 |

## Fold-local Train-only Median

| model                                         |   n_subjects |   n_positive |   n_negative |   accuracy |   roc_auc |   pr_auc |   brier_score |
|:----------------------------------------------|-------------:|-------------:|-------------:|-----------:|----------:|---------:|--------------:|
| ML_EEG_updated_no_selector_logistic_l1        |           19 |           10 |            9 |   0.736842 |  0.711111 | 0.775406 |      0.208001 |
| no_SSL_CNN_updated_sub05_sub28_seedensemble10 |           19 |           10 |            9 |   0.842105 |  0.811111 | 0.807984 |      0.177034 |
| residual_aware_SSL_CNN_seedmean10             |           19 |           10 |            9 |   0.842105 |  0.844444 | 0.836025 |      0.125887 |

Near-threshold exclusion rows report how many subjects were excluded before metrics were recomputed. These sensitivity checks are descriptive and must not replace the primary locked label without a new pre-specified analysis plan.