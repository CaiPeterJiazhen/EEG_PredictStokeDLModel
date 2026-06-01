# Clinical Baseline Results

All clinical baselines used patient-level LOSO-CV. Numeric imputation/scaling and categorical imputation/one-hot encoding were fit on the 18 training subjects within each fold only. Post-treatment variables (`FMA_post`, `MBI_post`, observed delta, residual, and label) were not used as predictors.

| model                             |   accuracy |   balanced_accuracy |   sensitivity |   specificity |   roc_auc |   pr_auc |   brier_score |
|:----------------------------------|-----------:|--------------------:|--------------:|--------------:|----------:|---------:|--------------:|
| FMA_pre_only_logistic             |   0.947368 |            0.944444 |           1   |      0.888889 | 0.888889  | 0.813207 |     0.0721859 |
| MBI_pre_only_logistic             |   0.736842 |            0.738889 |           0.7 |      0.777778 | 0.877778  | 0.889794 |     0.148355  |
| age_sex_duration_logistic         |   0.157895 |            0.155556 |           0.2 |      0.111111 | 0.0333333 | 0.357424 |     0.33812   |
| baseline_clinical_only_logistic   |   0.842105 |            0.838889 |           0.9 |      0.777778 | 0.911111  | 0.89877  |     0.104528  |
| qEEG_only_logistic                |   0.789474 |            0.783333 |           0.9 |      0.666667 | 0.888889  | 0.902233 |     0.145039  |
| no_SSL_CNN_seedmean10             |   0.842105 |            0.833333 |           1   |      0.666667 | 0.822222  | 0.789701 |     0.171074  |
| residual_aware_SSL_CNN_seedmean10 |   0.842105 |            0.833333 |           1   |      0.666667 | 0.844444  | 0.836025 |     0.125887  |
| ML_EEG_logistic_l1                |   0.842105 |            0.838889 |           0.9 |      0.777778 | 0.844444  | 0.877096 |     0.164352  |
| ML_EEG_logistic_l2                |   0.842105 |            0.838889 |           0.9 |      0.777778 | 0.833333  | 0.852691 |     0.176967  |
| ML_EEG_svm_rbf                    |   0.789474 |            0.788889 |           0.8 |      0.777778 | 0.844444  | 0.90402  |     0.15201   |

Reference rows include qEEG-only, ML EEG baselines, no-SSL CNN, and the final residual-aware SSL-CNN where corresponding locked predictions already existed. These rows are included for manuscript comparison only and do not change model selection.