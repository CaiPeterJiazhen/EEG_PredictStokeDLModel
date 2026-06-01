# Clinical Baseline Results

All clinical baselines used patient-level LOSO-CV. Numeric imputation/scaling and categorical imputation/one-hot encoding were fit on the 18 training subjects within each fold only. Post-treatment variables (`FMA_post`, `MBI_post`, observed delta, residual, and label) were not used as predictors.

| model                                         |   accuracy |   balanced_accuracy |   sensitivity |   specificity |   roc_auc |   pr_auc |   brier_score |
|:----------------------------------------------|-----------:|--------------------:|--------------:|--------------:|----------:|---------:|--------------:|
| FMA_pre_only_logistic                         |   0.947368 |            0.944444 |           1   |      0.888889 | 0.888889  | 0.813207 |     0.0721859 |
| MBI_pre_only_logistic                         |   0.736842 |            0.738889 |           0.7 |      0.777778 | 0.877778  | 0.889794 |     0.148355  |
| age_sex_duration_logistic                     |   0.157895 |            0.155556 |           0.2 |      0.111111 | 0.0333333 | 0.357424 |     0.33812   |
| baseline_clinical_only_logistic               |   0.842105 |            0.838889 |           0.9 |      0.777778 | 0.911111  | 0.89877  |     0.104528  |
| qEEG_only_logistic                            |   0.789474 |            0.783333 |           0.9 |      0.666667 | 0.888889  | 0.902233 |     0.145039  |
| no_SSL_CNN_updated_sub05_sub28_seedensemble10 |   0.842105 |            0.833333 |           1   |      0.666667 | 0.811111  | 0.807984 |     0.177034  |
| residual_aware_SSL_CNN_seedmean10             |   0.842105 |            0.833333 |           1   |      0.666667 | 0.844444  | 0.836025 |     0.125887  |
| ML_EEG_updated_no_selector_gaussian_nb        |   0.631579 |            0.633333 |           0.6 |      0.666667 | 0.633333  | 0.610526 |     0.368421  |
| ML_EEG_updated_no_selector_knn                |   0.473684 |            0.461111 |           0.7 |      0.222222 | 0.683333  | 0.765132 |     0.304094  |
| ML_EEG_updated_no_selector_logistic_l1        |   0.736842 |            0.733333 |           0.8 |      0.666667 | 0.711111  | 0.775406 |     0.208001  |
| ML_EEG_updated_no_selector_logistic_l2        |   0.684211 |            0.688889 |           0.6 |      0.777778 | 0.777778  | 0.839721 |     0.221418  |
| ML_EEG_updated_no_selector_random_forest      |   0.315789 |            0.3      |           0.6 |      0        | 0.377778  | 0.509012 |     0.270964  |
| ML_EEG_updated_no_selector_svm_linear         |   0.631579 |            0.633333 |           0.6 |      0.666667 | 0.666667  | 0.666551 |     0.250735  |
| ML_EEG_updated_no_selector_svm_rbf            |   0.210526 |            0.2      |           0.4 |      0        | 0.144444  | 0.393863 |     0.30865   |
| ML_EEG_updated_selectk100_gaussian_nb         |   0.526316 |            0.533333 |           0.4 |      0.666667 | 0.622222  | 0.601017 |     0.473684  |
| ML_EEG_updated_selectk100_knn                 |   0.631579 |            0.616667 |           0.9 |      0.333333 | 0.605556  | 0.583889 |     0.333333  |
| ML_EEG_updated_selectk100_logistic_l1         |   0.473684 |            0.472222 |           0.5 |      0.444444 | 0.444444  | 0.549555 |     0.306868  |
| ML_EEG_updated_selectk100_logistic_l2         |   0.684211 |            0.677778 |           0.8 |      0.555556 | 0.766667  | 0.793437 |     0.207165  |
| ML_EEG_updated_selectk100_random_forest       |   0.368421 |            0.366667 |           0.4 |      0.333333 | 0.316667  | 0.444739 |     0.287449  |
| ML_EEG_updated_selectk100_svm_linear          |   0.684211 |            0.677778 |           0.8 |      0.555556 | 0.677778  | 0.674231 |     0.216167  |
| ML_EEG_updated_selectk100_svm_rbf             |   0.684211 |            0.694444 |           0.5 |      0.888889 | 0.777778  | 0.771338 |     0.218677  |

Reference rows include qEEG-only, updated_sub05_sub28 PSD/WPLI ML baselines, updated_sub05_sub28 no-SSL CNN, and the final residual-aware SSL-CNN where corresponding locked predictions already existed. These rows are included for manuscript comparison only and do not change model selection.