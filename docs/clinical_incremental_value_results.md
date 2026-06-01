# Clinical Baseline And EEG Incremental Value

All clinical predictors are baseline-only: age, sex, duration, affected_hand, FMA_pre, and MBI_pre.
Post-treatment variables, observed/predicted deltas, residuals, and labels are blocked from model inputs.

## Clinical-only LOSO Models

| model                       | model_family   | feature_selection   |   n_subjects |   accuracy |   balanced_accuracy |   roc_auc |   pr_auc |   brier_score |
|:----------------------------|:---------------|:--------------------|-------------:|-----------:|--------------------:|----------:|---------:|--------------:|
| clinical_only_logistic_l2   | clinical_only  | none                |           19 |   0.842105 |            0.838889 |  0.911111 | 0.89877  |     0.104528  |
| clinical_only_gaussian_nb   | clinical_only  | none                |           19 |   0.894737 |            0.894444 |  0.888889 | 0.798012 |     0.070097  |
| clinical_only_logistic_l1   | clinical_only  | none                |           19 |   0.947368 |            0.944444 |  0.888889 | 0.798012 |     0.0714309 |
| clinical_only_svm_rbf       | clinical_only  | none                |           19 |   0.789474 |            0.783333 |  0.866667 | 0.784026 |     0.131867  |
| clinical_only_random_forest | clinical_only  | none                |           19 |   0.842105 |            0.844444 |  0.866667 | 0.782255 |     0.156682  |
| clinical_only_knn           | clinical_only  | none                |           19 |   0.842105 |            0.838889 |  0.85     | 0.787338 |     0.146199  |

## EEG-only And EEG+clinical Models

| model                               | model_family   | feature_selection   |   n_subjects |   accuracy |   balanced_accuracy |   roc_auc |   pr_auc |   brier_score |
|:------------------------------------|:---------------|:--------------------|-------------:|-----------:|--------------------:|----------:|---------:|--------------:|
| eeg_clinical_logistic_l1_selectk100 | eeg_clinical   | selectk100          |           19 |   0.894737 |            0.894444 |  0.9      | 0.848012 |      0.111132 |
| eeg_clinical_logistic_l2_selectk100 | eeg_clinical   | selectk100          |           19 |   0.684211 |            0.677778 |  0.8      | 0.82172  |      0.181256 |
| eeg_clinical_svm_rbf_selectk100     | eeg_clinical   | selectk100          |           19 |   0.631579 |            0.644444 |  0.788889 | 0.775108 |      0.21031  |
| eeg_only_svm_rbf_selectk100         | eeg_only       | selectk100          |           19 |   0.684211 |            0.694444 |  0.777778 | 0.771338 |      0.218677 |
| eeg_only_logistic_l2_none           | eeg_only       | none                |           19 |   0.684211 |            0.688889 |  0.777778 | 0.839721 |      0.221418 |
| eeg_only_logistic_l2_selectk100     | eeg_only       | selectk100          |           19 |   0.684211 |            0.677778 |  0.766667 | 0.793437 |      0.207165 |
| eeg_only_logistic_l1_none           | eeg_only       | none                |           19 |   0.736842 |            0.733333 |  0.711111 | 0.775406 |      0.208001 |
| eeg_only_knn_none                   | eeg_only       | none                |           19 |   0.473684 |            0.461111 |  0.683333 | 0.765132 |      0.304094 |
| eeg_only_svm_linear_selectk100      | eeg_only       | selectk100          |           19 |   0.684211 |            0.677778 |  0.677778 | 0.674231 |      0.216167 |
| eeg_only_svm_linear_none            | eeg_only       | none                |           19 |   0.631579 |            0.633333 |  0.666667 | 0.666551 |      0.250735 |
| eeg_only_gaussian_nb_none           | eeg_only       | none                |           19 |   0.631579 |            0.633333 |  0.633333 | 0.610526 |      0.368421 |
| eeg_only_gaussian_nb_selectk100     | eeg_only       | selectk100          |           19 |   0.526316 |            0.533333 |  0.622222 | 0.601017 |      0.473684 |
| eeg_only_knn_selectk100             | eeg_only       | selectk100          |           19 |   0.631579 |            0.616667 |  0.605556 | 0.583889 |      0.333333 |
| eeg_only_logistic_l1_selectk100     | eeg_only       | selectk100          |           19 |   0.473684 |            0.472222 |  0.444444 | 0.549555 |      0.306868 |
| eeg_only_random_forest_none         | eeg_only       | none                |           19 |   0.315789 |            0.3      |  0.377778 | 0.509012 |      0.270964 |
| eeg_only_random_forest_selectk100   | eeg_only       | selectk100          |           19 |   0.368421 |            0.366667 |  0.316667 | 0.444739 |      0.287449 |
| eeg_only_svm_rbf_none               | eeg_only       | none                |           19 |   0.210526 |            0.2      |  0.144444 | 0.393863 |      0.30865  |

## Paired Bootstrap Versus Best Clinical-only Model

| reference_model           | candidate_model                     | metric      |   metric_a |   metric_b |   difference |     ci_low |   ci_high |   p_value_two_sided |   n_bootstrap |
|:--------------------------|:------------------------------------|:------------|-----------:|-----------:|-------------:|-----------:|----------:|--------------------:|--------------:|
| clinical_only_logistic_l2 | eeg_clinical_logistic_l1_selectk100 | accuracy    |   0.842105 |   0.894737 |   0.0526316  | -0.105263  | 0.210526  |               0.732 |           500 |
| clinical_only_logistic_l2 | eeg_clinical_logistic_l1_selectk100 | roc_auc     |   0.911111 |   0.9      |  -0.0111111  | -0.1       | 0.0674621 |               1     |           500 |
| clinical_only_logistic_l2 | eeg_clinical_logistic_l1_selectk100 | brier_score |   0.104528 |   0.111132 |   0.00660449 | -0.0572081 | 0.0664126 |               0.86  |           500 |
| clinical_only_logistic_l2 | eeg_clinical_logistic_l2_selectk100 | accuracy    |   0.842105 |   0.684211 |  -0.157895   | -0.315789  | 0         |               0.084 |           500 |
| clinical_only_logistic_l2 | eeg_clinical_logistic_l2_selectk100 | roc_auc     |   0.911111 |   0.8      |  -0.111111   | -0.3       | 0.0296795 |               0.168 |           500 |
| clinical_only_logistic_l2 | eeg_clinical_logistic_l2_selectk100 | brier_score |   0.104528 |   0.181256 |   0.0767283  |  0.0104459 | 0.146341  |               0.016 |           500 |
| clinical_only_logistic_l2 | eeg_clinical_svm_rbf_selectk100     | accuracy    |   0.842105 |   0.631579 |  -0.210526   | -0.448684  | 0         |               0.072 |           500 |
| clinical_only_logistic_l2 | eeg_clinical_svm_rbf_selectk100     | roc_auc     |   0.911111 |   0.788889 |  -0.122222   | -0.372173  | 0.117917  |               0.308 |           500 |
| clinical_only_logistic_l2 | eeg_clinical_svm_rbf_selectk100     | brier_score |   0.104528 |   0.21031  |   0.105782   |  0.0115389 | 0.190972  |               0.028 |           500 |
| clinical_only_logistic_l2 | eeg_only_logistic_l1_none           | accuracy    |   0.842105 |   0.736842 |  -0.105263   | -0.368421  | 0.157895  |               0.576 |           500 |
| clinical_only_logistic_l2 | eeg_only_logistic_l1_none           | roc_auc     |   0.911111 |   0.711111 |  -0.2        | -0.488769  | 0.0844508 |               0.184 |           500 |
| clinical_only_logistic_l2 | eeg_only_logistic_l1_none           | brier_score |   0.104528 |   0.208001 |   0.103473   | -0.0055973 | 0.205572  |               0.06  |           500 |
| clinical_only_logistic_l2 | eeg_only_logistic_l2_none           | accuracy    |   0.842105 |   0.684211 |  -0.157895   | -0.421053  | 0.105263  |               0.348 |           500 |
| clinical_only_logistic_l2 | eeg_only_logistic_l2_none           | roc_auc     |   0.911111 |   0.777778 |  -0.133333   | -0.398782  | 0.0931818 |               0.3   |           500 |
| clinical_only_logistic_l2 | eeg_only_logistic_l2_none           | brier_score |   0.104528 |   0.221418 |   0.11689    | -0.0187525 | 0.273368  |               0.124 |           500 |
| clinical_only_logistic_l2 | eeg_only_svm_rbf_selectk100         | accuracy    |   0.842105 |   0.684211 |  -0.157895   | -0.368421  | 0.0526316 |               0.188 |           500 |
| clinical_only_logistic_l2 | eeg_only_svm_rbf_selectk100         | roc_auc     |   0.911111 |   0.777778 |  -0.133333   | -0.387689  | 0.133793  |               0.32  |           500 |
| clinical_only_logistic_l2 | eeg_only_svm_rbf_selectk100         | brier_score |   0.104528 |   0.218677 |   0.114149   |  0.0184693 | 0.196896  |               0.02  |           500 |

The paired bootstrap uses subject-level LOSO predictions and resamples subjects, not seeds or segments.