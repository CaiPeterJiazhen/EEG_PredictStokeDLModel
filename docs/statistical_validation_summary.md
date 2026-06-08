# Statistical Validation Summary

All uncertainty estimates use subject-level LOSO predictions. Bootstrap resampling is over subjects; no segment-level rows or seed rows are treated as independent observations.

## Model Confidence Intervals And Calibration

| model_name                                    |   n_subjects |   accuracy |   balanced_accuracy |   roc_auc |   pr_auc |   brier_score |       ece |   calibration_intercept |   calibration_slope |   binomial_accuracy_p |
|:----------------------------------------------|-------------:|-----------:|--------------------:|----------:|---------:|--------------:|----------:|------------------------:|--------------------:|----------------------:|
| residual_aware_SSL_CNN_seedmean10             |           19 |   0.842105 |            0.833333 |  0.844444 | 0.836025 |      0.125887 | 0.210788  |               -0.729695 |             1.31461 |            0.00221252 |
| ML_EEG_updated_no_selector_logistic_l1        |           19 |   0.736842 |            0.733333 |  0.711111 | 0.775406 |      0.208001 | 0.0595994 |                0.139481 |             1.07054 |            0.0317841  |

## Paired Model Comparisons

| model_a                                       | model_b                                       |   n_subjects | metric                   |   metric_a |   metric_b |   difference |      ci_low |     ci_high |   p_value_two_sided |   n_bootstrap |   a_correct_b_wrong |   a_wrong_b_correct |   n_discordant |   p_value |
|:----------------------------------------------|:----------------------------------------------|-------------:|:-------------------------|-----------:|-----------:|-------------:|------------:|------------:|--------------------:|--------------:|--------------------:|--------------------:|---------------:|----------:|
| ML_EEG_updated_no_selector_logistic_l1        | residual_aware_SSL_CNN_seedmean10             |           19 | accuracy                 |   0.736842 |   0.842105 |    0.105263  |  -0.105263  |   0.315789  |              0.4228 |          5000 |                 nan |                 nan |            nan |   nan     |
| ML_EEG_updated_no_selector_logistic_l1        | residual_aware_SSL_CNN_seedmean10             |           19 | balanced_accuracy        |   0.733333 |   0.833333 |    0.1       |  -0.10003   |   0.311111  |              0.3664 |          5000 |                 nan |                 nan |            nan |   nan     |
| ML_EEG_updated_no_selector_logistic_l1        | residual_aware_SSL_CNN_seedmean10             |           19 | sensitivity              |   0.8      |   1        |    0.2       |   0         |   0.5       |              0.2392 |          5000 |                 nan |                 nan |            nan |   nan     |
| ML_EEG_updated_no_selector_logistic_l1        | residual_aware_SSL_CNN_seedmean10             |           19 | specificity              |   0.666667 |   0.666667 |    0         |  -0.333333  |   0.333333  |              1      |          5000 |                 nan |                 nan |            nan |   nan     |
| ML_EEG_updated_no_selector_logistic_l1        | residual_aware_SSL_CNN_seedmean10             |           19 | roc_auc                  |   0.711111 |   0.844444 |    0.133333  |  -0.215909  |   0.47619   |              0.4576 |          5000 |                 nan |                 nan |            nan |   nan     |
| ML_EEG_updated_no_selector_logistic_l1        | residual_aware_SSL_CNN_seedmean10             |           19 | pr_auc                   |   0.775406 |   0.836025 |    0.0606187 |  -0.237948  |   0.371496  |              0.6648 |          5000 |                 nan |                 nan |            nan |   nan     |
| ML_EEG_updated_no_selector_logistic_l1        | residual_aware_SSL_CNN_seedmean10             |           19 | brier_score              |   0.208001 |   0.125887 |   -0.082114  |  -0.189461  |   0.0371969 |              0.1688 |          5000 |                 nan |                 nan |            nan |   nan     |
| ML_EEG_updated_no_selector_logistic_l1        | residual_aware_SSL_CNN_seedmean10             |           19 | mcnemar_hard_predictions | nan        | nan        |  nan         | nan         | nan         |            nan      |           nan |                   1 |                   3 |              4 |     0.625 |

## Permutation Tests

| model_name                                    | metric            |   observed |    p_value |   n_permutations |   n_valid_permutations |
|:----------------------------------------------|:------------------|-----------:|-----------:|-----------------:|-----------------------:|
| residual_aware_SSL_CNN_seedmean10             | accuracy          |   0.842105 | 0.0039992  |             5000 |                   5000 |
| residual_aware_SSL_CNN_seedmean10             | balanced_accuracy |   0.833333 | 0.00339932 |             5000 |                   5000 |
| residual_aware_SSL_CNN_seedmean10             | roc_auc           |   0.844444 | 0.00459908 |             5000 |                   5000 |
| residual_aware_SSL_CNN_seedmean10             | pr_auc            |   0.836025 | 0.015197   |             5000 |                   5000 |
| residual_aware_SSL_CNN_seedmean10             | brier_score       |   0.125887 | 0.9994     |             5000 |                   5000 |
| ML_EEG_updated_no_selector_logistic_l1        | accuracy          |   0.736842 | 0.0589882  |             5000 |                   5000 |
| ML_EEG_updated_no_selector_logistic_l1        | balanced_accuracy |   0.733333 | 0.0561888  |             5000 |                   5000 |
| ML_EEG_updated_no_selector_logistic_l1        | roc_auc           |   0.711111 | 0.0631874  |             5000 |                   5000 |
| ML_EEG_updated_no_selector_logistic_l1        | pr_auc            |   0.775406 | 0.054989   |             5000 |                   5000 |
| ML_EEG_updated_no_selector_logistic_l1        | brier_score       |   0.208001 | 0.960808   |             5000 |                   5000 |
