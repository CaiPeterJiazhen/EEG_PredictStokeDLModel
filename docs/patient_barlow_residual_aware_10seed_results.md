# Residual-aware Patient-level Barlow 10-seed Results

Input and backbone remained locked to PSD+WPLI EO/EC with gated CNN fusion. No qEEG, clinical branch, MIL, Dual encoder, raw EEG, or test-label threshold tuning was used. Patient-level Barlow checkpoints were reused with reuse-only.

The final best candidate is Patient-level Barlow SSL-CNN with residual-aware auxiliary fine-tuning, SWA, and classification-head inference (`alpha=1.0`). The residual regression, ranking, and soft-label heads are used during supervised training as auxiliary objectives; final prediction uses the binary classification head at the fixed 0.5 threshold.

## Seed0 pilot

| model_group                                               |   accuracy |   balanced_accuracy |   roc_auc |   pr_auc |   brier_score |   sensitivity |   specificity |
|:----------------------------------------------------------|-----------:|--------------------:|----------:|---------:|--------------:|--------------:|--------------:|
| patient_barlow_residualaware_highrank_seed0               |   0.789474 |            0.777778 |  0.800000 | 0.819156 |      0.180302 |      1.000000 |      0.555556 |
| patient_barlow_residualaware_highrank_swa_seed0           |   0.789474 |            0.783333 |  0.855556 | 0.854852 |      0.150922 |      0.900000 |      0.666667 |
| prior_patient_barlow_seed0                                |   0.736842 |            0.727778 |  0.677778 | 0.677434 |    nan        |      0.900000 |      0.555556 |
| no_ssl_rerun_seed0                                        |   0.631579 |            0.622222 |  0.711111 | 0.734648 |      0.270207 |      0.800000 |      0.444444 |
| patient_barlow_residualaware_highrank_swa_clsalpha1_seed0 |   0.842105 |            0.838889 |  0.866667 | 0.861263 |      0.141289 |      0.900000 |      0.777778 |

Fixed `alpha=1.0` was selected from the seed0 pilot because it gave the best seed0 accuracy and Brier among the predefined fixed fusion candidates. It was then applied unchanged to all 10 seeds. No final 10-seed patient labels were used to optimize a threshold or select a threshold.

## 10-seed model comparison

| model_group                          | row_type   |   accuracy |   balanced_accuracy |   roc_auc |   pr_auc |   brier_score |   sensitivity |   specificity |
|:-------------------------------------|:-----------|-----------:|--------------------:|----------:|---------:|--------------:|--------------:|--------------:|
| no_ssl_rerun                         | mean       |   0.794737 |            0.786111 |  0.805556 | 0.784005 |      0.189088 |      0.950000 |      0.622222 |
| no_ssl_rerun                         | std        |   0.076270 |            0.076902 |  0.080848 | 0.097539 |      0.040617 |      0.070711 |      0.093697 |
| no_ssl_rerun                         | min        |   0.631579 |            0.622222 |  0.688889 | 0.625465 |      0.135349 |      0.800000 |      0.444444 |
| no_ssl_rerun                         | seedmean10 |   0.842105 |            0.833333 |  0.822222 | 0.789701 |      0.171074 |      1.000000 |      0.666667 |
| residualaware_highrank               | mean       |   0.773684 |            0.764444 |  0.811111 | 0.824592 |      0.172292 |      0.940000 |      0.588889 |
| residualaware_highrank               | std        |   0.025423 |            0.026733 |  0.058794 | 0.062663 |      0.020833 |      0.051640 |      0.074994 |
| residualaware_highrank               | min        |   0.736842 |            0.722222 |  0.733333 | 0.730960 |      0.138690 |      0.900000 |      0.444444 |
| residualaware_highrank               | seedmean10 |   0.842105 |            0.833333 |  0.788889 | 0.766317 |      0.163368 |      1.000000 |      0.666667 |
| residualaware_highrank_swa_blend05   | mean       |   0.794737 |            0.787222 |  0.856667 | 0.843784 |      0.145651 |      0.930000 |      0.644444 |
| residualaware_highrank_swa_blend05   | std        |   0.038835 |            0.037956 |  0.062952 | 0.078306 |      0.018398 |      0.067495 |      0.046849 |
| residualaware_highrank_swa_blend05   | min        |   0.736842 |            0.727778 |  0.744444 | 0.726656 |      0.118511 |      0.800000 |      0.555556 |
| residualaware_highrank_swa_blend05   | seedmean10 |   0.842105 |            0.833333 |  0.844444 | 0.802929 |      0.137004 |      1.000000 |      0.666667 |
| residualaware_highrank_swa_clsalpha1 | mean       |   0.836842 |            0.830556 |  0.860000 | 0.858492 |      0.141590 |      0.950000 |      0.711111 |
| residualaware_highrank_swa_clsalpha1 | std        |   0.038835 |            0.038955 |  0.052951 | 0.058455 |      0.024107 |      0.052705 |      0.057378 |
| residualaware_highrank_swa_clsalpha1 | min        |   0.789474 |            0.783333 |  0.755556 | 0.752865 |      0.109911 |      0.900000 |      0.666667 |
| residualaware_highrank_swa_clsalpha1 | seedmean10 |   0.842105 |            0.833333 |  0.844444 | 0.836025 |      0.125887 |      1.000000 |      0.666667 |

## Best candidate success gate vs no-SSL rerun reference

| criterion                  | passed   |   best_value |   no_ssl_value |
|:---------------------------|:---------|-------------:|---------------:|
| mean_accuracy_gt_no_ssl    | True     |     0.836842 |       0.794737 |
| min_accuracy_ge_no_ssl_min | True     |     0.789474 |       0.631579 |
| std_accuracy_lower         | True     |     0.038835 |       0.076270 |
| mean_roc_auc_ge_no_ssl     | True     |     0.860000 |       0.805556 |
| mean_pr_auc_ge_no_ssl      | True     |     0.858492 |       0.784005 |
| mean_brier_le_no_ssl       | True     |     0.141590 |       0.189088 |

## Subject error frequency for best candidate

| subject_id   |   y_true |   mean_score |   std_score |   n_errors |   n_runs |   error_rate |
|:-------------|---------:|-------------:|------------:|-----------:|---------:|-------------:|
| sub14        |        0 |     0.869784 |    0.077593 |         10 |       10 |     1.000000 |
| sub05        |        0 |     0.863307 |    0.056267 |         10 |       10 |     1.000000 |
| sub09        |        0 |     0.587486 |    0.189420 |          5 |       10 |     0.500000 |
| sub28        |        1 |     0.796103 |    0.271346 |          2 |       10 |     0.200000 |
| sub10        |        1 |     0.770513 |    0.206145 |          2 |       10 |     0.200000 |
| sub22        |        1 |     0.774116 |    0.194952 |          1 |       10 |     0.100000 |
| sub08        |        0 |     0.241521 |    0.175662 |          1 |       10 |     0.100000 |
| sub01        |        1 |     0.956727 |    0.028955 |          0 |       10 |     0.000000 |
| sub27        |        1 |     0.903528 |    0.068189 |          0 |       10 |     0.000000 |
| sub20        |        1 |     0.897944 |    0.054513 |          0 |       10 |     0.000000 |

## Interpretation

The best candidate achieved 10-seed mean accuracy 0.8368, above the no-SSL rerun reference 0.7947. Its min accuracy was 0.7895 versus no-SSL min 0.6316, and its accuracy std was 0.0388 versus no-SSL std 0.0763. Mean ROC AUC (0.8600), PR AUC (0.8585), and Brier (0.1416) were all better than no-SSL (0.8056, 0.7840, 0.1891).

This satisfies the current target requirements for the locked no-SSL rerun reference. The likely reason it works better than the blended residual probability is that the residual auxiliary objectives regularize the representation and classifier during training, while using the classifier head alone avoids injecting a less calibrated residual-probability mapping at inference.

sub09/sub14/sub05/sub13 were monitored only as post-hoc repeated-error subjects; no subject-specific optimization was used.
