# Patient-level Barlow SSL-CNN + SAM/SWA Stabilized Fine-tuning

This run returns to Patient-level Barlow as the main SSL-CNN candidate. The input remains PSD+FC-WPLI EO/EC baseline EEG with gated CNN fusion; no qEEG branch, clinical branch, MIL, Dual encoder, raw EEG model, or architecture search was added. Fold-specific Patient-level Barlow encoders were cached first and then reused with `--reuse-ssl-encoders --reuse-only` for supervised fine-tuning.

## Seed0 pilot

| model_group                         |   accuracy |   balanced_accuracy |   sensitivity |   specificity |   roc_auc |   pr_auc |   brier_score |
|:------------------------------------|-----------:|--------------------:|--------------:|--------------:|----------:|---------:|--------------:|
| patient_barlow_swa_seed0            |   0.789474 |            0.777778 |           1   |      0.555556 |  0.788889 | 0.749463 |      0.175163 |
| patient_barlow_sam_seed0            |   0.526316 |            0.516667 |           0.7 |      0.333333 |  0.666667 | 0.764918 |      0.227151 |
| patient_barlow_staged_sam_swa_seed0 |   0.631579 |            0.622222 |           0.8 |      0.444444 |  0.788889 | 0.83373  |      0.191544 |
| prior_patient_barlow_seed0          |   0.736842 |            0.727778 |           0.9 |      0.555556 |  0.677778 | 0.677434 |    NA        |

Seed0 was positive for SWA-only: accuracy and balanced accuracy improved over the prior Patient-level Barlow seed0 reference. SAM-only and staged SAM+SWA were worse on accuracy. Therefore the 10-seed continuation used SWA-only as the best stabilized variant.

## 10-seed summary

| model_group                                   |   n_seeds |   mean_accuracy |   std_accuracy |   min_accuracy |   mean_balanced_accuracy |   mean_roc_auc |   mean_pr_auc |   mean_brier_score |   seedmean_accuracy |   seedmean_roc_auc |   seedmean_pr_auc |   seedmean_brier_score | notes                                                                                                    |
|:----------------------------------------------|----------:|----------------:|---------------:|---------------:|-------------------------:|---------------:|--------------:|-------------------:|--------------------:|-------------------:|------------------:|-----------------------:|:---------------------------------------------------------------------------------------------------------|
| patient_barlow_swa                            |        10 |        0.778947 |      0.0393859 |       0.736842 |                 0.768333 |       0.844444 |      0.850886 |             0.1699 |            0.789474 |           0.844444 |          0.865981 |               0.158111 | Best seed0 stabilized variant; Patient-level Barlow encoder reused, supervised SWA-only fine-tuning.     |
| no_ssl_cnn_reference_from_prior_locked_result |        10 |        0.794737 |    NA         |     NA        |               NA        |     NA        |    NA        |           NA      |            0.842105 |           0.811111 |          0.7824   |               0.1714   | Reference values from prior locked no-SSL document; paired predictions not available in this summarizer. |
| no_ssl_schemeA_updated_sub05_sub28            |        10 |        0.752632 |      0.0942972 |       0.578947 |                 0.743333 |       0.783333 |      0.768648 |           NA      |            0.842105 |           0.811111 |          0.807984 |             NA        | Reference from updated_sub05_sub28_10seed_no_ssl_barlow_cnn_summary.csv                                  |
| prior_patient_barlow_updated_sub05_sub28      |        10 |        0.752632 |      0.0528941 |       0.684211 |                 0.741667 |       0.738889 |      0.719375 |           NA      |            0.789474 |           0.7      |          0.63096  |             NA        | Reference from updated_sub05_sub28_10seed_no_ssl_barlow_cnn_summary.csv                                  |

## Seed stability

| row_type   |   accuracy |   balanced_accuracy |   roc_auc |   pr_auc |   brier_score |   sensitivity |   specificity |
|:-----------|-----------:|--------------------:|----------:|---------:|--------------:|--------------:|--------------:|
| mean       |  0.778947  |           0.768333  | 0.844444  | 0.850886 |     0.1699    |     0.97      |     0.566667  |
| std        |  0.0393859 |           0.0398338 | 0.0358323 | 0.045385 |     0.0214983 |     0.0458258 |     0.0598352 |
| min        |  0.736842  |           0.722222  | 0.788889  | 0.749463 |     0.131929  |     0.9       |     0.444444  |
| max        |  0.842105  |           0.833333  | 0.911111  | 0.918366 |     0.215932  |     1         |     0.666667  |

## Error frequency

| subject_id   |   y_true |   mean_score |   std_score |   n_errors |   n_runs |   error_rate |
|:-------------|---------:|-------------:|------------:|-----------:|---------:|-------------:|
| sub05        |        0 |    0.914028  |   0.0616703 |         10 |       10 |          1   |
| sub09        |        0 |    0.825006  |   0.107908  |         10 |       10 |          1   |
| sub14        |        0 |    0.850362  |   0.15449   |          9 |       10 |          0.9 |
| sub08        |        0 |    0.645096  |   0.129597  |          9 |       10 |          0.9 |
| sub28        |        1 |    0.634804  |   0.291325  |          2 |       10 |          0.2 |
| sub22        |        1 |    0.768309  |   0.185913  |          1 |       10 |          0.1 |
| sub07        |        0 |    0.0850982 |   0.159912  |          1 |       10 |          0.1 |
| sub01        |        1 |    0.979331  |   0.0262749 |          0 |       10 |          0   |

sub09/sub14 were monitored as post-hoc repeated-error subjects, not optimized directly.

| subject_id   |   y_true | model_group        |   seed |   y_score |   y_pred |   error |   mean_score |   std_score |   n_errors |   error_rate |
|:-------------|---------:|:-------------------|-------:|----------:|---------:|--------:|-------------:|------------:|-----------:|-------------:|
| sub09        |        0 | patient_barlow_swa |      0 |  0.764988 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |      1 |  0.771327 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |      2 |  0.913683 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |      3 |  0.593369 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |      4 |  0.822466 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |      5 |  0.940921 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |      7 |  0.854604 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |     13 |  0.839832 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |     21 |  0.967625 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub09        |        0 | patient_barlow_swa |     42 |  0.781246 |        1 |       1 |     0.825006 |    0.107908 |         10 |          1   |
| sub14        |        0 | patient_barlow_swa |      0 |  0.987816 |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |      1 |  0.956634 |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |      2 |  0.84742  |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |      3 |  0.876255 |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |      4 |  0.978971 |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |      5 |  0.852943 |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |      7 |  0.830818 |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |     13 |  0.455645 |        0 |       0 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |     21 |  0.78415  |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |
| sub14        |        0 | patient_barlow_swa |     42 |  0.932973 |        1 |       1 |     0.850362 |    0.15449  |          9 |          0.9 |

## Statistical comparison

Patient-level bootstrap and paired score/permutation comparisons were computed on seedmean10 predictions where local reference prediction files were available. Seeds are repeated training runs, not independent patients. The prior locked no-SSL reference is included as a metric-only row because its full paired seedmean prediction file was not available in this local summary set.

| comparison                                         | metric                               |   reference_value |   candidate_value |   candidate_minus_reference |   bootstrap_ci_low |   bootstrap_ci_high |   both_correct |   both_wrong |   permutation_p |
|:---------------------------------------------------|:-------------------------------------|------------------:|------------------:|----------------------------:|-------------------:|--------------------:|---------------:|-------------:|----------------:|
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | accuracy                             |          0.842105 |          0.789474 |                  -0.0526316 |        -0.157895   |          0          |             |           |              |
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | balanced_accuracy                    |          0.833333 |          0.777778 |                  -0.0555556 |        -0.166667   |          0          |             |           |              |
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | roc_auc                              |          0.811111 |          0.844444 |                   0.0333333 |        -0.0454545  |          0.142857   |             |           |              |
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | pr_auc                               |          0.807984 |          0.865981 |                   0.0579976 |        -0.0461446  |          0.177668   |             |           |              |
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | brier_score                          |          0.177034 |          0.158111 |                  -0.0189238 |        -0.059957   |          0.0206671  |             |           |              |
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | paired_correctness_counts            |          1        |          0        |                  -1         |                 |                  |             15 |            3 |              |
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | roc_auc_random_label_permutation     |                |                |                   0.0333333 |                 |                  |             |           |       0.69      |
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | pr_auc_random_label_permutation      |                |                |                   0.0579976 |                 |                  |             |           |       0.723333  |
| no_ssl_schemeA_updated_vs_patient_barlow_swa       | brier_score_random_label_permutation |                |                |                  -0.0189238 |                 |                  |             |           |       0.383333  |
| prior_patient_barlow_updated_vs_patient_barlow_swa | accuracy                             |          0.789474 |          0.789474 |                   0         |         0          |          0          |             |           |              |
| prior_patient_barlow_updated_vs_patient_barlow_swa | balanced_accuracy                    |          0.777778 |          0.777778 |                   0         |         0          |          0          |             |           |              |
| prior_patient_barlow_updated_vs_patient_barlow_swa | roc_auc                              |          0.7      |          0.844444 |                   0.144444  |        -0.0187175  |          0.377778   |             |           |              |
| prior_patient_barlow_updated_vs_patient_barlow_swa | pr_auc                               |          0.63096  |          0.865981 |                   0.235021  |        -0.00658716 |          0.386416   |             |           |              |
| prior_patient_barlow_updated_vs_patient_barlow_swa | brier_score                          |          0.203446 |          0.158111 |                  -0.0453353 |        -0.094245   |         -0.00215067 |             |           |              |
| prior_patient_barlow_updated_vs_patient_barlow_swa | paired_correctness_counts            |          0        |          0        |                   0         |                 |                  |             15 |            4 |              |
| prior_patient_barlow_updated_vs_patient_barlow_swa | roc_auc_random_label_permutation     |                |                |                   0.144444  |                 |                  |             |           |       0.14      |
| prior_patient_barlow_updated_vs_patient_barlow_swa | pr_auc_random_label_permutation      |                |                |                   0.235021  |                 |                  |             |           |       0.09      |
| prior_patient_barlow_updated_vs_patient_barlow_swa | brier_score_random_label_permutation |                |                |                  -0.0453353 |                 |                  |             |           |       0.0666667 |

## Interpretation

Patient-level Barlow + SWA reached 10-seed mean accuracy 0.7789, min accuracy 0.7368, and seedmean10 accuracy 0.7895. This improves over the prior Patient-level Barlow seed0 pilot and the updated prior Patient-level Barlow summary, but it still does not exceed the prior locked no-SSL reference mean accuracy of 0.7947. The main gain appears to be supervised stabilization from SWA rather than SAM: SAM-only was unstable in seed0, while SWA-only improved seed0 and was selected for 10 seeds.

The model is a promising supplementary stabilized SSL-CNN candidate, but based on the available locked reference it should not yet replace the no-SSL CNN as the main model. The next defensible step would be to inspect calibration and score distributions, not to add qEEG or search new architectures in this branch.
