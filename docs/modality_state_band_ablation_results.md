# Modality, State, And Band Ablation Results

All rows are patient-level LOSO tabular ablations over baseline EEG features. Feature selection, if enabled, is fitted inside each training fold only.

## Best Model Per Ablation

| ablation_name         | model                                      |   n_input_columns |   accuracy |   balanced_accuracy |   roc_auc |   pr_auc |   brier_score | explainability_alignment                                                                  |
|:----------------------|:-------------------------------------------|------------------:|-----------:|--------------------:|----------:|---------:|--------------:|:------------------------------------------------------------------------------------------|
| psd_only              | ablation_psd_only_logistic_l2              |               744 |   0.789474 |            0.788889 |  0.811111 | 0.840104 |      0.192946 | gate weights available; branch/state occlusion available                                  |
| ec_only               | ablation_ec_only_logistic_l2               |             11718 |   0.631579 |            0.633333 |  0.788889 | 0.815659 |      0.222793 | gate weights available; branch/state occlusion available                                  |
| psd_wpli              | ablation_psd_wpli_logistic_l2              |             23436 |   0.684211 |            0.677778 |  0.766667 | 0.793437 |      0.207165 | gate weights available; branch/state occlusion available                                  |
| motor_wpli_edges      | ablation_motor_wpli_edges_logistic_l2      |              6780 |   0.631579 |            0.627778 |  0.733333 | 0.81631  |      0.261126 | gate weights available; branch/state occlusion available; WPLI edge attribution available |
| psd_eo_only           | ablation_psd_eo_only_logistic_l2           |               372 |   0.684211 |            0.677778 |  0.733333 | 0.770837 |      0.267018 | gate weights available; branch/state occlusion available                                  |
| wpli_ec_only          | ablation_wpli_ec_only_logistic_l2          |             11346 |   0.631579 |            0.633333 |  0.711111 | 0.793279 |      0.230992 | gate weights available; branch/state occlusion available                                  |
| beta_medium_only      | ablation_beta_medium_only_logistic_l2      |              3906 |   0.736842 |            0.733333 |  0.711111 | 0.776232 |      0.232151 | gate weights available; branch/state occlusion available; PSD band attribution available  |
| psd_eo_wpli_ec        | ablation_psd_eo_wpli_ec_logistic_l2        |             11718 |   0.578947 |            0.583333 |  0.711111 | 0.793279 |      0.23944  | gate weights available; branch/state occlusion available                                  |
| wpli_only             | ablation_wpli_only_logistic_l2             |             22692 |   0.631579 |            0.627778 |  0.688889 | 0.750898 |      0.232211 | gate weights available; branch/state occlusion available                                  |
| beta_medium_beta_high | ablation_beta_medium_beta_high_logistic_l2 |              7812 |   0.631579 |            0.627778 |  0.577778 | 0.559787 |      0.279674 | gate weights available; branch/state occlusion available; PSD band attribution available  |
| beta_high_only        | ablation_beta_high_only_logistic_l2        |              3906 |   0.578947 |            0.577778 |  0.566667 | 0.561048 |      0.316967 | gate weights available; branch/state occlusion available; PSD band attribution available  |
| eo_only               | ablation_eo_only_logistic_l2               |             11718 |   0.315789 |            0.316667 |  0.3      | 0.466981 |      0.490112 | gate weights available; branch/state occlusion available                                  |

## Interpretation

Compare PSD-only, WPLI-only, and PSD+WPLI rows against gate weights and branch/state occlusion. Band-specific beta rows are support checks for whether attribution-localized beta features carry predictive signal when isolated. Motor WPLI edges are a targeted biological plausibility check, not a replacement for the full locked model.