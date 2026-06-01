# Network-Level Biomarker Validation

This analysis validates network/ROI-level attribution summaries from the locked final model. It does not promote any single WPLI edge as a confirmed biomarker.

Patient-level summaries average the 10 seed/fold attribution rows within each subject before correlation tests, so seeds are not treated as independent clinical samples.

## Strongest Network/ROI Associations

| feature family | summary | rho vs signed distance | p | FDR q | direction |
|---|---|---:|---:|---:|---|
| WPLI | wpli_beta_frontal_central | 0.073 | 0.7670 | 0.9744 | poor-recovery association |
| WPLI | wpli_beta_frontal_parietal | 0.122 | 0.6189 | 0.9744 | positive-class association |
| WPLI | wpli_beta_central_parietal | -0.090 | 0.7155 | 0.9744 | poor-recovery association |
| WPLI | wpli_beta_interhemispheric | 0.028 | 0.9091 | 0.9744 | positive-class association |
| WPLI | wpli_beta_motor_adjacent | 0.030 | 0.9035 | 0.9744 | poor-recovery association |
| PSD | psd_frontal_delta | -0.255 | 0.2913 | 0.9744 | poor-recovery association |

## Interpretation

These results are manuscript support for network-level patterns. They should be described as exploratory and hypothesis-generating because the cohort has 19 patients and lacks external validation.

Reported groups include WPLI beta frontal-central, frontal-parietal, central-parietal, interhemispheric, motor-adjacent summaries and PSD frontal delta, fronto-central beta, temporal delta, and motor beta-medium summaries.
