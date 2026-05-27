# sub09/sub14 Error Subject QC Audit

This audit compares sub09 and sub14 against the full 19-patient supervised cohort. A preprocessing recommendation is made only when an objective QC metric is an outlier (absolute z-score >= 3).

## Label Borderline Check

| subject_id | FMA_pre | FMA_post | observed_delta | predicted_delta | residual | label | distance_to_threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| sub09 | 6.000 | 11.000 | 5.000 | 42.000 | 37.000 | 0 | 35.500 |
| sub14 | 60.000 | 62.000 | 2.000 | 4.200 | 2.200 | 0 | 0.700 |

## QC Outlier Summary

- sub09: no objective EEG/metadata QC metric exceeded |z| >= 3.0.
- sub14: no objective EEG/metadata QC metric exceeded |z| >= 3.0.

## Metric Ranks For sub09/sub14

| metric | sub09 value/rank/z | sub14 value/rank/z |
|---|---:|---:|
| residual | 37 / 2 / 1.57 | 2.2 / 9 / -0.66 |
| distance_to_threshold | 35.5 / 2 / 1.59 | 0.7 / 17 / -0.83 |
| psd_mean | 2.038 / 7 / 0.13 | 1.53 / 11 / -0.26 |
| psd_std | 8.71 / 9 / -0.20 | 4.085 / 13 / -0.57 |
| wpli_mean | 0.1241 / 9 / -0.08 | 0.1354 / 6 / 0.37 |
| wpli_std | 0.1175 / 12 / -0.23 | 0.1228 / 10 / -0.01 |
| zero_count | 0 / 1 / 0.00 | 0 / 1 / 0.00 |
| extreme_z_score_count | 16 / 6 / -0.24 | 18 / 5 / -0.23 |
| psd_eo_ec_mean_abs_distance | 0.5663 / 14 / -0.46 | 1.284 / 8 / -0.15 |
| wpli_eo_ec_mean_abs_distance | 0.1146 / 5 / 0.79 | 0.09324 / 11 / -0.43 |
| channel_level_psd_outlier_count | 1 / 4 / -0.14 | 0 / 6 / -0.30 |
| edge_band_wpli_outlier_count | 38 / 10 / -0.42 | 31 / 11 / -0.46 |
| channel_level_psd_max_abs_z | 3.281 / 4 / 1.12 | 0.7027 / 19 / -1.26 |
| edge_band_wpli_max_abs_z | 3.677 / 11 / -0.13 | 3.671 / 12 / -0.16 |

## Interpretation

- sub09: classify as a hard/borderline case rather than a preprocessing failure; it is not especially close to the residual threshold and has no objective QC outlier.
- sub14: classify as a hard/borderline case rather than a preprocessing failure; it is near the residual threshold and has no objective QC outlier.
