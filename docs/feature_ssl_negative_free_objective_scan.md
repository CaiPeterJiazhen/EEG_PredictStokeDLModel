# Negative-Free Feature SSL Objective Scan

All runs used original baseline features under `runs/baseline_rerun_20260529`, PSD+WPLI inputs unchanged, gated CNN supervised transfer, 10 LOSO seed runs unless noted.

## Main 10-Seed Comparison

| model | summary | acc | bal_acc | roc_auc | pr_auc | brier | sens | spec | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| no_ssl_origfeat | seed_run_mean | 0.7947 | 0.7861 | 0.8044 | 0.7823 | 0.1893 | 0.9500 | 0.6222 | CNN baseline |
| barlow_deterministic | seed_run_mean | 0.7842 | 0.7744 | 0.7633 | 0.7497 | 0.2054 | 0.9600 | 0.5889 | existing Barlow table; seed0 differs from confirmed 0.7368 run |
| masked_barlow | seed_run_mean | 0.7842 | 0.7756 | 0.7644 | 0.7545 | 0.2158 | 0.9400 | 0.6111 | new latent mask + Barlow |
| vicreg | seed_run_mean | 0.7263 | 0.7211 | 0.7500 | 0.7444 | 0.2260 | 0.8200 | 0.6222 | negative-free VICReg |
| masked_vicreg | seed_run_mean | 0.7632 | 0.7572 | 0.7556 | 0.7411 | 0.2170 | 0.8700 | 0.6444 | latent mask + VICReg |
| no_ssl_multi_ssl_residual_cons014_detbridge004 | seed_run_mean | 0.8000 | 0.7906 | 0.8089 | 0.7865 | 0.1830 | 0.9700 | 0.6111 | existing composite residual; not significant vs no-SSL |
| no_ssl_multi_ssl_residual_cons014_detbridge004 | patient_seedmean10 | 0.8421 | 0.8333 | 0.8222 | 0.8187 | 0.1707 | 1.0000 | 0.6667 | existing composite residual; not significant vs no-SSL |
| no_ssl_origfeat | patient_seedmean10 | 0.8421 | 0.8333 | 0.8111 | 0.7824 | 0.1714 | 1.0000 | 0.6667 | CNN baseline |
| masked_barlow | patient_seedmean10 | 0.8421 | 0.8333 | 0.6778 | 0.6043 | 0.1917 | 1.0000 | 0.6667 | new latent mask + Barlow |
| vicreg | patient_seedmean10 | 0.8421 | 0.8333 | 0.7778 | 0.7544 | 0.1793 | 1.0000 | 0.6667 | negative-free VICReg |
| masked_vicreg | patient_seedmean10 | 0.8421 | 0.8333 | 0.7444 | 0.6822 | 0.1851 | 1.0000 | 0.6667 | latent mask + VICReg |

## Seed0 Bootstrap SSL Pilot

SimSiam was added as another negative-sample-free bootstrap objective. It uses the existing PSD+WPLI gated CNN encoder, a projection head, an online predictor, and stop-gradient between the two augmented EO/EC feature views. It does not use negative samples and does not introduce a new supervised architecture.

| model | acc | bal_acc | roc_auc | pr_auc | brier | sens | spec | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| no-SSL CNN seed0 | 0.6316 | 0.6222 | 0.6667 | 0.6528 | 0.2760 | 0.8000 | 0.4444 | original baseline feature rerun |
| Barlow SSL-CNN seed0 | 0.7368 | 0.7222 | 0.7444 | 0.6912 | 0.2007 | 1.0000 | 0.4444 | confirmed original-feature Barlow seed0 |
| BYOL SSL-CNN + EEG summary seed0 | 0.7368 | 0.7278 | 0.6778 | 0.7228 | 0.2646 | 0.9000 | 0.5556 | worse calibration/ranking than Barlow |
| SimSiam SSL-CNN seed0 | 0.6842 | 0.6722 | 0.7556 | 0.7976 | 0.2373 | 0.9000 | 0.4444 | negative pilot; adds a sub15 false negative |
| SimSiam SSL-CNN + no-ratio ROI residual seed0 | 0.6842 | 0.6722 | 0.7444 | 0.7828 | 0.2066 | 0.9000 | 0.4444 | calibration improves but classification does not |

SimSiam improved seed0 ROC/PR over no-SSL, but it is worse than the confirmed Barlow seed0 on accuracy, balanced accuracy, sensitivity, and Brier. The ROI residual improves calibration but does not change the decision errors. Do not run SimSiam 10 seeds unless another change first fixes the false-positive/false-negative tradeoff in seed0.

## Current Conclusion

Seed0 pilots for masked-barlow and masked-vicreg looked positive, but their 10-seed means did not outperform the no-SSL CNN baseline. VICReg improved seed0 ROC/PR but was not stable across seeds. BYOL and SimSiam are valid negative-sample-free bootstrap baselines, but their seed0 pilots do not justify 10-seed expansion. The existing residual composite slightly improves patient_seedmean10 ROC/PR/Brier over no-SSL, but previous patient-level bootstrap/permutation intervals cross zero, so it is not statistically significant.

The next viable direction should not be another simple global SSL objective swap. It should add fold-internal model selection/calibration or an interpretable low-capacity EEG biomarker branch with strict LOSO validation, because the current failures are mostly ranking/calibration instability and repeated high-confidence false positives rather than lack of seed0 capacity.
