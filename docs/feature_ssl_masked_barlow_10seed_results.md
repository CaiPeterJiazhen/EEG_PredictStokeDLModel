# Feature SSL Masked-Barlow 10-Seed Results

## Method

Implemented `masked-barlow`: a negative-sample-free Barlow Twins objective with latent modality masking over PSD and WPLI branch embeddings. The raw PSD and WPLI-FC feature inputs are unchanged.

Literature basis from AnySearch: CroSSL uses latent masking of modality-specific intermediate embeddings without negative-pair sampling; biomedical SSL surveys support Barlow/VICReg/BYOL-style negative-free representation learning for small labeled biosignal settings.

## Seed0 Gate

| model | acc | bal_acc | roc_auc | pr_auc | brier | sens | spec |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_ssl_origfeat | 0.6316 | 0.6222 | 0.6667 | 0.6528 | 0.2760 | 0.8000 | 0.4444 |
| barlow_ssl_confirmed_origfeat | 0.7368 | 0.7222 | 0.7444 | 0.6912 | 0.2007 | 1.0000 | 0.4444 |
| masked_barlow_ssl | 0.7895 | 0.7778 | 0.7667 | 0.7073 | 0.1923 | 1.0000 | 0.5556 |

Seed0 was positive: masked-barlow improved accuracy, balanced accuracy, ROC AUC, PR AUC, Brier score, and specificity relative to both no-SSL seed0 and the confirmed original-feature Barlow seed0.

## 10-Seed Outcome

| model | row_type | acc | bal_acc | roc_auc | pr_auc | brier | sens | spec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no_ssl_origfeat | seedrun_mean | 0.7947 | 0.7861 | 0.8044 | 0.7823 | 0.1893 | 0.9500 | 0.6222 |
| masked_barlow_ssl | seedrun_mean | 0.7842 | 0.7756 | 0.7644 | 0.7545 | 0.2158 | 0.9400 | 0.6111 |
| no_ssl_origfeat | seedmean10 | 0.8421 | 0.8333 | 0.8111 | 0.7824 | 0.1714 | 1.0000 | 0.6667 |
| masked_barlow_ssl | seedmean10 | 0.8421 | 0.8333 | 0.6778 | 0.6043 | 0.1917 | 1.0000 | 0.6667 |

Conclusion: this is not a final improvement. The 10-seed seed-run mean is below no-SSL on accuracy, balanced accuracy, ROC AUC, PR AUC, and Brier. The seedmean10 ensemble ties no-SSL on accuracy and balanced accuracy but has worse ROC AUC, PR AUC, and Brier.

## Post-hoc Hard Subjects

| model | subject | y_true | error_count | positive_pred_count | mean_y_score |
|---|---|---:|---:|---:|---:|
| masked_barlow_ssl | sub09 | 0 | 8 | 8 | 0.9487 |
| masked_barlow_ssl | sub14 | 0 | 8 | 8 | 0.9808 |
| no_ssl_origfeat | sub09 | 0 | 10 | 10 | 0.9319 |
| no_ssl_origfeat | sub14 | 0 | 9 | 9 | 0.8619 |

sub09/sub14 are monitored only as post-hoc repeated false-positive subjects, not as optimization targets.

## Files

- Comparison CSV: `runs/baseline_rerun_20260529/results/metrics/feature_ssl_masked_barlow_10seed_vs_no_ssl_summary.csv`
- Subject error frequency: `runs/baseline_rerun_20260529/results/metrics/feature_ssl_masked_barlow_10seed_subject_error_frequency.csv`
- sub09/sub14 summary: `runs/baseline_rerun_20260529/results/metrics/sub09_sub14_masked_barlow_10seed_scores_summary.csv`

Next direction: keep the negative-sample-free constraint, but avoid relying on simple score averaging from masked-barlow because it worsened ranking/calibration across seeds.
