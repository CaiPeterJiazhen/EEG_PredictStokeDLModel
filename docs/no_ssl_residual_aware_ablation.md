# no-SSL Residual-aware Multi-task Ablation

This ablation tests whether the residual-aware supervised objective alone explains the gain.

All models use the same PSD+WPLI EO/EC input and the same gated multimodal CNN backbone. The no-SSL residual-aware model is randomly initialized and does not load Patient-level Barlow checkpoints.

## Setup

| model_group | SSL encoder | supervised objective | SWA | inference |
|:--|:--|:--|:--|:--|
| standard_no_ssl_bce | no | BCE classification | no | classification score, 0.5 threshold |
| no_ssl_residualaware_highrank_swa_clsalpha1 | no | BCE + residual Huber + pairwise ranking + residual-distance soft label | yes | classification head, 0.5 threshold |
| patient_barlow_residualaware_highrank_swa_clsalpha1 | Patient-level Barlow | BCE + residual Huber + pairwise ranking + residual-distance soft label | yes | classification head, 0.5 threshold |

The residual head, ranking loss, and soft-label loss are used as training-time auxiliary objectives. Final inference uses the binary classification head. No final test-label threshold optimization is used.

## 10-seed Results

| model_group | row_type | accuracy | balanced_accuracy | roc_auc | pr_auc | brier_score | sensitivity | specificity |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| standard_no_ssl_bce | mean | 0.7947 | 0.7861 | 0.8056 | 0.7840 | 0.1891 | 0.9500 | 0.6222 |
| standard_no_ssl_bce | std | 0.0763 | 0.0769 | 0.0808 | 0.0975 | 0.0406 | 0.0707 | 0.0937 |
| standard_no_ssl_bce | min | 0.6316 | 0.6222 | 0.6889 | 0.6255 | 0.1353 | 0.8000 | 0.4444 |
| standard_no_ssl_bce | seedmean10 | 0.8421 | 0.8333 | 0.8222 | 0.7897 | 0.1711 | 1.0000 | 0.6667 |
| no_ssl_residualaware_highrank_swa_clsalpha1 | mean | 0.8158 | 0.8122 | 0.8989 | 0.9084 | 0.1304 | 0.8800 | 0.7444 |
| no_ssl_residualaware_highrank_swa_clsalpha1 | std | 0.0447 | 0.0437 | 0.0473 | 0.0525 | 0.0298 | 0.0789 | 0.0537 |
| no_ssl_residualaware_highrank_swa_clsalpha1 | min | 0.7368 | 0.7333 | 0.8111 | 0.8080 | 0.0800 | 0.8000 | 0.6667 |
| no_ssl_residualaware_highrank_swa_clsalpha1 | seedmean10 | 0.8947 | 0.8889 | 0.9222 | 0.9273 | 0.1105 | 1.0000 | 0.7778 |
| patient_barlow_residualaware_highrank_swa_clsalpha1 | mean | 0.8368 | 0.8306 | 0.8600 | 0.8585 | 0.1416 | 0.9500 | 0.7111 |
| patient_barlow_residualaware_highrank_swa_clsalpha1 | std | 0.0388 | 0.0390 | 0.0530 | 0.0585 | 0.0241 | 0.0527 | 0.0574 |
| patient_barlow_residualaware_highrank_swa_clsalpha1 | min | 0.7895 | 0.7833 | 0.7556 | 0.7529 | 0.1099 | 0.9000 | 0.6667 |
| patient_barlow_residualaware_highrank_swa_clsalpha1 | seedmean10 | 0.8421 | 0.8333 | 0.8444 | 0.8360 | 0.1259 | 1.0000 | 0.6667 |

## Interpretation

The residual-aware supervised objective is a major contributor. Without SSL, it improves mean accuracy from 0.7947 to 0.8158, reduces accuracy std from 0.0763 to 0.0447, raises min accuracy from 0.6316 to 0.7368, and substantially improves ROC AUC, PR AUC, and Brier.

Patient-level Barlow still adds value for the primary seed-level stability target. With the same residual-aware objective, Patient-level Barlow improves mean accuracy from 0.8158 to 0.8368, min accuracy from 0.7368 to 0.7895, and accuracy std from 0.0447 to 0.0388.

The tradeoff is that the no-SSL residual-aware model has stronger score-level ROC/PR/Brier and seedmean10 metrics than the SSL residual-aware model. For the current target, which prioritizes per-seed mean accuracy, min accuracy, and seed stability, Patient-level Barlow residual-aware remains the better main candidate. For score calibration or ensemble-style seedmean analysis, the no-SSL residual-aware result should be treated as an important competing ablation.

## Output Files

- `results/metrics/no_ssl_residualaware_ablation_10seed_summary.csv`
- `results/metrics/no_ssl_residualaware_highrank_swa_clsalpha1_10seed_summary.csv`
- `results/metrics/no_ssl_residualaware_highrank_swa_clsalpha1_subject_error_frequency.csv`
- `results/predictions/seedmean_no_ssl_residualaware_highrank_swa_clsalpha1_10seed.csv`
