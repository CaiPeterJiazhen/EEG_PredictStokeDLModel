# Residual-Barlow-CNN Tuning Check, 2026-06-11

## Purpose

This check tested whether Residual-Barlow-CNN could be tuned, without new SSL pretraining, to outperform the rerun Residual-aware CNN across patient-level seed-run metrics.

The tested Barlow model reused fold-specific Patient-level Barlow encoder checkpoints and only changed supervised fine-tuning settings.

## Baseline

Residual-aware CNN rerun:

`results/metrics/no_ssl_residualaware_highrank_swa_clsalpha1_rerun_20260611_10seed_summary.csv`

| Model | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1 | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Residual-aware CNN | 0.815789 | 0.812222 | 0.880000 | 0.744444 | 0.793438 | 0.833017 | 0.898889 | 0.908442 | 0.130424 |

## Pure Residual-Barlow Fine-Tuning

Best fine-tuning variant from the current check:

- Patient-level Barlow encoder reused fold-by-fold.
- `embedding_dim=32` per PSD/WPLI branch, fused embedding = 64.
- `variant=highrank`: `lambda_reg=0.3`, `lambda_rank=0.3`, `lambda_soft=0.1`.
- `rank_margin=0.5`.
- `lr=0.002`, `weight_decay=1e-5`.
- `SWA enabled`, `swa_start_epoch=50`, `swa_lr=0.00025`.
- Inference alpha fixed at `0.95`.

Artifacts:

- `results/metrics/resbarlow_tune_20260611_lowswa_alpha095_10seed_per_seed.csv`
- `results/metrics/resbarlow_tune_20260611_lowswa_alpha095_10seed_summary.csv`
- `results/predictions/dl_loso_predictions_resbarlow_tune_20260611_lowswa_alpha095_pilot_seed*.csv`

| Model | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1 | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Residual-Barlow-CNN tuned | 0.836842 | 0.830556 | 0.950000 | 0.711111 | 0.786072 | 0.859571 | 0.863333 | 0.862800 | 0.140854 |

Conclusion: pure Residual-Barlow-CNN improves accuracy, balanced accuracy, sensitivity, and F1, but it does not outperform Residual-aware CNN on specificity, precision, ROC AUC, PR AUC, or Brier score.

## Score-Level Feasibility Check

A read-only score-level scan tested whether Barlow adds complementary signal when the no-SSL model remains the main path:

`score = (1 - w) * no_ssl_score + w * residual_barlow_score`

This is not a pure Residual-Barlow-CNN result and should not be reported as such.

Artifacts:

- `results/metrics/resbarlow_tune_20260611_lowswa_alpha095_hybrid_weight_scan_10seed.csv`
- `results/metrics/no_ssl_plus_resbarlow_lowswa_alpha095_w010_10seed_summary.csv`
- `results/metrics/no_ssl_plus_resbarlow_lowswa_alpha095_w020_10seed_summary.csv`
- `results/predictions/dl_loso_predictions_no_ssl_plus_resbarlow_lowswa_alpha095_w010_10seed_all_seed_runs.csv`
- `results/predictions/dl_loso_predictions_no_ssl_plus_resbarlow_lowswa_alpha095_w020_10seed_all_seed_runs.csv`

Best threshold-preserving point from this scan:

| Model | Barlow weight | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1 | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL + Residual-Barlow score mix | 0.20 | 0.836842 | 0.832778 | 0.910000 | 0.755556 | 0.807529 | 0.854359 | 0.893333 | 0.895728 | 0.125912 |

This score mix improves accuracy, balanced accuracy, specificity, precision, F1, and Brier compared with the no-SSL rerun, but ROC AUC and PR AUC remain lower. Therefore it also does not satisfy the all-metric superiority target.

## Interpretation

The current data do not support the claim that pure Residual-Barlow-CNN can be made better than Residual-aware CNN across all patient-level metrics by light supervised fine-tuning alone.

The earlier record where Barlow-related models improved ROC AUC or Brier referred mainly to hybrid residual/consistency mechanisms, such as bounded logit-delta or no-SSL main path plus a small SSL consistency residual. Those are useful evidence that Barlow contains complementary information, but they are not the same as pure Residual-Barlow-CNN fine-tuning.

## Related Full PSD+WPLI Final-Ablation Result

The independent final-model ablation run contains a stronger `full_psd_wpli` result than the new `resbarlow_tune_20260611_lowswa_alpha095` fine-tuning check above:

`final_model_ablation_explainability_results_20260610/independent_train_gpu_main/results/tables/table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`

For the `full_psd_wpli` row, the model is `final_residual_aware_patient_barlow_ssl_cnn_highrank`, with `uses_ssl=True`, `uses_residual_aware_heads=True`, `uses_swa=True`, and `ssl_pretraining_scope=strict_loso_supervised_training_pool_only`.

Seed-run mean comparison against the no-SSL Residual-aware CNN rerun:

| Model | Accuracy | Balanced accuracy | Sensitivity | Specificity | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL Residual-aware CNN rerun | 0.815789 | 0.812222 | 0.880000 | 0.744444 | 0.898889 | 0.908442 | 0.130424 |
| final-ablation `full_psd_wpli` SSL-CNN | 0.810526 | 0.804444 | 0.920000 | 0.688889 | 0.908889 | 0.916613 | 0.127786 |

This `full_psd_wpli` result improves ROC AUC, PR AUC, Brier score, and sensitivity, but has slightly lower accuracy, balanced accuracy, and specificity. It should therefore be kept separate from the new `resbarlow_tune_20260611_lowswa_alpha095` run when discussing model selection.
