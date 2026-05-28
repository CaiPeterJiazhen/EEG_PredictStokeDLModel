# Segment Barlow 10-Seed Locked Results

## Why 10 Seeds

The prior 6-seed Segment Barlow result was already useful, but the supervised
cohort has only 19 patients. Extending to 10 seeds gives a more stable estimate
of training randomness without increasing the clinical sample size. Seeds are
therefore treated as repeated training runs, not as independent patients.

No architecture search or hyperparameter search was performed in this locked
evaluation. Attention MIL was not extended. Dual-encoder Segment Barlow remains
supplementary only.

## Protocol

Locked seed set:

```text
0, 1, 2, 3, 4, 5, 7, 13, 21, 42
```

Shared supervised protocol:

```text
architecture=multimodal
feature_kind=psd-fc-wpli
fusion=gated
encoder_kind=cnn
embedding_dim=32
dropout=0.0
supervised_lr=0.002
weight_decay=0.00001
epochs=100
patience=100
LOSO-CV=19 patient folds
```

Segment Barlow SSL protocol:

```text
objective=barlow
data_scope=all-patient
historical_unlabeled_pretraining=true
pretrain_epochs=20
projection_dim=32
feature_mask_prob=0.03
noise_std=0.02
lambda_latent=1.0
lambda_local=0.1
transfer_mode=finetune
```

Model groups:

- `no_ssl_stable_cnn`
- `psd_segbarlow_ssl_cnn`
- `wpli_segbarlow_ssl_cnn`
- `psd_wpli_segbarlow_equal_weight`
- `no_ssl_wpli_equal_weight` mixed supervised/SSL candidate
- `no_ssl_psd_wpli_equal_weight` supplementary mixed ensemble

## Checkpoint Reuse

Fold-specific SSL encoder checkpoints are saved under:

```text
results/checkpoints/ssl_encoders/
```

Checkpoint branch names are locked to `psd` and `wpli`. The `fc-wpli` token is
kept only as `segment_feature_kind`.

The 6 existing supervised prediction/metric CSVs were reused. They were not
rerun. The old cache state was:

- seed `0`: legacy PSD/WPLI checkpoint files existed, but lacked strict
  metadata fields such as `historical_unlabeled_pretraining`.
- seeds `1,2,3,7,13`: reusable PSD/WPLI checkpoints were missing.

The cache was refreshed with `--ssl-only-cache --save-ssl-encoders
--reuse-ssl-encoders`, so only SSL pretraining and encoder saving were run for
the existing 6 seeds. No new supervised predictions were written for those
seeds.

The final manifest is:

| branch | rows | coverage |
| --- | ---: | --- |
| psd | 190 | 10 seeds x 19 LOSO folds |
| wpli | 190 | 10 seeds x 19 LOSO folds |

All 380 checkpoint payloads were loaded and checked to contain only
`checkpoint_type`, `metadata`, `encoder_state_dict`, and optional
`prefixed_state_dict`; no projector, classifier, or supervised head keys were
found. Default ignore rules still cover `results/`, `*.pt`, `*.pth`, and
`*.ckpt`; the locked encoder checkpoints are force-added only for the explicit
GitHub upload requested after this evaluation.

## Outputs

Primary 10-seed files:

```text
results/metrics/segment_barlow_10seed_model_selection_summary.csv
results/metrics/segment_barlow_10seed_ensemble_threshold_summary.csv
results/metrics/segment_barlow_10seed_subject_error_frequency.csv
results/metrics/sub09_sub14_10seed_model_scores_summary.csv
results/metrics/segment_barlow_10seed_statistical_comparison.csv
results/predictions/ensemble_predictions_psd_wpli_segbarlow_equal_weight_10seed.csv
results/metrics/ensemble_metrics_psd_wpli_segbarlow_equal_weight_10seed.csv
```

## 10-Seed Seed-Run Summary

Fixed threshold `0.5`, summarized across 10 training seeds:

| model_group | acc mean | acc std | acc min | acc max | bal acc mean | ROC AUC mean | PR AUC mean | Brier mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_ssl_stable_cnn | 0.7895 | 0.0623 | 0.6842 | 0.8421 | 0.7800 | 0.8344 | 0.8430 | 0.1880 |
| psd_segbarlow_ssl_cnn | 0.7895 | 0.0577 | 0.6842 | 0.8947 | 0.7806 | 0.7789 | 0.7711 | 0.2117 |
| wpli_segbarlow_ssl_cnn | 0.8000 | 0.0516 | 0.6842 | 0.8421 | 0.7894 | 0.8267 | 0.8282 | 0.1901 |
| psd_wpli_segbarlow_equal_weight | 0.8053 | 0.0529 | 0.6842 | 0.8421 | 0.7961 | 0.8200 | 0.8284 | 0.1935 |
| no_ssl_wpli_equal_weight | 0.8000 | 0.0614 | 0.6316 | 0.8421 | 0.7900 | 0.8389 | 0.8453 | 0.1838 |
| no_ssl_psd_wpli_equal_weight | 0.8158 | 0.0485 | 0.6842 | 0.8421 | 0.8072 | 0.8322 | 0.8365 | 0.1886 |

The SSL-only PSD+WPLI ensemble has the highest fixed-threshold mean accuracy
among SSL-only candidates, while WPLI Segment Barlow is the strongest single
SSL branch on mean accuracy, ROC AUC, PR AUC, and Brier score.

## Seed-Mean Patient-Level Summary

Seed-mean probabilities evaluate the 19 patients once after averaging the 10
seed scores. Fixed threshold `0.5`:

| model_group | accuracy | balanced accuracy | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_ssl_stable_cnn | 0.8421 | 0.8333 | 0.8889 | 0.8974 | 0.1747 |
| psd_segbarlow_ssl_cnn | 0.8421 | 0.8333 | 0.8111 | 0.8247 | 0.1908 |
| wpli_segbarlow_ssl_cnn | 0.8421 | 0.8333 | 0.8333 | 0.8437 | 0.1845 |
| psd_wpli_segbarlow_equal_weight | 0.8421 | 0.8333 | 0.8333 | 0.8613 | 0.1868 |
| no_ssl_wpli_equal_weight | 0.8421 | 0.8333 | 0.8667 | 0.8881 | 0.1790 |
| no_ssl_psd_wpli_equal_weight | 0.8421 | 0.8333 | 0.8333 | 0.8530 | 0.1822 |

All fixed-threshold seed-mean candidates tie on accuracy and balanced
accuracy. The no-SSL baseline remains strongest on ROC AUC, PR AUC, and Brier.
This means the locked result should not be described as a clear SSL
outperformance over no-SSL.

## Threshold Calibration

Two threshold policies were reported:

- `fixed_0.5`: primary threshold with no label tuning.
- `leave_one_seed_out_oof`: for each held-out seed, the threshold is selected
  using only the other nine seed prediction sets.

No primary result directly optimizes the final 19-subject seed-mean threshold.
Weighted ensembles remain disabled because no strict inner/OOF weight-selection
mechanism is available.

The SSL-only PSD+WPLI ensemble improves from fixed seed-run mean accuracy
`0.8053` to leave-one-seed threshold seed-run mean accuracy `0.8158`. The
seed-mean patient-level accuracy remains `0.8421` under both policies.

## Statistical Comparison

The statistical comparison uses 19 patients as the bootstrap unit. Seeds are
not treated as independent patients.

Selected rows:

| analysis | comparison/model | metric | observed | 95% CI / p |
| --- | --- | --- | ---: | --- |
| bootstrap | no_ssl_stable_cnn | ROC AUC | 0.8889 | 0.6932 to 1.0000 |
| bootstrap | no_ssl_stable_cnn | PR AUC | 0.8974 | 0.7056 to 1.0000 |
| bootstrap | psd_wpli_segbarlow_equal_weight | PR AUC | 0.8613 | 0.6434 to 1.0000 |
| paired correctness | noSSL vs PSD | correctness difference | 0.0000 | sign-test p=1.0000 |
| paired correctness | noSSL vs WPLI | correctness difference | 0.0000 | sign-test p=1.0000 |
| paired correctness | noSSL vs PSD+WPLI | correctness difference | 0.0000 | sign-test p=1.0000 |
| paired score bootstrap | noSSL -> PSD+WPLI | ROC AUC difference | -0.0556 | -0.2051 to 0.0455 |
| random-label permutation | no_ssl_stable_cnn | balanced accuracy | 0.8333 | p=0.0022 |
| random-label permutation | psd_wpli_segbarlow_equal_weight | balanced accuracy | 0.8333 | p=0.0038 |

The paired correctness tests are flat because all fixed-threshold seed-mean
primary candidates classify the same number of patients correctly.

## Error Analysis

The 10-seed subject-error file is:

```text
results/metrics/segment_barlow_10seed_subject_error_frequency.csv
```

Focused sub09/sub14 rows:

| subject | model_group | n_errors / n_runs | mean score | distance to 0.5 |
| --- | --- | ---: | ---: | ---: |
| sub09 | no_ssl_stable_cnn | 9 / 10 | 0.9048 | 0.4048 |
| sub14 | no_ssl_stable_cnn | 10 / 10 | 0.8935 | 0.3935 |
| sub09 | psd_segbarlow_ssl_cnn | 10 / 10 | 0.9733 | 0.4733 |
| sub14 | psd_segbarlow_ssl_cnn | 9 / 10 | 0.9262 | 0.4262 |
| sub09 | wpli_segbarlow_ssl_cnn | 10 / 10 | 0.9524 | 0.4524 |
| sub14 | wpli_segbarlow_ssl_cnn | 10 / 10 | 0.9885 | 0.4885 |
| sub09 | psd_wpli_segbarlow_equal_weight | 10 / 10 | 0.9628 | 0.4628 |
| sub14 | psd_wpli_segbarlow_equal_weight | 10 / 10 | 0.9574 | 0.4574 |

Both subjects are true class `0` but receive high positive-class scores. PSD
Segment Barlow reduces sub14 from 10/10 errors to 9/10 errors, but neither WPLI
nor the equal-weight SSL ensemble reduces recurrent errors.

## Paper Recommendation

For a conservative paper narrative:

- Use WPLI Segment Barlow as the main single-branch SSL-CNN result because it
  is the strongest locked single SSL model on mean accuracy, ROC AUC, PR AUC,
  and Brier among SSL branches.
- Report PSD+WPLI equal-weight SSL ensemble as a secondary SSL stability
  result: it has the best SSL-only mean accuracy under fixed and OOF-threshold
  seed-run summaries, but it does not improve recurrent sub09/sub14 errors.
- Keep noSSL+WPLI and noSSL+PSD+WPLI as exploratory mixed supervised/SSL
  ensembles. They must not be described as SSL-only improvements.
- State clearly that no-SSL remains competitive and stronger on seed-mean ROC
  AUC, PR AUC, and Brier in this locked evaluation.
