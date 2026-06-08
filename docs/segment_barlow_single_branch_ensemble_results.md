# Segment Barlow Single-Branch Ensemble Results

## Goal

This run completes the latest Segment Barlow SSL model-selection step without
retraining supervised models. It reuses the existing six-seed LOSO prediction
CSVs for:

- no-SSL stable `psd-fc-wpli` gated CNN
- PSD Segment Barlow SSL-CNN transferred into the same patient-level
  `psd-fc-wpli` gated CNN
- FC/wPLI Segment Barlow SSL-CNN transferred into the same patient-level
  `psd-fc-wpli` gated CNN

The new scripts generate probability-level equal-weight ensembles, strict
non-test-optimized threshold summaries, per-subject error frequencies, and
small-sample statistical comparisons.

## Why Attention MIL Stops Here

The WPLI Segment Barlow attention MIL seed0 pilot failed as a main-line method:

| method | accuracy | balanced_accuracy | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: |
| WPLI Segment Barlow attention MIL seed0 | 0.5263 | 0.5167 | 0.4778 | 0.5202 |

It is therefore not expanded to six seeds. The result is kept as a negative
pilot and supplementary context only.

## Why Dual Encoder Is Not Main-Line

The seed0 dual PSD+WPLI Segment Barlow transfer did not improve accuracy or
balanced accuracy over the corresponding no-SSL and single-branch pilots:

| dual run | accuracy | balanced_accuracy | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: |
| dual finetune lr=0.002 | 0.6842 | 0.6722 | 0.7333 | 0.7557 |
| dual freeze lr=0.002 | 0.6316 | 0.6222 | 0.6667 | 0.7475 |
| dual finetune lr=0.001 | 0.6842 | 0.6722 | 0.6889 | 0.6551 |

The likely failure mode is branch conflict under very small sample size:
jointly fine-tuning two SSL-initialized encoders can disturb a supervised
fusion model that already has only 19 patient-level labels.

## Why Single-Branch SSL-CNN

Single-branch Segment Barlow pretraining is the safer main line because it
uses one SSL-initialized branch at a time while the supervised model still
receives both PSD and WPLI inputs. This preserves the established
patient-level `psd-fc-wpli` gated CNN surface while reducing the optimization
instability of dual encoder fine-tuning.

Six-seed fixed-threshold results:

| model_group | accuracy_mean | accuracy_std | accuracy_max | balanced_accuracy_mean | ROC AUC mean | PR AUC mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_ssl_stable_cnn | 0.7895 | 0.0608 | 0.8421 | 0.7806 | 0.8389 | 0.8453 |
| psd_segbarlow_ssl_cnn | 0.7982 | 0.0707 | 0.8947 | 0.7889 | 0.7796 | 0.7511 |
| wpli_segbarlow_ssl_cnn | 0.7982 | 0.0562 | 0.8421 | 0.7880 | 0.8185 | 0.8171 |

PSD and WPLI Segment Barlow both remain competitive, but neither single branch
dominates every metric.

## Why Probability Ensemble

The ensemble step tests whether PSD SSL and WPLI SSL are complementary after
supervised patient-level fine-tuning. Probability-level fusion is deliberately
simpler than feature-level dual fusion: each model makes a patient-level score,
then scores are averaged. This avoids forcing two pretrained encoders to share
one small supervised optimization problem.

Six-seed fixed-threshold ensemble summaries:

| model_group | accuracy_mean | accuracy_std | balanced_accuracy_mean | ROC AUC mean | PR AUC mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| psd_wpli_segbarlow_equal_weight | 0.8158 | 0.0588 | 0.8074 | 0.8185 | 0.8252 |

The SSL-only PSD+WPLI ensemble improves the six-seed mean accuracy over both
single SSL branches, while keeping accuracy std moderate. The noSSL+WPLI
seed-mean score gives the best fixed-threshold ROC AUC and PR AUC among the
reported seed-mean candidates, but it includes the supervised no-SSL model and
should be described separately from the SSL-only ensemble.

## Seed-Mean Patient-Level Results

These rows average probabilities across the six seeds at patient level, then
evaluate 19 subjects once. This is not treated as 114 independent samples.

| model_group | threshold_method | threshold | accuracy | balanced_accuracy | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_ssl_stable_cnn | fixed_0.5 | 0.5000 | 0.8421 | 0.8333 | 0.8778 | 0.8954 | 0.1684 |
| psd_segbarlow_ssl_cnn | fixed_0.5 | 0.5000 | 0.8421 | 0.8333 | 0.8444 | 0.8515 | 0.1833 |
| wpli_segbarlow_ssl_cnn | fixed_0.5 | 0.5000 | 0.7895 | 0.7778 | 0.8222 | 0.8228 | 0.1883 |
| psd_wpli_segbarlow_equal_weight | fixed_0.5 | 0.5000 | 0.8421 | 0.8333 | 0.8444 | 0.8515 | 0.1837 |

## Threshold Calibration

Two threshold policies are reported:

- `fixed_0.5`: primary fixed threshold, no label tuning.
- `leave_one_seed_out_oof`: a fallback OOF-style calibration because strict
  inner training-only threshold predictions are not available. For each held
  seed, the threshold is selected from the other five seed prediction sets and
  then applied to the held seed. The seed-mean rows use the median of those
  leave-one-seed thresholds.

No primary result searches for an optimal threshold on the final 19-subject
seed-mean predictions. Direct test-label optimized threshold selection is
implemented only as an exploratory-disabled path in code and is not reported as
a main result.

Seed-mean OOF-threshold highlights:

| model_group | threshold | accuracy | balanced_accuracy | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| psd_segbarlow_ssl_cnn | 0.5000 | 0.8421 | 0.8333 | 0.8444 | 0.8515 |
| wpli_segbarlow_ssl_cnn | 0.5120 | 0.8421 | 0.8333 | 0.8222 | 0.8228 |
| psd_wpli_segbarlow_equal_weight | 0.4989 | 0.8421 | 0.8333 | 0.8444 | 0.8515 |

OOF-weighted PSD/WPLI ensembling is disabled because no strict inner/OOF
weight-selection mechanism is available. Equal-weight ensembles are the primary
fusion results.

## Error Frequency

The per-subject error table is written to:

```text
results/metrics/segment_barlow_subject_error_frequency.csv
```

The focused sub09/sub14 summary is written to:

```text
results/metrics/sub09_sub14_model_scores_summary.csv
```

Both sub09 and sub14 remain difficult. Under fixed-threshold six-seed scoring,
sub09 is wrong in all six runs for every reported model group. sub14 is wrong
in all six runs for no-SSL, WPLI Segment Barlow, PSD+WPLI ensemble,
noSSL+WPLI ensemble, and noSSL+PSD+WPLI ensemble; PSD Segment Barlow reduces
sub14 to five errors out of six, but not enough to change the overall
conclusion.

## Statistical Comparison

The statistical comparison file is:

```text
results/metrics/segment_barlow_statistical_comparison.csv
```

It uses patient-level bootstrap resampling over 19 subjects, paired
correctness/sign comparisons, paired score bootstrap, and a random-label
permutation baseline. Seeds are not treated as independent patients.

Selected rows:

| analysis | model | metric | observed | 95% CI / p |
| --- | --- | --- | ---: | --- |
| bootstrap | psd_wpli_segbarlow_equal_weight | accuracy | 0.8421 | 0.6842 to 1.0000 |
| paired correctness | WPLI vs PSD+WPLI | correctness difference | 0.0526 | sign-test p=1.0000 |

The paired tests are intentionally conservative: with 19 subjects, most model
pairs differ on only zero or one subject at fixed threshold.

## Checkpoint Reuse

The updated checkpoint contract requires branch-specific SSL encoder
checkpoints under:

```text
results/checkpoints/ssl_encoders/
```

Checkpoint files are ignored by git through `results/`, `*.pt`, `*.pth`, and
`*.ckpt` patterns. They should not be committed.

Each future checkpoint must carry metadata including:

```text
branch
ssl_objective
segment_ssl_method
base_seed
effective_seed
fold_index
test_subject_id
excluded_subject_id
ssl_data_scope
historical_unlabeled_pretraining
segment_feature_kind
supervised_feature_kind
encoder_kind
embedding_dim
dropout
projection_dim
pretrain_epochs
pretrain_lr
batch_size
feature_mask_prob
noise_std
lambda_latent
lambda_local
n_ssl_segments
source_feature_manifest_hash
created_at
```

Current local cache status:

| cache family | files | seed coverage | strict reuse status |
| --- | ---: | --- | --- |
| `segssl_barlow_psd_*` | 19 | seed0 only | legacy metadata; not reusable under new strict validation |
| `segssl_barlow_wpli_*` | 19 | seed0 only | legacy metadata; not reusable under new strict validation |
| `segssl_wpli_barlow_foldstrict_v1_*` | 19 | seed0 only | legacy metadata; not reusable under new strict validation |

The six-seed prediction and metric CSVs are available and were used for model
selection. Encoder checkpoint reuse is not complete for seeds 1, 2, 3, 7, and
13. For those runs, record the status as: metrics available, encoder
checkpoint unavailable for reuse. To fill the cache later without rerunning
supervised transfer, use `scripts/17_train_segment_ssl_transfer.py` with
`--ssl-only-cache --save-ssl-encoders`.

## Output Files

Main generated files:

```text
results/predictions/ensemble_predictions_psd_wpli_segbarlow_equal_weight.csv
results/metrics/ensemble_metrics_psd_wpli_segbarlow_equal_weight.csv
results/metrics/segment_barlow_ensemble_threshold_summary.csv
results/metrics/segment_barlow_model_selection_summary.csv
results/metrics/segment_barlow_subject_error_frequency.csv
results/metrics/sub09_sub14_model_scores_summary.csv
results/metrics/segment_barlow_statistical_comparison.csv
```

New code and tests:

```text
src/eeg_recovery/training/segment_barlow_model_selection.py
scripts/22_ensemble_segment_barlow_predictions.py
scripts/23_compare_segment_barlow_models.py
tests/test_segment_barlow_ensemble.py
tests/test_threshold_calibration.py
tests/test_ssl_checkpoint_reuse.py
```

## Paper Recommendation

Use the single-branch Segment Barlow SSL-CNN results and the equal-weight
PSD+WPLI probability ensemble as the main SSL model-selection story. The
noSSL+WPLI seed-mean ensemble can be reported as a strong mixed supervised/SSL
candidate because it has the best ROC AUC and PR AUC in this selection run,
but it should not be described as an SSL-only improvement.

Supplementary ablations:

- Dual Segment Barlow encoder transfer
- WPLI attention MIL seed0 negative pilot
- OOF-threshold sensitivity analysis
- noSSL+PSD+WPLI mixed ensemble

Recommended explainability next steps:

- WPLI edge-band attribution for the WPLI branch
- PSD channel-frequency attribution for the PSD branch
- focused review of sub09 and sub14 because they remain the dominant repeated
  error subjects
