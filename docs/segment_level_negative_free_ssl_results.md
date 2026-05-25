# Segment-Level Negative-Free SSL Results

## Goal

This direction moves SSL pretraining from one subject-level EO/EC averaged
feature pair to many fixed-window EEG segments. The target transfer model is
still the established patient-level `psd-fc-wpli` gated CNN. Supervised
evaluation remains patient-level LOSO.

The implementation is intentionally negative-free:

- Segment Barlow Twins
- Segment VICReg
- optional masked latent prediction on top of either alignment objective

NT-Xent is not the primary method here because in-batch negatives are
semantically fragile in a small patient EEG cohort. Two patients can share
recovery-relevant neural structure, so forcing them apart can be a poor
pretext task even when a single seed looks strong.

## Implementation

Code added:

```text
src/eeg_recovery/features/segment_features.py
src/eeg_recovery/training/segment_ssl_dataset.py
src/eeg_recovery/training/train_segment_ssl.py
scripts/17_compute_segment_features.py
scripts/17_train_segment_ssl_transfer.py
tests/test_segment_features.py
tests/test_segment_ssl_dataset.py
tests/test_segment_ssl_training.py
```

Phase 1 now computes segment-level PSD and FC/wPLI caches:

```text
data/features/segment_level/psd/
data/features/segment_level/fc/
```

Each PSD segment cache stores:

```text
subject_id
subject_key
group
stage
state
segment_index
start_sample
end_sample
affected_hand
affected_hand_aligned
source_set_path
source_fdt_path
source_set_mtime_ns
source_fdt_mtime_ns
cache_fingerprint
```

Each FC segment cache stores `wpli` and `imaginary_coherence`, both shaped
`1891 x 6`, plus the fixed edge list, band names, band ranges, segment
metadata, affected-hand alignment flag, and source `.set/.fdt` provenance.

The cache freshness check uses source `.set` and `.fdt` size/mtime metadata,
so regenerated preprocessed source files do not silently reuse stale segment
caches. The transfer script supports PSD-only segment SSL, FC/wPLI-only
segment SSL, and joint PSD+FC-wPLI segment SSL. Joint PSD+FC-wPLI loading uses
an inner join on `subject_key/stage/state/segment_index`, so a segment is used
only when both PSD and FC caches exist.

## Leakage Controls

For each outer LOSO fold:

1. The current test patient is excluded from the segment SSL pool.
2. The segment scaler is fit only on the selected fold-local SSL segments.
3. Fine-tuning still uses patient-level supervised records, not independent
   segment labels.
4. Final predictions are written as 19 patient-level LOSO predictions.
5. Segment cache files live under `data/features/segment_level/`, which is
   already covered by `.gitignore` through `data/features/`.

If `--data-scope all-patient` or `all-patient-health` is used, post-treatment
patient EEG can enter SSL only for training-fold patients. Output tables mark
this as `historical_unlabeled_pretraining=True`. This is not a prospective
baseline-only setting.

## Commands

Compute PSD segments:

```powershell
python -B scripts\17_compute_segment_features.py --config configs\paths.example.yaml --data-scope all-patient --psd-window-seconds 4 --psd-overlap-fraction 0.5
```

Compute FC/wPLI segments:

```powershell
python -B scripts\17_compute_segment_features.py --config configs\paths.example.yaml --data-scope all-patient --feature-kind fc-wpli --fc-window-seconds 8 --fc-overlap-fraction 0.5
```

Compute PSD and FC/wPLI segments in one pass:

```powershell
python -B scripts\17_compute_segment_features.py --config configs\paths.example.yaml --data-scope all-patient --feature-kind psd-fc-wpli --psd-window-seconds 4 --psd-overlap-fraction 0.5 --fc-window-seconds 8 --fc-overlap-fraction 0.5
```

Run segment-level Barlow/VICReg transfer on the required seed set:

```powershell
python -B scripts\17_train_segment_ssl_transfer.py --config configs\paths.example.yaml --data-scope all-patient --device cuda --objective barlow vicreg --segment-feature-kind psd --supervised-feature-kind psd-fc-wpli --pretrain-epochs 20 --pretrain-batch-size 16 --embedding-dim 32 --projection-dim 32 --feature-mask-prob 0.03 --noise-std 0.02 --lambda-latent 1.0 --lambda-local 0.1 --supervised-epochs 100 --patience 100 --supervised-lr 0.002 --supervised-weight-decay 0.00001 --dropout 0 --transfer-mode finetune --seeds 0 1 2 3 7 13
```

For wPLI-only segment SSL transfer, change `--segment-feature-kind` to
`fc-wpli`. For joint PSD+FC-wPLI segment SSL, use `psd-fc-wpli` after both
cache directories have been generated.

Expected per-run output names use the `segssl_` prefix:

```text
results/predictions/dl_loso_predictions_segssl_<objective>_<settings>_seed<seed>.csv
results/metrics/dl_model_comparison_segssl_<objective>_<settings>_seed<seed>.csv
results/ssl/segment_ssl_history_segssl_<objective>_<settings>_seed<seed>.csv
results/training_logs/dl_loss_history_segssl_<objective>_<settings>_seed<seed>.csv
results/figures/dl_loss_curve_segssl_<objective>_<settings>_seed<seed>.png
```

The script also writes seed ensemble outputs plus:

```text
results/metrics/segssl_seed_summary_<objective>_<data_scope>_<segment_feature_kind>.csv
results/metrics/segssl_subject_error_frequency_<objective>_<data_scope>_<segment_feature_kind>.csv
results/metrics/segssl_transfer_summary_<segment_feature_kind>.csv
```

## Current Comparison Baselines

Initial PSD-only segment Barlow real runs are in progress. The final comparison
table should not be filled until the required seed set and any planned VICReg
or FC/wPLI segment runs complete. Segment-level FC/wPLI cache generation has
not been run yet. The comparison targets remain:

| Method | Shared-Seed Mean Accuracy | Accuracy Std | Minimum Accuracy | Ensemble Accuracy |
| --- | ---: | ---: | ---: | ---: |
| no-SSL stable protocol | 0.7193 | 0.1035 | 0.5789 | 0.7895 |
| multi-task VICReg | 0.7544 | 0.0430 | 0.6842 | 0.7895 |
| tuned feature-space Barlow | 0.7474 | 0.0544 | 0.6842 | 0.7895 |
| segment-level Barlow | pending | pending | pending | pending |
| segment-level VICReg | pending | pending | pending | pending |
| segment-level Barlow/VICReg + masked latent prediction | pending | pending | pending | pending |

After real runs complete, update this table using actual seed mean, seed std,
seed minimum, seed ensemble metrics, and per-subject error frequency. If the
new method does not improve the baseline, report that directly.

## Verification

The new targeted tests pass:

```text
python -B -m pytest tests/test_segment_features.py tests/test_segment_ssl_dataset.py tests/test_segment_ssl_training.py -v -p no:cacheprovider
```

Result during implementation:

```text
19 passed
```
