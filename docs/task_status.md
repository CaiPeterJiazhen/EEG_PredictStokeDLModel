# tACS EEG Proportional Recovery Task Status

This file tracks durable project status across threads and subagents. Update it after each completed task. Do not use chat history as the only status record.

## Current State

- Project context document created: `docs/project_context.md`
- Implementation plan created: `docs/implementation_plan.md`
- Tasks 1-10 source code and tests have been implemented.
- Formal PSD and FC feature files, Task 9 PSD-baseline ML result CSVs, and Task 10 DL result CSVs have been generated in the project directory.
- No cache, bytecode, or temporary files should remain in the project directory.
- Project directory is currently not a git repository.

## Task Board

| Task | Status | Owner | Last updated | Notes |
|---|---|---|---|---|
| 1. Project scaffold and configuration | Completed | Codex | 2026-05-13 | Created package layout, configs, `.gitignore`, config tests; `python -m pytest tests/test_config.py -v -p no:cacheprovider` passed with 4 tests |
| 2. Metadata parsing and label generation | Completed | Codex + subagent | 2026-05-13 | Created metadata readers, subject ID normalization, label generation, model-input helper; `python -B -m pytest tests/test_config.py tests/test_metadata_labels.py -v -p no:cacheprovider` passed with 13 tests |
| 3. EEG file indexing | Completed | Codex + subagent | 2026-05-13 | Created EEG file index records, EO/EC state parsing, patient/health grouping, supervised baseline coverage validation; `python -B -m pytest tests/test_config.py tests/test_metadata_labels.py tests/test_eeg_index.py -v -p no:cacheprovider` passed with 20 tests |
| 4. EEGLAB metadata and data reader | Completed | Codex + subagent | 2026-05-13 | Added `.set` metadata reader, `.fdt` float32 reader, length/context validation, stale datfile fallback; `python -B -m pytest tests/test_config.py tests/test_metadata_labels.py tests/test_eeg_index.py tests/test_eeglab_io.py -v -p no:cacheprovider` passed with 31 tests |
| 5. 62-channel mapping and hemisphere flip | Completed | Codex + subagent | 2026-05-14 | Added fixed 62-channel mapping config, immutable mirror map loader, affected-hand hemisphere flip, and malformed-config/order tests; `python -B -m pytest tests/test_config.py tests/test_metadata_labels.py tests/test_eeg_index.py tests/test_eeglab_io.py tests/test_channel_mapping.py -v -p no:cacheprovider` passed with 46 tests |
| 6. PSD feature extraction | Completed | Codex + subagent | 2026-05-14 | Added Welch PSD calculation, PSD writer, baseline PSD script, provenance metadata, and PSD tests; `python -B -m pytest tests/test_config.py tests/test_metadata_labels.py tests/test_eeg_index.py tests/test_eeglab_io.py tests/test_channel_mapping.py tests/test_psd_features.py -v -p no:cacheprovider` passed with 54 tests |
| 7. Functional connectivity feature extraction | Completed | Codex + subagent | 2026-05-14 | Added deterministic upper-triangle edge list, STFT-based wPLI and imaginary coherence, FC writer, baseline FC script, and FC tests; `python -B -m pytest tests/test_config.py tests/test_metadata_labels.py tests/test_eeg_index.py tests/test_eeglab_io.py tests/test_channel_mapping.py tests/test_psd_features.py tests/test_connectivity_shapes.py -v -p no:cacheprovider` passed with 61 tests |
| 8. LOSO split and leakage tests | Completed | Codex + subagent | 2026-05-14 | Added LOSO fold generation and fold-local transformer fitting guard with y/fit-param masking; `python -B -m pytest tests/test_config.py tests/test_metadata_labels.py tests/test_eeg_index.py tests/test_eeglab_io.py tests/test_channel_mapping.py tests/test_psd_features.py tests/test_connectivity_shapes.py tests/test_loso_leakage.py -v -p no:cacheprovider` passed with 74 tests |
| 9. Traditional ML baselines | Completed | Codex + subagent | 2026-05-14 | Added PSD/FC tabular feature loaders, fold-local baseline preprocessing, traditional ML registry with optional boosted-model skip records, LOSO training, output writers, and baseline tests; `python -B -m pytest tests/test_config.py tests/test_metadata_labels.py tests/test_eeg_index.py tests/test_eeglab_io.py tests/test_channel_mapping.py tests/test_psd_features.py tests/test_connectivity_shapes.py tests/test_loso_leakage.py tests/test_baseline_training.py -v -p no:cacheprovider` passed with 87 tests |
| 10. Deep learning models without SSL | Completed | Codex + subagent | 2026-05-14 | Added lightweight PyTorch PSD/FC encoders, dual-state, 3D, and multimodal fusion models, gated EO/EC fusion weights, supervised LOSO training with early stopping and CUDA device selection, DL output writers, manual training script, and model tests; `python -B -m pytest tests -v -p no:cacheprovider` passed with 103 tests |
| 11. Self-supervised pretraining | Pending | Unassigned | 2026-05-13 | Strict LOSO SSL exclusion must be supported |
| 12. Evaluation metrics and statistical tests | Pending | Unassigned | 2026-05-13 | Include bootstrap CI and permutation test |
| 13. Explainability | Pending | Unassigned | 2026-05-13 | PSD, FC, and clinical validation layers |
| 14. End-to-end pipeline and documentation | Pending | Unassigned | 2026-05-13 | README and pipeline runners |

## Fixed Decisions

- Use the 19 subjects in `F:\CJZFile\EEG_M1\19例患者脑电数据完整性检查.xlsx` as the supervised cohort.
- Use actual 62 EEG channels, not 64.
- Deleted channels are `M1` and `M2`.
- `*1.set` is EO.
- `*2.set` is EC.
- Main supervised EEG input uses baseline patient EEG only.
- Label threshold is residual median `1.5`.
- Positive label uses `Residual <= 1.5`.
- Patient-level LOSO-CV is mandatory.

## Open Questions

No blocking questions are currently open.

## Cleanliness Log

- 2026-05-13: Created only formal docs under `docs/`. No temporary files intentionally created.
- 2026-05-13: Completed Task 1 scaffold/configuration files and tests. Test cache/bytecode artifacts should be removed after verification.
- 2026-05-13: Completed Task 2 metadata parsing and label generation. Verified no cache directories or temporary project files remain after tests.
- 2026-05-13: Completed Task 3 EEG file indexing. Verified no cache directories, bytecode files, or temporary project files remain after tests.
- 2026-05-13: Completed Task 4 EEGLAB metadata and data reader. Verified no cache directories, bytecode files, or temporary project files remain after tests.
- 2026-05-14: Completed Task 5 channel mapping and hemisphere flip. Verified with no-cache test run and final scan found no cache, bytecode, or temporary project files.
- 2026-05-14: Completed Task 6 PSD feature extraction. Verified with no-cache test run and final scan found no PSD output directory, cache, bytecode, or temporary project files.
- 2026-05-14: Completed Task 7 functional connectivity feature extraction. Verified with no-cache test run and final scan found no FC output directory, cache, bytecode, or temporary project files.
- 2026-05-14: Completed Task 8 LOSO split and leakage tests. Verified with no-cache test run and final scan found no results directory, cache, bytecode, or temporary project files.
- 2026-05-14: Completed Task 9 traditional ML baselines. Verified Task 9 tests, script help, and Tasks 1-9 regression with no-cache runs; final scan found no results directory, cache, bytecode, or temporary project files.
- 2026-05-14: Ran real PSD baseline training. Generated 38 PSD feature files and Task 9 prediction/metric CSVs; final scan found no cache, bytecode, or temporary project files.
- 2026-05-14: Completed Task 10 deep learning models without SSL. Verified Task 10 tests, script help, full regression with no-cache runs, and spec/quality review approval; final scan found no cache, bytecode, or temporary project files.
- 2026-05-14: Ran real Task 10 PSD dual-state concat deep-learning LOSO training with `--device cuda`. Generated `dl_loso_predictions.csv` and `dl_model_comparison.csv`; final scan found no cache, bytecode, or temporary project files.
- 2026-05-14: Extended Task 10 deep learning support for FC-only `fc-both` and PSD+FC multimodal feature kinds (`psd-fc-wpli`, `psd-fc-icoh`, `psd-fc-both`). Verified with 103-test full regression and spec/quality review approval.
- 2026-05-14: Ran real CUDA DL concat experiments for `psd`, `fc-wpli`, `fc-icoh`, `fc-both`, `psd-fc-wpli`, `psd-fc-icoh`, and `psd-fc-both`. Preserved each run under result-specific CSV filenames and wrote `results/metrics/dl_model_comparison_all_concat.csv`.
