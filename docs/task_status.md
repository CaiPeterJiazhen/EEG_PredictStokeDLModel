# Task Status

## Explainability residual-aware SSL-CNN

- Explained seed/fold samples: 190
- Output directory: `results/explainability`
- Figure directory: `results/figures/explainability`
- Topomap directory: `results/figures/explainability/topomaps`
- Model: `residualaware_highrank_swa_clsalpha1`
- Attribution target: classification logit
- Attribution method: SmoothGrad-smoothed Integrated Gradients with 64 IG steps and 4 SmoothGrad samples
- Occlusion validation: branch, state, PSD band/channel/channel-band, WPLI band/edge/node/hemisphere/network groups
- Documentation: `docs/explainability_residual_aware_ssl_cnn_results.md`
- Checkpoints: supervised fold checkpoints are required locally and are not intended for commit.

## Manuscript support package

- EEG reference baselines: `docs/eeg_reference_baseline_results.md`
- Residual threshold sensitivity: `docs/residual_threshold_sensitivity.md`
- SSL data scope note: `docs/ssl_data_scope_sensitivity.md`
- Extended sanity note: `docs/explainability_sanity_check_extension.md`
- Network-level biomarker validation: `docs/network_level_biomarker_validation.md`
- Paper tables: `results/tables/`
- Paper figures: `results/figures/paper/`
- Final model remains `residualaware_highrank_swa_clsalpha1`; no new main model was trained or selected in this package.
