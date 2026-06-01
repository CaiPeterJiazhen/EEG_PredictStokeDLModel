# Explainability Upload Manifest

This commit uploads the analysis artifacts needed for web-based review of the residual-aware Patient-level Barlow SSL-CNN explainability run.

## Uploaded Summary Artifacts

Core document:

- `docs/explainability_residual_aware_ssl_cnn_results.md`
- `docs/task_status.md`

Summarized explainability tables:

- `results/explainability/explained_predictions.csv`
- `results/explainability/branch_state_gate_weights.csv`
- `results/explainability/psd_channel_band_importance.csv`
- `results/explainability/psd_frequency_importance.csv`
- `results/explainability/psd_channel_frequency_top_features.csv`
- `results/explainability/psd_seed_stability.csv`
- `results/explainability/wpli_top_edges.csv`
- `results/explainability/wpli_band_importance.csv`
- `results/explainability/wpli_node_importance.csv`
- `results/explainability/wpli_network_group_importance.csv`
- `results/explainability/wpli_seed_stability.csv`
- `results/explainability/occlusion_branch_state.csv`
- `results/explainability/occlusion_psd_band_channel.csv`
- `results/explainability/occlusion_wpli_edge_node_band.csv`
- `results/explainability/attribution_stability_summary.csv`
- `results/explainability/sanity_check_summary.csv`
- `results/explainability/psd_biomarker_validation.csv`
- `results/explainability/wpli_biomarker_validation.csv`

Figures:

- `results/figures/explainability/*.png`

## Not Uploaded

The raw long attribution tables are intentionally not uploaded because they are too large for practical GitHub review:

- `results/explainability/psd_attribution_long.csv` (~247 MB)
- `results/explainability/wpli_edge_attribution_long.csv` (~527 MB)

They remain available locally and can be regenerated with:

```bash
python -B scripts/31_explain_residual_aware_ssl_cnn.py --config configs/paths.example.yaml --device cuda --model-group residualaware_highrank_swa_clsalpha1 --seeds 0 1 2 3 4 5 7 13 21 42 --method integrated_gradients smoothgrad occlusion --target classification_logit --output-tag residualaware_10seed --ig-steps 64 --smoothgrad-samples 4
```

Checkpoint binaries are also intentionally not uploaded:

- `results/checkpoints/**/*.pt`
- `results/checkpoints/**/*.pth`
- `results/checkpoints/**/*.ckpt`

