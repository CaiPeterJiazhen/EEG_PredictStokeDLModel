# Task Status

## Explainability residual-aware SSL-CNN

- Explained seed/fold samples: 190
- Output directory: `results/explainability`
- Figure directory: `results/figures/explainability`
- Model: `residualaware_highrank_swa_clsalpha1`
- Attribution target: classification logit
- Attribution method: Integrated Gradients with 64 steps plus SmoothGrad with 4 samples
- Occlusion validation: branch, state, PSD band/channel/channel-band, WPLI band/edge/node/hemisphere/network groups
- Documentation: `docs/explainability_residual_aware_ssl_cnn_results.md`
- Checkpoints: supervised fold checkpoints are required locally and are not intended for commit.
