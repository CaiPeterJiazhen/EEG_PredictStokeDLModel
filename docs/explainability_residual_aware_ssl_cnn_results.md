# Explainability: residual-aware Patient-level Barlow SSL-CNN

## Model Explained

The explained model is `residualaware_highrank_swa_clsalpha1`: Patient-level Barlow SSL-CNN with residual-aware auxiliary fine-tuning, SWA, and classification-head inference. The input and backbone are locked to PSD+WPLI EO/EC gated CNN, with no qEEG branch, clinical branch, MIL module, or dual encoder.

All 10 seeds and 19 LOSO folds were analyzed: 190 seed/fold explanations. Supervised fold checkpoints were saved under `results/checkpoints/supervised/residualaware_highrank_swa_clsalpha1/` and are intentionally not part of the committed artifacts.

## Attribution Target

Attribution targets the binary classification logit. The residual regression, pairwise ranking, and soft-label heads are training-time auxiliary objectives only; they are not interpreted as the main prediction target.

## Methods

Integrated Gradients used a zero baseline after fold-local feature scaling and 64 interpolation steps. SmoothGrad averaged 4 noisy IG samples. Group occlusion zeroed branch, state, PSD band/channel/channel-band, and WPLI band/edge/node/interhemispheric/network groups. Sanity checks included classifier-head randomization and input permutation.

Primary output files:

- `results/explainability/psd_attribution_long.csv`
- `results/explainability/wpli_edge_attribution_long.csv`
- `results/explainability/occlusion_branch_state.csv`
- `results/explainability/attribution_stability_summary.csv`
- `results/explainability/psd_biomarker_validation.csv`
- `results/explainability/wpli_biomarker_validation.csv`

Figures are in `results/figures/explainability/`.

## PSD Findings

The largest locked-band PSD attributions were dominated by EO low-frequency delta and EO beta-high features. The top locked-band channel-frequency features were:

| state | channel | frequency_hz | band | mean signed attribution | mean absolute attribution |
|---|---:|---:|---|---:|---:|
| EO | F5 | 1.5 | Delta | -0.001444 | 0.001727 |
| EO | T7 | 3.0 | Delta | -0.001364 | 0.001649 |
| EO | TP7 | 24.0 | Beta High | -0.001006 | 0.001571 |
| EO | F5 | 2.0 | Delta | -0.001291 | 0.001559 |
| EO | FC5 | 3.5 | Delta | -0.001177 | 0.001531 |

Mean absolute attribution by locked PSD band:

| band | mean absolute attribution |
|---|---:|
| Beta High | 0.000362 |
| Delta | 0.000360 |
| Beta Low | 0.000311 |
| Beta Medium | 0.000305 |
| Theta | 0.000297 |
| Alpha | 0.000264 |

Some highest raw PSD rows fall into `Other` frequencies outside the locked Delta/Theta/Alpha/Beta definitions. These are reported in CSVs but are not promoted as primary biomarkers.

## WPLI Findings

WPLI explanations were strongest in EC beta-band connectivity, especially frontal-central and frontal-parietal edges. The top WPLI edge-band findings were:

| state | channel_i | channel_j | band | mean signed attribution | mean absolute attribution |
|---|---|---|---|---:|---:|
| EC | F8 | CP1 | Beta High | 0.000154 | 0.002128 |
| EC | FT7 | P2 | Beta Medium | -0.001331 | 0.002119 |
| EC | C5 | P2 | Beta Medium | -0.000529 | 0.002111 |
| EC | F4 | C5 | Beta High | -0.000062 | 0.002097 |
| EC | F8 | FC4 | Beta High | 0.000994 | 0.002095 |

Mean absolute WPLI attribution by band:

| band | mean absolute attribution |
|---|---:|
| Beta High | 0.000716 |
| Beta Medium | 0.000659 |
| Beta Low | 0.000658 |
| Alpha | 0.000523 |
| Theta | 0.000490 |
| Delta | 0.000464 |

The highest node-level WPLI importance concentrated around frontal and fronto-central channels: F8, F4, F6, CP4, F2, FC2, FC6, FZ, FP1, P2, PO7, and FC4.

## Branch And State Effects

Gate weights and occlusion show a split use of modalities:

| quantity | result |
|---|---:|
| PSD EO gate mean | 0.9448 |
| PSD EC gate mean | 0.0552 |
| WPLI EO gate mean | 0.0053 |
| WPLI EC gate mean | 0.9947 |
| WPLI branch occlusion mean delta_probability | 0.1149 |
| EC state occlusion mean delta_probability | 0.1090 |
| EO state occlusion mean delta_probability | -0.1009 |
| PSD branch occlusion mean delta_probability | -0.1322 |

Interpretation: the model primarily routes PSD through EO and WPLI through EC. Removing WPLI or EC lowers the positive-class score on average, while removing PSD or EO raises it on average. Since signed occlusion reflects score direction, not pure usefulness, loss deltas and attribution stability should be used together.

## Stability

PSD attribution maps were moderately stable across seeds, while WPLI map-level correlations were high but exact top-edge overlap was low:

| feature family | mean seed-pair Spearman | mean top-20 Jaccard |
|---|---:|---:|
| PSD | 0.7509 | 0.3178 |
| WPLI | 0.8888 | 0.0442 |

PSD top-k selection was strongest for EO TP7 Delta, EO T7 Delta, EO FPZ Beta High, EO F5 Beta High, EO FC5 Beta High, and related EO beta/delta features. WPLI should be interpreted more as a stable beta-band frontal/central/parietal network pattern than as a single definitive edge, because exact top-edge selection is seed-variable.

## Sanity Checks

Input permutation degraded attribution correlations, supporting input-specific attribution structure:

| sanity check | PSD EO mean correlation | WPLI EO mean correlation |
|---|---:|---:|
| input permutation | 0.1859 | 0.1678 |
| classifier randomization | 0.3384 | 0.4706 |

Classifier randomization was mixed: mean signed correlations dropped, but mean absolute correlations remained high for several samples. Therefore, the interpretation is reported conservatively. Features should be promoted only when they are stable across seeds and supported by occlusion or biomarker validation.

## Biomarker Validation

PSD top features did not survive FDR correction for signed-distance association. Several EO C5/FPZ/FC5 beta-high or delta features had low uncorrected group-permutation p-values, but the FDR-adjusted signed-distance p-values were not significant.

WPLI beta-high edges showed stronger association signals. The best signed-distance FDR values were near but above 0.05, including EC F6-FC2 Beta High, EC F4-C4 Beta High, and EC FC6-TP8 Beta High. This supports beta-band WPLI as a plausible candidate but not a confirmed biomarker in this n=19 cohort.

## Stroke EEG Interpretation

The pattern is biologically plausible but exploratory. Delta/theta slow activity can reflect injury-related cortical dysfunction, while beta-band frontal/central and motor-adjacent connectivity is compatible with motor recovery and neuromodulation hypotheses such as beta-band tACS. The strongest WPLI explanations being EC beta-band edges is consistent with a resting-state connectivity interpretation. These are model explanations and association checks, not causal evidence.

## Figures

Generated figures include:

- `psd_channel_frequency_heatmap_all.png`
- `psd_channel_frequency_heatmap_positive.png`
- `psd_channel_frequency_heatmap_negative.png`
- `psd_band_delta_top_channels.png`
- `psd_band_theta_top_channels.png`
- `psd_band_alpha_top_channels.png`
- `psd_band_beta_low_top_channels.png`
- `psd_band_beta_medium_top_channels.png`
- `psd_band_beta_high_top_channels.png`
- `wpli_connectome_top20_alpha.png`
- `wpli_connectome_top20_beta_medium.png`
- `wpli_connectome_top20_theta.png`
- `wpli_node_importance_top25.png`
- `branch_state_occlusion_barplot.png`
- `stability_topk_selection_frequency.png`
- `stability_seed_to_seed_correlation.png`
- `sub09_sub14_error_attribution_comparison.png`

Full scalp topomaps are represented as channel barplots because electrode coordinate layout is not implemented in this repository.

## Limitations

- Attribution is not causal.
- Cohort size is n=19.
- LOSO folds and seeds produce nontrivial variability.
- WPLI exact edge selection is unstable despite stable broader beta-band network importance.
- Classifier-randomization sanity checks are mixed.
- There is no external validation.

## Conclusion

The most defensible explanation is not a single channel or edge. The selected SSL-CNN appears to rely on EO PSD delta/beta features plus EC WPLI beta-band frontal-central/frontal-parietal connectivity. These patterns are plausible for stroke recovery EEG analysis and beta-band neuromodulation hypotheses, but should be treated as hypothesis-generating biomarkers requiring replication.
