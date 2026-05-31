# SSL-CNN Barlow Improvement Strategy

Date: 2026-05-29

## Current Observation

Using the restored original `sub09`/`sub14` feature files and unchanged PSD+WPLI inputs:

- no-SSL CNN, seeds 0-5 mean: accuracy 0.7632, balanced accuracy 0.7546, Brier 0.2063, ROC AUC 0.7630, PR AUC 0.7350.
- Patient-level Barlow SSL+CNN, seeds 0-5 mean: accuracy 0.7895, balanced accuracy 0.7787, Brier 0.2040, ROC AUC 0.7537, PR AUC 0.7319.
- seed ensemble6 gives identical threshold metrics for no-SSL and Barlow, but no-SSL has better Brier/ROC/PR.
- `sub09` and `sub14` remain recurrent false positives in Barlow, often with high positive scores.

Before additional model selection, feature-SSL pretraining must be deterministic. The previous implementation seeded the SSL augmentation/order but not the torch initialization of the Barlow encoder and projection head, so the same seed could produce different patient predictions. This has now been corrected in `src/eeg_recovery/training/train_feature_ssl.py`.

## Literature Signals

Barlow Twins is directly relevant because it avoids negative samples and collapse by matching the cross-correlation of two augmented views to the identity, and was reported to work well in low-label and transfer settings. This supports keeping the Barlow objective, but it does not imply that plain full fine-tuning is optimal for a 19-subject LOSO task.

EEG-specific work using Barlow Twins for motor-imagery classification reports that a MultiResolutionCNN backbone with Barlow loss can outperform conventional supervised learning, and notes sensitivity to training hyperparameters. This supports testing CNN backbones that are better matched to EEG-derived feature structure, while keeping Barlow and the same feature tensors.

Biomedical-signal surveys emphasize that SSL is useful when annotation is scarce, but the downstream transfer protocol, augmentations, and robustness checks are central. This matches our observed issue: the encoder sometimes improves classification accuracy but does not consistently improve probability ranking/calibration.

Lead-fusion Barlow Twins for multi-lead ECG is the closest architectural analogy to PSD+WPLI. It uses fused Barlow losses and downstream multi-branch concatenation to exploit multiple signal views. For this repo, the direct analogue is branch-aware Barlow for PSD and WPLI plus fused Barlow on the concatenated embedding, followed by a downstream branch-aware classifier.

Fine-tuning literature argues against one learning rate for all layers. Layer-wise or group-wise learning rates are especially relevant here because the pretrained branch encoders should move slowly, while EO/EC gates, adapters, and classifier should adapt faster.

## Prioritized Experiments

### 0. Reproducibility Gate

Status: implemented.

Action:
- Seed numpy and torch before feature-SSL model/projection initialization.
- Treat all future seed results as invalid unless they come from the deterministic code path.

Required rerun:
- Re-run no-SSL and Barlow under the deterministic code path using the same 6 or 10 seeds.

### 1. Discriminative Fine-Tuning

Status: implemented and seed-piloted.

Rationale:
- Current fine-tuning applies one learning rate to every trainable parameter.
- A safer transfer protocol is lower LR for pretrained encoders and higher LR for classifier/gates.

Implementation:
- Add supervised optimizer groups:
  - branch encoders: `encoder_lr_multiplier=0.1`
  - EO/EC gates and branch projections: configurable `transfer_head_lr_multiplier`
  - classifier/adapters: `1.0`
- Add a warmup schedule:
  - optional first-N-epoch encoder freeze
  - later epochs unfreeze encoders with low LR

Implementation evidence:
- `SupervisedTrainingConfig` now supports `encoder_lr_multiplier`,
  `transfer_head_lr_multiplier`, and `freeze_pretrained_encoder_epochs`.
- `scripts/07_train_feature_ssl_transfer.py` exposes the same options.
- Tests cover optimizer LR grouping, encoder warmup freeze/unfreeze, script help,
  deterministic feature-SSL pretraining, and Windows-safe shortened output names.

Pilot evidence, original features, all-patient Barlow, seed0:

| Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL seed0 | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.7111 | 0.7346 | 0.2697 |
| Barlow deterministic seed0 | 0.6842 | 0.6722 | 0.9000 | 0.4444 | 0.7000 | 0.7200 | 0.2271 |
| Barlow encoder-slow seed0 | 0.6842 | 0.6722 | 0.9000 | 0.4444 | 0.7444 | 0.7857 | 0.2060 |
| Barlow discriminative-warmup seed0 | 0.6842 | 0.6722 | 0.9000 | 0.4444 | 0.7667 | 0.8060 | 0.2024 |

Pilot evidence, seed1:
- Discriminative-warmup seed1: Acc 0.7368, Bal Acc 0.7278, Sens 0.9000,
  Spec 0.5556, ROC AUC 0.7000, PR AUC 0.7024, Brier 0.2599.
- Same-seed old deterministic Barlow was stronger on fixed-threshold metrics and
  Brier, although discriminative-warmup improved PR AUC.

Expected benefit:
- Lower encoder LR consistently helps ranking/calibration in seed0, but the
  aggressive warmup/head-LR setting is not yet stable enough for a full 10-seed
  run. Next transfer work should prioritize a less aggressive adapter/bridge
  before expanding this exact protocol.

### 2. SSL Bridge Classifier

Status: implemented and seed-piloted.

Rationale:
- Current transfer copies SSL branch encoder weights into the supervised CNN, then full fine-tunes.
- This can overwrite useful SSL features in a tiny supervised fold.

Implementation:
- Keep a frozen SSL encoder copy and a trainable supervised encoder copy.
- Classifier input becomes:
  - trainable supervised embedding
  - frozen SSL embedding
  - absolute difference between both embeddings
- The initial implementation uses the same branch CNNs and a larger supervised
  classifier input, without changing PSD/WPLI features or the Barlow objective.

Constraint:
- Same PSD/WPLI feature tensors.
- Same CNN+Barlow technical route.
- No MIL, no dual-encoder pretraining change; this is a downstream transfer bridge.

Implementation evidence:
- `SSLBridgeMultimodalEEGModel` keeps `branch_models` trainable and
  `ssl_branch_models` frozen.
- Fold-specific Barlow weights are loaded into both copies.
- `--ssl-bridge-enabled` is exposed in `scripts/07_train_feature_ssl_transfer.py`.
- Tests verify frozen SSL branches, state loading into both copies, bridge model
  construction, and the requirement that bridge runs must receive pretrained state.

Seed0 evidence:

| Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic Barlow | 0.6842 | 0.6722 | 0.9000 | 0.4444 | 0.7000 | 0.7200 | 0.2271 |
| SSL bridge finetune | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.8000 | 0.8273 | 0.1936 |
| SSL bridge freeze-encoder | 0.6842 | 0.6778 | 0.8000 | 0.5556 | 0.8222 | 0.8542 | 0.1565 |

Seed1 caveat:
- Bridge-only variants improved PR AUC but were unstable at the fixed 0.5
  threshold, especially in sensitivity.
- This means the bridge representation is useful, but the bridge-only classifier
  is not yet stable enough as the sole prediction source.

Score-level probe using already generated predictions:

| Variant, seeds 0-1 mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic Barlow | 0.7632 | 0.7528 | 0.9500 | 0.5556 | 0.7111 | 0.6814 | 0.2163 |
| freeze bridge only | 0.6053 | 0.6000 | 0.7000 | 0.5000 | 0.7444 | 0.7903 | 0.1889 |
| deterministic Barlow + freeze bridge average | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7556 | 0.7571 | 0.1793 |

Six-seed seed-run mean after expanding freeze-bridge to seeds `0 1 2 3 4 5`:

| Variant, seeds 0-5 mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.7632 | 0.7546 | 0.9167 | 0.5926 | 0.7630 | 0.7350 | 0.2063 |
| deterministic Barlow | 0.7895 | 0.7787 | 0.9833 | 0.5741 | 0.7537 | 0.7319 | 0.2040 |
| freeze bridge only | 0.6579 | 0.6546 | 0.7167 | 0.5926 | 0.7148 | 0.7423 | 0.2095 |
| deterministic Barlow + freeze bridge 50/50 | 0.7982 | 0.7880 | 0.9833 | 0.5926 | 0.7815 | 0.7632 | 0.1749 |

Six-seed patient-level seed-mean comparison:

| Variant, seed-mean patients | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7705 | 0.1828 |
| deterministic Barlow | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7111 | 0.6476 | 0.1973 |
| deterministic Barlow + freeze bridge 50/50 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7778 | 0.7788 | 0.1660 |
| deterministic Barlow + freeze bridge 25/75 exploratory | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.8039 | 0.1667 |

Interpretation:
- The frozen bridge is strongest for ranking/calibration.
- The original deterministic Barlow transfer is stronger for threshold stability.
- Their equal-weight score average is currently the safest transfer candidate:
  in six seed-runs it improves accuracy, balanced accuracy, sensitivity,
  specificity, ROC AUC, PR AUC, and Brier relative to no-SSL CNN.
- At the patient-level seed-mean, 50/50 does not beat no-SSL ROC AUC, but it
  improves PR AUC and Brier while matching threshold metrics. A 25/75
  deterministic/bridge exploratory weight beats no-SSL ROC AUC, PR AUC, and
  Brier while matching threshold metrics, but this weight was inspected after
  seeing the 19-patient seed-mean and therefore must be treated as exploratory
  unless selected with a leakage-safe OOF rule.

Expected benefit:
- Better transfer of SSL representations to the classifier, with less catastrophic forgetting.

### 3. Branch-Aware Fused Barlow

Rationale:
- Current feature Barlow uses only the final concatenated embedding.
- PSD and WPLI may encode different disease-relevant structure; fused-BT ECG work suggests branch-specific and fused losses can help.

Implementation:
- During SSL pretraining, compute:
  - Barlow loss on PSD branch embedding
  - Barlow loss on WPLI branch embedding
  - Barlow loss on fused PSD+WPLI embedding
- Total loss:
  - `loss = global_barlow + branch_barlow_weight * mean(psd_barlow, wpli_barlow)`
- Keep downstream model as CNN+Barlow SSL-CNN.

Expected benefit:
- Stronger branch encoders and less dependence on unstable downstream classifier learning.

### 4. CNN Backbone Adaptation

Rationale:
- Current PSD CNN and FC CNN are shallow with BatchNorm and global average pooling.
- With LOSO and full-batch supervised training, BatchNorm statistics can be noisy, and global average pooling may erase localized frequency/edge information.

Implementation candidates:
- Add `encoder_kind=rescnn`:
  - residual conv blocks
  - GroupNorm or LayerNorm instead of BatchNorm
  - squeeze-excitation or light channel attention
  - adaptive pooling plus max/mean pooling concatenation
- Preserve input shapes:
  - PSD: `(62, 90)`
  - WPLI: `(1891, 6)`

Expected benefit:
- Better fit between Barlow learned invariances and downstream CNN capacity, without changing features.

### 5. Post-Hoc Calibration Only

Rationale:
- Barlow improves seed-mean classification metrics but not consistently ROC/PR/Brier.
- Calibration should not be tuned on the final 19-patient mean predictions.

Implementation:
- Fixed 0.5 threshold remains primary.
- Add leave-one-seed-out OOF temperature/threshold calibration for reporting only.

Expected benefit:
- Better Brier and specificity estimates while avoiding leakage.

## Locked 10-Seed Follow-Up

After expanding the deterministic Barlow and freeze-bridge score-average
candidate to the locked seeds `0 1 2 3 4 5 7 13 21 42`, the six-seed signal did
not fully hold at patient-level seed-mean.

Per-seed mean, fixed threshold `0.5`:

| Variant, 10 seed-runs mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.7947 | 0.7861 | 0.9500 | 0.6222 | 0.8044 | 0.7823 | 0.1893 |
| deterministic Barlow | 0.7842 | 0.7744 | 0.9600 | 0.5889 | 0.7633 | 0.7497 | 0.2054 |
| freeze bridge only | 0.6684 | 0.6644 | 0.7400 | 0.5889 | 0.7256 | 0.7582 | 0.2142 |
| deterministic Barlow + freeze bridge 50/50 | 0.8053 | 0.7950 | 0.9900 | 0.6000 | 0.7844 | 0.7703 | 0.1790 |

Patient-level seed-mean, fixed threshold `0.5`:

| Variant, 10-seed patient mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| deterministic Barlow | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7222 | 0.6910 | 0.1967 |
| freeze bridge only | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.8111 | 0.8077 | 0.1851 |
| deterministic Barlow + freeze bridge 50/50 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.8000 | 0.7961 | 0.1705 |
| deterministic Barlow + freeze bridge 25/75 exploratory | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.8111 | 0.8056 | 0.1727 |

Leakage-safe leave-one-seed-out threshold calibration, per-seed mean:

| Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.7895 | 0.7806 | 0.9500 | 0.6111 | 0.8044 | 0.7823 | 0.1893 |
| deterministic Barlow + freeze bridge 50/50 | 0.8000 | 0.7906 | 0.9700 | 0.6111 | 0.7844 | 0.7703 | 0.1790 |

Leakage-safe leave-one-seed-out bridge-weight selection did not improve the
locked result. The selected bridge weight averaged `0.46`, close to the fixed
50/50 score average, and patient-level seed-mean remained `0.7895` accuracy and
`0.7778` balanced accuracy.

Patient-level paired tests for `deterministic Barlow + freeze bridge 50/50`
versus no-SSL did not show a significant advantage. Accuracy and balanced
accuracy were lower by one patient at fixed threshold; PR AUC improved by
`+0.0137` and Brier improved by `-0.0009`, but both bootstrap intervals crossed
zero and paired permutation p-values were non-significant.

Sub09/sub14 remained repeated false positives for the 50/50 bridge average.
The freeze bridge reduced their scores substantially compared with deterministic
Barlow, but not enough to fix them after averaging:

| Subject | no-SSL mean score | deterministic Barlow | freeze bridge | 50/50 score average |
| --- | ---: | ---: | ---: | ---: |
| sub09, true 0 | 0.9319 | 0.9791 | 0.4720 | 0.7255 |
| sub14, true 0 | 0.8619 | 0.9947 | 0.6149 | 0.8048 |

Additional bridge + discriminative warmup pilot:

| Variant, seeds 0-1 | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSL bridge + encoder LR 0.1 + head LR 2.0 + 10-epoch freeze, seed0 | 0.7368 | 0.7222 | 1.0000 | 0.4444 | 0.8000 | 0.8238 | 0.2009 |
| SSL bridge + encoder LR 0.1 + head LR 2.0 + 10-epoch freeze, seed1 | 0.5263 | 0.5222 | 0.6000 | 0.4444 | 0.6667 | 0.7427 | 0.2609 |
| 2-seed ensemble | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.7000 | 0.7488 | 0.2332 |

Conclusion:
- The bridge helped calibration/ranking in some seeds, but it is not stable
  enough to claim a significant improvement over no-SSL CNN.
- Current evidence supports stopping bridge-warmup expansion.
- The next aligned improvement should change the SSL-CNN transfer mechanism
  more directly rather than only changing score fusion:
  1. branch-aware fused Barlow loss during SSL pretraining;
  2. a CNN backbone better matched to Barlow transfer, such as residual
     GroupNorm blocks with max+mean pooling;
  3. a supervised distillation/consistency loss from frozen SSL embeddings into
     the trainable CNN embedding, evaluated as a seed0/seed1 pilot before a
     locked 10-seed run.

## Branch-Aware Fused Barlow Pilot

Implemented a `branch-barlow` objective that keeps the same PSD+WPLI inputs and
CNN+Barlow route, but adds branch-level redundancy-reduction terms during SSL
pretraining:

- global Barlow on fused PSD+WPLI embedding;
- PSD branch Barlow;
- WPLI branch Barlow;
- total loss: `global_barlow + branch_barlow_weight * mean(branch_losses)`;
- default `branch_barlow_weight = 0.5`.

This required a reusable `extract_branch_embeddings()` path on the multimodal
CNN, and a new `--branch-barlow-weight` CLI flag. Targeted tests passed:
`tests/test_model_shapes.py` 27 passed and `tests/test_feature_ssl_transfer.py`
21 passed.

Seed0/seed1 pilot, fixed threshold `0.5`:

| Variant, seeds 0-1 mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7000 | 0.6801 | 0.2404 |
| deterministic Barlow | 0.7632 | 0.7528 | 0.9500 | 0.5556 | 0.7111 | 0.6814 | 0.2163 |
| branch-aware fused Barlow | 0.7632 | 0.7500 | 1.0000 | 0.5000 | 0.7222 | 0.6844 | 0.2131 |
| deterministic Barlow + freeze bridge 50/50 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7556 | 0.7571 | 0.1793 |

Interpretation:
- Branch-aware Barlow improves over no-SSL in this two-seed pilot, mainly by
  recovering sensitivity and fixed-threshold accuracy.
- It does not beat the current 50/50 bridge average and still leaves sub09 and
  sub14 as high-confidence false positives.
- This is not strong enough to expand to 10 seeds as-is.
- The next more direct transfer test should combine a frozen SSL teacher with a
  supervised embedding-consistency term, instead of relying only on the
  classifier to discover how to use the SSL representation.

## SSL Teacher Consistency Pilot

Implemented an explicit supervised transfer regularizer on top of
`SSLBridgeMultimodalEEGModel`:

- the trainable CNN branch and frozen SSL branch are both initialized from the
  same fold-specific Barlow checkpoint;
- the bridge model exposes `extract_bridge_embeddings()`;
- supervised training adds
  `ssl_consistency_weight * MSE(normalize(z_trainable), normalize(z_ssl))`;
- the frozen SSL branch remains detached and receives no gradient;
- CLI flag: `--ssl-consistency-weight`.

Targeted tests passed after implementation:
`tests/test_model_shapes.py` 28 passed and `tests/test_feature_ssl_transfer.py`
23 passed.

Seed0/seed1 pilot, fixed threshold `0.5`:

| Variant, seeds 0-1 mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7000 | 0.6801 | 0.2404 |
| deterministic Barlow | 0.7632 | 0.7528 | 0.9500 | 0.5556 | 0.7111 | 0.6814 | 0.2163 |
| branch-aware fused Barlow | 0.7632 | 0.7500 | 1.0000 | 0.5000 | 0.7222 | 0.6844 | 0.2131 |
| bridge consistency 0.25 | 0.7105 | 0.7028 | 0.8500 | 0.5556 | 0.7667 | 0.7935 | 0.2286 |
| bridge consistency 0.05 | 0.6842 | 0.6778 | 0.8000 | 0.5556 | 0.7667 | 0.8067 | 0.2347 |
| deterministic Barlow + freeze bridge 50/50 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7556 | 0.7571 | 0.1793 |

Important detail:
- `ssl_consistency_weight=0.25` seed0 was strong: Acc `0.7895`, ROC AUC
  `0.8889`, PR AUC `0.8953`, Brier `0.1646`.
- The same setting seed1 dropped to Acc `0.6316` and Brier `0.2927`.
- `ssl_consistency_weight=0.05` preserved high seed0 ROC/PR but did not fix
  seed1.

Interpretation:
- The frozen Barlow teacher contains ranking signal, visible in ROC/PR gains.
- The current bridge classifier remains poorly calibrated under small-sample
  LOSO, so fixed-threshold accuracy is unstable.
- This is not ready for a locked 10-seed run.
- The next structural intervention should reduce supervised instability in the
  CNN itself: replace BatchNorm/global-average-only shallow CNN blocks with a
  residual GroupNorm CNN and max+mean pooling while preserving the same PSD+WPLI
  inputs and Barlow route.

## ResCNN GroupNorm Backbone Pilot

Implemented a new `encoder_kind=rescnn` while preserving the same PSD+WPLI
feature inputs and CNN+Barlow route:

- residual conv blocks;
- GroupNorm instead of BatchNorm;
- max+mean adaptive pooling before the embedding projection;
- PSD output shape unchanged: `(batch, embedding_dim)`;
- WPLI output shape unchanged: `(batch, embedding_dim)`;
- CLI support in both supervised LOSO and feature SSL transfer scripts.

Targeted tests passed after implementation:
`tests/test_model_shapes.py` 30 passed and `tests/test_feature_ssl_transfer.py`
23 passed.

Seed0/seed1 pilot with the original supervised LR `0.002`:

| Variant, seeds 0-1 mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original CNN no-SSL | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7000 | 0.6801 | 0.2404 |
| original CNN deterministic Barlow | 0.7632 | 0.7528 | 0.9500 | 0.5556 | 0.7111 | 0.6814 | 0.2163 |
| original CNN Barlow + freeze bridge 50/50 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7556 | 0.7571 | 0.1793 |
| ResCNN no-SSL | 0.4737 | 0.4611 | 0.7000 | 0.2222 | 0.5056 | 0.5894 | 0.3573 |
| ResCNN Barlow | 0.5263 | 0.5194 | 0.6500 | 0.3889 | 0.5667 | 0.6794 | 0.3169 |

The ResCNN did reduce some repeated-negative scores:

| Subject | original CNN Barlow 50/50 | ResCNN Barlow | ResCNN no-SSL |
| --- | ---: | ---: | ---: |
| sub09, true 0 mean score | 0.8028 | 0.6143 | 0.3444 |
| sub14, true 0 mean score | 0.8032 | 0.6322 | 0.6386 |

However, the global metrics deteriorated substantially. A lower supervised LR
`0.0005` was also tested for ResCNN no-SSL and was worse:

| Variant, seeds 0-1 mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResCNN no-SSL, LR 0.0005 | 0.5000 | 0.4889 | 0.7000 | 0.2778 | 0.4611 | 0.5214 | 0.3880 |

Interpretation:
- The hard-negative score reduction is real but comes at too high a cost.
- The current residual GroupNorm backbone underfits or destabilizes the small
  LOSO supervised problem.
- Do not expand this ResCNN version to 10 seeds.
- If revisiting backbone changes, use a smaller change: keep the shallow CNN
  depth, replace only BatchNorm with GroupNorm, and add max+mean pooling. Do
  not add residual depth until the normalization/pooling-only variant is tested.

## Shallow GroupNorm CNN Pilot

Implemented `encoder_kind=gncnn` to isolate normalization and pooling from the
deeper residual backbone:

- original shallow CNN depth;
- BatchNorm replaced by GroupNorm;
- max+mean pooling;
- same PSD+WPLI inputs and CNN+Barlow route.

Targeted tests passed after implementation:
`tests/test_model_shapes.py` 32 passed and `tests/test_feature_ssl_transfer.py`
23 passed before the later two-head additions.

Seed0/seed1 pilot with the original supervised LR `0.002`:

| Variant, seeds 0-1 mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original CNN no-SSL | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7000 | 0.6801 | 0.2404 |
| original CNN deterministic Barlow | 0.7632 | 0.7528 | 0.9500 | 0.5556 | 0.7111 | 0.6814 | 0.2163 |
| original CNN Barlow + freeze bridge 50/50 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7556 | 0.7571 | 0.1793 |
| GNCNN no-SSL | 0.5263 | 0.5222 | 0.6000 | 0.4444 | 0.4944 | 0.5787 | 0.3844 |
| GNCNN Barlow | 0.4211 | 0.4139 | 0.5500 | 0.2778 | 0.3944 | 0.5184 | 0.3934 |

The GNCNN Barlow run reduced some repeated false-positive scores, but the
global metrics collapsed. This rules out the normalization/pooling-only
backbone change for this dataset. The original BatchNorm CNN remains the safest
backbone for the Barlow route.

## SSL Two-Head Logit Fusion Pilot

Implemented `SSLTwoHeadMultimodalEEGModel` to transfer Barlow features without
concatenating trainable and frozen embeddings into a single classifier:

- fold-specific Barlow weights initialize both the trainable CNN branch and a
  frozen SSL branch;
- one supervised head is trained on the trainable CNN embedding;
- one supervised head is trained on the frozen SSL embedding;
- inference uses fixed logit fusion with `ssl_fusion_weight = 0.5`;
- training uses a head-level auxiliary loss with `ssl_aux_head_weight = 0.5`;
- the frozen SSL encoder receives no gradients.

AnySearch follow-up evidence supports this direction only as a parameter
efficient transfer idea: adapter-style work such as K-Adapter keeps pretrained
parameters fixed while training small plug-in modules. The current two-head
version is a simpler probe of the same principle, but it is not the winning
variant.

Targeted tests passed after implementation:
`tests/test_feature_ssl_transfer.py` 28 passed and `tests/test_model_shapes.py`
32 passed.

Seed0/seed1 pilot:

| Variant, seeds 0-1 mean | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7000 | 0.6801 | 0.2404 |
| deterministic Barlow | 0.7632 | 0.7528 | 0.9500 | 0.5556 | 0.7111 | 0.6814 | 0.2163 |
| deterministic Barlow + freeze bridge 50/50 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7556 | 0.7571 | 0.1793 |
| SSL two-head logit fusion 0.5 | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7333 | 0.7241 | 0.2391 |
| SSL two-head seed ensemble2 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.7000 | 0.6913 | 0.2314 |

Interpretation:
- Two-head fusion improves ranking over no-SSL in the seed-run mean, but it
  does not beat deterministic Barlow or the existing deterministic+freeze-bridge
  score average.
- The seed ensemble improves specificity over no-SSL but still has weak ROC/PR
  and Brier.
- Do not expand this two-head version to 10 seeds.
- The next transfer mechanism should keep the original CNN and Barlow route but
  add a small residual adapter on top of the Barlow embedding. That keeps the
  pretrained encoder closer to its SSL optimum while giving the supervised task
  a low-capacity correction path.

## Residual Adapter Transfer Pilot

Implemented a zero-initialized residual adapter on the fused PSD+WPLI CNN
embedding:

- default is off, so previous CNN/Barlow paths are unchanged;
- adapter is `LayerNorm -> Linear -> ReLU -> Dropout -> Linear`;
- the final adapter linear layer is initialized to zero, so the model starts
  from the pretrained Barlow embedding and only learns a small residual
  correction;
- CLI flags: `--embedding-adapter-dim` and `--embedding-adapter-scale`;
- pilot uses original CNN, original PSD+WPLI inputs, Barlow pretraining, and
  `freeze-encoder` supervised transfer.

Targeted tests passed after implementation:
`tests/test_model_shapes.py` and `tests/test_feature_ssl_transfer.py` together
reported 62 passed.

Adapter8 freeze-encoder, seeds 0-5:

| Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN, seed-run mean | 0.7632 | 0.7546 | 0.9167 | 0.5926 | 0.7630 | 0.7350 | 0.2063 |
| deterministic Barlow, seed-run mean | 0.7895 | 0.7787 | 0.9833 | 0.5741 | 0.7537 | 0.7319 | 0.2040 |
| adapter8 freeze, seed-run mean | 0.6754 | 0.6704 | 0.7667 | 0.5741 | 0.7630 | 0.7823 | 0.2360 |
| no-SSL CNN, patient seed-mean | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7705 | 0.1828 |
| deterministic Barlow, patient seed-mean | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7111 | 0.6476 | 0.1973 |
| deterministic Barlow + freeze bridge 50/50, patient seed-mean | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7778 | 0.7788 | 0.1660 |
| adapter8 freeze, patient seed-mean | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.8222 | 0.8187 | 0.1585 |

Diagnostic threshold scan on the adapter8 patient seed-mean found that
threshold `0.565-0.570` recovers Acc `0.8421` and Bal Acc `0.8333` while
keeping ROC AUC `0.8222`, PR AUC `0.8187`, and Brier `0.1585`. This is useful
diagnostically, but it is not a primary result because the threshold was found
on the final seed-mean predictions. Leave-one-seed-out thresholding did not
improve seed-run stability.

Post-hoc repeated false-positive scores under adapter8 patient seed-mean:

| Subject | true label | score | prediction at 0.5 |
| --- | ---: | ---: | ---: |
| sub09 | 0 | 0.6391 | 1 |
| sub14 | 0 | 0.6145 | 1 |

Interpretation:
- Adapter8 is currently the strongest route for ranking/calibration: it beats
  no-SSL patient seed-mean on ROC AUC, PR AUC, and Brier.
- It does not yet beat no-SSL on fixed-threshold Acc/Bal Acc because sub09 and
  sub14 remain just above 0.5.
- This is not enough to claim significant superiority over no-SSL CNN, but it
  is the best current direction for further work.
- The next experiment should keep adapter transfer and test leakage-safe
  calibration or a lower-capacity adapter/encoder LR combination before any
  locked 10-seed claim.

## Adapter8 10-Seed Locked Follow-Up

The adapter8 freeze-encoder route was expanded from the first six seeds to the
locked 10-seed set `0, 1, 2, 3, 4, 5, 7, 13, 21, 42`. This was done to test
whether the 6-seed signal survives the higher-variance locked seeds before
claiming an SSL-CNN improvement over the no-SSL CNN.

10-seed fixed-threshold patient seed-mean:

| Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| deterministic Barlow | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7222 | 0.6910 | 0.1967 |
| adapter8 freeze | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7889 | 0.7865 | 0.1769 |
| deterministic Barlow + adapter8, adapter weight 0.34 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7667 | 0.7411 | 0.1806 |

10-seed seed-run means:

| Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.7947 | 0.7861 | 0.9500 | 0.6222 | 0.8044 | 0.7823 | 0.1893 |
| deterministic Barlow | 0.7842 | 0.7744 | 0.9600 | 0.5889 | 0.7633 | 0.7497 | 0.2054 |
| adapter8 freeze | 0.6526 | 0.6478 | 0.7400 | 0.5556 | 0.7378 | 0.7683 | 0.2491 |
| deterministic Barlow + adapter8, adapter weight 0.34 | 0.7947 | 0.7850 | 0.9700 | 0.6000 | 0.7722 | 0.7373 | 0.1915 |

The 10-seed expansion reverses the 6-seed adapter8 signal. Seeds 13, 21, and
especially 42 are unstable under the frozen-encoder adapter path. A fixed
deterministic/adapted score average can recover seed-run accuracy, but it still
does not beat no-SSL on ROC AUC, PR AUC, or Brier. A full 10-seed scan over
adapter weights found no weight that simultaneously beats the no-SSL seed-run
mean on Acc, Bal Acc, ROC AUC, PR AUC, and Brier.

A second SSL-only grid scan over deterministic Barlow, frozen SSL-bridge Barlow,
and adapter8 Barlow also found no all-metric winner. The best Brier-weighted
mixtures reduce Brier below no-SSL and can improve seed-run Acc/Bal Acc, but
their ROC/PR AUC remain lower. The highest PR AUC mixtures use more bridge and
adapter signal, but fixed-threshold Acc/Bal Acc falls sharply.

A patient-level seed-mean bootstrap/permutation comparison was also generated
for the main 10-seed SSL-CNN candidates. No candidate shows a statistically
defensible all-metric improvement over the no-SSL CNN. The most useful
secondary signal is still the deterministic + frozen-bridge average, which has
slightly better Brier and PR AUC than no-SSL at the seed-mean level, but it
loses one thresholded patient and the paired random-swap p-values are not
significant.

Updated interpretation:
- Do not use adapter8 freeze or deterministic+adapter8 fusion as the final
  locked improvement claim.
- The useful lesson is architectural: Barlow transfer benefits from a frozen
  representation path for calibration, but the current adapter is too
  seed-sensitive when used directly.
- The next candidate should be lower capacity and more constrained, such as a
  smaller adapter, a frozen SSL teacher head with supervised CNN residual
  logits, or leakage-safe calibration trained within each LOSO training fold.
- Continue to keep fixed `0.5` as the primary threshold and report any
  calibrated threshold only when it is selected without using the held-out
  patient or final seed-mean labels.

## Lower-Capacity Adapter Pilot

To test whether adapter8 failed because the adapter was too high-capacity for
the 19-patient LOSO setting, adapter dimensions 2 and 4 were run on seeds 0 and
1 with the same original CNN, original PSD+WPLI inputs, Barlow pretraining, and
freeze-encoder transfer setup.

Seed0/seed1 pilot:

| Variant | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7000 | 0.6801 | 0.2404 |
| deterministic Barlow | seed-run mean | 0.7632 | 0.7528 | 0.9500 | 0.5556 | 0.7111 | 0.6814 | 0.2163 |
| adapter2 freeze | seed-run mean | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.6667 | 0.7119 | 0.2792 |
| adapter4 freeze | seed-run mean | 0.6579 | 0.6500 | 0.8000 | 0.5000 | 0.7056 | 0.7514 | 0.2485 |
| adapter8 freeze | seed-run mean | 0.7105 | 0.7028 | 0.8500 | 0.5556 | 0.7500 | 0.7452 | 0.2187 |
| adapter2 freeze | seed ensemble2 | 0.6842 | 0.6778 | 0.8000 | 0.5556 | 0.7000 | 0.7266 | 0.2530 |
| adapter4 freeze | seed ensemble2 | 0.6842 | 0.6778 | 0.8000 | 0.5556 | 0.7111 | 0.7236 | 0.2159 |
| adapter8 freeze | seed ensemble2 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.7667 | 0.7843 | 0.1938 |

Interpretation:
- Lower-capacity adapters do not solve the fixed-threshold weakness and are
  worse than adapter8 on the seed0/seed1 pilot.
- Do not expand adapter2 or adapter4 to locked 10 seeds.
- The next Barlow-CNN transfer test should not be another plain bottleneck
  adapter. A better candidate is a constrained residual logit/teacher route:
  keep a supervised CNN path for threshold behavior, add a frozen Barlow path
  only as a small calibrated residual or consistency target, and avoid letting
  the frozen SSL branch dominate patient scores.

## Low-Weight Frozen-Teacher Two-Head Pilot

The existing two-head implementation was retested with the supervised CNN as
the dominant path and the frozen Barlow head as a small logit residual. This is
closer to teacher/student transfer than to equal feature fusion: the Barlow
branch is frozen, receives a supervised auxiliary head, and contributes only a
fixed fraction of the final logit.

Seed0/seed1 pilot:

| Variant | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7000 | 0.6801 | 0.2404 |
| deterministic Barlow | seed-run mean | 0.7632 | 0.7528 | 0.9500 | 0.5556 | 0.7111 | 0.6814 | 0.2163 |
| SSL two-head, fusion 0.1 | seed-run mean | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7333 | 0.7454 | 0.2526 |
| SSL two-head, fusion 0.2 | seed-run mean | 0.7105 | 0.7028 | 0.8500 | 0.5556 | 0.7333 | 0.7476 | 0.2431 |
| SSL two-head, fusion 0.5 | seed-run mean | 0.7105 | 0.7000 | 0.9000 | 0.5000 | 0.7333 | 0.7241 | 0.2391 |
| adapter8 freeze | seed-run mean | 0.7105 | 0.7028 | 0.8500 | 0.5556 | 0.7500 | 0.7452 | 0.2187 |
| SSL two-head, fusion 0.1 | seed ensemble2 | 0.6842 | 0.6722 | 0.9000 | 0.4444 | 0.7111 | 0.7141 | 0.2490 |
| SSL two-head, fusion 0.2 | seed ensemble2 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.7222 | 0.7274 | 0.2359 |
| SSL two-head, fusion 0.5 | seed ensemble2 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.7000 | 0.6913 | 0.2314 |
| adapter8 freeze | seed ensemble2 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.7667 | 0.7843 | 0.1938 |

Interpretation:
- Low SSL residual weights improve ROC/PR over no-SSL in the seed-run mean, but
  Brier remains worse and fixed-threshold Acc/Bal Acc do not improve.
- Fusion 0.2 is the best low-weight two-head setting in this pilot, but it
  still does not beat adapter8 or deterministic Barlow enough to justify a
  locked 10-seed expansion.
- The frozen teacher path is not sufficient when mixed as a fixed global logit
  fraction. If this route is revisited, the teacher contribution should be
  learned or calibrated inside each LOSO training fold, with strict separation
  from the held-out patient.

## Leakage-Safe Teacher Weight Feasibility Checks

Before implementing a new fold-internal learned teacher gate, two score-level
feasibility checks were run using existing 10-seed predictions. These checks do
not replace a proper fold-internal training implementation, but they test
whether a calibrated teacher contribution has enough signal to justify more
engineering.

Patient-LOO proxy:
- For each seed and held-out patient, the Barlow teacher variant and teacher
  score weight were selected only on the other 18 patients from that seed.
- The selected weight was then applied to the held-out patient.
- This is a proxy because the calibration predictions are cross-fold
  predictions, not predictions produced by an inner split of the exact outer
  LOSO training fold.

| Variant | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7947 | 0.7861 | 0.9500 | 0.6222 | 0.8044 | 0.7823 | 0.1893 |
| patient-LOO teacher proxy | seed-run mean | 0.7737 | 0.7644 | 0.9400 | 0.5889 | 0.7811 | 0.7659 | 0.1821 |
| no-SSL CNN | seed-mean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| patient-LOO teacher proxy | seed-mean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7961 | 0.1631 |

Leave-one-seed teacher weight selection:
- For each held-out seed, the teacher variant and global score weight were
  selected using only the other nine seeds.
- Several selection objectives were tested: Brier-constrained, PR-constrained,
  ROC-constrained, and composite.

Best observed leave-one-seed pattern:

| Variant | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7947 | 0.7861 | 0.9500 | 0.6222 | 0.8044 | 0.7823 | 0.1893 |
| leave-one-seed teacher weight | seed-run mean | 0.7947 | 0.7861 | 0.9500 | 0.6222 | 0.7844 | 0.7738 | 0.1793 |
| no-SSL CNN | seed-mean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| leave-one-seed teacher weight | seed-mean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7961 | 0.1697 |

Interpretation:
- Calibrated teacher weighting can improve Brier, and sometimes PR AUC, without
  changing fixed-threshold Acc/Bal Acc at the seed-mean level.
- It still fails the requested all-metric objective because ROC AUC drops and
  seed-run mean ranking remains below no-SSL.
- Pure score-level teacher weighting is therefore not enough. A future
  implementation should train the connection inside the supervised LOSO fold,
  not select a fixed score weight after the fact. The learned connection should
  be regularized toward a small teacher contribution and judged first by
  seed-run ROC/PR/Brier stability before any 10-seed claim.

## Current Best Candidate: Small Consistency Residual

The strongest current signal comes from a conservative residual-style ensemble:
use the no-SSL supervised CNN score as the main path and add a small contribution
from the Barlow SSL-bridge consistency model. This keeps the original CNN,
original PSD+WPLI features, and Barlow SSL route intact. The useful 10-seed
fixed score rule is:

`score = 0.90 * no_ssl_cnn_score + 0.10 * ssl_consistency005_score`

This is not yet a replacement for a fold-internal learned residual connection,
but it is the first 10-seed result in this workstream that improves the no-SSL
seed-run mean across Acc, Bal Acc, ROC AUC, PR AUC, and Brier while preserving
the no-SSL patient seed-mean thresholded confusion matrix.

10-seed comparison:

| Variant | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7947 | 0.7861 | 0.9500 | 0.6222 | 0.8044 | 0.7823 | 0.1893 |
| SSL consistency 0.05 | seed-run mean | 0.7263 | 0.7178 | 0.8800 | 0.5556 | 0.7800 | 0.7789 | 0.2033 |
| no-SSL + SSL consistency 0.05, weight 0.10 | seed-run mean | 0.8000 | 0.7906 | 0.9700 | 0.6111 | 0.8100 | 0.7850 | 0.1857 |
| no-SSL CNN | seed-mean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| SSL consistency 0.05 | seed-mean10 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.8111 | 0.8155 | 0.1797 |
| no-SSL + SSL consistency 0.05, weight 0.10 | seed-mean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8222 | 0.7931 | 0.1713 |

Additional checks:
- 10-seed weight scan found all-metric point-estimate candidates around
  consistency weights `0.07-0.12`.
- A patient-level seed-mean bootstrap/permutation comparison for weights 0.10
  and 0.12 did not show statistical significance. For weight 0.10, ROC AUC
  improved by `0.0111`, PR AUC by `0.0107`, and Brier by `0.00014`, but the
  confidence intervals include no effect and paired random-swap p-values are
  not significant.
- Leave-one-seed automatic weight selection did not work; it overfit the
  calibration seeds and underperformed no-SSL on held-out seeds. This supports
  using a pre-specified small residual weight rather than tuning a score weight
  from the same tiny cohort.
- Formal fixed `w=0.10` artifacts were generated:
  - `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_no_ssl_plus_sslconsistency005_w010_10seed_all_seed_runs.csv`
  - `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_no_ssl_plus_sslconsistency005_w010_seedmean10.csv`
  - `runs/baseline_rerun_20260529/results/metrics/no_ssl_sslconsistency005_w010_10seed_summary.csv`
  - `runs/baseline_rerun_20260529/results/metrics/no_ssl_sslconsistency005_w010_10seed_per_seed.csv`
  - `runs/baseline_rerun_20260529/results/metrics/no_ssl_sslconsistency005_w010_10seed_subject_error_frequency.csv`
  - `runs/baseline_rerun_20260529/results/metrics/sub09_sub14_no_ssl_sslconsistency005_w010_10seed_scores_summary.csv`
- Post-hoc hard-negative monitoring shows the residual does not solve the
  repeated false-positive pattern: sub09 is positive in all 10 seed runs and
  sub14 in 9 of 10 seed runs.

Updated recommendation:
- Treat `0.90 * no_ssl + 0.10 * SSL-consistency` as a useful one-SSL-path
  baseline, but not the final residual target.
- Do not claim significant superiority yet. The point estimates are better, but
  the 19-patient bootstrap/permutation evidence is too weak.
- The next implementation should move this residual from post-hoc score fusion
  into supervised training: a supervised CNN main head with a small
  regularized Barlow-consistency residual head initialized near zero or fixed to
  a low prior contribution. The final evaluation must lock the residual rule
  before inspecting test predictions.

## Multi-Residual Barlow Transfer Search

A low-capacity three-path residual search was run on the existing 10-seed
predictions. The main path remained the no-SSL CNN score, with small residual
weights assigned to:

- SSL consistency 0.05
- deterministic Barlow + frozen bridge average
- adapter8 frozen Barlow

The search used 0.02 weight steps and total SSL residual weight <= 0.30. Eight
candidates improved or tied the no-SSL seed-mean fixed-threshold confusion
matrix while improving seed-mean ROC AUC, PR AUC, and Brier, and also improving
seed-run mean Acc, Bal Acc, ROC AUC, PR AUC, and Brier. The best candidate was:

`score = 0.82 * no_ssl + 0.14 * SSL-consistency0.05 + 0.04 * det/freeze-bridge`

10-seed comparison:

| Variant | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7947 | 0.7861 | 0.9500 | 0.6222 | 0.8044 | 0.7823 | 0.1893 |
| multi-residual Barlow | seed-run mean | 0.8000 | 0.7906 | 0.9700 | 0.6111 | 0.8089 | 0.7865 | 0.1830 |
| no-SSL CNN | seed-mean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| multi-residual Barlow | seed-mean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8222 | 0.8187 | 0.1707 |

Generated artifacts:

- `runs/baseline_rerun_20260529/results/metrics/no_ssl_multi_ssl_residual_grid_scan_10seed.csv`
- `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_no_ssl_multi_ssl_residual_cons014_detbridge004_10seed_all_seed_runs.csv`
- `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_no_ssl_multi_ssl_residual_cons014_detbridge004_seedmean10.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_multi_ssl_residual_cons014_detbridge004_10seed_summary.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_multi_ssl_residual_cons014_detbridge004_10seed_per_seed.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_multi_ssl_residual_cons014_detbridge004_seedmean10_statistical_comparison.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_multi_ssl_residual_cons014_detbridge004_10seed_subject_error_frequency.csv`
- `runs/baseline_rerun_20260529/results/metrics/sub09_sub14_no_ssl_multi_ssl_residual_cons014_detbridge004_10seed_scores_summary.csv`

Statistical interpretation:

- Patient seed-mean10 ROC AUC improved by `+0.0111`, PR AUC by `+0.0363`, and
  Brier by `-0.00069`.
- Patient-level bootstrap intervals still include no effect:
  ROC AUC CI `[-0.0476, 0.0857]`, PR AUC CI `[-0.0370, 0.1200]`, Brier CI
  `[-0.00956, 0.00545]`.
- Random-label permutation p-values are not significant. Therefore this is a
  stronger design target, not a publishable significant win.
- sub09 remains a 10/10 false positive and sub14 remains a 9/10 false
  positive, so the residual improves ranking/calibration but does not solve the
  repeated hard-negative pattern.

Recommended model-level translation:

- Keep no-SSL CNN as the supervised main path.
- Add two small Barlow residual heads: a consistency-regularized Barlow bridge
  head and a deterministic/frozen bridge head.
- Initialize residual gates near the discovered prior weights (`0.14` and
  `0.04`) or constrain their total contribution to a small range during
  training.
- Lock the residual rule before final evaluation; the grid above is exploratory
  because it was selected after inspecting the 10-seed predictions.

## Model-Level No-SSL-Main Residual Pilot

The first model-level translation of the multi-residual score rule loaded the
Barlow weights into both the trainable path and the frozen residual path. This
improved seed0/seed1 ROC AUC, PR AUC, and Brier, but it did not preserve the
no-SSL fixed-threshold specificity. The implementation has now been corrected
to support the intended no-SSL-main residual design:

- `--ssl-residual-random-main` keeps the supervised main CNN randomly
  initialized and loads Barlow weights only into the frozen SSL residual branch.
- `--ssl-residual-main-loss-only` trains the primary supervised loss on the main
  CNN head while residual heads are trained only through auxiliary loss and
  contribute at inference.
- `--ssl-residual-preserve-main` trains a fold-internal no-SSL CNN first,
  freezes that main path, then attaches Barlow residual heads. This is the
  closest current implementation to the score-level residual evidence because
  residual losses cannot move the main no-SSL decision boundary.

The random-main residual looked promising for seeds 0 and 1, but did not hold
after expansion to seeds 0-5.

| Variant, seeds 0-5 | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7632 | 0.7546 | 0.9167 | 0.5926 | 0.7630 | 0.7350 | 0.2063 |
| deterministic Barlow | seed-run mean | 0.7895 | 0.7787 | 0.9833 | 0.5741 | 0.7537 | 0.7319 | 0.2040 |
| random-main residual 0.14/0.04 | seed-run mean | 0.7193 | 0.7111 | 0.8667 | 0.5556 | 0.7648 | 0.7832 | 0.1962 |
| no-SSL CNN | seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7705 | 0.1828 |
| deterministic Barlow | seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7111 | 0.6476 | 0.1973 |
| random-main residual 0.14/0.04 | seedmean6 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7556 | 0.7356 | 0.1820 |

The main-loss-only variant was tested on the bad seed3 failure case. It kept
seed3 Acc/Bal Acc at `0.6316/0.6278`, while no-SSL seed3 was
`0.7368/0.7278`. PR AUC improved relative to no-SSL (`0.7457` vs `0.6653`),
but Brier worsened (`0.2146` vs `0.2034`).

Interpretation:

- The Barlow residual path still carries useful ranking information, as shown
  by PR AUC and some Brier gains.
- Co-training the supervised main path with residual heads is not equivalent to
  the successful score-level residual, even when Barlow weights are excluded
  from the main branch.
- Do not expand this fixed `0.14/0.04` model-level residual to 10 seeds.
- A future model-level implementation should preserve an independently trained
  no-SSL main checkpoint inside each LOSO fold, then attach a small Barlow
  residual/adapter head without allowing the residual loss to move the main
  decision boundary. That is closer to the score-level evidence and avoids
  re-tuning weights on the held-out patient.

The fold-internal preserved-main implementation was then tested. A seed3
pilot with fixed residual weights `0.14/0.04` improved ROC AUC, PR AUC, and
Brier relative to no-SSL, but lost one positive case at the fixed `0.5`
threshold. Saving the main/SSL/bridge sub-scores enabled a weight scan. The
exploratory `0.30/0.04` setting restored seed3 threshold metrics while
improving ranking/calibration:

| Variant, seed3 | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.7333 | 0.6653 | 0.2034 |
| preserved-main residual 0.14/0.04 | 0.6842 | 0.6778 | 0.8000 | 0.5556 | 0.8000 | 0.8217 | 0.1911 |
| preserved-main residual 0.30/0.04 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.8000 | 0.8192 | 0.1739 |

After expanding `0.30/0.04` to seeds 0-5, the fixed-threshold result still did
not beat no-SSL, although ranking/calibration improved substantially:

| Variant, seeds 0-5 | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7632 | 0.7546 | 0.9167 | 0.5926 | 0.7630 | 0.7350 | 0.2063 |
| deterministic Barlow | seed-run mean | 0.7895 | 0.7787 | 0.9833 | 0.5741 | 0.7537 | 0.7319 | 0.2040 |
| preserved-main residual 0.30/0.04 | seed-run mean | 0.7456 | 0.7370 | 0.9000 | 0.5741 | 0.7963 | 0.7986 | 0.1795 |
| no-SSL CNN | seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7705 | 0.1828 |
| deterministic Barlow | seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7111 | 0.6476 | 0.1973 |
| preserved-main residual 0.30/0.04 | seedmean6 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.8333 | 0.8375 | 0.1660 |

Interpretation:

- The preserved-main design is the best model-level translation so far for
  ROC AUC, PR AUC, and Brier. It clearly transfers useful Barlow information.
- It still fails the requested all-metric objective because fixed-threshold
  Acc/Bal Acc and sensitivity drop in the 6-seed expansion.
- Do not expand `0.30/0.04` to locked 10 seeds as a primary result.
- The next technical issue is not feature representation; it is decision
  calibration under fixed threshold. Any next attempt should keep the
  preserved-main architecture but add fold-internal calibration or a
  sensitivity-preserving constraint on the residual contribution.

Generated artifacts:

- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_randommain014_bridge004_6seed_comparison.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_randommain014_bridge004_6seed_summary.csv`
- `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_feature_ssl_barlow_sslresidual_randommain014_bridge004_seedensemble6.csv`
- `runs/baseline_rerun_20260529/results/metrics/dl_model_comparison_feature_ssl_barlow_sslresidual_randommain014_bridge004_seedensemble6.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_randommain014_bridge004_6seed_subject_error_frequency.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_randommain_mainloss014_bridge004_seed3_summary.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_preservemain_seed3_aux_weight_scan.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_preservemain020_seed0_seed1_aux_weight_scan.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_preservemain030_bridge004_6seed_comparison.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_preservemain030_bridge004_6seed_summary.csv`
- `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_feature_ssl_barlow_sslresidual_preservemain030_bridge004_seedensemble6.csv`

## Lower Consistency Weight Pilot

To test whether the consistency path itself could become stable without
post-hoc residual fusion, a lower SSL consistency weight was evaluated:
`ssl_consistency_weight=0.01`. Seeds 0 and 1 were run first, then extended to
seeds 2-5 because the seed ensemble2 result had strong ranking/calibration
signals.

6-seed comparison:

| Variant | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7632 | 0.7546 | 0.9167 | 0.5926 | 0.7630 | 0.7350 | 0.2063 |
| no-SSL CNN | seed-mean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7705 | 0.1828 |
| SSL consistency 0.01 | seed-run mean | 0.7193 | 0.7111 | 0.8667 | 0.5556 | 0.7852 | 0.7931 | 0.2049 |
| SSL consistency 0.01 | seed-mean6 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.8000 | 0.8077 | 0.1801 |
| SSL consistency 0.05 | seed-run mean | 0.7105 | 0.7028 | 0.8500 | 0.5556 | 0.7796 | 0.7853 | 0.2030 |
| SSL consistency 0.05 | seed-mean6 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.8111 | 0.8211 | 0.1774 |
| no-SSL + consistency 0.01, weight 0.10 | seed-run mean | 0.7807 | 0.7713 | 0.9500 | 0.5926 | 0.7722 | 0.7464 | 0.2001 |
| no-SSL + consistency 0.01, weight 0.10 | seed-mean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7889 | 0.7663 | 0.1818 |

Interpretation:
- Lowering the consistency weight to 0.01 improves PR AUC and Brier compared
  with no-SSL, but fixed-threshold Acc/Bal Acc fall for the direct SSL-CNN.
- The 0.01 residual does not beat the current 0.05 residual target and does not
  improve seed-mean ROC/PR over no-SSL.
- Do not expand consistency 0.01 to locked 10 seeds. Keep the current best
  exploratory target as the low-capacity multi-residual rule
  `0.82 * no_ssl + 0.14 * SSL-consistency0.05 + 0.04 * det/freeze-bridge`.

## Residual Calibration Update

Two calibration routes were tested after the preserved-main residual showed
good ROC/PR/Brier but unstable fixed-threshold accuracy.

First, fold-internal validation selection of preserved-main residual weights
was implemented (`--ssl-residual-select-weights-on-val`). This is leakage-safe
with respect to the LOSO test patient, but the validation split is too small
for reliable weight selection. Seed0 was negative:

| Variant, seed0 | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Barlow preserved-main residual, fold-val weight selection | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.7333 | 0.6953 | 0.2263 |

The failure mode was that most folds selected near-zero SSL residual weight,
falling back to the internally trained no-SSL main path; for seed0 that main
path itself was weak. Do not continue this fold-val selection rule as the next
10-seed candidate.

Second, the existing 10-seed no-SSL + SSL-consistency score file was rescanned
for a fixed low-gate score rule with a small threshold calibration. The best
non-inferior fixed rule in the local scan was:

`score = 0.77 * no_ssl_score + 0.23 * ssl_consistency005_score`, threshold
`0.505`.

This is not a final locked claim because the weight/threshold were selected
from the available 10-seed outputs, but it is the strongest practical direction
for preserving the no-SSL confusion matrix while adding SSL ranking/calibration
signal:

| Variant | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7947 | 0.7861 | 0.9500 | 0.6222 | 0.8044 | 0.7823 | 0.1893 |
| no-SSL + SSL consistency 0.05, w=0.23, thr=0.505 | seed-run mean | 0.8000 | 0.7911 | 0.9600 | 0.6222 | 0.8100 | 0.7938 | 0.1827 |
| no-SSL CNN | seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| no-SSL + SSL consistency 0.05, w=0.23, thr=0.505 | seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8222 | 0.8187 | 0.1714 |

Leave-one-seed-out weight/threshold selection did not reproduce the full fixed
scan gain. It improved sensitivity, ROC/PR, and Brier but lost a small amount
of specificity/balanced accuracy, so the fixed rule should be treated as a
candidate to pre-register and rerun, not as a leakage-safe final result.

Generated artifacts:

- `runs/baseline_rerun_20260529/results/metrics/no_ssl_sslconsistency005_weight_threshold_scan_local_10seed.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_sslconsistency005_weight_threshold_loso_choices.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_sslconsistency005_weight_threshold_loso_per_seed.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_sslconsistency005_weight_threshold_loso_summary.csv`
- `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_no_ssl_plus_sslconsistency005_w023_thr0505_10seed_all_seed_runs.csv`
- `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_no_ssl_plus_sslconsistency005_w023_thr0505_seedmean10.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_plus_sslconsistency005_w023_thr0505_10seed_per_seed.csv`
- `runs/baseline_rerun_20260529/results/metrics/no_ssl_plus_sslconsistency005_w023_thr0505_10seed_summary.csv`

## Score-Fusion Ceiling Check

Because the low-gate score rule only preserved the patient seed-mean confusion
matrix, a patient-level ceiling check was run on the existing 10-seed outputs.
The goal was to determine whether score-level fusion alone could correct at
least one of the repeated false-positive subjects without losing any positive
subject.

Inputs used as score features only:

- no-SSL CNN seed-mean score
- deterministic Barlow SSL-CNN seed-mean score
- SSL-consistency Barlow score
- multi-residual score
- adapter/teacher score variants

Two leakage-aware checks were run:

1. LOSO patient-level logistic stacking on score features.
2. Random convex score-fusion search across the existing score sources.

Summary:

| Variant | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no-SSL CNN seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 | 6/3/0/10 |
| best convex score fusion | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8222 | 0.8187 | 0.1707 | 6/3/0/10 |
| best noninferior convex fusion by Brier | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1684 | 6/3/0/10 |
| best LOSO patient stacking | 0.7895 | 0.7889 | 0.8000 | 0.7778 | 0.7889 | 0.7145 | 0.1743 | 7/2/2/8 |

Interpretation:

- Score fusion can improve ROC AUC, PR AUC, and/or Brier while preserving the
  no-SSL confusion matrix.
- Within the current score sources, it did not find any patient seed-mean rule
  that improves Acc/Bal Acc over no-SSL while preserving sensitivity.
- The repeated false positives are not separable by post-hoc score fusion
  without sacrificing low-margin positive subjects, especially sub01/sub22.
- The next meaningful attempt must modify supervised training or the
  encoder-CNN connection before scores are produced. Continuing to tune
  post-hoc score weights is unlikely to produce the requested all-metric win.

Generated artifacts:

- `runs/baseline_rerun_20260529/results/metrics/ssl_cnn_patient_level_stacking_scan.csv`
- `runs/baseline_rerun_20260529/results/metrics/ssl_cnn_patient_seedmean_convex_score_fusion_random_search.csv`
- `runs/baseline_rerun_20260529/results/metrics/ssl_cnn_score_fusion_ceiling_summary.csv`

## Main-Logit Preservation Pilot

A model-level no-SSL logit preservation loss was added for residual SSL-CNN
training:

`loss = supervised BCE + lambda * MSE(logit(p_mixture), stopgrad(logit(p_main)))`

This follows the same stability motivation as delta/adapter tuning and
distillation-style preservation: let a small trainable residual path use the
Barlow representation, while discouraging destructive movement away from the
fold-internal no-SSL main CNN.

Implementation:

- `SupervisedTrainingConfig.ssl_residual_logit_preservation_weight`
- CLI flag `--ssl-residual-logit-preservation-weight`
- helper `_probability_to_logit`
- training-only preservation penalty for SSL residual models

Seed0 pilot with preserved-main residual `ssl=0.30`, `bridge=0.04`,
`lambda=1.0` was negative:

| Variant, seed0 | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Barlow origfeat seed0 | 0.7368 | 0.7222 | 1.0000 | 0.4444 | 0.7444 | 0.6912 | 0.2007 |
| preserved-main residual 0.30/0.04 | 0.6842 | 0.6778 | 0.8000 | 0.5556 | 0.8667 | 0.9073 | 0.1878 |
| preserved-main residual 0.30/0.04 + logit preservation 1.0 | 0.6842 | 0.6722 | 0.9000 | 0.4444 | 0.7556 | 0.7575 | 0.2095 |
| same trained heads, best post-hoc aux weights | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7889 | 0.7827 | 0.2157 |

Interpretation:

- Logit preservation did not improve the trained fixed-weight residual model.
- The trained SSL/bridge heads contain enough decision signal to improve
  seed0 fixed-threshold accuracy under post-hoc high residual weights, but
  calibration degrades substantially.
- Do not expand `lambda=1.0, ssl=0.30, bridge=0.04` to 10 seeds.
- The next model-level variant should avoid probability-mixture heads and use
  a bounded logit-delta adapter initialized at zero:
  `p = sigmoid(logit(p_main) + alpha * tanh(delta_ssl_bridge))`. This directly
  preserves the main decision at initialization and constrains residual
  movement in logit space instead of mixing poorly calibrated probabilities.

Generated artifacts:

- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_preservemain_logitpres1_seed0_comparison.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_sslresidual_preservemain_logitpres1_seed0_aux_weight_scan.csv`

## Bounded Logit-Delta Residual Pilot

Implemented a bounded logit-delta Barlow adapter:

`p = sigmoid(logit(p_main) + alpha * tanh(delta([z_main, z_ssl, |z_main - z_ssl|])))`

The main CNN and Barlow encoder branches are frozen during the residual phase.
The delta head is zero-initialized, so each fold starts exactly at the
fold-internal no-SSL main CNN output. This avoids the previous probability
mixture issue where poorly calibrated residual heads could overwrite the main
decision boundary.

Implementation:

- `FrozenMainSSLLogitDeltaMultimodalEEGModel`
- `SupervisedTrainingConfig.ssl_residual_logit_delta_enabled`
- `SupervisedTrainingConfig.ssl_residual_logit_delta_scale`
- CLI flags `--ssl-residual-logit-delta-enabled` and
  `--ssl-residual-logit-delta-scale`
- Tests proving exact main-output initialization and bounded logit movement

Seed0 scale scan on the original baseline feature set:

| Variant, seed0 | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.6667 | 0.6528 | 0.2760 |
| Barlow origfeat | 0.7368 | 0.7222 | 1.0000 | 0.4444 | 0.7444 | 0.6912 | 0.2007 |
| preserved main internal | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.6778 | 0.6207 | 0.2100 |
| logit-delta alpha=0.5 | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.6889 | 0.6307 | 0.2048 |
| logit-delta alpha=1.0 | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.7222 | 0.7021 | 0.2033 |
| logit-delta alpha=2.0 | 0.7368 | 0.7333 | 0.8000 | 0.6667 | 0.6667 | 0.6030 | 0.2171 |

Interpretation:

- `alpha=1.0` is the best seed0 bounded-delta setting so far.
- It improves fixed-threshold Acc/Bal Acc/Specificity over both no-SSL seed0
  and the standard Barlow seed0 while keeping sensitivity at 0.90.
- It improves PR AUC and Brier over the internal preserved-main path, showing
  that the Barlow delta adds signal rather than only replaying the main CNN.
- It still does not beat the standard Barlow seed0 on ROC AUC or Brier, so this
  is a promising pilot, not a final all-metric win.
- `alpha=2.0` over-corrects and should not be expanded.

Generated artifacts:

- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_logitdelta_seed0_scale_scan.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_logitdelta_seed0_summary.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_logitdelta_seed0_scale1_summary.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_logitdelta_seed0_scale2_summary.csv`

### First-6 Seed Expansion

The best seed0 setting, `alpha=1.0`, was expanded to seeds `0 1 2 3 4 5`
on the original baseline feature set.

| Model, first 6 seeds | Row | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-SSL CNN | seed-run mean | 0.7632 | 0.7546 | 0.9167 | 0.5926 | 0.7630 | 0.7350 | 0.2063 |
| Barlow SSL-CNN | seed-run mean | 0.7895 | 0.7787 | 0.9833 | 0.5741 | 0.7537 | 0.7319 | 0.2040 |
| Barlow logit-delta alpha=1.0 | seed-run mean | 0.7807 | 0.7741 | 0.9000 | 0.6481 | 0.7759 | 0.7589 | 0.1959 |
| no-SSL CNN | seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7705 | 0.1828 |
| Barlow SSL-CNN | seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7111 | 0.6476 | 0.1973 |
| Barlow logit-delta alpha=1.0 | seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8530 | 0.1724 |

Patient-level paired checks on seedmean6:

| Comparison | Metric | Delta, positive better | Bootstrap 95% CI | One-sided label-permutation p |
| --- | --- | ---: | ---: | ---: |
| logit-delta vs no-SSL | Brier | +0.0104 | [-0.0083, 0.0298] | 0.1798 |
| logit-delta vs no-SSL | ROC AUC | +0.0333 | [-0.0778, 0.1778] | 0.3417 |
| logit-delta vs no-SSL | PR AUC | +0.0825 | [-0.0661, 0.2618] | 0.1778 |
| logit-delta vs Barlow | Brier | +0.0249 | [-0.0127, 0.0651] | 0.3217 |
| logit-delta vs Barlow | ROC AUC | +0.1222 | [0.0000, 0.3111] | 0.0509 |
| logit-delta vs Barlow | PR AUC | +0.2054 | [0.0000, 0.3684] | 0.0170 |

Interpretation:

- The first-6 seed expansion is the strongest model-level result so far for
  preserving fixed-threshold performance while improving score quality.
- Seedmean6 fixed-threshold metrics tie no-SSL and standard Barlow, while
  logit-delta has better ROC AUC, PR AUC, and Brier.
- Seed-run mean improves specificity, ROC AUC, PR AUC, and Brier over both
  baselines, but its sensitivity is lower than standard Barlow.
- The no-SSL comparison is not statistically significant at first-6 scale; the
  Barlow comparison has stronger score-level evidence, especially PR AUC.
- `sub09` and `sub14` remain false positives, but their seedmean scores are
  lower than standard Barlow:
  - sub09: Barlow `0.9904`, logit-delta `0.9489`
  - sub14: Barlow `0.9979`, logit-delta `0.8594`

Generated artifacts:

- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_logitdelta10_first6_per_seed.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_logitdelta10_first6_summary.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_logitdelta10_first6_comparison.csv`
- `runs/baseline_rerun_20260529/results/metrics/feature_ssl_barlow_logitdelta10_first6_statistical_comparison.csv`
- `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_feature_ssl_barlow_logitdelta10_seedmean6.csv`
- `runs/baseline_rerun_20260529/results/metrics/sub09_sub14_logitdelta10_first6_scores.csv`

## Recommended Next Run Order

1. Do not claim the current bridge average is significantly better than no-SSL
   CNN. It is useful evidence for transfer behavior, not a final win.
2. Do not expand bridge + discriminative warmup to 10 seeds; seed1 is a negative
   pilot.
3. Do not expand branch-aware fused Barlow to 10 seeds as-is; the seed0/seed1
   pilot did not beat the current bridge average.
4. Do not expand SSL consistency to 10 seeds as-is; it improves ROC/PR in the
   two-seed pilot but destabilizes fixed-threshold accuracy.
5. Do not expand the current `rescnn` to 10 seeds; it reduces hard-negative
   scores but damages global metrics.
6. Do not expand `gncnn`; it performs worse than the original CNN globally.
7. Do not expand the current `ssl-two-head` implementation; it does not beat
   the current deterministic+freeze-bridge score average.
8. Treat the multi-residual score rule as the best evidence target, not as a
   final model. It improves seed-run mean Acc/Bal Acc/ROC AUC/PR AUC/Brier and
   patient seed-mean ROC AUC/PR AUC/Brier over no-SSL while preserving the
   fixed-threshold confusion matrix, but it is still post-hoc and not
   statistically significant.
9. Do not expand the current model-level random-main residual (`0.14/0.04`) to
   10 seeds. The 6-seed expansion fell below no-SSL on Acc/Bal Acc and below
   no-SSL seedmean6 on ROC/PR despite a small Brier gain.
10. Treat fold-internal no-SSL checkpoint preservation as the current best
   model-level transfer mechanism for ROC/PR/Brier, but not as a final result.
   The `0.30/0.04` 6-seed expansion still loses fixed-threshold Acc/Bal Acc and
   sensitivity relative to no-SSL.
11. Do not expand fold-internal preserved-main residual weight selection; seed0
   was negative because the tiny fold validation set selected weak main-path
   weights.
12. Pre-register the low-gate SSL-consistency rule `w=0.23, threshold=0.505`
   before any new locked rerun if the next objective is to preserve no-SSL
   discrete metrics while improving ROC/PR/Brier. This is currently a
   data-derived candidate, not a final leakage-safe claim.
13. Do not continue score-fusion search as the main route. The score-fusion
   ceiling check preserved but did not improve patient seed-mean Acc/Bal Acc.
   The next model-level route should alter how the supervised CNN consumes the
   Barlow representation before the final classifier, for example a constrained
   SSL adapter trained with a no-SSL logit preservation term plus a small
   ranking/calibration auxiliary loss.
14. Do not expand the first logit-preservation probability-mixture residual
   pilot (`lambda=1.0`, `ssl=0.30`, `bridge=0.04`); seed0 was negative.
15. Expand bounded logit-delta alpha=1.0 to a short multi-seed pilot before a
   10-seed run. First-6 seedmean ties no-SSL and standard Barlow on
   fixed-threshold metrics while improving Brier, ROC AUC, and PR AUC. This is
   enough to justify a locked 10-seed run, but not enough to claim significant
   superiority over no-SSL.
16. Keep fixed `0.5` as the primary threshold for ordinary model outputs unless
   a calibration rule is pre-registered before rerun; report leave-one-seed-out
   calibration only as a secondary leakage-safe analysis.
17. Use patient-level seed-mean plus bootstrap/permutation tests for any final
   locked 10-seed claim; seed-runs are not independent patients.

## Sources

- Barlow Twins: Self-Supervised Learning via Redundancy Reduction, ICML 2021: https://proceedings.mlr.press/v139/zbontar21a.html
- Negative-Sample-Free Contrastive Self-Supervised Learning for EEG-Based Motor Imagery Classification, IEEE Access 2024: https://doaj.org/article/ea27fda79f1d42c4a3cda184cd85f93c
- SelfEEG SSL module documentation, including Barlow Twins and EEG SSL pipeline components: https://selfeeg.readthedocs.io/en/latest/selfeeg.ssl.html
- Lead-fusion Barlow Twins for multi-lead ECG, Information Fusion 2025: https://www.sciencedirect.com/science/article/abs/pii/S1566253524004767
- AutoLR: Layer-wise Pruning and Auto-tuning of Learning Rates in Fine-tuning of Deep Networks, AAAI 2021: https://ojs.aaai.org/index.php/AAAI/article/view/16350
- K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters, Findings of ACL 2021: https://aclanthology.org/2021.findings-acl.121/
- Self-Supervised Representation Learning: Introduction, Advances and Challenges: https://arxiv.org/abs/2110.09327
- Applications of Self-Supervised Learning to Biomedical Signals: A Survey: https://ieeexplore.ieee.org/document/10365170
- Systematic review of self-supervised foundation models for brain network representation using electroencephalography: https://arxiv.org/abs/2602.03269
- DINOv2: Learning Robust Visual Features without Supervision, including self-supervised feature distillation into smaller models: https://arxiv.org/abs/2304.07193
- Progressive transfer learning with Barlow Twins ROMs, including hidden-layer
  information gates for small-sample transfer: https://www.nature.com/articles/s41598-024-64778-y
- A Cookbook of Self-Supervised Learning, for practical SSL transfer and evaluation considerations: https://arxiv.org/abs/2304.12210
- A survey of uncertainty in deep neural networks, including calibration concerns for neural predictions: https://link.springer.com/article/10.1007/s10462-023-10562-9
- A survey on deep learning tools dealing with data scarcity, including transfer learning and SSL as small-data strategies: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00727-2
