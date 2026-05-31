# SSL-CNN Complementarity Audit and Next Direction

Date: 2026-05-30

## Scope

This audit checks whether the current negative-sample-free SSL-CNN results contain enough patient-level complementarity to plausibly beat the no-SSL CNN baseline on this 19-patient LOSO task.

Reference no-SSL baseline:

- Predictions: `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_no_ssl_origfeat_seedmean10.csv`
- Metrics: accuracy `0.8421`, balanced accuracy `0.8333`, sensitivity `1.0000`, specificity `0.6667`, ROC AUC `0.8111`, PR AUC `0.7824`, Brier `0.1714`
- Wrong subjects: `sub05`, `sub09`, `sub14`

Generated audit files:

- Full audit: `runs/baseline_rerun_20260529/results/metrics/ssl_cnn_candidate_complementarity_audit.csv`
- Filtered audit: `runs/baseline_rerun_20260529/results/metrics/ssl_cnn_complementarity_filtered_summary.csv`

## Current Evidence

Stable SSL-only candidates do not correct the no-SSL error set at the fixed 0.5 decision threshold. The top stable SSL-only rows all preserve the same wrong subjects: `sub05`, `sub09`, and `sub14`.

| Candidate | Acc | BalAcc | Sens | Spec | ROC AUC | PR AUC | Brier | Wrong subjects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no-SSL seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 | sub05; sub09; sub14 |
| Barlow freezebridge seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7778 | 0.7788 | 0.1660 | sub05; sub09; sub14 |
| Barlow det25/freezebridge75 ensemble6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.8039 | 0.1667 | sub05; sub09; sub14 |
| Barlow seedensemble5 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8530 | 0.1691 | sub05; sub09; sub14 |
| VICReg seedensemble10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7778 | 0.7544 | 0.1793 | sub05; sub09; sub14 |
| Masked-VICReg seedensemble10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7444 | 0.6822 | 0.1851 | sub05; sub09; sub14 |
| Masked-Barlow seedensemble10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.6778 | 0.6043 | 0.1917 | sub05; sub09; sub14 |

The best current SSL-side result remains a low-weight constrained residual over masked-VICReg plus interpretable ROI summaries:

- `masked_vicreg_seedmean10_eegsummary_roi_noratio_nolesion_nowpligraph_nospectralshape_constrained_loww_fixed05_aucpr`
- Accuracy and balanced accuracy match no-SSL: `0.8421` and `0.8333`
- ROC AUC, PR AUC, and Brier improve numerically: `0.8333`, `0.8489`, `0.1599`
- Prior patient-level bootstrap/permutation intervals crossed zero, so this is not a significant improvement.

## Why Single-Seed Wins Are Not Enough

Some single-seed SSL or no-SSL runs correct one of `sub05/sub09/sub14`, but they introduce a new error or rely on choosing a seed after seeing the final 19-patient results. For example:

- A single VICReg seed corrected `sub14`, but introduced `sub28` as a new error.
- A single masked-VICReg seed corrected `sub09`, but introduced `sub07`.
- A single no-SSL seed corrected `sub14` without a new error, but that is not an SSL-specific effect and cannot be selected as a main result without a pre-registered seed policy.

This means the current SSL family has weak stable complementarity with no-SSL. It mostly shifts ranking/calibration, not patient-level decisions.

## Literature Signals From AnySearch

Useful sources found during this audit:

- Barlow Twins is explicitly negative-sample-free and has EEG evidence in motor-imagery classification: `Negative-Sample-Free Contrastive Self-Supervised Learning for Electroencephalogram-Based Motor Imagery Classification`, IEEE Access 2024, surfaced via `https://github.com/dlcjfgmlnasa/Barlow_Twins_EEG`.
- Biomedical signal SSL surveys support negative-sample-free or generative SSL for label-scarce biosignal settings: `Applications of Self-Supervised Learning to Biomedical Signals: A Survey`, IEEE.
- GMAEEG directly motivates graph-aware masked autoencoding for EEG: `GMAEEG: A Self-Supervised Graph Masked Autoencoder for EEG Representation Learning`, IEEE JBHI 2024, PubMed `https://pubmed.ncbi.nlm.nih.gov/39146173/`.
- Stroke recovery literature supports connectivity-based analysis of motor networks after stroke: `Reorganization of cerebral networks after stroke`, Brain 2011, and related connectivity reviews.

## Design Implication

Do not spend more runs on plain Barlow/VICReg/BYOL/SimSiam variants unless the pretext task changes materially. The existing runs already show:

- Barlow seed0 can reproduce `0.7368`, but its stable seedmean results do not exceed no-SSL classification.
- Masked-Barlow seed0 was promising, but 10-seed patient-level ranking/calibration degraded.
- Masked reconstruction and masked reconstruction plus VICReg have already been tested; the latter looked promising at seed0 but failed to generalize across 10 seeds.
- EEG-derived residual features help calibration modestly but have not produced significant gains.

The next plausible experiment should use the fact that WPLI is a graph, not a dense image. A reasonable next candidate is:

`graph-regularized masked WPLI autoencoding + PSD masked reconstruction + weak VICReg/Barlow invariance`

Constraints for this candidate:

- Keep PSD and WPLI-FC input tensors unchanged.
- Use no negative samples.
- Keep the supervised model as a CNN or CNN-hybrid: the CNN branches remain the supervised backbone, while graph masking is an SSL auxiliary/pretraining task.
- Use fold-internal LOSO pretraining only; no held-out subject in SSL for that fold.
- Keep interpretability through branch gates, feature/edge masking importances, and grouped PSD-band/WPLI-edge ablation.
- Start with seed0 only. Run 10 seeds only if seed0 improves balanced accuracy, Brier, or fixes a repeated false positive without creating an offsetting false negative.

## Current Recommendation

The current result set does not prove that SSL-CNN significantly beats CNN. It proves that negative-sample-free SSL can match no-SSL classification and sometimes improve ranking/calibration, but the repeated false positives remain stable.

The next experiment should be graph-aware rather than another feature-vector SSL objective. If graph-aware masked SSL also preserves the same `sub05/sub09/sub14` error set, the limiting factor is likely cohort size/label separability rather than the specific SSL objective.
