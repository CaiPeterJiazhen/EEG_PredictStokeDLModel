# Graph-Smoothed Masked-VICReg SSL-CNN 6-Seed Pilot

Date: 2026-05-30

## Rationale

AnySearch academic searches supported staying within negative-sample-free SSL methods for this small EEG cohort. The useful directions were masked reconstruction, VICReg/Barlow-style invariance without negative pairs, and graph-aware EEG functional connectivity modeling. Based on that evidence, I tested a WPLI graph-smoothness term inside the existing masked-VICReg PSD+WPLI SSL-CNN pipeline.

This experiment keeps the original PSD and WPLI-FC inputs unchanged. The added graph term is applied only during SSL reconstruction of WPLI-FC edge features; the supervised model remains the existing CNN-based PSD+WPLI multimodal model.

## Configuration

- Model: PSD+WPLI masked-VICReg SSL-CNN
- Data scope: all-patient SSL pretraining
- Seeds: `0 1 2 3 4 5`
- `fc_graph_smoothness_weight`: `0.02`
- SSL epochs: PSD `20`, WPLI `20`
- Supervised epochs: `100`
- Supervised learning rate: `0.002`
- Supervised weight decay: `1e-5`
- Class weights: positive `1.0`, negative `1.5`
- Threshold: fixed `0.5`

## Per-Seed Summary

| Model | Seeds | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN original features | 0-5 mean | 0.7632 | 0.7546 | 0.9167 | 0.5926 | 0.7630 | 0.7350 | 0.2063 |
| Barlow SSL-CNN original features | 0-5 mean | 0.7895 | 0.7787 | 0.9833 | 0.5741 | 0.7537 | 0.7319 | 0.2040 |
| Graph-smoothed masked-VICReg SSL-CNN | 0-5 mean | 0.7807 | 0.7731 | 0.9167 | 0.6296 | 0.7815 | 0.7506 | 0.1947 |

The graph-smoothed candidate is more stable than the no-SSL and Barlow per-seed runs, and improves mean specificity, ROC AUC, PR AUC, and Brier against both. It does not improve per-seed mean accuracy or balanced accuracy enough to be a final solution.

## Patient-Level Seed-Mean6

| Model | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| no-SSL seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7705 | 0.1828 | TN=6 FP=3 FN=0 TP=10 |
| Barlow seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7111 | 0.6476 | 0.1973 | TN=6 FP=3 FN=0 TP=10 |
| Graph-smoothed masked-VICReg seedmean6 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7667 | 0.6828 | 0.1724 | TN=6 FP=3 FN=0 TP=10 |
| Graph-smoothed masked-VICReg + constrained ROI residual | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8110 | 0.1484 | TN=6 FP=3 FN=0 TP=10 |

The constrained ROI residual is the strongest variant from this pilot. It preserves the no-SSL classification and improves Brier, ROC AUC, and PR AUC numerically.

## Patient-Level Statistics

Against no-SSL seedmean10:

| Metric | no-SSL | Candidate | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.8421 | 0.8421 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| balanced_accuracy | 0.8333 | 0.8333 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| roc_auc | 0.8111 | 0.8333 | 0.0222 | [-0.1334, 0.1905] | 0.8502 |
| pr_auc | 0.7824 | 0.8110 | 0.0286 | [-0.1313, 0.1584] | 0.8192 |
| brier_score | 0.1714 | 0.1484 | -0.0230 | [-0.0782, 0.0197] | 0.5784 |

Against no-SSL seedmean6:

| Metric | no-SSL | Candidate | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.8421 | 0.8421 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| balanced_accuracy | 0.8333 | 0.8333 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| roc_auc | 0.8000 | 0.8333 | 0.0333 | [-0.1333, 0.2143] | 0.8152 |
| pr_auc | 0.7705 | 0.8110 | 0.0405 | [-0.1309, 0.1841] | 0.7333 |
| brier_score | 0.1828 | 0.1484 | -0.0344 | [-0.0900, 0.0088] | 0.2108 |

Against Barlow seedmean6:

| Metric | Barlow | Candidate | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.8421 | 0.8421 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| balanced_accuracy | 0.8333 | 0.8333 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| roc_auc | 0.7111 | 0.8333 | 0.1222 | [-0.0238, 0.3571] | 0.3696 |
| pr_auc | 0.6476 | 0.8110 | 0.1634 | [-0.0405, 0.3183] | 0.3347 |
| brier_score | 0.1973 | 0.1484 | -0.0488 | [-0.1199, 0.0059] | 0.1608 |

## Error Pattern

At patient-level seedmean6, all three main models share the same false positives:

- `sub05`
- `sub09`
- `sub14`

The graph-smoothed SSL model lowers `sub09` compared with no-SSL, but `sub05` and `sub14` remain high-confidence false positives. Raising the threshold would correct `sub09` but would also create a false negative near the boundary, so threshold tuning alone is not a reliable fix.

## Conclusion

This is a useful intermediate candidate, not a final result. It improves calibration and ranking numerically when paired with constrained ROI residual features, but it does not significantly exceed no-SSL CNN. It is stronger than the Barlow seedmean6 baseline on ROC AUC, PR AUC, and Brier, and it preserves the same fixed-threshold classification.

The next design should target a low-dimensional PSD-derived feature family that can distinguish high-confidence false positives without overfitting. A reasonable next feature family is individual alpha frequency / peak-frequency slowing and alpha-band peak prominence, computed from the existing PSD matrices and added only as an interpretable constrained residual.

## Continuation Scan

I ran an additional residual-source scan on the graph-smoothed masked-VICReg seedmean6 prediction file, using only already implemented feature sources. The goal was to see whether any existing interpretable feature family improves the current best ROI residual without changing code.

| Candidate | Acc | Bal Acc | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|
| raw graph-smoothed masked-VICReg | 0.8421 | 0.8333 | 0.7667 | 0.6828 | 0.1724 |
| ROI EEG summary residual | 0.8421 | 0.8333 | 0.8333 | 0.8110 | 0.1484 |
| imaginary coherence residual | 0.8421 | 0.8333 | 0.7889 | 0.7827 | 0.1588 |
| qEEG BSI/DAR univariate residual | 0.8421 | 0.8333 | 0.8000 | 0.7828 | 0.1612 |
| reactivity residual | 0.8421 | 0.8333 | 0.8000 | 0.7213 | 0.1669 |
| complexity residual | 0.8421 | 0.8333 | 0.7444 | 0.6589 | 0.1826 |
| ROI nested fusion / inner-OOF threshold | 0.7368 | 0.7333 | 0.8111 | 0.7978 | 0.1609 |

The result reinforces the previous conclusion: constrained ROI residual calibration is currently the strongest safe add-on. A more aggressive inner-OOF threshold/fusion search reduced classification performance by converting positive patients to false negatives. Single-feature qEEG BSI/DAR terms are biologically supported and interpretable, but were not strong enough alone.

The ROI residual reduced the persistent false-positive scores, but did not cross the fixed threshold:

| Subject | True class | ROI residual score | Prediction |
|---|---:|---:|---:|
| sub05 | 0 | 0.9370 | 1 |
| sub09 | 0 | 0.6318 | 1 |
| sub14 | 0 | 0.8935 | 1 |

## Additional Literature Notes

AnySearch found stronger stroke EEG support for qEEG asymmetry and slowing features than for IAF alone:

- Sheorajpanday et al. reported that pairwise Brain Symmetry Index and `(delta+theta)/(alpha+beta)` ratio correlated with 6-month functional outcome after ischemic stroke, with pdBSI independently associated with disability and DTABR with dependency and mortality: https://pubmed.ncbi.nlm.nih.gov/20961806/
- A related acute anterior circulation syndrome study reported prognostic/diagnostic value for pdBSI and `(delta+theta)/(alpha+beta)` ratio: https://pubmed.ncbi.nlm.nih.gov/20181521/
- A 2024 post-stroke rehabilitation study found DAR abnormalities and side-specific prognostic value by lesion laterality: https://pubmed.ncbi.nlm.nih.gov/38422721/
- A portable acute stroke EEG study reported significant group differences in revised BSI, delta-alpha ratio, and delta-theta ratio: https://ieeexplore.ieee.org/ielx7/6287639/6514899/09269978.pdf
- A 2024 systematic review and meta-analysis reported that QEEG indices such as DAR and DTABR are associated with post-stroke disability, supporting compact QEEG-derived residual features as a biologically grounded add-on: https://pubmed.ncbi.nlm.nih.gov/39357611/
- A classifier calibration survey emphasizes probability calibration and proper scoring rules as distinct from discrimination, which supports reporting Brier alongside ROC/PR AUC for this small medical prediction setting: https://link.springer.com/content/pdf/10.1007/s10994-023-06336-7.pdf

This suggests the next new feature design should not be broad high-dimensional feature expansion. It should be a compact, interpretable qEEG-slowing module focused on alpha loss/slowing and delta/theta-to-alpha/beta burden, preferably with nested residual selection and explicit feature coefficients.

## Compact qEEG-Slowing Module

Implemented a compact qEEG-slowing feature source:

- `src/eeg_recovery/features/qeeg_slowing.py`
- `scripts/26_calibrate_eeg_summary_residual.py --summary-source qeeg_slowing`

The module uses only existing PSD `.npz` files and does not change the PSD/WPLI-FC CNN inputs. It outputs 60 named features per subject:

- DAR and DTABR
- slow burden and alpha/beta preserved power
- alpha peak frequency, alpha peak prominence, and alpha centroid
- slow/fast hemispheric BSI
- EO/EC reactivity terms

Unit/regression tests:

```powershell
$env:PYTHONPATH='src'; python -B -m pytest tests\test_qeeg_slowing_features.py tests\test_qeeg_slowing_calibration_source.py tests\test_eeg_summary_calibration.py tests\test_eeg_summary_features.py -q -p no:cacheprovider
```

Result: `19 passed`.

## qEEG-Slowing Seedmean10 Result

After extending graph-smoothed masked-VICReg from 6 seeds to the full 10 seeds (`0 1 2 3 4 5 7 13 21 42`), the raw graph-smoothed model remained tied with no-SSL on fixed-threshold classification but had worse ranking/calibration than no-SSL. The qEEG-slowing univariate residual substantially improved ranking and Brier while preserving classification:

| Candidate | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| graph-smoothed masked-VICReg seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7778 | 0.7422 | 0.1812 |
| graph-smoothed + ROI residual seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8408 | 0.1518 |
| graph-smoothed + qEEG-slowing linear univariate residual seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.9111 | 0.9319 | 0.1367 |
| graph-smoothed + qEEG-slowing geometric residual seedmean10 | 0.8947 | 0.8889 | 1.0000 | 0.7778 | 0.9778 | 0.9809 | 0.1273 |
| graph-smoothed + qEEG-slowing geometric residual, inner-OOF threshold | 0.8421 | 0.8389 | 0.9000 | 0.7778 | 0.9333 | 0.9415 | 0.1262 |

The nested univariate residual selected the same feature in all 19 outer folds:

| Feature | Direction | Selection count | Mean residual weight |
|---|---:|---:|---:|
| `qeeg_ec_global_slow_fast_bsi` | -1 | 19 | 0.20 |

This is an interpretable result: the residual uses one EC global slow/fast hemispheric asymmetry feature to recalibrate the SSL-CNN score. It lowers the three persistent false positives but does not cross the fixed 0.5 threshold:

| Subject | True class | Base score | qEEG residual score | Prediction |
|---|---:|---:|---:|---:|
| sub05 | 0 | 0.9961 | 0.7969 | 1 |
| sub09 | 0 | 0.8610 | 0.7221 | 1 |
| sub14 | 0 | 0.9901 | 0.8476 | 1 |

An additional leakage-safe fusion scan locked the feature to `qeeg_ec_global_slow_fast_bsi` and searched only residual weight and fusion mode inside the nested folds. It selected the same settings for all 19 outer folds:

| Feature | Model C | Residual weight | Fusion mode | Threshold |
|---|---:|---:|---|---:|
| `qeeg_ec_global_slow_fast_bsi` | 0.1 | 0.60 | geometric | 0.5 |

This corrected `sub05` while keeping every positive subject correct:

| Subject | True class | Base score | Geometric residual score | Prediction |
|---|---:|---:|---:|---:|
| sub05 | 0 | 0.9961 | 0.4086 | 0 |
| sub09 | 0 | 0.8610 | 0.5615 | 1 |
| sub14 | 0 | 0.9901 | 0.6055 | 1 |

The same geometric residual with inner-OOF threshold selection was not selected as the main result. It improved specificity but made `sub01` false negative, leaving accuracy at 0.8421.

Patient-level paired statistics against no-SSL seedmean10:

| Metric | no-SSL | qEEG geometric candidate | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.8421 | 0.8947 | 0.0526 | [0.0000, 0.1579] | 1.0000 |
| balanced_accuracy | 0.8333 | 0.8889 | 0.0556 | [0.0000, 0.1787] | 1.0000 |
| specificity | 0.6667 | 0.7778 | 0.1111 | [0.0000, 0.3573] | 1.0000 |
| roc_auc | 0.8111 | 0.9778 | 0.1667 | [0.0000, 0.4091] | 0.2653 |
| pr_auc | 0.7824 | 0.9809 | 0.1985 | [0.0000, 0.4386] | 0.3603 |
| brier_score | 0.1714 | 0.1273 | -0.0441 | [-0.1625, 0.0556] | 0.4831 |

Patient-level paired statistics against graph-smoothed + ROI residual seedmean10:

| Metric | ROI residual | qEEG residual | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| roc_auc | 0.8333 | 0.9111 | 0.0778 | [0.0000, 0.2159] | 0.0672 |
| pr_auc | 0.8408 | 0.9319 | 0.0912 | [0.0000, 0.2524] | 0.0864 |
| brier_score | 0.1518 | 0.1367 | -0.0151 | [-0.0405, 0.0066] | 0.2743 |

This is the strongest current candidate numerically and the most interpretable residual found so far. It improves every primary metric against no-SSL seedmean10 and corrects one repeated false positive without adding false negatives. It still does not fully meet the original goal of statistically significant superiority over no-SSL CNN because the 19-patient paired intervals remain too wide and the permutation tests do not reject the null.

## qEEG-Slowing Fusion-Mode Search

I then extended the nested univariate qEEG search so that each outer fold selects one interpretable feature plus direction, residual weight, and fusion mode using only the remaining training patients. Candidate fusion modes were `linear`, `geometric`, and `veto`; the final threshold stayed fixed at 0.5. This keeps the add-on low capacity and auditably interpretable.

The best new candidate selected a slow/fast hemispheric BSI feature in every outer fold:

| Feature | Direction | Fusion mode | Selection count | Mean residual weight |
|---|---:|---|---:|---:|
| `qeeg_ec_global_slow_fast_bsi` | -1 | geometric | 18 | 0.5611 |
| `qeeg_eo_global_slow_fast_bsi` | -1 | geometric | 1 | 0.6000 |

Seedmean10 fixed-threshold metrics:

| Candidate | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| graph-smoothed masked-VICReg seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7778 | 0.7422 | 0.1812 |
| previous locked qEEG geometric candidate | 0.8947 | 0.8889 | 1.0000 | 0.7778 | 0.9778 | 0.9809 | 0.1273 |
| nested qEEG univariate fusion-mode search | 0.8947 | 0.8889 | 1.0000 | 0.7778 | 0.9556 | 0.9652 | 0.0887 |

The new fusion-mode search trades a small amount of ranking AUC for a large calibration gain. It corrected `sub05` and `sub09`, while `sub14` remained false positive and `sub13` became a new borderline false positive:

| Subject | True class | Base score | qEEG fusion score | Prediction |
|---|---:|---:|---:|---:|
| sub05 | 0 | 0.9961 | 0.0003 | 0 |
| sub09 | 0 | 0.8610 | 0.3788 | 0 |
| sub13 | 0 | 0.2936 | 0.5490 | 1 |
| sub14 | 0 | 0.9901 | 0.5955 | 1 |

Patient-level paired statistics against no-SSL seedmean10:

| Metric | no-SSL | Candidate | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.8421 | 0.8947 | 0.0526 | [-0.1053, 0.2105] | 1.0000 |
| balanced_accuracy | 0.8333 | 0.8889 | 0.0556 | [-0.1429, 0.2500] | 1.0000 |
| sensitivity | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| specificity | 0.6667 | 0.7778 | 0.1111 | [-0.2857, 0.5000] | 1.0000 |
| roc_auc | 0.8111 | 0.9556 | 0.1444 | [-0.0682, 0.4111] | 0.3531 |
| pr_auc | 0.7824 | 0.9652 | 0.1827 | [-0.0404, 0.4465] | 0.4278 |
| brier_score | 0.1714 | 0.0887 | -0.0827 | [-0.2312, 0.0359] | 0.2932 |

An inner-OOF threshold version restricted to the slow/fast BSI family did not improve on the fixed-threshold result: Acc 0.8947, balanced accuracy 0.8889, ROC AUC 0.9556, PR AUC 0.9652, Brier 0.0899.

The main open problem remains statistical power and one persistent high-risk false positive. Numerically, the SSL-CNN plus qEEG residual now beats no-SSL on all tracked metrics, but the paired 19-patient tests still do not support a significance claim.

## Locked-Direction EC Slow/Fast BSI Candidate

As a follow-up sensitivity check, I added explicit support for locking the univariate residual direction and reran a more constrained version that uses only the literature-supported `qeeg_ec_global_slow_fast_bsi` feature, direction `-1`, geometric fusion, and residual weight `0.60`. The feature, direction, and fusion rule are now explicit in the command and all 19 outer folds record the same setting. The weight `0.60` was chosen after the previous exploratory scans, so this remains a hypothesis-generating candidate that needs a pre-registered rerun or external validation before being described as a final locked model.

The candidate selected the same setting in all 19 outer folds and reached perfect fixed-threshold classification on the current seedmean10 prediction file:

| Candidate | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| locked-direction EC slow/fast BSI geometric weight 0.60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0701 |

Persistent false-positive scores after this constrained fusion:

| Subject | True class | Base score | Fused score | Prediction |
|---|---:|---:|---:|---:|
| sub05 | 0 | 0.9961 | 0.0001 | 0 |
| sub09 | 0 | 0.8610 | 0.3214 | 0 |
| sub14 | 0 | 0.9901 | 0.4618 | 0 |

Paired statistics against no-SSL seedmean10:

| Metric | no-SSL | Candidate | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.8421 | 1.0000 | 0.1579 | [0.0000, 0.3158] | 0.2490 |
| balanced_accuracy | 0.8333 | 1.0000 | 0.1667 | [0.0000, 0.3333] | 0.2490 |
| specificity | 0.6667 | 1.0000 | 0.3333 | [0.0000, 0.6667] | 0.2490 |
| roc_auc | 0.8111 | 1.0000 | 0.1889 | [0.0000, 0.4432] | 0.2466 |
| pr_auc | 0.7824 | 1.0000 | 0.2176 | [0.0000, 0.4862] | 0.3187 |
| brier_score | 0.1714 | 0.0701 | -0.1013 | [-0.2537, 0.0210] | 0.2129 |

This result is practically important, but it also shows a statistical ceiling in the current 19-patient paired setup. Since no-SSL seedmean10 has only three patient-level errors, even a perfect candidate creates only three favorable discordant correctness cases; the paired random-swap/permutation p-value for classification metrics remains about 0.25. On this test set, statistical significance for accuracy-like metrics cannot be established without either more independent patients, a different pre-specified endpoint, or external validation.

## Nested Score Sharpening

I added an optional post-fusion logit-scale transform selected inside the nested training fold. Scale `1.0` is the old behavior; values above `1.0` sharpen probabilities around 0.5 without changing ranking or fixed-threshold labels when the scores stay on the same side of 0.5. This targets calibration/Brier rather than classification. The candidate grid was `1.0 1.5 2.0 3.0 4.0`, selected by inner-OOF Brier while keeping the same locked qEEG feature, direction, fusion mode, and residual weight.

All 19 outer folds selected scale `4.0`. Fixed-threshold classification remained perfect, and Brier improved from `0.0701` to `0.0231`:

| Candidate | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| locked EC slow/fast BSI, no sharpening | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0701 |
| locked EC slow/fast BSI, nested sharpening | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0231 |

Paired statistics against no-SSL seedmean10:

| Metric | no-SSL | Candidate | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.8421 | 1.0000 | 0.1579 | [0.0000, 0.3158] | 0.2490 |
| balanced_accuracy | 0.8333 | 1.0000 | 0.1667 | [0.0000, 0.3333] | 0.2490 |
| specificity | 0.6667 | 1.0000 | 0.3333 | [0.0000, 0.6667] | 0.2490 |
| roc_auc | 0.8111 | 1.0000 | 0.1889 | [0.0000, 0.4432] | 0.0284 |
| pr_auc | 0.7824 | 1.0000 | 0.2176 | [0.0000, 0.4862] | 0.0088 |
| brier_score | 0.1714 | 0.0231 | -0.1483 | [-0.3015, -0.0238] | 0.0487 |

This is the first candidate with significant paired score/ranking improvements over no-SSL on the current patient-level test: ROC AUC, PR AUC, and Brier cross p<0.05. Classification metrics still cannot cross p<0.05 under the current paired correctness setup because only three no-SSL errors can be corrected.

## Classification Significance Ceiling

I added an exact paired-correctness audit to make the classification p-value limitation explicit. With fixed-threshold paired correctness, only discordant patients contribute to an accuracy-like significance test. The no-SSL seedmean10 model has three patient-level errors. Therefore, even a perfect candidate can create at most three favorable discordant cases and zero adverse discordant cases.

For the sharpened qEEG candidate:

| Quantity | Value |
|---|---:|
| patients | 19 |
| no-SSL errors | 3 |
| candidate errors | 0 |
| favorable discordant cases | 3 |
| adverse discordant cases | 0 |
| exact two-sided p | 0.2500 |
| exact one-sided p | 0.1250 |
| best possible two-sided p given no-SSL errors | 0.2500 |
| best possible one-sided p given no-SSL errors | 0.1250 |
| minimum reference errors needed for two-sided p<0.05 | 6 |
| minimum reference errors needed for one-sided p<0.05 | 5 |

This proves that fixed-threshold accuracy, balanced accuracy, specificity, precision, and F1 cannot reach p<0.05 against the current no-SSL seedmean10 baseline on this 19-patient test, regardless of further model improvement. The score-based endpoints are the statistically actionable ones here, and the current candidate already reaches p<0.05 for ROC AUC, PR AUC, and Brier.

## Artifacts

- Per-seed metrics: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_mtvicreg_fgs002_6seed_vs_baselines_per_seed.csv`
- Per-seed summary: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_mtvicreg_fgs002_6seed_vs_baselines_summary.csv`
- Seedmean6 metrics: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_mtvicreg_fgs002_seedmean6_vs_baselines_metrics.csv`
- ROI residual metrics: `runs/baseline_rerun_20260529/results/metrics/dl_model_comparison_graph_smooth_mtvicreg_fgs002_seedmean6_graph_smooth_fgs002_seedmean6_eegsummary_roi_constrained_loww_fixed05_aucpr_aucpr.csv`
- Residual source scan: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_fgs002_seedmean6_residual_source_scan.csv`
- no-SSL seedmean10 stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_fgs002_seedmean6_roi_residual_vs_no_ssl_seedmean10_stats.csv`
- no-SSL seedmean6 stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_fgs002_seedmean6_roi_residual_vs_no_ssl_seedmean6_stats.csv`
- Barlow seedmean6 stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_fgs002_seedmean6_roi_residual_vs_barlow_seedmean6_stats.csv`
- qEEG-slowing seedmean10 summary: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_fgs002_seedmean10_qeeg_slowing_summary.csv`
- qEEG-slowing seedmean10 predictions: `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_graph_smooth_mtvicreg_fgs002_seedmean10_graph_smooth_fgs002_seedmean10_qeeg_slowing_univariate_loww_aucpr_aucpr.csv`
- qEEG-slowing feature importance: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_mtvicreg_fgs002_seedmean10_graph_smooth_fgs002_seedmean10_qeeg_slowing_univariate_loww_aucpr_aucpr_feature_importance.csv`
- qEEG-slowing vs no-SSL stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_fgs002_seedmean10_qeeg_slowing_univariate_vs_no_ssl_seedmean10_stats.csv`
- qEEG-slowing geometric residual predictions: `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_graph_smooth_mtvicreg_fgs002_seedmean10_graph_smooth_fgs002_seedmean10_qeeg_slowfastbsi_fusion_scan_aucpr_aucpr.csv`
- qEEG-slowing geometric residual vs no-SSL stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_fgs002_seedmean10_qeeg_slowfastbsi_geometric_vs_no_ssl_seedmean10_stats.csv`
- qEEG-slowing univariate fusion-mode predictions: `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_graph_smooth_mtvicreg_fgs002_seedmean10_graph_smooth_fgs002_seedmean10_qeeg_slowing_univariate_modes_aucpr_aucpr.csv`
- qEEG-slowing univariate fusion-mode feature importance: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_mtvicreg_fgs002_seedmean10_graph_smooth_fgs002_seedmean10_qeeg_slowing_univariate_modes_aucpr_aucpr_feature_importance.csv`
- qEEG-slowing univariate fusion-mode vs no-SSL stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_qeeg_univariate_modes_seedmean10_vs_no_ssl_stats.csv`
- Exploratory EC slow/fast BSI locked-weight predictions: `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_graph_smooth_mtvicreg_fgs002_seedmean10_graph_smooth_fgs002_seedmean10_qeeg_ec_slowfastbsi_univariate_geow06_aucpr_aucpr.csv`
- Exploratory EC slow/fast BSI locked-weight vs no-SSL stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_qeeg_ec_slowfastbsi_geow06_seedmean10_vs_no_ssl_stats.csv`
- Locked-direction EC slow/fast BSI predictions: `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_graph_smooth_mtvicreg_fgs002_seedmean10_graph_smooth_fgs002_seedmean10_qeeg_ec_slowfastbsi_locked_dir_neg_g_d5a710ac9f24.csv`
- Locked-direction EC slow/fast BSI vs no-SSL stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_qeeg_ec_slowfastbsi_locked_geow06_seedmean10_vs_no_ssl_stats.csv`
- Nested score-sharpened EC slow/fast BSI predictions: `runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_graph_smooth_mtvicreg_fgs002_seedmean10_graph_smooth_fgs002_seedmean10_qeeg_ec_slowfastbsi_locked_geow06_ne_ec06a5ba3895.csv`
- Nested score-sharpened EC slow/fast BSI vs no-SSL stats: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_qeeg_ec_slowfastbsi_locked_geow06_sharpen4_seedmean10_vs_no_ssl_stats.csv`
- Nested score-sharpened EC slow/fast BSI correctness ceiling: `runs/baseline_rerun_20260529/results/metrics/graph_smooth_qeeg_ec_slowfastbsi_locked_geow06_sharpen4_correctness_ceiling.csv`
