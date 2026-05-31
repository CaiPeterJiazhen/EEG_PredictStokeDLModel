# ROI EEG Summary Features and Nested Calibration

Date: 2026-05-30

## Rationale

AnySearch literature search supported two useful constraints for the next iteration:

- Negative-sample-free SSL remains appropriate for biomedical signals, but current Barlow/VICReg scans did not give stable 10-seed gains on this small 19-patient LOSO task.
- Stroke EEG papers repeatedly emphasize spectral power, hemispheric asymmetry/BSI, laterality, and resting-state FC as clinically meaningful biomarkers, so interpretable auxiliary features should be ROI/band/hemisphere named rather than a larger opaque branch.

Sources searched:

- Applications of Self-Supervised Learning to Biomedical Signals: A Survey: https://ieeexplore.ieee.org/ielx7/6287639/6514899/10365170.pdf
- EEG Biomarkers Related With the Functional State of Stroke Patients: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00582/full
- Biomarkers of stroke recovery using EEG-based resting-state functional connectivity: https://pmc.ncbi.nlm.nih.gov/articles/PMC13107850/
- HealthSOS stroke prognostics EEG result: https://ieeexplore.ieee.org/ielx7/6287639/6514899/09269978.pdf
- Few-shot EEG classification survey covering SSL/DA/TL: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1421922/pdf
- Early resting-state EEG prediction after stroke; DAR, BSI, and BSIdir were evaluated, with theta-band BSI adding prognostic value: https://pubmed.ncbi.nlm.nih.gov/33248434/

## Implemented

- Extended `src/eeg_recovery/features/eeg_summary.py` from global summaries to 186 named features:
  - global PSD mean and BSI by EO/EC and frequency band
  - ROI PSD mean and BSI by frontal/central/temporal/parietal/occipital regions
  - WPLI mean, EC-EO absolute delta, left-intra/right-intra/interhemispheric means, and intra-hemispheric asymmetry by band/state
- Added optional stroke qEEG ratio/relative-power features, increasing the full summary vector to 294 named features:
  - delta-alpha ratio, delta-theta ratio, theta-beta ratio
  - delta+theta over alpha+beta ratio
  - global and ROI relative band power
- Added lesion-aligned directional features from the already hemisphere-aligned PSD/WPLI files:
  - PSD ipsilesional and contralesional mean power by state/band/ROI
  - PSD ipsi-contra signed asymmetry
  - WPLI ipsilesional and contralesional intra-hemispheric mean connectivity
  - WPLI intra-hemispheric signed asymmetry
- Added WPLI graph theory features from the unchanged WPLI-FC matrices:
  - global node strength mean
  - weighted global efficiency
  - weighted clustering coefficient
  - ipsilesional and contralesional strength means
  - strength signed asymmetry
- Added PSD spectral-shape features from the unchanged PSD matrices:
  - spectral entropy
  - spectral centroid
  - spectral spread
  - 95% spectral edge frequency
- Extended `src/eeg_recovery/training/eeg_summary_calibration.py` with nested inner-OOF threshold selection.
- Added summary feature regex filtering so the best non-ratio ROI feature set can be reproduced with `--exclude-feature-regex "relative|ratio"`.
- Added nested-selected low-capacity probability fusion modes:
  - `linear`: weighted average of CNN and summary probabilities
  - `geometric`: probability geometric mean, which penalizes disagreement more strongly
  - `veto`: only lowers the CNN probability when the summary model is less positive
- Added tests for ROI feature names and leakage-safe threshold selection.

## Results

Patient seed-mean baselines:

| Model | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8111 | 0.7824 | 0.1714 |
| masked-VICReg SSL-CNN seedmean10 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7444 | 0.6822 | 0.1851 |
| masked-VICReg + ROI nested calibration, fixed 0.5 | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.8556 | 0.8829 | 0.1679 |
| masked-VICReg + ROI nested calibration, inner-OOF threshold | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.8444 | 0.8711 | 0.1714 |
| masked-VICReg + constrained low-weight ROI residual, fixed 0.5 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.8333 | 0.8489 | 0.1599 |
| masked-VICReg + constrained low-weight full qEEG ROI residual, fixed 0.5 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7667 | 0.7594 | 0.1683 |
| masked-VICReg + lesion-aligned directional residual only, fixed 0.5 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7444 | 0.6761 | 0.1857 |
| masked-VICReg + ROI + lesion-aligned residual, fixed 0.5 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.7889 | 0.7827 | 0.1675 |
| no-SSL + ROI + lesion-aligned residual, fixed 0.5 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.8444 | 0.8362 | 0.1550 |
| masked-VICReg + WPLI graph residual only, fixed 0.5 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7000 | 0.6310 | 0.1936 |
| no-SSL + WPLI graph residual only, fixed 0.5 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7667 | 0.6828 | 0.1710 |
| masked-VICReg + PSD spectral-shape residual only, fixed 0.5 | 0.7895 | 0.7778 | 1.0000 | 0.5556 | 0.6889 | 0.6210 | 0.1975 |
| no-SSL + PSD spectral-shape residual only, fixed 0.5 | 0.8421 | 0.8333 | 1.0000 | 0.6667 | 0.7444 | 0.6677 | 0.1776 |
| masked-VICReg + nested univariate PSD BSI/gamma residual, fixed 0.5 | 0.8421 | 0.8389 | 0.9000 | 0.7778 | 0.7889 | 0.7321 | 0.1829 |
| masked-VICReg + sensitivity-constrained univariate PSD BSI/gamma residual, fixed 0.5 | 0.8421 | 0.8389 | 0.9000 | 0.7778 | 0.8000 | 0.7363 | 0.1757 |
| no-SSL + nested univariate PSD BSI/gamma residual, fixed 0.5 | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.8111 | 0.7928 | 0.1811 |
| no-SSL + sensitivity-constrained univariate PSD BSI/gamma residual, fixed 0.5 | 0.7368 | 0.7278 | 0.9000 | 0.5556 | 0.7444 | 0.7462 | 0.1935 |
| masked-VICReg + nested fusion search, fixed 0.5 | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.7667 | 0.6960 | 0.1815 |
| masked-VICReg + BSI-only nested fusion search, fixed 0.5 | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.7778 | 0.7803 | 0.1807 |
| masked-VICReg + WPLI-only nested fusion search, fixed 0.5 | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.6444 | 0.5857 | 0.2174 |
| masked-VICReg + BSI/WPLI asymmetry nested fusion search, fixed 0.5 | 0.7895 | 0.7833 | 0.9000 | 0.6667 | 0.8111 | 0.7896 | 0.1701 |

Seed0 CNN-hybrid pilot:

| Model | Acc | Bal Acc | Sens | Spec | ROC AUC | PR AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-SSL CNN seed0 | 0.6316 | 0.6222 | 0.8000 | 0.4444 | 0.6667 | 0.6528 | 0.2760 |
| original Barlow SSL-CNN seed0 | 0.7368 | 0.7222 | 1.0000 | 0.4444 | 0.7444 | 0.6912 | 0.2007 |
| Barlow SSL-CNN + ROI summary branch seed0 | 0.6316 | 0.6278 | 0.7000 | 0.5556 | 0.7111 | 0.8128 | 0.2642 |

## Interpretation

The ROI features contain useful ranking information: masked-VICReg + nested ROI calibration improved ROC AUC, PR AUC, and Brier against both masked-VICReg and the no-SSL seedmean baseline. However, fixed and inner-OOF thresholded classifications both reduced accuracy and balanced accuracy. The seed0 CNN-hybrid ROI branch was also negative relative to the original Barlow seed0 result.

The best current candidate is masked-VICReg + constrained low-weight ROI residual with a fixed 0.5 threshold. It preserves no-SSL seedmean10 accuracy, balanced accuracy, sensitivity, and specificity while improving ROC AUC, PR AUC, and Brier. This is still not a significant improvement over CNN:

| Metric | no-SSL | Candidate | Delta | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.8421 | 0.8421 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| balanced_accuracy | 0.8333 | 0.8333 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| roc_auc | 0.8111 | 0.8333 | 0.0222 | [-0.1190, 0.1667] | 0.8494 |
| pr_auc | 0.7824 | 0.8489 | 0.0664 | [-0.1010, 0.2188] | 0.7345 |
| brier_score | 0.1714 | 0.1599 | -0.0115 | [-0.0645, 0.0295] | 0.7301 |

This is not yet a valid final improvement over CNN. The strongest evidence is that ROI/BSI/WPLI graph features should be used as a constrained residual/ranking auxiliary signal, not as a fully trainable high-dimensional supervised branch on 19 patients.

Adding qEEG ratio and relative-power features was negative for the SSL candidate under the same constrained low-weight fixed-threshold protocol. Lesion-aligned directional features were also negative as a residual-calibration input: they improved some no-SSL ranking/calibration values but reduced specificity and accuracy by adding an extra false positive. WPLI graph theory features are biologically plausible but negative in this nested residual protocol: they preserved classification but worsened masked-VICReg ROC/PR/Brier. PSD spectral-shape features were also negative, reducing masked-VICReg classification and ranking. For now, the reproducible candidate should exclude `relative`, `ratio`, lesion-aligned directional terms, graph theory terms, and spectral-shape terms:

```powershell
python -B scripts/26_calibrate_eeg_summary_residual.py --config configs/paths.baseline_rerun_20260529.yaml --base-predictions runs/baseline_rerun_20260529/results/predictions/dl_loso_predictions_feature_ssl_all-patient_psd-fc-wpli_gated_cnn_masked-vicreg_finetune_seedensemble10_pre50_temp0_2_noise0_02_f7c9f5f4e5f4.csv --base-name masked_vicreg_seedmean10 --selection-objective aucpr --candidate-weights 0 0.05 0.10 0.15 0.20 0.25 0.30 --preserve-base-classification --exclude-feature-regex "relative|ratio|ipsilesional|contralesional|signed_asymmetry|global_strength|global_efficiency|weighted_clustering|spectral_entropy|spectral_centroid|spectral_spread|spectral_edge" --output-tag eegsummary_roi_noratio_nolesion_nowpligraph_nospectralshape_constrained_loww_fixed05
```

## Fusion Diagnostic

The stronger fusion modes were tested because the linear residual lowered repeated false-positive scores without crossing the 0.5 decision boundary. The diagnostic result is negative for the main objective:

- no-SSL + no-ratio ROI residual lowered sub09 from `0.9319` to `0.7550`, but sub05 and sub14 remained high-confidence false positives.
- masked-VICReg + no-ratio ROI residual lowered sub09 from `0.8952` to `0.6956`, but sub05 and sub14 remained false positives.
- A diagnostic oracle grid over `linear/geometric/veto`, weights `0.00-1.00`, and thresholds `0.25-0.75` could at best trade one false positive for one false negative. For masked-VICReg, veto fusion corrected sub09 but made sub20 false negative, leaving accuracy at `0.8421` and only moving balanced accuracy from `0.8333` to `0.8389`.
- Nested fusion selection reproduced the same failure mode more conservatively: fixed-threshold accuracy fell to `0.7895` because one positive patient was lost.

This means the current ROI summary features are useful for calibration and ranking, but they are not sufficient as a hard false-positive veto on this 19-patient cohort.

## Lesion-Aligned Diagnostic

The feature files already contain `hemisphere_aligned=True`, produced by flipping channels to a common affected-hand convention before PSD/WPLI extraction. Therefore, after alignment, left-labeled channels can be interpreted as ipsilesional/affected-side channels and right-labeled channels as contralesional channels. This made it possible to add direction-aware features without changing the PSD/WPLI model inputs.

The diagnostic was negative for classification. Directional features assigned a high summary score to `sub08`, turning it into an additional false positive in both no-SSL and masked-VICReg residual calibration. For masked-VICReg, the ROI + lesion-aligned residual errors were `sub05`, `sub08`, `sub09`, and `sub14`; the earlier no-lesion ROI residual had only `sub05`, `sub09`, and `sub14`. This suggests the direction-aware features are plausible explanatory covariates but too unstable for this small cohort's residual classifier.

## WPLI Graph Diagnostic

AnySearch found graph-theory support for EEG FC features in stroke and EEG network analysis. A recent chronic stroke EEG study reported reduced ipsilesional alpha/low-beta node strength and clustering coefficient, with low-beta power and delta network redistribution related to motor function. A graph-theory EEG review also identifies clustering, path length/global integration, and small-world properties as common EEG FC descriptors.

The implemented graph summary is intentionally low-capacity and interpretable: per state/band WPLI matrices are converted to node strength, weighted efficiency, weighted clustering, ipsilesional/contralesional strength, and signed strength asymmetry. In nested residual calibration, however, WPLI graph features were negative for the SSL candidate. They preserved the same 3 false positives but reduced masked-VICReg ranking and calibration: ROC AUC `0.7000`, PR AUC `0.6310`, Brier `0.1936`. They should remain as documented explanatory features, but not as part of the current best residual.

## PSD Spectral-Shape Diagnostic

AnySearch did not identify a strong stroke-specific entropy result, but EEG small-sample classification literature commonly uses handcrafted complexity and spectral-shape descriptors. A low-cost PSD-only version was added to avoid raw EEG entropy computation: spectral entropy, centroid, spread, and 95% spectral edge frequency, computed globally and by ROI.

The nested residual result was negative. For masked-VICReg, spectral-shape residuals reduced specificity from `0.6667` to `0.5556` and worsened ROC/PR/Brier. For no-SSL, classification was unchanged but ROC/PR/Brier were worse. These features remain available for exploratory error analysis, but they should be excluded from the current best reproducible candidate.

## Univariate Biomarker Diagnostic

An oracle scan suggested that a single feature family could reduce the repeated false-positive pattern, especially `psd_eo_temporal_beta_bsi` and `psd_eo_central_gamma_contralesional_mean`. Because oracle feature choice would leak the full patient set, a nested single-feature selector was added:

- each outer test patient is held out;
- inside the outer training set, inner OOF predictions select exactly one summary feature, one direction, and one residual weight;
- the outer test feature score is computed as a percentile rank against only the outer training patients;
- no model is allowed to see the outer test label when choosing the biomarker.

The balanced-accuracy objective produced a weak specificity-positive result for masked-VICReg: specificity increased from `0.6667` to `0.7778`, and balanced accuracy increased from `0.8333` to `0.8389`. However, sensitivity fell from `1.0000` to `0.9000`, accuracy stayed at `0.8421`, and Brier/ROC/PR all worsened versus the best low-weight ROI residual. The main selected features were:

| Feature | Direction | Selection Count |
|---|---:|---:|
| `psd_eo_central_gamma_contralesional_mean` | -1 | 10 |
| `psd_eo_temporal_beta_bsi` | -1 | 8 |
| `psd_eo_delta_bsi` | -1 | 1 |

The selected model corrected some false positives but introduced `sub22` as a false negative. The `aucpr` objective was unstable and should not be used: masked-VICReg accuracy fell to `0.4211`. This confirms that single-feature biomarkers are interpretable and useful for error analysis, but they still do not satisfy the main goal of improving all metrics over the no-SSL CNN.

A sensitivity-preserving constraint was then added to the nested selector (`--min-sensitivity-delta 0.0`). This improved masked-VICReg Brier relative to the unconstrained univariate run (`0.1757` vs `0.1829`) and made the selected feature more stable: `psd_eo_central_gamma_contralesional_mean` was selected in 17 of 19 outer folds. However, outer-fold sensitivity still dropped to `0.9000`, because inner-OOF sensitivity preservation did not fully generalize to the held-out patient. The same selected pattern was harmful for no-SSL. Therefore this constraint is useful for sensitivity-aware diagnostics, but it is not sufficient for a main result.

Top masked-VICReg ROI calibration features by mean absolute coefficient:

| Rank | Feature |
|---:|---|
| 1 | `psd_eo_parietal_gamma_mean` |
| 2 | `psd_eo_central_gamma_mean` |
| 3 | `psd_eo_temporal_beta_bsi` |
| 4 | `wpli_eo_beta_high_intra_asymmetry` |
| 5 | `psd_eo_gamma_mean` |
| 6 | `wpli_ec_alpha_intra_asymmetry` |
| 7 | `psd_ec_central_beta_bsi` |
| 8 | `psd_ec_beta_bsi` |

## Decision

Do not run 10 seeds for the Barlow + ROI summary CNN branch. The seed0 pilot is negative.

Next best direction: keep negative-free SSL-CNN as the base, but add a low-capacity, nested-selected residual/ranking head using ROI EEG features. The current evidence favors fixed clinical threshold 0.5 with a small residual weight grid (`0.00-0.30`) over inner-OOF threshold search, because threshold search degraded classification in the outer folds.
