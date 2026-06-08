# TRIPOD+AI-Oriented Reporting Checklist

This checklist translates the current manuscript package into prediction-model reporting requirements. It is not a formal journal form, but it flags items that reviewers of a small AI prognosis study are likely to inspect.

| Reporting item | Current manuscript status | Evidence or required action |
|---|---|---|
| Title identifies prediction target and study design | Present | Title states residual-aware SSL EEG model and pilot LOSO design. |
| Abstract reports participants, predictors, outcome, validation, and main metrics | Present | `docs/manuscript_residual_aware_ssl_cnn_nature_polished.md` |
| Clinical setting and intervention | Partly present | tACS protocol added; recruitment dates, clinical site, and concurrent rehabilitation still needed. |
| Participants and eligibility | Partly present | n=19 supervised and n=28 EEG pool reported; inclusion/exclusion criteria still needed. |
| Outcome definition | Present | FMA-UE proportional-recovery residual and cohort median threshold 1.5 reported. |
| Predictor definition | Present for derived predictors | Baseline EO/EC PSD and WPLI described; `.set` sampling rate, channel count, and duration audited; acquisition hardware and preprocessing details before `.set` files still needed. |
| Sample size | Present | n=19 supervised, 10/9 class split, n=28 EEG pool. |
| Missing data handling | Present with caveat | Non-supervised and follow-up availability are described; de-identified participant-flow and source-note categories are provided in `results/tables/participant_flow_safety_source_notes.csv` and Supplementary Table 7. Formal adverse-event monitoring and protocol-level missing-data handling still require author confirmation. |
| Model architecture | Present | Multimodal gated CNN, patient-level Barlow SSL, residual/ranking auxiliary heads. |
| Data leakage safeguards | Present | Patient-level LOSO, fold-local scaling/feature selection, no seed/segment independence; machine-checkable audit in `docs/prediction_validation_integrity_audit.md`. |
| Performance measures | Present | Accuracy, balanced accuracy, ROC-AUC, PR-AUC, Brier, calibration summaries. |
| Uncertainty estimates | Present | Subject-level bootstrap intervals, paired comparisons, Supplementary Table 9 performance precision audit, and Supplementary Figure 3 precision-boundary visualization. |
| Internal validation | Present | LOSO with 10 seeds; no external validation. |
| Calibration | Present | Brier score and reliability/calibration metrics described. |
| Explainability | Present with limitations | SmoothGrad-IG, occlusion, topomaps, WPLI connectomes; hypothesis-generating framing. |
| Code availability | Draft present | Needs final archive DOI and commit hash. |
| Data availability | Draft present | Needs ethics, consent, repository DOI, and access route. |
| Bias and applicability risks | Present | Small cohort, no external validation, strong clinical-only exploratory baseline, median-derived label threshold; conservative PROBAST/TRIPOD+AI-oriented audit provided in `docs/probast_tripod_ai_risk_audit.md` and Supplementary Table 8, with performance precision boundaries in Supplementary Table 9 and Supplementary Figure 3. |
| Claim-to-evidence alignment | Present | Central manuscript claims are mapped to evidence strength, acceptable wording, and overclaims to avoid in `docs/claim_strength_audit.md` and Supplementary Table 10. |
| Model reporting card | Present | Intended use, validation safeguards, reproducibility actions, deployment boundaries, and unsupported uses are summarized in `docs/model_reporting_card.md` and Supplementary Table 11. |

## Priority Fixes Before Submission

1. Add ethics approval, recruitment, and consent details.
2. Add EEG acquisition and preprocessing details upstream of the current preprocessed EEGLAB files.
3. Decide whether to deposit derived PSD/WPLI features or only subject-level predictions and tables.
4. Select target journal and convert this checklist into its required reporting form if needed.
