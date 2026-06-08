# Citation Claim Coverage Audit

Overall status: **PASS**

This audit checks whether the main manuscript's key literature-dependent claims are supported by the selected reference set and whether high-risk interpretive phrases remain properly bounded.

## Summary

- Selected references: 25
- Cited references in manuscript: 25 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
- Invalid citation numbers: none
- Uncited selected references: none
- Claim requirements passing: 14/14
- Overclaim boundary warnings: 0

## Claim-Level Coverage

| Area | Status | Expected refs | Found refs | Support boundary |
| --- | --- | --- | --- | --- |
| Proportional recovery and PREP2 context | PASS | 1, 2, 3 | 1, 2, 3, 22 | Supports recovery-rule framing; does not validate the cohort-median residual threshold. |
| tACS stroke neuromodulation context | PASS | 22 | 1, 2, 3, 22 | Supports frequency-specific tACS context only; not a claim of tACS treatment efficacy in this dataset. |
| EEG biomarkers and connectivity prognosis | PASS | 7, 8, 9, 16 | 4, 5, 6, 7, 8, 9, 10, 16, 23, 24, 25 | Supports EEG biomarker plausibility; not an external validation of this model. |
| WPLI method rationale | PASS | 10 | 4, 5, 6, 7, 8, 9, 10, 16, 23, 24, 25 | Supports WPLI choice for phase synchronization in scalp EEG. |
| Stroke recovery ML and deep learning | PASS | 4, 5, 6, 23, 24 | 4, 5, 6, 7, 8, 9, 10, 16, 23, 24, 25 | Supports broader ML/DL prognosis context; not a direct comparator claim. |
| EEG deep learning feasibility | PASS | 25 | 4, 5, 6, 7, 8, 9, 10, 16, 23, 24, 25 | Supports feasibility of EEG-derived neural signatures, with treatment-response prognosis left unresolved. |
| Prediction-model reporting guidance | PASS | 13, 14, 15 | 13, 14, 15 | Supports TRIPOD/TRIPOD+AI/PROBAST reporting and risk-of-bias framing. |
| Explanation alignment with prior physiology | PASS | 7, 8, 9, 16 | 7, 8, 9, 16 | Supports biological plausibility, not causal interpretation of saliency or WPLI edges. |
| Clinical predictor discussion boundary | PASS | 1, 2, 3, 4, 5, 6, 23, 24 | 1, 2, 3, 4, 5, 6, 23, 24 | Supports clinical-predictor context while preserving the no-incremental-EEG-value boundary. |
| Methods WPLI justification | PASS | 10 | 10, 11, 12 | Supports WPLI method selection and volume-conduction/sample-bias rationale. |
| MNE-Python EEG processing and visualization | PASS | 11, 12 | 10, 11, 12 | Supports software provenance for EEG processing and topographic visualization. |
| Self-supervised Barlow Twins pretraining | PASS | 17 | 17 | Supports redundancy-reduction SSL method provenance. |
| Attribution methods | PASS | 18, 19 | 18, 19 | Supports attribution-method provenance; not a claim that saliency is causal. |
| Software stack | PASS | 11, 12, 20, 21 | 11, 12, 20, 21 | Supports named software packages used in the analysis workflow. |

## Anchor Evidence

### Proportional recovery and PREP2 context

- Status: PASS
- Paragraph anchor: Upper-limb recovery after stroke varies substantially between patients, even among individuals with similar baseline motor impairment. The proportional-recovery framework and related biomarker algorithms have shown th...

### tACS stroke neuromodulation context

- Status: PASS
- Paragraph anchor: Upper-limb recovery after stroke varies substantially between patients, even among individuals with similar baseline motor impairment. The proportional-recovery framework and related biomarker algorithms have shown th...

### EEG biomarkers and connectivity prognosis

- Status: PASS
- Paragraph anchor: Resting-state EEG is attractive for this purpose because it is non-invasive, low cost, and sensitive to post-stroke changes in oscillatory power and functional connectivity. Prior work has linked EEG biomarkers to mot...

### WPLI method rationale

- Status: PASS
- Paragraph anchor: Resting-state EEG is attractive for this purpose because it is non-invasive, low cost, and sensitive to post-stroke changes in oscillatory power and functional connectivity. Prior work has linked EEG biomarkers to mot...

### Stroke recovery ML and deep learning

- Status: PASS
- Paragraph anchor: Resting-state EEG is attractive for this purpose because it is non-invasive, low cost, and sensitive to post-stroke changes in oscillatory power and functional connectivity. Prior work has linked EEG biomarkers to mot...

### EEG deep learning feasibility

- Status: PASS
- Paragraph anchor: Resting-state EEG is attractive for this purpose because it is non-invasive, low cost, and sensitive to post-stroke changes in oscillatory power and functional connectivity. Prior work has linked EEG biomarkers to mot...

### Prediction-model reporting guidance

- Status: PASS
- Paragraph anchor: The objective of this study was to test whether residual-aware self-supervised multimodal EEG learning could support patient-level prediction of upper-limb proportional recovery in a pilot post-stroke cohort. We repor...

### Explanation alignment with prior physiology

- Status: PASS
- Paragraph anchor: These findings align with prior reports linking post-stroke motor recovery to oscillatory beta activity, resting-state EEG features, and motor-network functional connectivity [7-9,16]. However, the explanation analyse...

### Clinical predictor discussion boundary

- Status: PASS
- Paragraph anchor: The exploratory clinical analyses show that baseline clinical variables were highly predictive in this cohort. This is consistent with the broader recovery literature, where baseline impairment and related clinical pr...

### Methods WPLI justification

- Status: PASS
- Paragraph anchor: PSD features were computed after affected-side alignment using Welch's method with a Hann window, 0.5 Hz frequency resolution, 50% overlap, density scaling, and constant detrending. The fixed PSD grid contained 90 bin...

### MNE-Python EEG processing and visualization

- Status: PASS
- Paragraph anchor: PSD features were computed after affected-side alignment using Welch's method with a Hann window, 0.5 Hz frequency resolution, 50% overlap, density scaling, and constant detrending. The fixed PSD grid contained 90 bin...

### Self-supervised Barlow Twins pretraining

- Status: PASS
- Paragraph anchor: The final model was a multimodal CNN with separate branches for PSD and WPLI inputs from eyes-open and eyes-closed states. Branch embeddings were fused through a gated representation layer into a 32-dimensional embedd...

### Attribution methods

- Status: PASS
- Paragraph anchor: Model explanation used SmoothGrad-smoothed integrated gradients for feature attribution [18,19]. Additional support analyses included branch/state occlusion, topographic projection of channel-frequency attributions, W...

### Software stack

- Status: PASS
- Paragraph anchor: Analyses used Python, PyTorch, scikit-learn, MNE-Python, and project-specific scripts [11,12,20,21]. Figure generation for this manuscript used `scripts/45_make_nature_manuscript_figures.py`. The generated figure mani...

## Overclaim Boundary Scan

| Phrase | Status | Sentence |
| --- | --- | --- |
| clinical use | PASS | These findings are exploratory and require larger prospective external validation before clinical use. |
| external validation | PASS | These findings are exploratory and require larger prospective external validation before clinical use. |
| superiority | PASS | Paired bootstrap comparisons quantified favorable numerical differences but did not support definitive superiority in this small cohort. |
| causal | PASS | They should be treated as hypothesis-generating biomarkers rather than causal evidence for a specific channel, frequency, or network edge. |
| external validation | PASS | By contrast, the independent benefit of self-supervised pretraining remains unresolved and should be tested in larger EEG pools with external validation. |
| incremental value | PASS | The current data do not establish that EEG adds robust incremental value over clinical variables alone. |
| external validation | PASS | Instead, the EEG model should be viewed as a candidate neurophysiological prognostic marker that requires larger prospective testing, multimodal clinical integration, and extern... |
| external validation | PASS | The supervised sample size was small, with 19 labeled patients and no external validation cohort. |
| incremental value | PASS | Clinical-only models performed strongly, so claims of EEG incremental value are not justified by the current data. |
| clinical use | PASS | A model reporting card in Supplementary Table 11 summarizes intended use, validation safeguards, reproducibility actions, and unsupported clinical uses. |
| clinical deployment | PASS | These results support further prospective validation rather than immediate clinical deployment. |
| causal | PASS | Attribution analyses were performed on the trained model outputs and were interpreted as model-dependent associations rather than causal neurophysiological mechanisms. |

## Interpretation

- The current citation layer supports the major background, method, reporting-guideline, software, and interpretation claims checked here.
- The audit does not replace target-journal reference-style formatting or author approval of the reference list.
- If new acquisition, preprocessing, intervention, or clinical deployment claims are added later, this audit should be rerun and the selected references should be updated.
