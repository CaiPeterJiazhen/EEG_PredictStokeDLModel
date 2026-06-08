# Residual-aware EEG learning for post-stroke proportional recovery

## Abstract

### Objective

Accurate prediction of upper-limb recovery after stroke could support earlier rehabilitation stratification, but small labeled cohorts limit EEG-based prognostic models. This study tested residual-aware self-supervised EEG learning for patient-level proportional-recovery prediction after stroke.

### Approach

We developed a multimodal SSL-CNN integrating baseline resting-state power spectral density and weighted phase-lag index connectivity from eyes-open and eyes-closed EEG before transcranial alternating current stimulation. The supervised cohort included 19 patients with baseline and post-treatment Fugl-Meyer Assessment of the upper extremity scores. Labels were residuals between expected and observed motor improvement. Evaluation used patient-level leave-one-subject-out cross-validation, ten seeds, bootstrap intervals, permutation testing, paired comparisons, ablations, and model explanation.

### Main results

The residual-aware SSL-CNN achieved accuracy of 84.2%, balanced accuracy of 83.3%, ROC-AUC of 0.844, PR-AUC of 0.836, and Brier score of 0.126. It showed numerically higher ROC-AUC and lower Brier score than a PSD+WPLI logistic-regression EEG baseline. Versus a matched no-SSL CNN, hard-label accuracy was unchanged, while ranking and calibration metrics improved directionally. Paired differences were not statistically definitive. Ablations identified residual-aware auxiliary supervision as the most consistent training signal; self-supervised pretraining had no stable independent gain. Explanations localized state- and frequency-specific PSD patterns and beta-band connectivity.

### Significance

Residual-aware self-supervised EEG learning provides a compact framework for using continuous recovery information while retaining binary prognostic inference. The findings are exploratory and require external validation before clinical deployment.

## Introduction

Upper-limb recovery after stroke varies substantially between patients, even among individuals with similar baseline motor impairment. The proportional-recovery framework and related biomarker algorithms have shown that much of this variability can be summarized by the relation between baseline impairment, corticospinal tract integrity, and subsequent motor improvement [1-3]. tACS is an emerging non-invasive neuromodulation approach with frequency-specific effects on motor-network activity after stroke [22]. However, many rehabilitation studies lack scalable biomarkers that can be repeatedly acquired at the bedside and integrated into machine-learning prediction models.

Resting-state EEG is attractive for this purpose because it is non-invasive, low cost, and sensitive to post-stroke changes in oscillatory power and functional connectivity. Prior work has linked EEG biomarkers to motor impairment and rehabilitation response, including beta-band oscillatory activity, resting-state connectivity, and EEG features used in brain-computer-interface rehabilitation settings [7-9,16]. WPLI reduces zero-lag coupling effects that can inflate connectivity estimates in electrophysiological recordings, making it a useful connectivity feature for scalp EEG prediction studies [10]. Recent stroke-prognosis studies further suggest that machine learning, deep learning, and explainable multimodal models can extract clinically meaningful information from neuroimaging, clinical variables, and EEG features [4-6,23,24]. EEG deep-learning models have also been tested for stroke-severity assessment, supporting the broader feasibility of EEG-derived neural signatures while leaving treatment-response prognosis unresolved [25].

Despite this promise, EEG prognosis studies face three recurring methodological risks. First, labeled stroke cohorts are often small, increasing the danger of overfitting and unstable estimates. Second, segment-level EEG samples can create artificial sample sizes if model evaluation is not performed at the patient level. Third, binary labels derived from proportional recovery can discard continuous information contained in the residual between expected and observed motor gain. We therefore designed a residual-aware self-supervised CNN that pretrains a patient-level representation using unlabeled EEG and then supervises the network with both the binary recovery label and auxiliary residual-ranking information.

The objective of this study was to test whether residual-aware self-supervised multimodal EEG learning could support patient-level prediction of upper-limb proportional recovery in a pilot post-stroke cohort. We report the model according to prediction-model guidance, including subject-level cross-validation, calibration-oriented scores, bootstrap uncertainty, permutation testing, ablation experiments, and model explanation [13-15].

## Results

### Cohort and outcome definition

The final supervised cohort comprised 19 patients with baseline EEG and complete baseline and follow-up motor assessments. Mean age was 64.7 years (SD 6.5), 11 patients were female, and 11 had left-sided affected upper limbs. Baseline FMA was 40.5 (SD 23.8), follow-up FMA was 45.5 (SD 23.2), and observed FMA improvement was 5.0 points (SD 4.1). The residual between predicted and observed FMA improvement had a median of 1.5 points. Ten patients were assigned to the proportional-recovery group and nine to the poor-recovery group. Descriptive cohort statistics are provided in Table 1.

The all-patient EEG pool contained 28 patients, of whom 19 were included in supervised leave-one-subject-out (LOSO) evaluation and nine were available only for unsupervised/self-supervised or descriptive use. Follow-up FMA and Modified Barthel Index (MBI) values were available for 20 of the 28 EEG-indexed patients.

### Residual-aware SSL-CNN performance

The final residual-aware SSL-CNN achieved accuracy of 0.842, balanced accuracy of 0.833, sensitivity of 1.000, specificity of 0.667, ROC-AUC of 0.844, PR-AUC of 0.836, and Brier score of 0.126 in patient-level LOSO evaluation (Table 2; Figure 2). The no-SSL CNN reference achieved the same hard-label accuracy and balanced accuracy, but lower ROC-AUC (0.811), PR-AUC (0.808), and a higher Brier score (0.177). A conventional PSD+WPLI logistic-regression EEG baseline achieved accuracy of 0.737, balanced accuracy of 0.733, ROC-AUC of 0.711, PR-AUC of 0.775, and Brier score of 0.208.

Paired bootstrap comparisons quantified favorable numerical differences but did not support definitive superiority in this small cohort. Relative to the no-SSL CNN, the residual-aware SSL-CNN had identical accuracy and balanced accuracy. ROC-AUC was higher by 0.033 (95% bootstrap interval -0.144 to 0.214; two-sided p = 0.788). PR-AUC was higher by 0.028 (-0.170 to 0.224; p = 0.824), and Brier score was lower by 0.051 (-0.121 to 0.004; p = 0.078). Relative to the logistic-regression EEG baseline, accuracy was higher by 0.105 (-0.105 to 0.316; p = 0.423). ROC-AUC was higher by 0.133 (-0.216 to 0.476; p = 0.458), and Brier score was lower by 0.082 (-0.189 to 0.037; p = 0.169). McNemar testing of hard predictions did not show a significant difference between the residual-aware SSL-CNN and the logistic-regression EEG baseline (discordant counts 1 versus 3; p = 0.625). There were no discordant hard predictions between the residual-aware and no-SSL CNN models. A performance precision audit in Supplementary Table 9 and Supplementary Figure 3 further shows that one changed hard prediction would move accuracy by 5.3 percentage points. Final-model bootstrap intervals remained wide for accuracy, ROC-AUC, PR-AUC, and Brier score.

Permutation tests using subject-level labels supported above-chance discrimination for the final model, with p = 0.004 for accuracy, p = 0.003 for balanced accuracy, p = 0.005 for ROC-AUC, and p = 0.015 for PR-AUC. Brier score was interpreted descriptively and through paired bootstrap comparisons because lower values indicate better performance and the stored permutation table used a common upper-tail direction.

### Exploratory clinical and incremental analyses

Baseline clinical variables alone were highly informative in this cohort. A clinical-only logistic model using baseline age, sex, disease duration, affected side, FMA, and MBI achieved ROC-AUC of 0.911, PR-AUC of 0.899, and Brier score of 0.105. Adding EEG features to clinical variables did not provide a stable incremental gain over the best clinical-only model in paired bootstrap analyses (Supplementary Table 4; Supplementary Figure 4). These analyses are reported as exploratory support analyses because the primary modelling question was EEG-based prediction and the sample size was insufficient for a definitive comparison of clinical, EEG-only, and multimodal clinical-plus-EEG models.

### Ablation and robustness analyses

Ablation experiments separated the contribution of residual-aware auxiliary supervision from the contribution of self-supervised pretraining. The matched no-SSL CNN with residual-aware auxiliary heads achieved seed-mean accuracy of 0.895, balanced accuracy of 0.889, ROC-AUC of 0.922, PR-AUC of 0.927, and Brier score of 0.110 in the current ablation table. The final patient-level Barlow SSL plus residual-aware model achieved accuracy of 0.842, balanced accuracy of 0.833, ROC-AUC of 0.844, PR-AUC of 0.836, and Brier score of 0.126. The patient-level Barlow SSL model without residual-aware heads was less stable, with ensemble ROC-AUC of 0.700 and PR-AUC of 0.631. Taken together, these ablations support the use of continuous residual information during supervised fine-tuning. They do not prove that Barlow-style self-supervised pretraining adds a stable performance gain in this small cohort (Table 3; Figure 3).

Feature-family ablations showed that PSD-only models retained useful discrimination (ROC-AUC 0.811), whereas WPLI-only and restricted connectivity feature sets were less accurate but still carried ranking information in some configurations. Eyes-closed features were more informative than eyes-open-only features in the current feature-selection analysis. Because these ablations used small LOSO folds and multiple feature subsets, they are interpreted as mechanistic support rather than as independent confirmatory tests.

### Model explanation and EEG biomarker localization

Model explanation combined SmoothGrad-smoothed integrated gradients, branch/state occlusion, EEG topographic maps, WPLI connectome summaries, and network-level attribution summaries (Figure 4; Supplementary Figure 2). The most prominent PSD contributions involved eyes-open low-frequency fronto-central channels and higher-frequency peri-central or temporal channels. Connectivity explanations emphasized eyes-closed beta-band WPLI edges, including frontal, central, and temporal node pairs. Several top-ranked WPLI edges showed associations between attribution magnitude and residual-related distance measures before multiple-comparison correction, while false-discovery-adjusted support was weaker.

These findings align with prior reports linking post-stroke motor recovery to oscillatory beta activity, resting-state EEG features, and motor-network functional connectivity [7-9,16]. However, the explanation analyses are model-dependent and associative. They should be treated as hypothesis-generating biomarkers rather than causal evidence for a specific channel, frequency, or network edge.

## Discussion

This pilot study tested a residual-aware self-supervised multimodal EEG model for predicting upper-limb proportional recovery after stroke. The final model achieved promising patient-level discrimination in LOSO evaluation and showed numerically better ranking and calibration-oriented scores than an updated PSD+WPLI logistic-regression EEG baseline. Compared with an otherwise matched no-SSL CNN, residual-aware SSL did not improve hard-label accuracy but moved ROC-AUC, PR-AUC, and Brier score in the favorable direction. The paired comparisons did not reach definitive significance, which is expected given the 19-patient supervised cohort and should temper interpretation.

The main methodological contribution is the use of residual-aware supervision. Binary proportional-recovery labels simplify clinical interpretation, but they can discard the magnitude and direction of deviation from expected recovery. Training auxiliary residual and ranking objectives allowed the network to use continuous recovery information while preserving a binary classification endpoint at inference. The ablation results indicate that this residual-aware component carried the most consistent training signal in the current dataset. By contrast, the independent benefit of self-supervised pretraining remains unresolved and should be tested in larger EEG pools with external validation.

Self-supervised patient-level representation learning was used to reduce reliance on labeled outcomes and exploit unlabeled patient EEG. Unlike segment-level augmentation, the evaluation unit remained the patient throughout cross-validation. This distinction is critical in EEG prognosis, where leakage from subject-specific EEG structure can substantially inflate performance if train-test partitioning is performed at the segment level. Our statistical workflow therefore used subject-level LOSO predictions for all reported metrics, bootstrap intervals, permutation tests, and paired comparisons.

The exploratory clinical analyses show that baseline clinical variables were highly predictive in this cohort. This is consistent with the broader recovery literature, where baseline impairment and related clinical predictors are central to recovery prognosis [1-3]. It is also consistent with prior machine-learning studies showing that clinical and multimodal variables can predict upper-limb impairment or recovery after stroke therapy [4-6,23,24]. The current data do not establish that EEG adds robust incremental value over clinical variables alone. Instead, the EEG model should be viewed as a candidate neurophysiological prognostic marker that requires larger prospective testing, multimodal clinical integration, and external validation.

The explainability analyses suggest that the network relied on physiologically plausible EEG features, including state-dependent PSD patterns and beta-band connectivity. This convergence with prior EEG and connectivity studies increases biological plausibility, but it does not remove the need for validation. Saliency and occlusion analyses can be sensitive to model architecture, preprocessing, and correlated inputs. Their strongest value here is to define testable neurophysiological hypotheses for future cohorts.

This study has several limitations. The supervised sample size was small, with 19 labeled patients and no external validation cohort. The clinical endpoint and residual threshold were derived within the available cohort, and the final label definition should be pre-specified in future work. Clinical-only models performed strongly, so claims of EEG incremental value are not justified by the current data. Explainability analyses were exploratory and not corrected strongly enough to support definitive channel-level or edge-level biomarkers. Finally, the self-supervised learning pool included additional EEG-indexed patients, but the scope and timing of those recordings should be fully reported in the final submission. These limitations, along with participant, predictor, outcome, analysis, missing-data, explainability, and reproducibility risks, are summarized in a conservative PROBAST/TRIPOD+AI-oriented audit in Supplementary Table 8. A claim-strength audit in Supplementary Table 10 maps central manuscript statements to supporting evidence, acceptable wording, and overclaims to avoid. A model reporting card in Supplementary Table 11 summarizes intended use, validation safeguards, reproducibility actions, and unsupported clinical uses.

In summary, residual-aware self-supervised EEG learning provided a compact and interpretable framework for patient-level prediction of proportional upper-limb recovery after stroke. The model achieved promising discrimination and Brier-score performance in a pilot LOSO cohort while preserving conservative subject-level validation. These results support further prospective validation rather than immediate clinical deployment.

## Methods

### Study cohort

The supervised cohort included 19 stroke patients with baseline resting-state EEG and complete baseline and post-treatment FMA-UE measurements. Project records define a common tACS protocol for all patients. Stimulation targeted the primary motor cortex contralateral to the affected hand, with C3 stimulation for right-hand impairment and C4 stimulation for left-hand impairment. The protocol used a 20 Hz stimulation frequency, 1000 microampere intensity, 20 min per session, once daily for 14 consecutive days over 2 weeks. The post-treatment outcome assessment was recorded immediately after the final tACS session. The M1 clinical source workbook contained 29 patient records; 28 were indexed in the current EEG directory and 19 formed the final supervised labelled cohort. Nine EEG-indexed patients were retained only for descriptive or self-supervised analyses, and one clinical workbook patient was not indexed in the current EEG data directory. Source-workbook notes for non-supervised or non-indexed entries included missing post-stimulation EEG/follow-up data, non-treatment or discharge, compliance/cognitive-communication difficulty, EEG-cap heat discomfort, post-session discomfort, and one post-session discomfort note followed by next-day hypertension. These notes are treated as participant-flow source notes rather than a complete adverse-event monitoring dataset.


### Outcome definition

The target endpoint was proportional recovery status based on the residual between expected and observed FMA-UE improvement. Expected improvement was computed using the proportional-recovery rule:

`Predicted Delta FMA = 0.7 x (66 - baseline FMA-UE)`.

Observed improvement was computed as:

`Observed Delta FMA = post-treatment FMA-UE - baseline FMA-UE`.

`Residual = predicted Delta FMA - observed Delta FMA`.

Patients with residuals at or below the supervised-cohort median residual threshold of 1.5 points were assigned to the proportional-recovery group, and patients with residuals above the threshold were assigned to the poor-recovery group. The resulting label distribution was 10 proportional-recovery and 9 poor-recovery patients. This median-derived threshold is reported as a cohort-specific modelling endpoint rather than as an externally validated clinical cut-off.

### EEG preprocessing and feature extraction

Resting-state EEG features were extracted separately for eyes-open and eyes-closed recordings acquired before tACS. Analyses began from project-provided preprocessed EEGLAB `.set/.fdt` files. The available analysis materials did not include the original acquisition log or preprocessing protocol. Acquisition hardware, online reference, filtering, re-reference, artifact rejection, and bad-channel handling before export therefore remain author-confirmed fields before submission. Within the manuscript feature pipeline, no additional temporal filtering, artifact rejection, channel interpolation, or bad-channel removal was performed after loading the provided preprocessed files. Feature scripts instead validated finite 62-channel continuous arrays, fixed channel order, and minimum window lengths before PSD and connectivity extraction. The source project records indicate that the preprocessed EEGLAB files contained continuous data with one trial per file. These files had a sampling rate of 250 Hz and 62 retained channels after removal of M1 and M2. Across the 38 supervised baseline EO/EC files, recording duration averaged 188.4 s and ranged from 101.0 to 247.8 s. File-name state rules assigned baseline `*1.set` files to eyes-open recordings and baseline `*2.set` files to eyes-closed recordings. The fixed 62-channel order was checked against the project channel map.

PSD features were computed after affected-side alignment using Welch's method with a Hann window, 0.5 Hz frequency resolution, 50% overlap, density scaling, and constant detrending. The fixed PSD grid contained 90 bins from 0.5 to 45 Hz for each of the 62 channels in each state. Functional connectivity was computed after the same affected-side alignment using short-time Fourier transforms with 2 s Hann windows and 50% overlap. WPLI and imaginary coherence were computed for the deterministic upper-triangle channel-pair list. This produced 1,891 edges by six frequency bands per state: delta (1-3 Hz), theta (4-7 Hz), alpha (8-13 Hz), beta-low (13-18 Hz), beta-medium (18-21 Hz), and beta-high (21-30 Hz). WPLI was selected because it reduces the influence of volume conduction and sample-size bias in phase-synchronization estimates [10]. EEG preprocessing and topographic visualization used MNE-Python where applicable [11,12].

Feature matrices were aligned to a common affected-hand convention before PSD and connectivity computation. Patients with right-hand impairment were left unchanged, whereas patients with left-hand impairment were mirrored so that the representation corresponded to right-hand impairment, C3 stimulation, and left-hemisphere stimulation-side convention. PSD and WPLI branches were standardized within training folds, and all feature selection for baseline models was performed inside LOSO training folds to avoid test-fold leakage.

### Model architecture

The final model was a multimodal CNN with separate branches for PSD and WPLI inputs from eyes-open and eyes-closed states. Branch embeddings were fused through a gated representation layer into a 32-dimensional embedding. Patient-level Barlow Twins self-supervised learning was used to learn redundancy-reduced EEG representations from patient EEG without using supervised labels [17]. The supervised fine-tuning stage combined a binary classification loss with auxiliary residual-aware losses, including signed residual distance and pairwise recovery ranking objectives. Signed distance was defined as `1.5 - residual`, so positive values indicated recovery closer to or beyond the median proportional-recovery endpoint and negative values indicated poor recovery. Stochastic weight averaging was used during final training where specified by the pipeline. At inference, predictions were generated from the classification head only; residual and ranking heads regularized training but were not used to set a post hoc test threshold.

Baseline models included conventional PSD+WPLI machine-learning classifiers, a no-SSL CNN with the same core architecture, and ablated CNN variants without residual-aware auxiliary heads. Exploratory clinical models used baseline-only variables: age, sex, disease duration, affected side, baseline FMA, and baseline MBI. Post-treatment outcomes, observed improvement, predicted improvement, residuals, and labels were excluded from all model inputs.

### Cross-validation and statistical analysis

The primary validation scheme was patient-level LOSO cross-validation. For the CNN models, training was repeated across ten random seeds, and manuscript-facing predictions were summarized as the pre-specified seed mean or seed ensemble output listed in the locked result tables. No segment-level row and no seed-level row was treated as an independent patient.

Primary metrics were accuracy, balanced accuracy, sensitivity, specificity, ROC-AUC, PR-AUC, and Brier score. Uncertainty estimates used 5,000 subject-level bootstrap resamples where available. Paired model comparisons also resampled subjects. McNemar tests were applied to paired hard predictions. Permutation tests used subject-level label permutations and 5,000 permutations. Reporting follows TRIPOD/TRIPOD+AI principles and acknowledges PROBAST-relevant bias risks in sample size, outcome definition, and validation design [13-15]. A conservative PROBAST/TRIPOD+AI-oriented risk audit was completed to make bias, applicability, and remaining author-confirmation requirements explicit before submission.

### Model explanation

Model explanation used SmoothGrad-smoothed integrated gradients for feature attribution [18,19]. Additional support analyses included branch/state occlusion, topographic projection of channel-frequency attributions, WPLI edge-level summaries, and network-level aggregation. Attribution analyses were performed on the trained model outputs and were interpreted as model-dependent associations rather than causal neurophysiological mechanisms.

### Software

Analyses used Python, PyTorch, scikit-learn, MNE-Python, and project-specific scripts [11,12,20,21]. Figure generation for this manuscript used `scripts/45_make_nature_manuscript_figures.py`. The generated figure manifest is stored at `results/figures/nature/figure_manifest.csv`.

## Data availability

De-identified derived data supporting the manuscript will be deposited in a citable repository before final journal submission or publication. The deposit will include subject-level analysis tables, locked LOSO model predictions, bootstrap, permutation and paired-comparison outputs, participant-flow source notes, figure-source summaries, and validation and artifact-quality audit tables. It will also include the conservative PROBAST/TRIPOD+AI risk audit, performance precision audit, claim-strength audit, and numeric-claim source trace audit. Additional deposited materials will include the AI model reporting card, source-workbook author metadata audit, author-field replacement map, patient-record PDF text-layer audit, and source-data workbook. The current workbook contains 36 worksheets, including a 446-field data dictionary for 37 derived CSV or figure-manifest files. Raw EEG recordings, minimally processed EEG files, identifiable clinical source records, and any directly linkable participant-level source files are not publicly released in this draft. These materials contain human-participant data and may be subject to institutional review board, consent, privacy, and data-use restrictions. Access to restricted raw or minimally processed data should be reviewed by the responsible institution. Qualified researchers may request access from the corresponding author or institutional data-access committee after ethics approval and completion of a data-use agreement, subject to the restrictions in the original consent and ethics approval.


## Code availability

The modelling, statistical validation, source-data assembly, figure-generation, manuscript-generation, audit, model-card, and package-assembly scripts will be deposited with the derived-data package or in a linked public code repository before final journal submission or publication. The final code archive should include the exact commit hash or version tag, environment metadata, random seeds, and locked prediction CSVs. It should also include scripts that reproduce the manuscript tables, figures, source-data workbook, and audit reports from the deposited derived data.

## Tables

Table 1. Cohort characteristics. Source file: `results/tables/table1_cohort_characteristics.csv`.

Table 2. Main model performance. Source file: `results/tables/table2_main_model_performance.csv`.

Table 3. Ablation analysis. Source file: `results/tables/table3_ablation.csv`.

Table 4. Explainability biomarkers. Source file: `results/tables/table4_explainability_biomarkers.csv`.

## Figure legends

Figure 1. Participant flow, study design, and residual-aware SSL-CNN architecture. The M1 clinical source workbook contained 29 patient records, of which 28 were indexed in the current EEG directory and 19 formed the final labelled LOSO cohort. The participant-flow panel also shows the 9 EEG-indexed patients retained outside supervised outcome modelling, the 1 clinical-source record not indexed in the current EEG directory, and the 10/9 proportional-recovery versus poor-recovery endpoint split. The model integrated eyes-open and eyes-closed PSD and WPLI branches, patient-level Barlow self-supervised pretraining, residual-aware auxiliary supervision, and classification-head inference. Figure file: `results/figures/nature/figure1_study_design_model.png`.

Figure 2. Primary performance, uncertainty, and calibration. The residual-aware SSL-CNN showed numerically higher ROC-AUC and PR-AUC and a lower Brier score relative to the updated PSD+WPLI logistic-regression EEG baseline. It also showed favorable score metrics relative to the no-SSL CNN despite identical hard-label accuracy. All uncertainty estimates used subject-level resampling. Figure file: `results/figures/nature/figure2_performance_calibration.png`.

Figure 3. Robustness, ablation, and threshold sensitivity. Ablation analyses compared traditional PSD+WPLI machine learning, a no-SSL CNN, patient-level Barlow SSL without residual-aware heads, no-SSL residual-aware CNN, and the combined residual-aware SSL-CNN. The ablations support residual-aware auxiliary supervision as a useful training signal but do not establish an independent performance gain from self-supervised pretraining in this cohort. Feature-family analyses summarize state, modality, band, and connectivity contributions. Figure file: `results/figures/nature/figure3_robustness_ablation.png`.

Figure 4. Explainability and neurophysiological interpretation. SmoothGrad-smoothed integrated gradients, occlusion, topographic maps, and WPLI connectome summaries localized model contributions to state-dependent PSD patterns and beta-band functional connectivity. The results are interpreted as model-dependent and hypothesis-generating. Figure file: `results/figures/nature/figure4_explainability_neurophysiology.png`.

Supplementary Figure 1. Repeated-error subject analysis. Post-hoc subject-level error summaries identify patients repeatedly misclassified across model variants or seeds. This analysis is exploratory and should be used to guide future cohort review rather than to revise labels post hoc. Figure file: `results/figures/nature/supplementary_error_subjects.png`.

Supplementary Figure 2. MNE-rendered WPLI connectivity attribution maps. The top 20 WPLI attribution edges were rendered for each EEG state and frequency band using the fixed 62-channel scalp layout. Red and blue edges indicate the sign of the mean attribution, and line width scales with mean absolute attribution. These maps are exploratory visual summaries of model explanation results and should not be interpreted as independently validated network biomarkers. Figure file: `results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_contact_sheet.png`.

Supplementary Figure 3. Performance precision and validation-boundary audit. Panel a shows final-model bootstrap intervals for primary performance metrics. Panel b shows paired bootstrap differences relative to the no-SSL CNN and logistic-regression EEG baseline. Zero indicates no difference, and Brier-score negative differences favour the final model. Panel c shows one-case movement in accuracy, sensitivity, and specificity for the 19-patient LOSO cohort. Figure file: `results/figures/nature/supplementary_performance_precision.png`.

Supplementary Figure 4. Exploratory clinical baseline and EEG incremental-value audit. Panel a compares selected clinical-only, EEG-plus-clinical, and EEG-only model metrics. Panel b shows paired bootstrap differences for EEG-plus-clinical candidates relative to the clinical-only logistic model; zero indicates no difference, and positive Brier-score differences favour the clinical-only model. Panel c summarizes directional point estimates across all exploratory EEG-only and EEG-plus-clinical candidates after transforming each metric so positive values favour the candidate. Panel d plots ROC-AUC benefit against Brier-score benefit relative to the clinical-only model. Figure file: `results/figures/nature/supplementary_clinical_incremental_value.png`.

## References

1. Prabhakaran S, Zarahn E, Riley C, Speizer A, Chong JY, Lazar RM, et al. Inter-individual variability in the capacity for motor recovery after ischemic stroke. Neurorehabil Neural Repair. 2008;22:64-71. doi:10.1177/1545968307305302
2. Byblow WD, Stinear CM, Barber PA, Petoe MA, Ackerley SJ. Proportional recovery after stroke depends on corticomotor integrity. Ann Neurol. 2015;78:848-859. doi:10.1002/ana.24472
3. Stinear CM, Byblow WD, Ackerley SJ, Smith M, Borges VM, Barber PA. PREP2: a biomarker-based algorithm for predicting upper limb function after stroke. Ann Clin Transl Neurol. 2017;4:811-820. doi:10.1002/acn3.488
4. Lin PJ, Zhai X, Li W, Li T, Cheng D, Li C, et al. A transferable deep learning prognosis model for predicting stroke patients' recovery in different rehabilitation trainings. IEEE J Biomed Health Inform. 2022;26:6003-6011. doi:10.1109/JBHI.2022.3205436
5. White A, Saranti M, d'Avila Garcez A, Hope TM, Price CJ, Bowman H. Predicting recovery following stroke: deep learning, multimodal data and feature selection using explainable AI. NeuroImage Clin. 2024;43:103638. doi:10.1016/j.nicl.2024.103638
6. Lassi M, Dalise S, Privitera L, Giannini N, Mancuso M, Azzollini V, et al. Enhancing upper limb motor recovery prediction after acute stroke using EEG and subacute data. APL Bioeng. 2026;10:016108. doi:10.1063/5.0287165
7. Mane R, Chew E, Phua KS, Ang KK, Robinson N, Vinod AP, et al. Prognostic and monitory EEG-biomarkers for BCI upper-limb stroke rehabilitation. IEEE Trans Neural Syst Rehabil Eng. 2019;27:1654-1664. doi:10.1109/TNSRE.2019.2924742
8. Saes M, Meskers CGM, Daffertshofer A, van Wegen EEH, Kwakkel G. Are early measured resting-state EEG parameters predictive for upper limb motor impairment six months poststroke? Clin Neurophysiol. 2021;132:56-62. doi:10.1016/j.clinph.2020.09.031
9. Tang CW, Hsiao FJ, Lee PL, Tsai YA, Hsu YF, Chen WT, et al. Beta-oscillations reflect recovery of the paretic upper limb in subacute stroke. Neurorehabil Neural Repair. 2020;34:450-462. doi:10.1177/1545968320913502
10. Vinck M, Oostenveld R, van Wingerden M, Battaglia F, Pennartz CMA. An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. NeuroImage. 2011;55:1548-1565. doi:10.1016/j.neuroimage.2011.01.055
11. Gramfort A. MEG and EEG data analysis with MNE-Python. Front Neurosci. 2013;7:267. doi:10.3389/fnins.2013.00267
12. Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, et al. MNE software for processing MEG and EEG data. NeuroImage. 2014;86:446-460. doi:10.1016/j.neuroimage.2013.10.027
13. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. Ann Intern Med. 2015;162:55-63. doi:10.7326/M14-0697
14. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378
15. Wolff RF, Moons KGM, Riley RD, Whiting PF, Westwood M, Collins GS, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. Ann Intern Med. 2019;170:51-58. doi:10.7326/M18-1376
16. Zhang Y, Gong G, Liu G, Xu S, Zeng J. Functional connectivity between non-motor and motor networks predicts motor recovery changes after stroke. Sci Rep. 2025;15:41448. doi:10.1038/s41598-025-19860-4
17. Zbontar J, Jing L, Misra I, LeCun Y, Deny S. Barlow Twins: self-supervised learning via redundancy reduction. Proceedings of the 38th International Conference on Machine Learning. 2021. https://proceedings.mlr.press/v139/zbontar21a.html
18. Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. Proceedings of the 34th International Conference on Machine Learning. 2017. https://proceedings.mlr.press/v70/sundararajan17a.html
19. Smilkov D, Thorat N, Kim B, Viegas F, Wattenberg M. SmoothGrad: removing noise by adding noise. arXiv. 2017. https://arxiv.org/abs/1706.03825
20. Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: an imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems. 2019. https://papers.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
21. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
22. Yuan K, Chen C, Lou WT, Khan A, Ti ECH, Lau CCY, et al. Differential effects of 10 and 20 Hz brain stimulation in chronic stroke: a tACS-fMRI study. IEEE Trans Neural Syst Rehabil Eng. 2022;30:455-464. doi:10.1109/TNSRE.2022.3153353
23. Tozlu C, Edwards D, Boes A, Labar D, Tsagaris KZ, Silverstein J, et al. Machine learning methods predict individual upper-limb motor impairment following therapy in chronic stroke. Neurorehabil Neural Repair. 2020;34:428-439. doi:10.1177/1545968320909796
24. AlArfaj AA, Hosni Mahmoud HA, Hafez AM. A deep learning model for stroke patients' motor function prediction. Appl Bionics Biomech. 2022;2022:1-9. doi:10.1155/2022/8645165
25. Singh S, Dawar D, Mehmood E, Pandian JD, Sahonta R, Singla S, et al. Determining diagnostic utility of EEG for assessing stroke severity using deep learning models. Biomed Eng Adv. 2024;7:100121. doi:10.1016/j.bea.2024.100121
