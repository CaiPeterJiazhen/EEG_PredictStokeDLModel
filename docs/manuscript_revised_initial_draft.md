# Residual-aware self-supervised EEG learning predicts post-tACS proportional upper-limb recovery after stroke

## Abstract

Prediction of upper-limb recovery after stroke could support earlier stratification of neuromodulation and rehabilitation, but EEG-based prognostic modelling is constrained by small labelled cohorts, high-dimensional predictors and unstable validation. We developed a residual-aware self-supervised convolutional neural network (SSL-CNN) to predict whether patients would follow a proportional-recovery or poor-recovery trajectory after transcranial alternating current stimulation (tACS). The model used baseline eyes-open and eyes-closed resting-state EEG features, including power spectral density (PSD) and weighted phase-lag index (WPLI) connectivity. The supervised cohort comprised 19 stroke patients with complete baseline and post-treatment Fugl-Meyer Assessment of the upper extremity (FMA-UE) scores. Nine additional EEG-indexed patients who could not form complete supervised labels were retained for self-supervised representation learning, allowing baseline EEG data that would otherwise be unusable for binary outcome training to contribute to the encoder.

The final residual-aware SSL-CNN achieved a ten-seed mean accuracy of 0.837, minimum accuracy of 0.789 and accuracy standard deviation of 0.039 in patient-level leave-one-subject-out validation. Mean ROC-AUC was 0.860, mean PR-AUC was 0.858 and mean Brier score was 0.142. In the locked patient-level prediction used for the confusion matrix, the model correctly identified 10 of 10 proportional-recovery patients and 6 of 9 poor-recovery patients. Traditional EEG machine-learning baselines showed more limited hard-label discrimination, and an otherwise matched no-SSL CNN showed larger seed-to-seed variability. Ablation results supported residual-aware auxiliary supervision as the most consistent training signal, whereas self-supervised pretraining alone did not establish a stable independent gain. Model explanations localized predictive information to state-dependent PSD patterns and beta-band WPLI connectivity.

These results indicate that residual-aware EEG representation learning can use continuous recovery information while preserving a clinically interpretable binary endpoint, and can extract physiologically interpretable candidate EEG recovery features from baseline recordings. Because the supervised cohort was small and external validation is not yet available, model performance, probability calibration and candidate PSD/WPLI biomarkers require confirmation in larger prospective tACS cohorts.

## Introduction

Upper-limb impairment remains one of the most disabling consequences of stroke, and the trajectory of recovery differs substantially between patients. Some patients regain a large fraction of their initial motor deficit, whereas others show limited improvement despite apparently similar clinical starting points and comparable rehabilitation exposure. The proportional-recovery framework formalizes one part of this variability by relating baseline impairment to expected subsequent motor gain [1,2]. Biomarker-based algorithms such as PREP2 further show that recovery prediction can be strengthened when behavioural measures are combined with neurophysiological information [3]. Nevertheless, proportional recovery is not a complete explanation of post-stroke motor outcome. Patients may deviate from the expected trajectory, and prediction is especially challenging when the question is whether an individual patient will respond favourably to a specific neuromodulatory intervention.

Transcranial alternating current stimulation (tACS) is a non-invasive stimulation approach that can entrain or modulate oscillatory activity in motor networks. Frequency-specific stimulation has been studied in chronic stroke and related motor-network contexts [22]. In principle, tACS could be incorporated into individualized rehabilitation if patients likely to achieve meaningful upper-limb recovery could be identified before treatment. In practice, treatment-response prediction remains underdeveloped. Clinical scales are indispensable for outcome assessment, but they do not directly describe the oscillatory and network state of the injured motor system. A scalable pretreatment neurophysiological marker would therefore be useful for stratifying patients in future tACS trials and for generating testable hypotheses about neural mechanisms of recovery.

Resting-state EEG is well suited to this role. It is non-invasive, comparatively inexpensive and repeatable in rehabilitation settings. EEG power features capture frequency-specific activity, whereas functional connectivity measures capture interactions between scalp-recorded neural signals. In stroke, these two feature families are attractive because motor recovery is unlikely to be represented by a single electrode or a single frequency band. A patient may show altered local oscillatory activity, altered interregional coupling, or both. Prior stroke work has linked EEG biomarkers to upper-limb rehabilitation outcomes, resting-state impairment prediction and beta-band recovery-related activity [7-9]. Resting-state cortical EEG rhythms and network measures have also been associated with upper-limb motor function in chronic stroke [26], and functional connectivity between motor and non-motor networks has been associated with motor recovery changes after stroke [16]. WPLI is particularly useful in scalp EEG because it reduces the influence of zero-lag coupling, volume conduction and sample-size bias in phase-synchronization estimates [10]. Recent machine-learning and deep-learning studies further support the feasibility of individualized stroke recovery prediction using clinical, neuroimaging or EEG-derived information [4-6,23-25]. However, the specific problem of predicting post-tACS proportional upper-limb recovery from baseline EEG remains unresolved.

Three methodological limitations motivate the present study. First, traditional EEG machine-learning pipelines often flatten high-dimensional PSD and connectivity features into vectors and then rely on feature selection or regularization. This strategy can be useful, but it may discard the structure of channel-frequency maps and connectivity matrices. It also makes the prediction problem depend heavily on which features survive preprocessing and fold-specific selection. Second, CNNs can preserve more of this structure, but labelled stroke cohorts are often small, and model estimates can become sensitive to random seed, fold composition and preprocessing choices. This issue is especially important in EEG, where many time windows can be extracted from one patient. If validation is performed at the segment level rather than the patient level, subject-specific signal structure can leak into the test set and inflate apparent performance. Third, proportional-recovery labels are usually converted into a binary endpoint, but the residual between expected and observed improvement is continuous. A patient just below the classification threshold and a patient far below it receive the same binary label, although the underlying recovery information differs.

Self-supervised learning offers a way to reduce reliance on labelled outcomes by learning patient-level EEG representations from unlabelled data. This is especially relevant in tACS cohorts, where some patients may have baseline EEG but incomplete follow-up, incomplete treatment exposure or time-point combinations that do not yield the final supervised proportional-recovery label. These data cannot be used as labelled outcome samples, but they can still help the encoder learn stable EEG structure if the validation scheme prevents outcome leakage. In this setting, the aim of self-supervision is not to create additional labels, but to increase the amount of EEG available for representation learning. Residual-aware training addresses the complementary problem: it keeps the binary proportional-recovery endpoint for inference while using continuous residual distance and recovery ranking as auxiliary training signals.

Here we developed a residual-aware SSL-CNN for baseline EEG prediction of proportional upper-limb recovery after tACS. The model integrated eyes-open and eyes-closed PSD and WPLI features from pre-treatment EEG. The study makes four contributions. First, it compares traditional EEG machine-learning baselines, a no-SSL CNN, an SSL-CNN without residual-aware heads, a no-SSL residual-aware CNN and the final residual-aware SSL-CNN under patient-level leave-one-subject-out validation. Second, it uses Barlow Twins self-supervised pretraining to include baseline EEG from patients who could not form complete supervised labels, increasing the patient-level EEG distribution available to the encoder. Third, it augments the binary proportional-recovery label with continuous residual distance, pairwise ranking and soft-label objectives so that training better reflects the continuous structure of the recovery endpoint. Fourth, it links model prediction to testable EEG features through integrated gradients, SmoothGrad, occlusion, PSD topomaps and WPLI connectivity analyses. Evaluation used ten random seeds, accuracy stability metrics, ROC-AUC, PR-AUC, Brier score, bootstrap uncertainty, permutation testing and a final confusion matrix.

## Materials and Methods

### Study Design

This was a single-centre stroke tACS prognostic modelling study using baseline resting-state EEG to predict post-treatment upper-limb proportional-recovery status. The workflow was: baseline EEG acquisition, tACS treatment, FMA-UE and MBI assessment after 14 treatment sessions, residual-defined outcome labelling, EEG feature construction, model training and patient-level validation. All primary model inputs were derived from pre-treatment EEG. Baseline clinical variables were used for cohort description and outcome definition, but they were not used as primary model inputs in this revised manuscript.

The study ethics statement is retained as a formal placeholder for author insertion: this study was approved by [IRB name and approval number to be inserted]. Written informed consent was obtained from all participants or their legally authorized representatives.

Figure 1 summarizes the overall workflow, including the clinical source frame, the 19-patient labelled supervised cohort, the additional EEG-indexed patients used for self-supervised learning, baseline EEG acquisition, tACS, FMA-UE outcome definition, PSD/WPLI feature extraction, model comparison and patient-level validation. The figure emphasizes that post-treatment FMA-UE was used only to define the proportional-recovery label, whereas all model inputs were restricted to pre-treatment EEG. It also separates the self-supervised EEG pool from the supervised validation cohort at the patient level.

### Participants

The M1 clinical source workbook contained 29 patient records. Twenty-eight patients were indexed in the EEG directory. Nineteen patients had the baseline EEG and complete baseline and post-treatment FMA-UE information needed to define supervised proportional-recovery labels. These 19 patients formed the patient-level leave-one-subject-out (LOSO) validation cohort. Ten patients were assigned to the proportional-recovery group and nine to the poor-recovery group.

Patients outside the supervised cohort were not treated as labelled outcome samples. Their exclusion from supervised training reflected incomplete experimental completion or absence of a complete proportional-recovery label. According to the author-provided classification, these records may be publicly grouped as baseline-only, baseline plus immediate time-point, or baseline plus immediate plus final time-point records that did not enter the current supervised label analysis. Their baseline EEG could still be used in the self-supervised learning pool to increase the amount of patient EEG available for representation learning.

Table 1 summarizes all 29 M1 clinical source records and separates the 19-patient supervised labelled cohort from the nine additional EEG-indexed patients used only for unlabelled/self-supervised representation learning. Continuous variables are summarized as mean (+/-SD), and categorical variables are summarized as percentages. P values compare the supervised labelled cohort with the additional EEG-indexed SSL pool using Welch's t-test, Mann-Whitney U tests or Fisher's exact tests as appropriate. These tests characterize the data source and missingness structure rather than provide confirmatory baseline inference.

**Table 1. Patients' information.** Values are reported for all M1 clinical source records, the supervised labelled cohort and the additional EEG-indexed SSL pool. P values compare the supervised labelled cohort with the additional EEG-indexed SSL pool. One clinical source record had no current EEG index and is included only in the all-record column. Continuous values are mean (+/-SD); a single available post-treatment value is shown as value (n=1). One disease-duration value recorded in days was converted to months using days/30.

| Measure | All clinical records | Supervised labelled cohort | Additional EEG-indexed SSL pool | P |
|:--|:--:|:--:|:--:|:--:|
| Subject | 29 | 19 | 9 |  |
| **Demographics** |  |  |  |  |
| Gender (woman) | 48% | 58% | 33% | 0.42 |
| Age, years | 63.72 (+/-8.94) | 64.74 (+/-6.49) | 60.11 (+/-12.07) | 0.31 |
| Course of disease, months | 37.69 (+/-21.59) | 36.26 (+/-17.76) | 42.46 (+/-29.33) | 0.86 |
| **Clinical measurements** |  |  |  |  |
| Affected upper limb, left | 55% | 58% | 56% | 1 |
| Affected upper limb, right | 45% | 42% | 44% |  |
| FMA-UE before treatment | 38.66 (+/-24.19) | 40.47 (+/-23.83) | 38.11 (+/-25.53) | 0.98 |
| FMA-UE after 14 sessions | 44.86 (+/-23.72) | 45.47 (+/-23.23) | 66.00 (n=1) |  |
| Observed FMA-UE improvement | 4.67 (+/-4.03) | 5.00 (+/-4.07) | 0.00 (n=1) |  |
| MBI before treatment | 56.55 (+/-21.26) | 55.53 (+/-19.92) | 62.22 (+/-22.93) | 0.46 |
| MBI after 14 sessions | 76.19 (+/-22.58) | 76.84 (+/-21.49) | 100.00 (n=1) |  |
| **Data availability** |  |  |  |  |
| Baseline EEG indexed | 97% | 100% | 100% |  |
| Complete post-treatment FMA-UE | 72% | 100% | 11% | <0.001 |
| Complete supervised label | 66% | 100% | 0% | design |

### Outcome Labels

The primary endpoint was proportional-recovery status after tACS, derived from FMA-UE. Expected improvement for patient \(i\) was defined as:

```text
Expected improvement_i = 0.7 x (66 - FMA_pre_i)
```

Observed improvement was:

```text
Observed improvement_i = FMA_post_i - FMA_pre_i
```

The proportional-recovery residual was:

```text
Residual_i = Expected improvement_i - Observed improvement_i
```

The supervised-cohort median residual was used as the cohort-specific threshold:

```text
tau = median(Residual_i)
```

The binary label was:

```text
y_i = 1, if Residual_i <= tau
y_i = 0, if Residual_i > tau
```

Here, \(y_i = 1\) indicates proportional recovery and \(y_i = 0\) indicates poor recovery. For residual-aware auxiliary training, the signed residual distance was defined as:

```text
d_i = tau - Residual_i
```

Positive values of \(d_i\) indicate recovery closer to or beyond the proportional-recovery threshold, and negative values indicate poorer recovery. Residual, signed-distance and ranking targets were used only during supervised training, not as test-time model inputs.

### tACS Protocol

tACS was delivered to the primary motor cortex contralateral to the affected hand. Right-hand impairment corresponded to C3 stimulation, and left-hand impairment corresponded to C4 stimulation. Stimulation used a Neuroscan 1x1 transcranial electrical stimulator (DC-STIMULATOR PLUS). The stimulation frequency was 20 Hz, intensity was 1000 microampere and duration was 20 min per session. Patients received one session per day for 14 sessions. Stimulation impedance was maintained below 30 kOhm. FMA-UE and MBI were assessed before treatment and after completion of the 14 tACS sessions. The formal assessor and blinding statement is retained as a placeholder: FMA-UE and MBI were assessed by [assessor qualification and blinding status to be inserted].

### EEG Preprocessing

EEG was acquired before tACS using a Compumedics Neuroscan SynAmps2 64-channel EEG system with a standard 10-20 electrode layout. Resting-state recordings included eyes-open (EO) and eyes-closed (EC) conditions. Preprocessing was performed in EEGLAB. The preprocessing pipeline used average reference, high-pass filtering at 0.5 Hz, low-pass filtering at 45 Hz and removal of 50 Hz line noise. Independent component analysis was performed in EEGLAB. Bad channels and artefactual segments/components were manually rejected in EEGLAB.

The current analysis used preprocessed EEGLAB `.set/.fdt` files. Feature extraction scripts validated finite continuous data arrays, fixed channel order and minimum recording length. Where 62 EEG channels were retained for model input, reference channels such as M1/M2 were not used as EEG predictors. Before PSD and connectivity extraction, features were aligned to a common affected-side convention. Patients with left-hand impairment were mirrored so that the model represented stimulation-side and affected-side information consistently across patients. This alignment was intended to reduce learning of trivial left-right differences.

### EEG Features

EEG features were constructed separately for EO and EC. PSD features were computed from 0.5 to 45 Hz using Welch's method and organized as channel-frequency matrices. Connectivity features were computed using WPLI, which reduces zero-lag phase-synchronization effects related to volume conduction and sample-size bias [10]. Connectivity features were organized by channel-pair edges and frequency bands. The frequency bands were delta, theta, alpha, beta-low, beta-medium and beta-high. The main EEG input contained PSD and WPLI information from both EO and EC, without clinical predictors.

### Machine Learning

Traditional EEG-ML baselines used flattened PSD+WPLI EO+EC features. The baseline set included logistic regression with L1 regularization, logistic regression with L2 regularization and RBF-kernel support-vector machine after SelectK=100 feature selection. Feature selection and scaling were performed inside the LOSO training folds to avoid test-fold leakage. These models define the performance boundary of conventional high-dimensional EEG feature modelling without CNN structure, SSL pretraining or residual-aware auxiliary objectives.

### Self-supervised Learning

Self-supervised learning was used to learn patient-level EEG representations without recovery labels. This step allowed baseline EEG from patients who could not enter supervised outcome training to contribute to encoder pretraining. Barlow Twins was selected instead of negative-pair contrastive learning or reconstruction-based autoencoding for three reasons. First, the cohort is small at the patient level, and explicit negative samples or memory queues may treat neurophysiologically similar patients as negatives. Barlow Twins learns invariance from the cross-correlation matrix between two augmented views of the same patient and does not require a large negative set. Second, its redundancy-reduction term decorrelates representation dimensions, which is useful for high-dimensional EEG features in small samples. Third, compared with reconstructing noisy high-dimensional EEG inputs, the Barlow objective more directly encourages a compact representation that can transfer to proportional-recovery classification. Two augmented views were generated from the same patient's EEG feature representation and passed through a shared encoder \(f(\cdot)\) and projection head \(g(\cdot)\):

```text
z_i^A = g(f(x_i^A)),  z_i^B = g(f(x_i^B))
```

The cross-correlation matrix between the two projected batches was:

```text
C_jk =
sum_b z^A_bj z^B_bk /
sqrt(sum_b (z^A_bj)^2) sqrt(sum_b (z^B_bk)^2)
```

The Barlow-style redundancy-reduction loss was:

```text
L_SSL = sum_j (1 - C_jj)^2 + lambda sum_j sum_{k != j} C_jk^2
```

The diagonal term encourages invariance between two views of the same patient-level EEG representation. The off-diagonal term reduces redundancy between representation dimensions, following the Barlow Twins principle of redundancy reduction [17]. SSL pretraining was followed by supervised fine-tuning on the 19-patient labelled cohort. Figure 2 shows the unlabelled EEG pool, two-view augmentation, shared encoder, cross-correlation matrix and SSL loss. In each LOSO fold, the held-out test patient was excluded before scaling, augmentation and encoder pretraining.

### CNN and Residual-aware Learning

The deep model used separate branches for PSD and WPLI features across EO and EC states. Branch embeddings were combined through a gated fusion layer into a shared representation. The classification head predicted proportional-recovery probability. Residual-aware auxiliary heads used the continuous signed residual distance and recovery-order information during training. Figure 3 shows that PSD and WPLI branches first learn EO/EC state-specific embeddings and then fuse them into a shared patient-level multimodal EEG representation. The total training loss can be written as:

```text
L_total = L_BCE(y_i, p_i)
        + alpha L_residual(d_i, d_hat_i)
        + beta L_rank
        + gamma L_soft
```

\(L_BCE\) denotes binary cross-entropy for the proportional-recovery label, \(L_residual\) denotes the signed-distance residual loss, \(L_rank\) denotes a pairwise recovery-ranking objective and \(L_soft\) denotes soft-label supervision derived from residual proximity where used by the training configuration. At inference, predictions were generated from the binary classification head only. Residual and ranking heads regularized training but did not use post-treatment information during testing.

### Validation

The primary validation scheme was patient-level LOSO cross-validation. No EEG segment, seed-level row or augmented view was treated as an independent patient. CNN models were trained across ten random seeds. The primary stability-oriented metrics were mean seed accuracy, minimum seed accuracy and accuracy standard deviation. Additional metrics were balanced accuracy, sensitivity, specificity, ROC-AUC, PR-AUC and Brier score.

Uncertainty was assessed using subject-level bootstrap intervals where available. Paired comparisons used subject-level resampling. McNemar tests were used for paired hard predictions. Permutation tests used subject-level label permutations. Calibration was assessed with Brier score and calibration curves. The locked patient-level prediction was used to provide a confusion matrix for the final model. Seed-averaged or ensemble predictions are treated only as sensitivity or descriptive locked-prediction analyses, not as the primary clinical-use metric. Within-cohort pre-post changes in FMA-UE and MBI were assessed using paired tests. Paired t-tests were used when the paired differences were compatible with approximate normality, and Wilcoxon signed-rank tests were used otherwise.

### Explainability

Model explanation was performed after training and validation. Integrated gradients and SmoothGrad were used to estimate the contribution of PSD channel-frequency features and WPLI edge features to proportional-recovery probability [18,19]. Branch and state occlusion quantified the change in prediction loss after removing PSD, WPLI, EO or EC inputs. PSD attributions were mapped by frequency band to 62-channel scalp topographies, and WPLI attributions were summarized by frequency band as the highest-absolute-attribution connectivity edges. Topographic and connectivity figures were rendered with MNE-Python [11,12]. These analyses report the distribution of EEG features used by the model and are interpreted as exploratory candidate biomarkers.

### Software

Analyses used Python, PyTorch, scikit-learn, MNE-Python and EEGLAB [11,12,20,21]. Reporting follows TRIPOD, TRIPOD+AI and PROBAST principles for clinical prediction models [13-15].

## Results

### Cohort

The final supervised cohort contained 19 patients. Mean age was 64.74 years (SD 6.49), and 11 patients were female. Eight patients had right-sided affected upper-limb impairment, and 11 had left-sided impairment. Baseline FMA-UE was 40.47 (SD 23.83), post-treatment FMA-UE after 14 tACS sessions was 45.47 (SD 23.23), and observed FMA-UE improvement was 5.00 points (SD 4.07; Wilcoxon signed-rank p < 0.001 for the paired pre-post change). The proportional-recovery residual was 12.87 (SD 16.23) across the supervised cohort. Baseline MBI was 55.53 (SD 19.92), and post-treatment MBI was 76.84 (SD 21.49; paired t-test p < 0.001 for the paired pre-post change).

The proportional-recovery and poor-recovery groups were similar in age (64.10 vs 65.44 years; p = 0.658), disease duration (35.40 vs 37.22 months; p = 1.000), sex distribution (p = 1.000) and affected side (p = 1.000). As expected from the residual-based outcome definition, the groups differed in baseline motor status and disability. The proportional-recovery group had higher baseline FMA-UE (59.60 vs 19.22; p < 0.001), higher post-treatment FMA-UE (64.10 vs 24.78; p < 0.001), higher baseline MBI (68.00 vs 41.67; p = 0.002) and higher post-treatment MBI (93.00 vs 58.89; p < 0.001). Observed FMA-UE improvement was not significantly different between groups (4.50 vs 5.56 points; p = 0.480), whereas the proportional-recovery residual sharply separated the two groups (-0.02 vs 27.19; p < 0.001). These results indicate that the binary label primarily captured whether observed improvement was proportional to the residual recovery potential rather than absolute improvement alone. The full source-frame patient information table is reported in the Participants section (Table 1).

### ML Baselines

Traditional EEG-ML baselines provided limited hard-label discrimination for proportional recovery. The PSD+WPLI logistic-regression L1 baseline achieved accuracy 0.737, balanced accuracy 0.733, ROC-AUC 0.711, PR-AUC 0.775 and Brier score 0.208. Logistic regression with L2 regularization achieved accuracy 0.684, balanced accuracy 0.689, ROC-AUC 0.778, PR-AUC 0.840 and Brier score 0.221. SVM RBF with SelectK=100 achieved accuracy 0.684, balanced accuracy 0.694, ROC-AUC 0.778, PR-AUC 0.771 and Brier score 0.219. These results indicate that traditional EEG-ML models retained some ranking information but did not provide stable hard-label classification or probability calibration in this small high-dimensional cohort. The model scorecard further shows that conventional models did not dominate across accuracy, sensitivity, specificity, ROC-AUC, PR-AUC and Brier score simultaneously (Table 2; Fig. 4C).

### CNN and SSL

CNN modelling improved the ability to use complete EEG matrices but remained seed-sensitive without residual-aware training. In the paired ten-seed comparison using seeds 0, 1, 2, 3, 4, 5, 6, 7, 8 and 13, the no-SSL CNN achieved a mean accuracy of 0.753, balanced accuracy of 0.743, ROC-AUC of 0.783 and PR-AUC of 0.769. The minimum accuracy across seeds was 0.579, and the accuracy standard deviation was 0.094. Patient-level Barlow SSL without residual-aware heads had the same mean accuracy of 0.753, balanced accuracy of 0.742, ROC-AUC of 0.739 and PR-AUC of 0.719. Its accuracy standard deviation was 0.053, and its minimum accuracy was 0.684. Thus, SSL alone did not improve mean accuracy or ranking metrics in the paired comparison, but it reduced seed-to-seed variability and increased the worst-seed accuracy (Table 3; Fig. 5A).

### Residual-Aware

Residual-aware auxiliary supervision provided the strongest training signal in the ablation structure. The no-SSL CNN with residual-aware heads achieved a ten-seed mean accuracy of 0.816, balanced accuracy of 0.812, ROC-AUC of 0.899, PR-AUC of 0.908 and Brier score of 0.130. The final patient-level Barlow SSL plus residual-aware model achieved a ten-seed mean accuracy of 0.837, balanced accuracy of 0.831, ROC-AUC of 0.860, PR-AUC of 0.858 and Brier score of 0.142. Its minimum accuracy was 0.789, and its accuracy standard deviation was 0.039. Thus, the final model showed the most favourable stability profile among the core deep models, although the current ablations do not prove that SSL has an independent performance gain beyond residual-aware supervision (Table 3; Fig. 5B,C).

### Final Model

The final residual-aware SSL-CNN supported patient-level prediction of proportional recovery with conservative uncertainty reporting. In the locked hard prediction used for the confusion matrix, the model correctly classified 10 of 10 proportional-recovery patients and 6 of 9 poor-recovery patients. The confusion matrix was TP = 10, FN = 0, FP = 3 and TN = 6, corresponding to sensitivity 1.000, specificity 0.667 and accuracy 0.842. The locked prediction had balanced accuracy 0.833, ROC-AUC 0.844, PR-AUC 0.836 and Brier score 0.126.

In the model-to-model scorecard, the final model provided the most complete performance profile among clinically relevant EEG-only candidates: it retained perfect sensitivity in the locked prediction, improved calibration relative to the traditional baselines and matched or exceeded the hard-label performance of the comparison models. The otherwise matched no-SSL CNN reached the same locked accuracy and balanced accuracy but had lower ROC-AUC (0.811), lower PR-AUC (0.808) and no paired-source Brier estimate in the ten-seed file. These values are reported alongside the ten-seed stability metrics because a single changed patient would move accuracy by 5.3 percentage points in a 19-patient LOSO cohort. Figure 4A shows a stepwise but high-ranking ROC curve, Figure 4B shows no false negatives at the fixed threshold, Figure 4C-a shows the metric-wise model comparison as grouped bars and Figure 4C-b shows that the final model had the lowest Brier score. The final model's advantage is therefore best understood as a more balanced combination of ranking, calibration and stability rather than a single hard-label metric.

The training curves provide an additional check on optimization stability (Fig. 4D). Across ten seeds and 19 LOSO folds, the mean training total loss decreased from 1.115 at epoch 1 to 0.201 at epoch 50 and 0.195 at epoch 100. The mean validation total loss decreased from 0.927 to 0.700 at epoch 50 and stabilized at 0.684 by epoch 100. Validation BCE decreased from 0.693 to 0.460, and the weighted residual contribution decreased from 0.164 to 0.117. The weighted soft-label term increased from 0.069 to approximately 0.107, but its contribution was small and reflected the tension between residual-proximity soft supervision and hard-label convergence rather than divergence of the overall objective.

### Ablations

Feature-family ablations indicated that PSD features retained useful discriminatory information. PSD-only features achieved accuracy 0.789, balanced accuracy 0.789, ROC-AUC 0.811, PR-AUC 0.840 and Brier score 0.193. WPLI-only features achieved accuracy 0.632, balanced accuracy 0.628, ROC-AUC 0.689, PR-AUC 0.751 and Brier score 0.232. This suggests that scalp power patterns were the strongest single feature family for hard-label discrimination, while WPLI connectivity alone contained weaker but still non-random recovery-ranking information.

State and frequency ablations gave a more nuanced picture. EO-only PSD+WPLI features performed poorly in the current analysis (accuracy 0.316, ROC-AUC 0.300), whereas EC-only PSD+WPLI features retained ranking information (accuracy 0.632, ROC-AUC 0.789, PR-AUC 0.816). Beta-medium-only features achieved accuracy 0.737 and ROC-AUC 0.711, outperforming beta-high-only features. Motor-related WPLI edges achieved ROC-AUC 0.733 and PR-AUC 0.816 despite the lower hard-label accuracy. Figure 5A shows that the residual-aware SSL-CNN has the tightest ten-seed accuracy distribution; Figure 5B identifies residual-aware supervision as the most stable ablation component; and Figure 5C-D show that PSD-only and EC-only inputs retain useful ranking information despite lower dimensionality. These ablations are exploratory and should be interpreted as evidence about candidate EEG information sources rather than as independently validated biomarkers (Table 4; Fig. 5).

### Explainability

Model explanations localized predictive information to state-dependent PSD patterns and beta-band connectivity. The gated fusion weights indicated a division of labour between feature families and behavioural states. In the PSD branch, the EO embedding received the larger average gate weight, whereas in the WPLI branch the EC embedding dominated. Occlusion analyses were directionally consistent with this state dependence: removing EC or WPLI features produced larger average loss perturbations than removing EO or PSD features, while PSD topographic attributions remained concentrated in EO channel-frequency maps. These results suggest that the final model used EO PSD and EC connectivity in a complementary rather than interchangeable manner.

The PSD attribution maps emphasized EO features over temporal, frontal and peri-central scalp regions. The highest-ranking PSD features included TP7, C5, F5, FPZ and FC5 in gamma, beta-high and delta ranges. Figure 6A shows broader and clearer spatial attribution in the EO row than in the EC row across the seven frequency bands from delta to gamma, with localized frontal, temporal and central maxima rather than a single isolated electrode. A nominal validation analysis linked the signed residual-distance association of EO F5 delta attribution to outcome structure, but this association should be treated as hypothesis-generating because the cohort is small and multiple testing correction is conservative. The PSD results align with prior EEG studies in which resting-state rhythms and beta-band activity were related to post-stroke motor impairment or recovery [7-9,26].

The WPLI attribution maps were concentrated in EC beta-band connectivity. The top-ranked edges involved frontal, central and parietal nodes, including F8-CP1, FT7-P2, C5-P2, F4-C5 and F8-FC4. Figure 6B shows that EC beta-medium and beta-high edges are more concentrated than EO edges, mainly linking frontal-central, central-parietal and bilateral channel regions. Both positive and negative signed attributions appear, indicating that the model used connectivity patterns associated with higher proportional-recovery probability and patterns associated with poorer recovery. Network-level summaries highlighted beta-band frontal-central, motor-adjacent and interhemispheric connectivity, each showing nominal residual-distance associations before correction for multiple exploratory tests. This pattern resembles the interpretive structure used in prior deep-learning prognosis work, where spatially distributed frontal, central and parietal neurophysiological features were treated as candidate recovery factors rather than as single causal mechanisms [4]. It is also compatible with reports linking motor recovery to cortical connectivity and non-motor-to-motor network interactions after stroke [16,26].

## Discussion

This study tested whether baseline resting-state EEG could predict post-tACS proportional upper-limb recovery after stroke. The final residual-aware SSL-CNN achieved the most stable ten-seed accuracy distribution among the core deep models and retained high sensitivity in the locked patient-level prediction. Traditional EEG-ML baselines retained partial ranking information but provided weaker hard-label discrimination and calibration. Taken together, the performance, ablation and explanation analyses indicate that leakage-controlled structured EEG representation learning can support recovery prediction and generate interpretable candidate PSD/WPLI biomarkers.

The cohort statistics help define the clinical context of the prediction task. The proportional-recovery and poor-recovery groups were similar in age, disease duration, sex and affected side, but they differed in baseline FMA-UE and MBI. This pattern is expected because the residual endpoint is derived from baseline impairment, expected improvement and observed improvement. Clinical variables were used to describe the cohort and define the recovery residual, whereas the main model input was restricted to pre-treatment EEG. The current results therefore address whether EEG-only features can predict a clinically defined recovery phenotype. The relative value of clinical-only, EEG-only and combined predictors should be tested in future external validation designs.

The residual-aware training strategy is the central methodological contribution. Proportional recovery is convenient as a binary clinical endpoint, but the underlying residual is continuous. A patient close to the median residual threshold and a patient far below it have different recovery information, even if both receive the same binary label. By incorporating signed residual distance, recovery ranking and residual-proximity soft labels during training, the model used more of the outcome structure while preserving a simple binary head for inference. This design follows the logic of modern prognosis modelling: the training objective should match the clinical construct as closely as possible, and the reported prediction should remain interpretable for patient-level decision support.

The structured CNN architecture also addresses a limitation of conventional EEG prognosis pipelines. Flattened PSD and WPLI vectors are high dimensional, and fold-specific feature selection can make the model depend on unstable subsets of channels, frequency bins and edges. The CNN used here retained the two-dimensional structure of PSD channel-frequency maps and the organized edge-frequency structure of WPLI connectivity. This does not mean that deep learning is automatically superior in small EEG cohorts. Rather, the ablation results suggest that structured models are useful when they are paired with strict patient-level validation, seed stability reporting and explicit checks against information leakage.

Barlow SSL contributes mainly through patient-level representation learning and improved use of incomplete EEG records. Compared with contrastive methods that rely on many negative samples, Barlow Twins learns invariance and redundancy reduction through the cross-correlation matrix between two augmented views, a better match for small patient-level EEG cohorts with substantial inter-patient heterogeneity. Compared with reconstruction-based self-supervision, it does not require the model to reconstruct high-dimensional EEG inputs; instead, it encourages a compact representation that can transfer to proportional-recovery classification. Patient-level Barlow SSL also allowed baseline EEG from patients without complete supervised proportional-recovery labels to contribute to encoder pretraining. This is a practical advantage in rehabilitation and neuromodulation studies, where incomplete treatment completion, missing follow-up or non-standard time-point combinations are common. In the paired seed comparison, SSL without residual-aware heads reduced seed-to-seed variability and improved worst-seed accuracy, but it did not improve mean accuracy, ROC-AUC or PR-AUC over the no-SSL CNN. The current evidence therefore supports SSL as a component for using unlabelled EEG and stabilizing representation learning; its independent performance gain requires larger cohorts.

The explanation analyses provide neurophysiological context. PSD attributions were strongest in EO features over temporal, frontal and peri-central channels, while WPLI attributions were strongest in EC beta-band connectivity involving frontal, central, parietal and interhemispheric edges. These findings are compatible with prior EEG rehabilitation studies linking resting-state rhythms, beta oscillations and motor-network connectivity to post-stroke upper-limb impairment or recovery [7-9,16,26]. They are also conceptually consistent with the interpretive approach of transferable deep-learning prognosis work, in which spatially distributed physiological features are analysed as candidate recovery factors rather than as single-channel biomarkers [4]. The present attribution maps should therefore be used to pre-specify future EEG hypotheses, such as whether frontal-central beta connectivity or EO frontal/temporal power improves patient stratification in larger tACS cohorts.

Several limitations are important. First, the labelled cohort contained only 19 patients, and a one-patient change shifts accuracy by 5.3 percentage points. This is why the manuscript reports locked predictions, ten-seed summaries, seed variability and conservative uncertainty language. Second, the residual threshold was defined by the median residual within this cohort; future studies should pre-specify the threshold, estimate it in a training cohort or validate it externally. Third, the additional EEG pool supports SSL pretraining but does not replace labelled external validation. Fourth, explanation results are model-dependent and were not confirmed after all exploratory multiple-comparison corrections. They should not be interpreted as causal neural mechanisms or as treatment targets. Fifth, ethics approval number, assessor qualification, blinding status, electrode-size details, adverse-event reporting and repository accession information remain to be inserted before submission.

In summary, this study provides an EEG-only framework for predicting post-tACS proportional upper-limb recovery and for generating candidate PSD and WPLI biomarkers. The findings justify prospective validation and model comparison in larger cohorts, especially designs that compare clinical-only, EEG-only and combined models and pre-specify the EO PSD and EC beta-connectivity hypotheses generated here.

## Conclusion

Baseline resting-state EEG combined with residual-aware self-supervised CNN modelling can provide patient-level prediction of post-tACS upper-limb proportional-recovery status in a stroke tACS cohort. The approach uses complete PSD and WPLI EEG structure, incorporates unlabelled EEG through SSL and preserves continuous residual information during training. These findings provide a reproducible modelling framework and candidate neurophysiological hypotheses for EEG-guided tACS response prediction.

## Data Availability

De-identified derived data supporting the manuscript will be deposited in a citable repository before final journal submission or publication. The deposit should include subject-level cohort tables, locked LOSO predictions, ten-seed model summaries, bootstrap and permutation outputs, participant-flow source notes, figure-source summaries, explainability tables, validation audits, model-reporting cards and figure manifests. Raw EEG recordings, minimally processed EEG files and directly identifiable clinical source records are not publicly released in this draft because they contain human-participant data and may be subject to institutional review board, consent, privacy and data-use restrictions. Access to restricted raw or minimally processed data should be reviewed by the responsible institution. Qualified researchers may request access from the corresponding author or institutional data-access committee after ethics approval and completion of a data-use agreement, subject to the original consent and institutional restrictions. Repository DOI, licence, access committee name and final dataset version remain to be inserted.

## Code Availability

The modelling, statistical validation, MNE topomap/connectivity rendering, figure-generation and manuscript table-generation scripts will be deposited in a public code repository or linked code archive before final journal submission or publication. The final archive should include the exact commit hash or version tag, environment metadata, random seeds, locked prediction CSVs and scripts required to reproduce the manuscript tables and figures from the deposited derived data.

## Tables

**Table 2. Traditional EEG machine-learning baselines.** Source file: `results/tables/table2_main_model_performance.csv`. All models use EEG PSD+WPLI EO+EC features only.

| Model                 | Feature selection                   | Accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC | PR-AUC | Brier |
|:----------------------|:------------------------------------|---------:|------------------:|------------:|------------:|--------:|-------:|------:|
| Logistic L1           | None                                | 0.737    | 0.733             | 0.800       | 0.667       | 0.711   | 0.775  | 0.208 |
| Logistic L2           | None                                | 0.684    | 0.689             | 0.600       | 0.778       | 0.778   | 0.840  | 0.221 |
| SVM RBF + SelectK=100 | SelectK=100 inside LOSO train folds | 0.684    | 0.694             | 0.500       | 0.889       | 0.778   | 0.771  | 0.219 |

**Table 3. Deep model stability and residual-aware ablation.** Source file: `results/metrics/core_ablation_10seed_summary.csv`. The No-SSL CNN and SSL-CNN without residual heads rows use the paired seed set 0, 1, 2, 3, 4, 5, 6, 7, 8 and 13 from `results/metrics/updated_sub05_sub28_10seed_no_ssl_barlow_cnn_summary.csv`. Metrics are ten-seed summaries unless otherwise indicated. NA indicates that Brier score was not available in that paired source file.

| Model                             | Mean accuracy | Min accuracy | Accuracy SD | Balanced accuracy | ROC-AUC | PR-AUC | Brier |
|:----------------------------------|--------------:|-------------:|------------:|------------------:|--------:|-------:|:------|
| No-SSL CNN                        | 0.753         | 0.579        | 0.094       | 0.743             | 0.783   | 0.769  | NA    |
| SSL-CNN, no residual heads        | 0.753         | 0.684        | 0.053       | 0.742             | 0.739   | 0.719  | NA    |
| No-SSL CNN + residual-aware heads | 0.816         | 0.737        | 0.045       | 0.812             | 0.899   | 0.908  | 0.130 |
| Residual-aware SSL-CNN            | 0.837         | 0.789        | 0.039       | 0.831             | 0.860   | 0.858  | 0.142 |

**Table 4. Feature, state and band ablation.** Source file: `results/tables/table3_ablation.csv`. Rows are exploratory EEG-only ablations.

| Ablation         | Input features | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Brier |
|:-----------------|---------------:|---------:|------------------:|--------:|-------:|------:|
| PSD only         | 744            | 0.789    | 0.789             | 0.811   | 0.840  | 0.193 |
| WPLI only        | 22,692         | 0.632    | 0.628             | 0.689   | 0.751  | 0.232 |
| PSD + WPLI       | 23,436         | 0.684    | 0.678             | 0.767   | 0.793  | 0.207 |
| EO only          | 11,718         | 0.316    | 0.317             | 0.300   | 0.467  | 0.490 |
| EC only          | 11,718         | 0.632    | 0.633             | 0.789   | 0.816  | 0.223 |
| Beta medium only | 3,906          | 0.737    | 0.733             | 0.711   | 0.776  | 0.232 |
| Beta high only   | 3,906          | 0.579    | 0.578             | 0.567   | 0.561  | 0.317 |
| Motor WPLI edges | 6,780          | 0.632    | 0.628             | 0.733   | 0.816  | 0.261 |

**Table 5. EEG explanation biomarkers.** Source files: `results/explainability/psd_channel_band_importance.csv` and `results/explainability/wpli_top_edges.csv`. Rows show the top five PSD and top five WPLI features by mean absolute attribution.

| Family | State | Band        | Feature | Signed attribution | Abs attribution |
|:-------|:------|:------------|:--------|-------------------:|----------------:|
| PSD    | EO    | Gamma       | TP7     | -0.000892          | 0.001330        |
| PSD    | EO    | Gamma       | C5      | -0.000846          | 0.001245        |
| PSD    | EO    | Beta High   | TP7     | -0.000749          | 0.001228        |
| PSD    | EO    | Beta High   | FPZ     | -0.000943          | 0.001159        |
| PSD    | EO    | Delta       | F5      | -0.000817          | 0.001137        |
| WPLI   | EC    | Beta High   | F8-CP1  | 0.000154           | 0.002128        |
| WPLI   | EC    | Beta Medium | FT7-P2  | -0.001331          | 0.002119        |
| WPLI   | EC    | Beta Medium | C5-P2   | -0.000529          | 0.002111        |
| WPLI   | EC    | Beta High   | F4-C5   | -0.000062          | 0.002097        |
| WPLI   | EC    | Beta High   | F8-FC4  | 0.000994           | 0.002095        |

## Figure Legends

**Figure 1. Overall framework.** Source cohort, labelled supervised cohort, self-supervised EEG pool, baseline EEG acquisition, 14-session tACS treatment, outcome assessment, PSD/WPLI feature extraction, model-family comparison and patient-level validation. Figure file: `results/figures/revised_initial/figure1_overall_framework_provided.png`.

**Figure 2. Self-supervised learning framework.** Unlabelled EEG use, two-view augmentation, shared encoder, cross-correlation matrix and Barlow-style redundancy-reduction loss. The figure emphasizes that EEG records outside the complete supervised outcome set can still enlarge representation learning. Figure file: `results/figures/revised_initial/figure2_ssl_framework_provided.png`.

**Figure 3. CNN and residual-aware learning.** EO/EC PSD and WPLI branches, gated fusion, shared embedding, binary proportional-recovery head and auxiliary residual-distance, ranking and soft-label heads. Only the binary head is used for test-time inference. Figure file: `results/figures/revised_initial/figure3_cnn_residual_aware_provided.png`.

**Figure 4A. Final-model ROC curve.** Single-curve ROC plot for the locked patient-level seedmean10 residual-aware SSL-CNN prediction. ROC-AUC = 0.844. The curve is stepwise because the analysis contains 19 patients. Figure file: `results/figures/revised_initial/figure4a_final_model_roc.png`.

**Figure 4B. Final-model confusion matrix.** Fixed-threshold patient-level confusion matrix for the final residual-aware SSL-CNN. TP = 10, FN = 0, FP = 3 and TN = 6; sensitivity = 1.00, specificity = 0.67 and accuracy = 0.84. Figure file: `results/figures/revised_initial/figure4b_final_model_confusion_matrix.png`.

**Figure 4C-a. Model metric histogram.** Grouped patient-level bars comparing traditional EEG-ML baselines, the standard-seed no-SSL CNN, Barlow CNN and the final residual-aware SSL-CNN across accuracy, balanced accuracy, sensitivity, specificity, ROC-AUC and PR-AUC. Figure file: `results/figures/revised_initial/figure4c_a_model_metric_histogram.png`.

**Figure 4C-b. Brier calibration error.** Patient-level Brier scores for the same model set, with lower values indicating better probability calibration. Figure file: `results/figures/revised_initial/figure4c_b_brier_calibration.png`.

**Figure 4D. Final-model loss curves.** Training and validation loss curves for the final residual-aware SSL-CNN, summarized across ten random seeds and 19 LOSO folds. The figure shows total loss, classification loss and weighted residual-aware auxiliary loss components; the dashed line marks the start of stochastic weight averaging. Figure file: `results/figures/revised_initial/figure4d_final_model_loss_curves.png`.

**Figure 5. Stability and ablation.** Multi-panel summary of seed-level stability and feature ablations. Panel a shows seed-level accuracy distributions for the core deep models; panel b shows a metric scorecard for residual-aware and SSL ablations, with colours normalized within each metric column and Brier score inverted; panel c ranks feature, state and band ablations by accuracy and ROC-AUC; panel d shows input dimensionality versus ROC-AUC, with point size encoding PR-AUC. This figure emphasizes stability, calibration and information efficiency rather than a single bar chart. Figure file: `results/figures/revised_initial/figure5_stability_ablation.png`.

**Figure 6A. PSD topomap explainability.** EO and EC rows show signed PSD attribution topomaps across delta, theta, alpha, beta-low, beta-medium, beta-high and gamma frequency bands. Figure file: `results/figures/revised_initial/figure6a_psd_topomap_bands.png`.

**Figure 6B. WPLI connectivity explainability.** EO and EC rows show the top-20 WPLI edges for each frequency band; red edges indicate positive signed attribution, blue edges indicate negative signed attribution, and line width indicates absolute attribution magnitude. Figure file: `results/figures/revised_initial/figure6b_wpli_connectivity_bands.png`.

**Supplementary Figure 1. Residual threshold and outcome definition.** Distribution of expected improvement, observed improvement, proportional-recovery residual and the supervised-cohort median threshold.

**Supplementary Figure 2. MNE PSD topomap contact sheet.** Complete EO/EC by frequency-band topographic attribution maps rendered from mean signed PSD attribution.

**Supplementary Figure 3. MNE WPLI connectivity contact sheet.** Complete EO/EC by frequency-band WPLI top-20 edge attribution maps. Red edges indicate non-negative mean signed attribution, blue edges indicate negative mean signed attribution and line width scales with mean absolute attribution.

**Supplementary Figure 4. Performance precision.** Bootstrap intervals, paired model differences and one-patient movement sensitivity for the 19-patient LOSO cohort.

## References

1. Prabhakaran S, Zarahn E, Riley C, Speizer A, Chong JY, Lazar RM, et al. Inter-individual variability in the capacity for motor recovery after ischemic stroke. Neurorehabil Neural Repair. 2008;22:64-71. doi:10.1177/1545968307305302
2. Byblow WD, Stinear CM, Barber PA, Petoe MA, Ackerley SJ. Proportional recovery after stroke depends on corticomotor integrity. Ann Neurol. 2015;78:848-859. doi:10.1002/ana.24472
3. Stinear CM, Byblow WD, Ackerley SJ, Smith M, Borges VM, Barber PA. PREP2: a biomarker-based algorithm for predicting upper limb function after stroke. Ann Clin Transl Neurol. 2017;4:811-820. doi:10.1002/acn3.488
4. Lin PJ, Zhai X, Li W, Li T, Cheng D, Li C, et al. A transferable deep learning prognosis model for predicting stroke patients' recovery in different rehabilitation trainings. IEEE J Biomed Health Inform. 2022;26:6003-6011. doi:10.1109/JBHI.2022.3205436
5. White A, Saranti M, d'Avila Garcez A, Hope TMH, Price CJ, Bowman H. Predicting recovery following stroke: deep learning, multimodal data and feature selection using explainable AI. NeuroImage Clin. 2024;43:103638. doi:10.1016/j.nicl.2024.103638
6. Lassi M, Dalise S, Privitera L, Giannini N, Mancuso M, Azzollini V, et al. Enhancing upper limb motor recovery prediction after acute stroke using EEG and subacute data. APL Bioeng. 2026;10:016108. doi:10.1063/5.0287165
7. Mane R, Chew E, Phua KS, Ang KK, Robinson N, Vinod AP, et al. Prognostic and monitory EEG-biomarkers for BCI upper-limb stroke rehabilitation. IEEE Trans Neural Syst Rehabil Eng. 2019;27:1654-1664. doi:10.1109/TNSRE.2019.2924742
8. Saes M, Meskers CGM, Daffertshofer A, van Wegen EEH, Kwakkel G. Are early measured resting-state EEG parameters predictive for upper limb motor impairment six months poststroke? Clin Neurophysiol. 2021;132:56-62. doi:10.1016/j.clinph.2020.09.031
9. Tang CW, Hsiao FJ, Lee PL, Tsai YA, Hsu YF, Chen WT, et al. Beta-oscillations reflect recovery of the paretic upper limb in subacute stroke. Neurorehabil Neural Repair. 2020;34:450-462. doi:10.1177/1545968320913502
10. Vinck M, Oostenveld R, van Wingerden M, Battaglia F, Pennartz CMA. An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. NeuroImage. 2011;55:1548-1565. doi:10.1016/j.neuroimage.2011.01.055
11. Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci. 2013;7:267. doi:10.3389/fnins.2013.00267
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
25. Singh S, Dawar D, Mehmood E, Pandian JD, Sahonta R, Singla S, Amit Batra, Cheruvu S, et al. Determining diagnostic utility of EEG for assessing stroke severity using deep learning models. Biomed Eng Adv. 2024;7:100121. doi:10.1016/j.bea.2024.100121
26. Zhang JJ, Bai Z, Fong KNK. Resting-state cortical electroencephalogram rhythms and network in patients after chronic stroke. J NeuroEngineering Rehabil. 2024;21:32. doi:10.1186/s12984-024-01328-7
