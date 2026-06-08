# Journal of Neural Engineering Cover Letter Template

Date: [Month Day, Year]

Dear Editor,

Please consider our manuscript, "Residual-aware EEG learning for post-stroke proportional recovery," for publication as a [Paper / Research Paper] in *Journal of Neural Engineering*.

This manuscript addresses a practical neural-engineering problem in post-stroke rehabilitation: whether baseline resting-state EEG can support patient-level modelling of upper-limb proportional recovery after a common transcranial alternating-current stimulation protocol. In a pilot supervised cohort of 19 participants, we developed a leakage-controlled leave-one-subject-out EEG modelling pipeline that combines PSD and WPLI features with residual-aware auxiliary supervision. The residual-aware SSL-CNN achieved balanced accuracy of 0.833, ROC-AUC of 0.844, PR-AUC of 0.836, and Brier score of 0.126. We present these findings as exploratory because the cohort is small and external validation is not yet available.

The manuscript is a close fit for *Journal of Neural Engineering* because it combines neural signal processing, neuro-rehabilitation, neuromodulation-response stratification, and transparent machine-learning evaluation. Its main contributions are:

1. A residual-aware outcome formulation anchored to proportional motor-recovery expectations rather than a purely dichotomous raw change score.
2. A subject-level EEG modelling workflow that uses affected-side alignment, PSD and connectivity features, and leave-one-subject-out validation to reduce common leakage risks in small neurorehabilitation datasets.
3. A prepared source-data and reproducibility package, including a 36-worksheet source-data workbook, a 446-field data dictionary for 37 derived CSV or figure-manifest files, figure-source summaries, statistical audits, participant-flow source notes, a conservative PROBAST/TRIPOD+AI risk audit, a performance precision audit, a claim-strength audit, a numeric-claim source trace audit, an AI model reporting card, a source-workbook author metadata audit, patient-record PDF text-layer audit, reporting checklists, manuscript-source integrity checks, author-field replacement mapping, and visual QA outputs.

We have deliberately avoided claiming clinical deployment readiness or proven EEG incremental value over clinical variables. Instead, the manuscript frames the model as a rigorously audited pilot analysis that identifies a reproducible analysis route and testable neurophysiological hypotheses for larger external cohorts.

Author-confirmed statements to insert before upload: [The manuscript is not under consideration elsewhere, and all authors have approved this submission.] [Confirm whether any related manuscripts, preprints, conference abstracts, or overlapping datasets exist.] [The study was approved by Ethics Committee or Institutional Review Board name, approval number, approval date.] [All participants provided written informed consent / consent waiver details.] [Add clinical trial or study registration number if applicable, or provide a submission-ready statement explaining why the study was not registered.]

Declarations requiring author confirmation before upload:

- Ethics approval: [committee name, approval number, approval date].
- Informed consent: [written consent / waiver details, including consent for EEG, tACS, clinical assessments, and data sharing].
- Competing interests: [The authors declare no competing interests / Insert competing-interest statement].
- Funding: [Insert funder names and grant numbers, or state that no specific funding was received].
- Data availability: De-identified derived source-data tables, locked LOSO model predictions, bootstrap/permutation/paired-comparison outputs, participant-flow source notes, figure-source summaries, validation and artifact-quality audit tables, the PROBAST/TRIPOD+AI risk audit, performance precision audit, claim-strength audit, numeric-claim source trace audit, AI model reporting card, source-workbook author metadata audit, patient-record PDF text-layer audit, the author-field replacement map, and the 36-worksheet source-data workbook will be made available at [repository name and DOI/accession]. Raw EEG, minimally processed EEG, and clinical source records are [controlled access / not publicly available] because [ethics, consent, privacy, or institutional restriction].
- Code availability: Analysis, statistical-validation, figure-generation, source-data assembly, manuscript-generation, audit, and package-assembly scripts will be available at [repository name, DOI/accession or URL, version, licence].
- Author contributions: [Insert CRediT or journal-required contribution statement].
- Acknowledgements: [Insert names and permissions, or state none].

The corresponding author is:

[Name, degree]  
[Institution]  
[Postal address]  
[Email]  
[Phone, optional]

Suggested reviewers, if requested:

- [Reviewer name, institution, email, rationale]
- [Reviewer name, institution, email, rationale]
- [Reviewer name, institution, email, rationale]

Excluded reviewers, if any:

- [Reviewer name, institution, reason]

Thank you for considering our manuscript.

Sincerely,

[Corresponding author name]  
On behalf of all authors

## Backup-Target Adaptation Notes

Use the document above as the JNE-first version. If the target changes:

- For IEEE TNSRE, recast the first paragraph around rehabilitation engineering, model reproducibility, and IEEE-style technical contribution.
- For NeuroImage: Clinical, foreground baseline EEG biomarkers and clinical neurophysiology rather than neural-engineering workflow.
- For Communications Medicine, emphasize transparent source data, reporting-guideline alignment, code availability, and restrained pilot framing.
- For Brain Stimulation, submit only after the intervention, safety, adverse-event, and tACS dose/montage details are complete.
