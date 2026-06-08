# Data Availability And FAIR Audit

## Data Availability

De-identified subject-level analysis tables, locked model predictions, statistical output tables, figure source summaries, quality-audit tables, author-field replacement mapping, and analysis code will be deposited in a citable repository before submission. The source-data workbook prepared for this manuscript is `outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx`. It currently contains 36 worksheets, including the source-data dictionary, participant-flow source notes, conservative PROBAST/TRIPOD+AI risk audit, performance precision audit, claim-strength audit, numeric-claim source trace audit, AI model reporting card, source-workbook author metadata audit, patient-record PDF text-layer audit, methods provenance, validation-integrity audit, manuscript-source integrity audit, reference metadata audit, local-reference positioning matrix, DOCX visual QA metrics, author-required evidence trace, author-field replacement map, EEG metadata summaries, manuscript tables, statistical outputs, and figure manifests. Figure exports are available in `results/figures/nature/`, and the figure manifest is available in `results/figures/nature/figure_manifest.csv`. A repository-facing README draft is available at `docs/repository_readme_for_deposit.md`, and the machine-readable source-data dictionary is available at `results/tables/source_data_dictionary.csv`; it currently defines 446 fields across 37 derived CSV or figure-manifest files.

Raw EEG recordings and individual-level clinical source records are not publicly released in this draft because they contain human-participant data and may be subject to consent, institutional review board, and data-use restrictions. Access to restricted raw or minimally processed data should be reviewed by the responsible institution. Qualified researchers may request access from the corresponding author or institutional data-access committee after ethics approval and completion of a data-use agreement.

## Repository And Citation Actions

| Action | Status | Notes |
|---|---|---|
| Deposit derived tables, statistics, and audit tables | Ready after author approval | Include `ResidualAware_SSL_CNN_Source_Data.xlsx`, `results/tables/`, and `results/statistics/`. |
| Deposit source-data dictionary and repository README | Ready after author approval | Include `results/tables/source_data_dictionary.csv` and `docs/repository_readme_for_deposit.md`. |
| Deposit figure source and exports | Ready after figure approval | Include TIFF/PDF/SVG/PNG exports and figure manifest. |
| Deposit code | Needs cleanup before archive | Include `scripts/`, `src/`, environment file, exact commit hash, and random seeds. |
| Deposit raw EEG | Author decision required | Needs consent/ethics confirmation and de-identification route. |
| Deposit clinical source records | Author decision required | Likely controlled access only. |
| Create dataset citation | Pending repository DOI | Add formal DataCite-style citation after Zenodo/OSF/institutional DOI is available. |

## Repository Record Metadata

| Metadata field | Current draft or required input |
|---|---|
| Dataset title | Source data and reproducibility package for "Residual-aware EEG learning for post-stroke proportional recovery" |
| Creators and ORCID IDs | Author confirmation required. |
| Related article | Add journal, manuscript identifier, and article DOI after submission or publication. |
| Description | Derived subject-level analysis tables, locked LOSO predictions, statistical-validation outputs, participant-flow source notes, figure source summaries, validation and artifact-quality audit tables, source-data workbook, source-data dictionary, figure exports, and scripts needed to regenerate the deposited derived outputs. |
| Keywords | stroke; electroencephalography; proportional recovery; upper-limb recovery; self-supervised learning; convolutional neural network; residual-aware learning; leave-one-subject-out validation; tACS; PSD; WPLI |
| Repository platform | Author decision required. |
| DOI or accession | Pending repository deposit. |
| Version and release date | Add after package freeze. |
| Data licence or terms | Author and institutional confirmation required before public release. |
| Code licence | Author and institutional confirmation required before public release. |
| Restricted data route | Data-access committee or corresponding-author contact, request documents, ethics approval requirement, data-use agreement, review process, and allowed reuse scope are required. |

## Dataset Citation Draft

[Authors]. Source data and reproducibility package for "Residual-aware EEG learning for post-stroke proportional recovery". [Repository]. [Version]. [Year]. [DOI/accession].

This citation is a template only. It should not be finalized until the repository record, version, DOI/accession, creators, and licence or access terms are confirmed.

## Dataset Inventory

| Dataset | Current location | Supports | Suggested access route |
|---|---|---|---|
| Cohort characteristics | `results/tables/table1_cohort_characteristics.csv` | Table 1, cohort description | Public derived source data |
| Main model performance | `results/tables/table2_main_model_performance.csv` | Table 2, Figure 2 | Public derived source data |
| Confidence intervals | `results/statistics/model_metric_confidence_intervals.csv` | Uncertainty reporting | Public derived source data |
| Pairwise comparisons | `results/statistics/model_pairwise_comparisons.csv` | Paired bootstrap and McNemar results | Public derived source data |
| Permutation tests | `results/statistics/model_permutation_tests.csv` | Above-chance testing | Public derived source data |
| Ablation analysis | `results/tables/table3_ablation.csv` | Table 3, Figure 3 | Public derived source data |
| Explainability biomarkers | `results/tables/table4_explainability_biomarkers.csv` | Table 4, Figure 4 | Public derived source data |
| Prediction-validation integrity audit | `results/tables/prediction_validation_integrity_audit.csv` | Patient-level validation, leakage-safeguard, bootstrap, permutation, and seed-summary checks | Public reproducibility audit |
| PROBAST/TRIPOD+AI risk audit | `results/tables/probast_tripod_ai_risk_audit.csv`; `docs/probast_tripod_ai_risk_audit.md` | Conservative risk-of-bias, applicability, mitigation, and author-action audit for prediction-model reporting | Public reproducibility audit |
| Performance precision audit | `results/tables/performance_precision_audit.csv`; `docs/performance_precision_audit.md` | Validation resolution, bootstrap interval width, paired-comparison precision, and above-chance testing boundaries | Public reproducibility audit |
| Claim-strength audit | `results/tables/claim_strength_audit.csv`; `docs/claim_strength_audit.md` | Evidence-to-wording alignment for central manuscript claims and overclaims to avoid | Public reproducibility audit |
| AI model reporting card | `results/tables/model_reporting_card.csv`; `docs/model_reporting_card.md` | Intended use, validation safeguards, reproducibility actions, deployment boundaries, and unsupported uses | Public reproducibility audit |
| Source-workbook author metadata audit | `results/tables/source_workbook_author_metadata_audit.csv`; `docs/source_workbook_author_metadata_audit.md` | Non-identifying scan of source workbooks for author-required protocol and submission metadata evidence | Public submission-support audit after author review |
| Patient-record PDF text-layer audit | `results/tables/patient_record_pdf_text_audit.csv`; `docs/patient_record_pdf_text_audit.md` | Non-identifying audit of 31 patient-record PDFs for page counts, text-layer availability, and OCR/manual-review need | Public submission-support audit after author review |
| Author-required evidence trace | `results/tables/author_required_evidence_trace.csv`; `docs/author_required_evidence_trace.md` | Evidence boundary for ethics, consent, registration, intervention, EEG acquisition, preprocessing, and repository fields | Public submission-support metadata after author review |
| Manuscript-source integrity audit | `results/tables/manuscript_integrity_audit.csv`; `docs/manuscript_integrity_audit.md` | Citation numbering, numeric claim, table-source, figure-export, and clean-variant checks | Public reproducibility audit |
| Reference metadata audit | `results/tables/reference_metadata_audit.csv`; `docs/reference_metadata_audit.md` | DOI/URL metadata checks for the 25-reference list | Public reference audit |
| Local reference positioning matrix | `results/tables/local_reference_positioning_matrix.csv`; `docs/local_reference_positioning_matrix.md` | Conservative use of the author-provided local prognostic-model PDF folder | Public citation-positioning audit after author review |
| DOCX visual QA metrics | `results/tables/docx_visual_qa_metrics.csv`; `outputs/doc_visual_qa/` | Word COM PDF export, page rendering, and nonblank visual checks for manuscript DOCX files | Public submission-support audit |
| Submission artifact quality audit | `results/tables/submission_artifact_quality_audit.csv`; `docs/submission_artifact_quality_audit.md` | DOCX structure, source workbook sheets, figure raster, visual QA, reference metadata, and package-integrity checks | Public submission-support audit |
| EEG metadata audit | `results/tables/eeg_recording_metadata_audit.csv`; `results/tables/eeg_recording_summary.csv` | EEG acquisition metadata summary | Public non-identifying metadata |
| Clinical workbook structure audit | `results/tables/clinical_workbook_structure_audit.csv` | Source-workbook provenance without patient-level values | Internal audit or public non-identifying metadata after author review |
| Source-data dictionary | `results/tables/source_data_dictionary.csv` | Field-level reuse and repository metadata for 446 fields across 37 derived CSV or figure-manifest files | Public repository metadata after author review |
| Repository README draft | `docs/repository_readme_for_deposit.md` | Repository-level reuse and access instructions | Public repository metadata after author review |
| Figure exports | `results/figures/nature/` | Figures 1-4, Supplementary Figure 1, and Supplementary Figure 3 performance-precision audit | Public figure source package |
| MNE WPLI connectivity exports | `results/figures/explainability/mne_wpli_connectivity/` | Supplementary Figure 2 and connectivity visual audit | Public figure source package after author approval |
| Raw EEG | Project data source directory | Feature extraction and model training | Controlled access or not shared, pending ethics |
| Clinical source records | Clinical workbook/source records | Outcome labels and covariates | Controlled access or not shared, pending ethics |

## FAIR Audit

| FAIR element | Current status | Required improvement |
|---|---|---|
| Findable | Derived files have stable project paths but no repository DOI yet | Deposit final package and cite DOI/accession. |
| Accessible | Derived outputs can be shared locally; raw data access route unresolved | Define public vs controlled-access datasets. |
| Interoperable | CSV, XLSX, PNG, TIFF, PDF, SVG, Python scripts, repository README draft, and a 446-field machine-readable data dictionary are standard formats | Finalize repository DOI/licence/access-route fields before public release. |
| Reusable | Locked tables and scripts support reuse, but ethics/licence terms are missing | Add licence for code, data-use terms, and consent-compatible restrictions. |

## Missing Information / Risk Flags

- Ethics approval institution and approval number are missing.
- Informed-consent wording and whether it permits public sharing of derived EEG features are missing.
- The final repository, DOI, licence, and access committee/contact are missing.
- Recruitment dates, acquisition protocol, and raw EEG de-identification route need confirmation.
- The residual threshold of 1.5 is cohort-median-derived in the current project context and should be reported as such, not as an externally validated clinical threshold.
- EEG recording duration is now estimated from EEGLAB metadata, and the manuscript feature pipeline is documented as starting from preprocessed EEGLAB `.set/.fdt` files without additional filtering/artifact rejection. Acquisition hardware, electrode cap/montage description, online reference, raw preprocessing filters, and artifact rejection criteria before export are not yet available in the current manuscript package.

## 中文核对

- 请确认伦理审批单位、审批号和知情同意书中关于数据共享的原文。
- 请决定哪些数据可以公开：原始 EEG、预处理 EEG、PSD/WPLI 派生特征、受试者级预测、统计表、图表源数据、代码。
- 请提供通讯作者或数据访问委员会的联系人，以及是否需要数据使用协议。
- 投稿前建议把派生数据和代码打包到 Zenodo、OSF 或学校机构仓储，并获得 DOI。
