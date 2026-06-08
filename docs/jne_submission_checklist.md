# Journal of Neural Engineering Submission Checklist

Source basis: `docs/target_journal_strategy.md`, public source check dated 2026-06-02. This checklist is a practical upload map for the prepared JNE-first package. CAS partition, JCR quartile, APC agreements, and institutional reimbursement eligibility still require author-side verification.

## Recommended Upload Set

| Item | Prepared file | Current status | Action before upload |
|---|---|---|---|
| Main manuscript, JNE structured abstract | `output/doc/ResidualAware_SSL_CNN_JNE_Structured_Manuscript.docx` | Draft ready | Replace all author-query content with verified ethics, consent, author, funding, and data/code statements. |
| Clean placeholder manuscript | `output/doc/ResidualAware_SSL_CNN_JNE_Clean_Placeholder.docx` | Draft ready | Use as the closest upload candidate after author fields are completed. |
| Supplementary Information | `output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx` | Draft ready | Confirm all supplementary tables and figures remain allowed by the final submission system. |
| Figure exports | `results/figures/nature/` and `results/figures/explainability/mne_wpli_connectivity/` | Draft ready | Upload journal-preferred formats; keep SVG/PDF/TIFF/PNG archive for editorial requests. |
| Source-data workbook | `outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx` | Draft ready; 36 worksheets with a 446-field data dictionary, participant-flow source notes, PROBAST/TRIPOD+AI risk audit, performance precision audit, claim-strength audit, numeric-claim source trace audit, AI model reporting card, source-workbook author metadata audit, patient-record PDF text-layer audit, author-field replacement map, author minimal completion pack, and audit sheets | Confirm whether it should be uploaded as source data, supplementary data, or deposited in a repository. |
| Reference manager file | `docs/citation_artifacts/selected_references.ris` | Draft ready | Reformat references if JNE submission or production requests a different style. |
| Cover letter | `docs/cover_letter_template.md` | JNE-first draft ready | Replace all bracketed placeholders and remove backup-target notes before upload. |
| Final declaration templates | `docs/jne_final_declaration_templates.md` | Ready-to-paste template set | Select the template matching the verified author/institutional facts and replace bracketed fields. |
| Author-required information form | `docs/author_required_information_form.md`; `output/doc/Author_Required_Information_Form.docx` | Author input required | Complete before final manuscript replacement. |
| Target-journal strategy | `docs/target_journal_strategy.md` | Internal decision support | Do not upload unless requested. |
| Audits and QA outputs | `docs/submission_readiness_audit.md`; `docs/submission_artifact_quality_audit.md`; `docs/manuscript_integrity_audit.md`; `docs/docx_visual_qa_report.md` | Internal QA | Keep for author/editorial response; upload only if requested. |

## JNE-Specific Requirements To Check

| Requirement | Current evidence | Status | Author action |
|---|---|---|---|
| Structured abstract headings | JNE manuscript variant uses Objective, Approach, Main results, and Significance. | Ready | Do not rename these headings. |
| Article length/page limit | Current strategy notes JNE papers are normally no more than 12,000 words or 14 journal pages. | Likely ready | Re-check after final author fields and formatting changes. |
| Human-participant ethics | Project files do not provide final ethics committee name, approval number, or approval date. | Blocking | Supply exact ethics approval statement. |
| Declaration of Helsinki wording | Not safely inferable from project files. | Blocking | Confirm whether the study complied with Declaration of Helsinki principles. |
| Informed consent | Consent wording and data-sharing scope are not verified. | Blocking | Supply final informed-consent statement. |
| Trial or study registration | No registration number was found in the project audit. | Blocking if applicable | Provide registration number, or a defensible statement that the study was not registered and why. |
| Data availability | Draft now maps derived tables, locked LOSO predictions, statistical outputs, participant-flow source notes, figure-source summaries, audit tables, PROBAST/TRIPOD+AI risk audit, performance precision audit, claim-strength audit, AI model reporting card, source-workbook author metadata audit, patient-record PDF text-layer audit, author-field replacement mapping, and the 36-worksheet workbook; repository DOI/accession, licence, and restricted-access route are missing. | Blocking | Choose repository and public/restricted data scope. |
| Code availability | Reproducibility scripts cover modelling, statistics, figures, source-data assembly, manuscript generation, audits, and package assembly; public repository, version, and licence are missing. | Blocking | Provide repository URL/DOI and code licence. |
| Competing interests | Not verified. | Blocking | Confirm no competing interests or provide statement. |
| Funding | Not verified. | Blocking | Provide grant names/numbers or state no specific funding. |
| Author list and affiliations | Not present in final form. | Blocking | Provide final author order, affiliations, ORCID IDs, and corresponding author details. |
| Author contributions | Not inferable from project files. | Blocking | Provide CRediT or journal-required contribution statement. |
| EEG acquisition details | Feature pipeline details are audited, but hardware, reference, impedance, and pre-`.set` preprocessing remain incomplete. | Blocking for Methods quality | Supply acquisition and preprocessing details from original records. |
| tACS intervention details | Core protocol is available from project notes, but device, electrode dimensions, concurrent rehabilitation, and safety summary remain incomplete. | Blocking for neuromodulation reporting | Confirm or correct all intervention and safety details. |

## Minimum Finalization Sequence

1. Complete `docs/author_required_information_form.md`.
2. Use `docs/jne_final_declaration_templates.md` to draft final ethics, consent, registration, Data Availability, Code Availability, funding, competing-interest, and contribution statements.
3. Replace placeholder declarations in `docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md`.
4. Regenerate the JNE DOCX if manuscript text changes.
5. Re-run manuscript-source integrity, DOCX visual QA, reference metadata audit, and package assembly.
6. Inspect the final DOCX/PDF visually before upload.
7. Upload the JNE manuscript, Supplementary Information, figure files, source data or repository links, cover letter, and any required declarations.

## Claims To Preserve During Submission

- The study is exploratory and based on a supervised cohort of 19 participants.
- There is no external validation cohort.
- The residual-aware SSL-CNN improved ranking/calibration-oriented metrics, but it did not improve hard-label accuracy over the no-SSL CNN.
- The current data do not prove incremental clinical value of EEG over clinical variables.
- Explainability and connectivity findings are hypothesis-generating.
- The residual threshold is cohort-median-derived and should not be presented as a validated clinical cut-off.
