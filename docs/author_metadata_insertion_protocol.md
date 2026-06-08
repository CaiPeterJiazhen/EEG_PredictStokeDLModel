# Author Metadata Final-Insertion Protocol

Metadata file: `F:\CJZProjectFile\EEG_PredictStokeDLModel\outputs\manuscript_package\author_submission_metadata_from_intake.json`

This dry-run protocol maps author-approved metadata to final manuscript, declaration, cover-letter, repository, and submission-checklist replacements. It deliberately does not edit manuscript files. Final replacement should be performed only after strict metadata validation passes and the author confirms the supplied statements.

## Readiness

- Strict validation exit code: 1
- Strict validation ready: no
- Fields with any supplied metadata and evidence source: 0
- Fields blocked from insertion: 19

## Validator Output

```text
F:\CJZProjectFile\EEG_PredictStokeDLModel\docs\author_submission_metadata_validation_report.md
metadata_fields=19
complete_fields=1
incomplete_blocking_fields=12
incomplete_high_priority_fields=6
```

## Blocked Insertions

| Field | Priority | Reason or preview | Target files |
|---|---|---|---|
| target_journal_reference_style | high | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/cover_letter_template.md; docs/jne_submission_checklist.md |
| author_list_affiliations | high | No field-specific author metadata supplied. | docs/cover_letter_template.md; manuscript title page if added; submission portal fields |
| author_contributions | high | No field-specific author metadata supplied. | docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final declarations |
| ethics_approval | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| informed_consent | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/data_availability_and_fair_audit.md |
| trial_or_study_registration | blocking_if_applicable | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| study_site_dates_design | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/author_required_information_form.md |
| eligibility_stroke_timing | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/tripod_ai_reporting_checklist.md |
| tacs_device_electrodes | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| concurrent_rehabilitation | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/target_journal_strategy.md |
| tacs_safety_adverse_events | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md |
| fma_assessors_timing | high | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md |
| eeg_hardware_reference_impedance | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/eeg_metadata_audit.md |
| resting_state_instructions | high | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md |
| raw_eeg_preprocessing | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/methods_gap_resolution_from_project_files.md |
| data_repository_doi_scope | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/data_availability_and_fair_audit.md; docs/repository_readme_for_deposit.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| code_repository_license | blocking | No field-specific author metadata supplied. | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/repository_readme_for_deposit.md |
| funding_competing_acknowledgements | high | No field-specific author metadata supplied. | docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final manuscript declarations |
| suggested_opposed_reviewers | optional | No field-specific author metadata supplied. | docs/cover_letter_template.md; submission portal |

## Ready Insertions

None.

## Final Replacement Workflow

1. Fill `docs/author_submission_metadata_template.json` with author-approved values and evidence sources.
2. Run `python scripts/63_validate_author_submission_metadata.py --strict`.
3. Run `python scripts/64_build_author_metadata_insertion_protocol.py --strict` to produce a ready-only insertion plan.
4. Replace only the fields marked ready, using the target files and replacement actions listed below.
5. Regenerate JNE and clean variants, DOCX files, source-data workbook, visual QA, manuscript integrity audit, artifact audit, and submission package.
6. Confirm no unverified placeholders remain in clean submission files before upload.

## Full Insertion Map

| Field | Priority | Status | Target files | Source metadata keys | Replacement action | Verification |
|---|---|---|---|---|---|---|
| target_journal_reference_style | high | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/cover_letter_template.md; docs/jne_submission_checklist.md | none supplied | Confirm whether JNE remains the first target. Reformat references only after the final journal is selected. | Run reference metadata audit and regenerate DOCX after final style changes. |
| author_list_affiliations | high | blocked | docs/cover_letter_template.md; manuscript title page if added; submission portal fields | none supplied | Insert final author metadata consistently across all submission surfaces. | Check author order, affiliations, ORCID IDs, and corresponding author match across manuscript, cover letter, and portal. |
| author_contributions | high | blocked | docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final declarations | none supplied | Replace contribution template with author-approved role assignments. | Confirm every author has at least one contribution and all initials map to the final author list. |
| ethics_approval | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md | none supplied | Replace author-query text with a verified ethics approval statement. | Run author evidence audit and manuscript-source integrity audit; confirm no unverified ethics placeholders remain. |
| informed_consent | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/data_availability_and_fair_audit.md | none supplied | Select the correct consent template and adapt data-sharing restrictions accordingly. | Confirm Data Availability restrictions match consent wording and repository access route. |
| trial_or_study_registration | blocking_if_applicable | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md | none supplied | Insert registration number or explicit non-registration statement without implying a registered trial if none exists. | Check JNE abstract/end-of-abstract registration requirements if the study qualifies as a clinical trial. |
| study_site_dates_design | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/author_required_information_form.md | none supplied | Add a concise study-design sentence and site/date window before the cohort-count paragraph. | Confirm design wording is consistent with registration, ethics approval, and limitations. |
| eligibility_stroke_timing | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/tripod_ai_reporting_checklist.md | none supplied | Insert protocol-level eligibility and timing details; preserve limitation if any source detail remains unavailable. | Update TRIPOD+AI checklist and rerun manuscript integrity audit. |
| tacs_device_electrodes | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md | none supplied | Append missing device/electrode details to the existing tACS protocol sentence. | Confirm frequency/intensity/session count still match project design and update safety text if needed. |
| concurrent_rehabilitation | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/target_journal_strategy.md | none supplied | Clarify whether the endpoint reflects tACS alone or tACS plus conventional rehabilitation. | Check Introduction/Discussion claims do not imply isolated tACS effects if rehabilitation was concurrent. |
| tacs_safety_adverse_events | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md | none supplied | Add a short safety/tolerability paragraph or explicitly state that safety data were not available. | For neuromodulation journals, confirm safety statement is present before upload. |
| fma_assessors_timing | high | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md | none supplied | Add assessment-procedure sentence before the proportional-recovery formula. | Confirm no post-treatment data leak into model inputs. |
| eeg_hardware_reference_impedance | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/eeg_metadata_audit.md | none supplied | Add acquisition paragraph before the sentence stating analyses began from preprocessed EEGLAB files. | Check acquisition details do not conflict with 250 Hz, 64-channel source design, and 62 retained channels after M1/M2 removal. |
| resting_state_instructions | high | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md | none supplied | Add task-instruction sentence near the EO/EC file-name state-rule sentence. | Confirm state labels and recording durations remain consistent with metadata audit. |
| raw_eeg_preprocessing | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/methods_gap_resolution_from_project_files.md | none supplied | Replace the current missing-protocol caveat with verified upstream preprocessing details while retaining the verified feature-pipeline boundary. | Run methods provenance audit and confirm feature scripts still add no extra filtering/artifact rejection after loading. |
| data_repository_doi_scope | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/data_availability_and_fair_audit.md; docs/repository_readme_for_deposit.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md | none supplied | Replace repository placeholders and select the correct public-vs-controlled Data Availability template. | Confirm DOI resolves, restricted route is specific, and raw/minimally processed EEG is not over-shared. |
| code_repository_license | blocking | blocked | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/repository_readme_for_deposit.md | none supplied | Replace code placeholders and add final commit/version information. | Run package assembly and confirm scripts needed to regenerate tables, figures, workbook, audits, and package are included. |
| funding_competing_acknowledgements | high | blocked | docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final manuscript declarations | none supplied | Replace declaration templates with author-approved final statements. | Confirm all authors have approved final declarations before upload. |
| suggested_opposed_reviewers | optional | blocked | docs/cover_letter_template.md; submission portal | none supplied | Fill only if requested or useful for the target journal; remove template section if not used. | Check suggested reviewers are not close collaborators and meet journal conflict rules. |
