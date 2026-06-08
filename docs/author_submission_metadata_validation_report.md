# Author Submission Metadata Validation Report

Metadata file: `F:\CJZProjectFile\EEG_PredictStokeDLModel\outputs\manuscript_package\author_submission_metadata_from_intake.json`

This report validates whether author-approved metadata are ready for final insertion into the manuscript, declarations, cover letter, repository README, and submission checklist. It does not verify the truth of supplied values; ethics, consent, registration, safety, and repository statements still require author and institutional confirmation.

## Summary

- Fields checked: 19
- Complete fields: 1
- Incomplete blocking or blocking-if-applicable fields: 12
- Incomplete high-priority fields: 6
- Final insertion readiness: no

## Blocking Fields

| Field | Missing required keys | Alternative requirement | Replacement action |
|---|---|---|---|
| ethics_approval | ethics_committee_name, approval_number, approval_date, applicable_site_or_sites, ready_to_paste_statement_en, evidence_source | none | Replace author-query text with a verified ethics approval statement. |
| informed_consent | consent_route, who_provided_consent, written_or_waived, covered_eeg_tacs_clinical_assessments, covered_data_sharing, ready_to_paste_statement_en, evidence_source | none | Select the correct consent template and adapt data-sharing restrictions accordingly. |
| trial_or_study_registration | ready_to_paste_statement_en, evidence_source | provide one of: registry_name + registration_number OR non_registration_reason_if_applicable | Insert registration number or explicit non-registration statement without implying a registered trial if none exists. |
| study_site_dates_design | hospital_or_department, recruitment_start_date, recruitment_end_date, follow_up_or_last_assessment_window, prospective_or_retrospective, single_or_multicentre, ready_to_paste_methods_sentence_en, evidence_source | none | Add a concise study-design sentence and site/date window before the cohort-count paragraph. |
| eligibility_stroke_timing | inclusion_criteria, exclusion_criteria, stroke_subtype_criteria, time_since_stroke_to_eeg, evidence_source | none | Insert protocol-level eligibility and timing details; preserve limitation if any source detail remains unavailable. |
| tacs_device_electrodes | device_model, electrode_dimensions, confirmed_target_frequency_intensity_duration_sessions, evidence_source | none | Append missing device/electrode details to the existing tACS protocol sentence. |
| concurrent_rehabilitation | was_conventional_rehabilitation_delivered, frequency, duration_per_session, main_training_content, ready_to_paste_methods_sentence_en, evidence_source | none | Clarify whether the endpoint reflects tACS alone or tACS plus conventional rehabilitation. |
| tacs_safety_adverse_events | adverse_event_summary, tolerability_summary, withdrawals_or_discontinuations, ready_to_paste_statement_en, evidence_source | none | Add a short safety/tolerability paragraph or explicitly state that safety data were not available. |
| eeg_hardware_reference_impedance | amplifier_model, acquisition_software, cap_system, original_channel_count, online_reference, ground, impedance_threshold, evidence_source | none | Add acquisition paragraph before the sentence stating analyses began from preprocessed EEGLAB files. |
| raw_eeg_preprocessing | raw_filtering, notch_filtering, rereference, artifact_rejection, ica_or_eye_muscle_artifact_handling, bad_channel_handling, eeglab_set_fdt_export_rules, evidence_source | none | Replace the current missing-protocol caveat with verified upstream preprocessing details while retaining the verified feature-pipeline boundary. |
| data_repository_doi_scope | repository_name, doi_or_accession, version, data_license, public_data_scope, restricted_data_scope, controlled_access_contact_or_committee, request_review_requirements, ready_to_paste_data_availability_en, evidence_source | none | Replace repository placeholders and select the correct public-vs-controlled Data Availability template. |
| code_repository_license | code_repository_url_or_doi, version_or_commit_hash, code_license, ready_to_paste_code_availability_en, evidence_source | none | Replace code placeholders and add final commit/version information. |

## High-Priority Fields

| Field | Missing required keys | Alternative requirement | Replacement action |
|---|---|---|---|
| target_journal_reference_style | target_journal, article_type, reference_style, evidence_source | none | Confirm whether JNE remains the first target. Reformat references only after the final journal is selected. |
| author_list_affiliations | author_order_full_names, affiliations, corresponding_author_name, corresponding_author_email, evidence_source | none | Insert final author metadata consistently across all submission surfaces. |
| author_contributions | credit_roles_by_author, all_authors_approved_final_manuscript, evidence_source | none | Replace contribution template with author-approved role assignments. |
| fma_assessors_timing | assessor_training_or_credentials, assessor_blinding, baseline_assessment_timing, post_treatment_assessment_timing, evidence_source | none | Add assessment-procedure sentence before the proportional-recovery formula. |
| resting_state_instructions | eyes_open_eyes_closed_order, target_duration_per_condition, fixation_or_eye_instruction, evidence_source | none | Add task-instruction sentence near the EO/EC file-name state-rule sentence. |
| funding_competing_acknowledgements | funding_sources_and_grant_numbers, competing_interests_statement, evidence_source | none | Replace declaration templates with author-approved final statements. |

## Full Field Map

| Field | Priority | Status | Missing required keys | Alternative requirement | Target files |
|---|---|---|---|---|---|
| target_journal_reference_style | high | incomplete | target_journal, article_type, reference_style, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/cover_letter_template.md; docs/jne_submission_checklist.md |
| author_list_affiliations | high | incomplete | author_order_full_names, affiliations, corresponding_author_name, corresponding_author_email, evidence_source | none | docs/cover_letter_template.md; manuscript title page if added; submission portal fields |
| author_contributions | high | incomplete | credit_roles_by_author, all_authors_approved_final_manuscript, evidence_source | none | docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final declarations |
| ethics_approval | blocking | incomplete | ethics_committee_name, approval_number, approval_date, applicable_site_or_sites, ready_to_paste_statement_en, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| informed_consent | blocking | incomplete | consent_route, who_provided_consent, written_or_waived, covered_eeg_tacs_clinical_assessments, covered_data_sharing, ready_to_paste_statement_en, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/data_availability_and_fair_audit.md |
| trial_or_study_registration | blocking_if_applicable | incomplete | ready_to_paste_statement_en, evidence_source | provide one of: registry_name + registration_number OR non_registration_reason_if_applicable | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| study_site_dates_design | blocking | incomplete | hospital_or_department, recruitment_start_date, recruitment_end_date, follow_up_or_last_assessment_window, prospective_or_retrospective, single_or_multicentre, ready_to_paste_methods_sentence_en, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/author_required_information_form.md |
| eligibility_stroke_timing | blocking | incomplete | inclusion_criteria, exclusion_criteria, stroke_subtype_criteria, time_since_stroke_to_eeg, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/tripod_ai_reporting_checklist.md |
| tacs_device_electrodes | blocking | incomplete | device_model, electrode_dimensions, confirmed_target_frequency_intensity_duration_sessions, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| concurrent_rehabilitation | blocking | incomplete | was_conventional_rehabilitation_delivered, frequency, duration_per_session, main_training_content, ready_to_paste_methods_sentence_en, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/target_journal_strategy.md |
| tacs_safety_adverse_events | blocking | incomplete | adverse_event_summary, tolerability_summary, withdrawals_or_discontinuations, ready_to_paste_statement_en, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md |
| fma_assessors_timing | high | incomplete | assessor_training_or_credentials, assessor_blinding, baseline_assessment_timing, post_treatment_assessment_timing, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md |
| eeg_hardware_reference_impedance | blocking | incomplete | amplifier_model, acquisition_software, cap_system, original_channel_count, online_reference, ground, impedance_threshold, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/eeg_metadata_audit.md |
| resting_state_instructions | high | incomplete | eyes_open_eyes_closed_order, target_duration_per_condition, fixation_or_eye_instruction, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md |
| raw_eeg_preprocessing | blocking | incomplete | raw_filtering, notch_filtering, rereference, artifact_rejection, ica_or_eye_muscle_artifact_handling, bad_channel_handling, eeglab_set_fdt_export_rules, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/methods_gap_resolution_from_project_files.md |
| data_repository_doi_scope | blocking | incomplete | repository_name, doi_or_accession, version, data_license, public_data_scope, restricted_data_scope, controlled_access_contact_or_committee, request_review_requirements, ready_to_paste_data_availability_en, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/data_availability_and_fair_audit.md; docs/repository_readme_for_deposit.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| code_repository_license | blocking | incomplete | code_repository_url_or_doi, version_or_commit_hash, code_license, ready_to_paste_code_availability_en, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/repository_readme_for_deposit.md |
| funding_competing_acknowledgements | high | incomplete | funding_sources_and_grant_numbers, competing_interests_statement, evidence_source | none | docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final manuscript declarations |
| suggested_opposed_reviewers | optional | complete | none | none | docs/cover_letter_template.md; submission portal |

## Use After Completion

1. Fill `docs/author_submission_metadata_template.json` with verified author-approved information.
2. Run `python scripts/63_validate_author_submission_metadata.py --strict`.
3. If strict validation passes, use `results/tables/author_field_replacement_map.csv` to replace the corresponding manuscript, declaration, cover-letter, and repository fields.
4. Regenerate DOCX files, source-data workbook, visual QA, artifact audit, and the submission package.
