# Author Minimal Completion Pack

This pack compresses the final author-metadata blocker into the smallest actionable answer set. It is generated from the same validation rules used by `scripts/63_validate_author_submission_metadata.py`, so it should stay aligned with the strict submission gate.

## Summary

- Current metadata source: `F:\CJZProjectFile\EEG_PredictStokeDLModel\outputs\manuscript_package\author_submission_metadata_from_intake.json`
- Fields needing author action: 18
- Blocking or blocking-if-applicable fields: 12
- High-priority fields: 6
- Minimal answer JSON skeleton: `outputs/manuscript_package/author_minimal_completion_answers.json`
- Machine-readable CSV: `results/tables/author_minimal_completion_pack.csv`

## How To Use

1. Use the tables below to answer each field with verified author, ethics, protocol, device, repository, or institutional evidence.
2. Review project-prefill suggestions, but copy them only after the corresponding author confirms they are accurate.
3. For the fastest plain-text route, fill `docs/author_quick_response_request_zh.md` or copy its 18 numbered fields into a reply.
4. For the fastest XLSX route, fill `Minimal_Completion.author_response` and `Minimal_Completion.evidence_source` in `outputs/manuscript_package/Author_Submission_Metadata_Intake.xlsx`; columns N:P provide formula-based completion status, should not be edited manually, and are ignored by the importer.
5. Alternatively, transfer approved answers into `docs/author_submission_metadata_template.json` or the full `Author_Input` sheet.
6. Run `python scripts/66_import_author_metadata_intake_workbook.py`, then `python scripts/63_validate_author_submission_metadata.py --strict` and `python scripts/64_build_author_metadata_insertion_protocol.py --strict`.
7. Only after both strict checks pass should clean manuscripts, declarations, Data Availability, Code Availability, cover letter, and repository README be finalized.

## Blocking Fields

| Field | Author action | Required keys still to fill | Project-prefill suggestions | Target files |
|---|---|---|---|---|
| code_repository_license | 提供代码仓储 URL/DOI、版本或 commit hash、代码许可证和可投稿英文声明。 | code_repository_url_or_doi, version_or_commit_hash, code_license, ready_to_paste_code_availability_en, evidence_source, environment_or_runtime_notes | evidence_source: docs/repository_readme_for_deposit.md; outputs/submission_package_20260601/submission_package_manifest.csv; environment_or_runtime_notes: The local package includes scripts for feature extraction, model evaluation, statistical validation, figure generation, source-... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/repository_readme_for_deposit.md |
| concurrent_rehabilitation | 确认是否并行常规康复，及其频率、单次时长、训练内容和一致性。 | was_conventional_rehabilitation_delivered, frequency, duration_per_session, main_training_content, ready_to_paste_methods_sentence_en, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/target_journal_strategy.md |
| data_repository_doi_scope | 提供数据仓储平台、DOI/访问号、版本、许可、公开范围和受控访问流程。 | repository_name, doi_or_accession, version, data_license, public_data_scope, restricted_data_scope, controlled_access_contact_or_committee, request_review_requirements, ready_to_paste_data_availability_en, evidence_source | public_data_scope: Prepared public/derived materials include subject-level derived analysis tables, locked LOSO predictions, bootstrap/permutation...; restricted_data_scope: Raw EEG recordings, minimally processed EEG files, identifiable clinical source records, and directly linkable participant-leve...; evidence_source: docs/data_availability_and_fair_audit.md; docs/repository_readme_for_deposit.md; docs/data_availability_and_fair_audit.md; docs... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/data_availability_and_fair_audit.md; docs/repository_readme_for_deposit.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| eeg_hardware_reference_impedance | 提供 EEG 放大器、采集软件、电极帽、在线参考、地线和阻抗阈值。 | amplifier_model, acquisition_software, cap_system, original_channel_count, online_reference, ground, impedance_threshold, evidence_source, raw_cnt_header_boundary_project_evidence | original_channel_count: Project design describes 64-channel EEG; the current analysis retained 62 channels after M1/M2 removal.; evidence_source: F:/CJZFile/EEG_M1/Patient_tACS_M1_EEG representative raw CNT headers; docs/methods_detail_provenance.md; docs/project_context.m...; raw_cnt_header_boundary_project_evidence: Representative raw CNT headers expose Version 3.0 and extended 10-20/10-10 style labels including AF3, AF4, CP3, CPZ, FC3, FCZ,... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/eeg_metadata_audit.md |
| eligibility_stroke_timing | 提供纳入排除标准、卒中亚型规则和发病至 EEG/tACS 的时间定义。 | inclusion_criteria, exclusion_criteria, stroke_subtype_criteria, time_since_stroke_to_eeg, evidence_source, source_workbook_clinical_fields_project_evidence | time_since_stroke_to_eeg: The M1-group clinical source workbook contains a disease-duration field. In the final 19-patient supervised cohort, disease dur...; evidence_source: F:/CJZFile/EEG_M1/脑卒中患者信息记录表.xlsx; F:/CJZFile/EEG_M1/19例患者脑电数据完整性检查.xlsx; docs/cohort_characteristics.md; F:/CJZFile/EEG_M1/脑卒中...; source_workbook_clinical_fields_project_evidence: The M1 patient-information workbook contains non-identifying clinical columns for subject ID, age, disease duration, sex, affec... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/tripod_ai_reporting_checklist.md |
| ethics_approval | 提供伦理委员会全称、批件号、批准日期、适用地点和可投稿英文伦理声明。 | ethics_committee_name, approval_number, approval_date, applicable_site_or_sites, ready_to_paste_statement_en, evidence_source, patient_record_pdf_text_audit_project_evidence | evidence_source: docs/patient_record_pdf_text_audit.md; results/tables/patient_record_pdf_text_audit.csv; patient_record_pdf_text_audit_project_evidence: A non-identifying audit of local patient/healthy record-book PDFs inspected 31 PDFs and 277 pages. Extractable text was minimal... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| informed_consent | 说明知情同意路径、签署对象、书面/豁免状态、覆盖 EEG/tACS/临床评估和数据共享的范围。 | consent_route, who_provided_consent, written_or_waived, covered_eeg_tacs_clinical_assessments, covered_data_sharing, ready_to_paste_statement_en, evidence_source, patient_record_pdf_text_audit_project_evidence | evidence_source: docs/patient_record_pdf_text_audit.md; results/tables/patient_record_pdf_text_audit.csv; patient_record_pdf_text_audit_project_evidence: A non-identifying audit of local patient/healthy record-book PDFs inspected 31 PDFs and 277 pages. Extractable text was minimal... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/data_availability_and_fair_audit.md |
| raw_eeg_preprocessing | 提供导出 .set/.fdt 前的滤波、陷波、重参考、坏道、ICA/伪迹和导出规则。 | raw_filtering, notch_filtering, rereference, artifact_rejection, ica_or_eye_muscle_artifact_handling, bad_channel_handling, eeglab_set_fdt_export_rules, evidence_source, segmentation_or_continuous_export_rules, raw_cnt_header_boundary_project_evidence | eeglab_set_fdt_export_rules: Manuscript analyses start from project-provided preprocessed EEGLAB .set/.fdt files. After loading those files, the feature pip...; evidence_source: F:/CJZFile/EEG_M1/Patient_tACS_M1_EEG representative raw CNT headers; docs/methods_detail_provenance.md; docs/eeg_metadata_audi...; segmentation_or_continuous_export_rules: The current indexed EEGLAB files are continuous one-trial recordings loaded as channels by samples.; raw_cnt_header_boundary_project_evidence: Representative raw CNT headers expose Version 3.0 and extended 10-20/10-10 style labels including AF3, AF4, CP3, CPZ, FC3, FCZ,... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/methods_gap_resolution_from_project_files.md |
| study_site_dates_design | 提供医院/科室、招募起止日期、末次评估窗口、前瞻/回顾和单/多中心设计。 | hospital_or_department, recruitment_start_date, recruitment_end_date, follow_up_or_last_assessment_window, prospective_or_retrospective, single_or_multicentre, ready_to_paste_methods_sentence_en, evidence_source, patient_raw_cnt_recording_window_project_evidence | follow_up_or_last_assessment_window: The project design defines outcome assessment immediately after the final tACS session; no longer-term follow-up window is docu...; evidence_source: F:/CJZFile/EEG_M1/Patient_tACS_M1_EEG raw CNT headers and file inventory; tacs_eeg_proportional_recovery_project_design.md; doc...; patient_raw_cnt_recording_window_project_evidence: Raw patient CNT files currently available under Patient_tACS_M1_EEG span 2024-01-12 to 2025-08-14 across 379 files (即时 83, 基线 1... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/author_required_information_form.md |
| tacs_device_electrodes | 补充 tACS 设备型号、电极尺寸/材料，并确认当前刺激方案。 | device_model, electrode_dimensions, confirmed_target_frequency_intensity_duration_sessions, evidence_source | confirmed_target_frequency_intensity_duration_sessions: Project records define a common tACS protocol: stimulation over contralateral M1, C3 for right-hand impairment and C4 for left-...; evidence_source: docs/methods_detail_provenance.md; tacs_eeg_proportional_recovery_project_design.md | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |
| tacs_safety_adverse_events | 提供不良事件、耐受性、退出/中止和安全监测方法的汇总。 | adverse_event_summary, tolerability_summary, withdrawals_or_discontinuations, ready_to_paste_statement_en, evidence_source, missing_data_note_counts_project_evidence, patient_record_pdf_text_audit_project_evidence | withdrawals_or_discontinuations: The M1-group patient-information source workbook records 29 M1-group entries and 9 rows with missing EEG or follow-up data. Rec...; evidence_source: F:/CJZFile/EEG_M1/脑卒中患者信息记录表.xlsx; docs/patient_record_pdf_text_audit.md; results/tables/patient_record_pdf_text_audit.csv; missing_data_note_counts_project_evidence: The M1 patient-information workbook contains 29 M1 rows; 9 rows have nonblank missing-data notes and 9 rows have nonblank dropo...; patient_record_pdf_text_audit_project_evidence: A non-identifying audit of local patient/healthy record-book PDFs inspected 31 PDFs and 277 pages. Extractable text was minimal... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md |
| trial_or_study_registration | 提供注册平台和注册号；如果未注册，提供作者认可的未注册说明。 | ready_to_paste_statement_en, evidence_source, non_registration_reason_if_applicable, registration_number, registry_name | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md |

## High-Priority Fields

| Field | Author action | Required keys still to fill | Project-prefill suggestions | Target files |
|---|---|---|---|---|
| author_contributions | 按 CRediT 或目标期刊格式填写每位作者贡献，并确认所有作者批准终稿。 | credit_roles_by_author, all_authors_approved_final_manuscript, evidence_source | none | docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final declarations |
| author_list_affiliations | 提供最终作者顺序、单位、通讯作者和邮箱。 | author_order_full_names, affiliations, corresponding_author_name, corresponding_author_email, evidence_source | none | docs/cover_letter_template.md; manuscript title page if added; submission portal fields |
| fma_assessors_timing | 补充 FMA-UE 评估者资质、盲法状态和治疗前后评估时间点。 | assessor_training_or_credentials, assessor_blinding, baseline_assessment_timing, post_treatment_assessment_timing, evidence_source, fma_ue_version_or_scoring_reference, source_workbook_scale_fields_project_evidence | baseline_assessment_timing: Baseline FMA-UE was assessed before the tACS intervention and before baseline EEG feature extraction.; post_treatment_assessment_timing: Post-treatment FMA-UE was assessed after the final tACS session according to the current project context.; evidence_source: F:/CJZFile/EEG_M1/脑卒中患者信息记录表.xlsx; F:/CJZFile/EEG_M1/M1组病历记录表.xlsx; docs/methods_detail_provenance.md; docs/project_context.md;...; fma_ue_version_or_scoring_reference: The current label-generation code and project context use the FMA-UE upper-extremity maximum score of 66 for proportional-recov...; source_workbook_scale_fields_project_evidence: Clinical source workbooks contain scale fields including MMSE, 治疗前BBT, 治疗前FMA, 治疗前MBI, 治疗后BBT, 治疗后FMA, 治疗后MBI. They do not iden... | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md |
| funding_competing_acknowledgements | 提供基金号、资助方角色、利益冲突和致谢内容。 | funding_sources_and_grant_numbers, competing_interests_statement, evidence_source | none | docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final manuscript declarations |
| resting_state_instructions | 确认睁眼/闭眼顺序、目标时长、注视说明和困倦监测。 | eyes_open_eyes_closed_order, target_duration_per_condition, fixation_or_eye_instruction, evidence_source | eyes_open_eyes_closed_order: Current project context maps baseline *1.set files to eyes-open resting state and *2.set files to eyes-closed resting state. Th...; target_duration_per_condition: Across the 38 supervised baseline EO/EC files, recording duration averaged 188.4 s and ranged from 101.0 to 247.8 s.; evidence_source: docs/project_context.md; docs/eeg_metadata_audit.md; results/tables/eeg_recording_metadata_audit.csv; docs/eeg_metadata_audit.md | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md |
| target_journal_reference_style | 确认最终投稿期刊、文章类型和参考文献格式。 | target_journal, article_type, reference_style, evidence_source | none | docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/cover_letter_template.md; docs/jne_submission_checklist.md |

## Field Details

### author_contributions

- Priority: `high`
- Author action: 按 CRediT 或目标期刊格式填写每位作者贡献，并确认所有作者批准终稿。
- Replacement target: Replace contribution template with author-approved role assignments.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| credit_roles_by_author | yes | blank | none | Author-approved response required for final insertion. |
| all_authors_approved_final_manuscript | yes | blank | none | Author-approved response required for final insertion. |
| evidence_source | yes | blank | none | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |

### author_list_affiliations

- Priority: `high`
- Author action: 提供最终作者顺序、单位、通讯作者和邮箱。
- Replacement target: Insert final author metadata consistently across all submission surfaces.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| author_order_full_names | yes | blank | none | Author-approved response required for final insertion. |
| affiliations | yes | blank | none | Author-approved response required for final insertion. |
| corresponding_author_name | yes | blank | none | Author-approved response required for final insertion. |
| corresponding_author_email | yes | blank | none | Author-approved response required for final insertion. |
| evidence_source | yes | blank | none | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |

### code_repository_license

- Priority: `blocking`
- Author action: 提供代码仓储 URL/DOI、版本或 commit hash、代码许可证和可投稿英文声明。
- Replacement target: Replace code placeholders and add final commit/version information.
- Statement pattern: Example pattern only: The analysis code is available at [repository URL or DOI], version [tag/commit], under [software licence].

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| code_repository_url_or_doi | yes | blank | none | Author-approved response required for final insertion. |
| version_or_commit_hash | yes | blank | none | Author-approved response required for final insertion. |
| code_license | yes | blank | none | Author-approved response required for final insertion. |
| ready_to_paste_code_availability_en | yes | blank | none | 含仓储、版本和许可的最终英文 Code Availability 句子。 |
| evidence_source | yes | blank | docs/repository_readme_for_deposit.md; outputs/submission_package_20260601/submission_package_manifest.csv | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| environment_or_runtime_notes | project_prefill_only | blank | The local package includes scripts for feature extraction, model evaluation, statistical validation, figure generation, source-data assembly, manuscript gene... | Author-approved response required for final insertion. |

### concurrent_rehabilitation

- Priority: `blocking`
- Author action: 确认是否并行常规康复，及其频率、单次时长、训练内容和一致性。
- Replacement target: Clarify whether the endpoint reflects tACS alone or tACS plus conventional rehabilitation.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| was_conventional_rehabilitation_delivered | yes | blank | none | Author-approved response required for final insertion. |
| frequency | yes | blank | none | Author-approved response required for final insertion. |
| duration_per_session | yes | blank | none | Author-approved response required for final insertion. |
| main_training_content | yes | blank | none | Author-approved response required for final insertion. |
| ready_to_paste_methods_sentence_en | yes | blank | none | 可直接放入 Methods 的最终英文句子。 |
| evidence_source | yes | blank | none | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |

### data_repository_doi_scope

- Priority: `blocking`
- Author action: 提供数据仓储平台、DOI/访问号、版本、许可、公开范围和受控访问流程。
- Replacement target: Replace repository placeholders and select the correct public-vs-controlled Data Availability template.
- Statement pattern: Controlled-access pattern only: Human-participant raw or linkable data are not publicly available because [privacy/consent/ethics reason]. Qualified researchers may request access from [institution/committee] subject to [ethics approval/data-use agreement]. Derived source data are available at [repository DOI/accession].

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| repository_name | yes | blank | none | Author-approved response required for final insertion. |
| doi_or_accession | yes | blank | none | Author-approved response required for final insertion. |
| version | yes | blank | none | Author-approved response required for final insertion. |
| data_license | yes | blank | none | Author-approved response required for final insertion. |
| public_data_scope | yes | blank | Prepared public/derived materials include subject-level derived analysis tables, locked LOSO predictions, bootstrap/permutation/paired-comparison outputs, fi... | Author-approved response required for final insertion. |
| restricted_data_scope | yes | blank | Raw EEG recordings, minimally processed EEG files, identifiable clinical source records, and directly linkable participant-level source files remain restrict... | Author-approved response required for final insertion. |
| controlled_access_contact_or_committee | yes | blank | none | Author-approved response required for final insertion. |
| request_review_requirements | yes | blank | none | Author-approved response required for final insertion. |
| ready_to_paste_data_availability_en | yes | blank | none | 含 DOI、访问路径和限制原因的最终英文 Data Availability 句子。 |
| evidence_source | yes | blank | docs/data_availability_and_fair_audit.md; docs/repository_readme_for_deposit.md; docs/data_availability_and_fair_audit.md; docs/repository_readme_for_deposit... | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |

### eeg_hardware_reference_impedance

- Priority: `blocking`
- Author action: 提供 EEG 放大器、采集软件、电极帽、在线参考、地线和阻抗阈值。
- Replacement target: Add acquisition paragraph before the sentence stating analyses began from preprocessed EEGLAB files.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| amplifier_model | yes | blank | none | Author-approved response required for final insertion. |
| acquisition_software | yes | blank | none | Author-approved response required for final insertion. |
| cap_system | yes | blank | none | Author-approved response required for final insertion. |
| original_channel_count | yes | blank | Project design describes 64-channel EEG; the current analysis retained 62 channels after M1/M2 removal. | Author-approved response required for final insertion. |
| online_reference | yes | blank | none | Author-approved response required for final insertion. |
| ground | yes | blank | none | Author-approved response required for final insertion. |
| impedance_threshold | yes | blank | none | Author-approved response required for final insertion. |
| evidence_source | yes | blank | F:/CJZFile/EEG_M1/Patient_tACS_M1_EEG representative raw CNT headers; docs/methods_detail_provenance.md; docs/project_context.md; configs/channel_mapping.yaml | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| raw_cnt_header_boundary_project_evidence | project_prefill_only | blank | Representative raw CNT headers expose Version 3.0 and extended 10-20/10-10 style labels including AF3, AF4, CP3, CPZ, FC3, FCZ, FP1, FP2, FPZ. The inspected ... | Author-approved response required for final insertion. |

### eligibility_stroke_timing

- Priority: `blocking`
- Author action: 提供纳入排除标准、卒中亚型规则和发病至 EEG/tACS 的时间定义。
- Replacement target: Insert protocol-level eligibility and timing details; preserve limitation if any source detail remains unavailable.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| inclusion_criteria | yes | blank | none | Author-approved response required for final insertion. |
| exclusion_criteria | yes | blank | none | Author-approved response required for final insertion. |
| stroke_subtype_criteria | yes | blank | none | Author-approved response required for final insertion. |
| time_since_stroke_to_eeg | yes | blank | The M1-group clinical source workbook contains a disease-duration field. In the final 19-patient supervised cohort, disease duration averaged 36.3 days, medi... | Author-approved response required for final insertion. |
| evidence_source | yes | blank | F:/CJZFile/EEG_M1/脑卒中患者信息记录表.xlsx; F:/CJZFile/EEG_M1/19例患者脑电数据完整性检查.xlsx; docs/cohort_characteristics.md; F:/CJZFile/EEG_M1/脑卒中患者信息记录表.xlsx; F:/CJZFile/EEG_M... | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| source_workbook_clinical_fields_project_evidence | project_prefill_only | blank | The M1 patient-information workbook contains non-identifying clinical columns for subject ID, age, disease duration, sex, affected side, pre/post FMA, pre/po... | Author-approved response required for final insertion. |

### ethics_approval

- Priority: `blocking`
- Author action: 提供伦理委员会全称、批件号、批准日期、适用地点和可投稿英文伦理声明。
- Replacement target: Replace author-query text with a verified ethics approval statement.
- Statement pattern: Example pattern only: The study was approved by [ethics committee] ([approval number], [approval date]) and was conducted in accordance with the Declaration of Helsinki.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| ethics_committee_name | yes | blank | none | Author-approved response required for final insertion. |
| approval_number | yes | blank | none | Author-approved response required for final insertion. |
| approval_date | yes | blank | none | Author-approved response required for final insertion. |
| applicable_site_or_sites | yes | blank | none | Author-approved response required for final insertion. |
| ready_to_paste_statement_en | yes | blank | none | 可直接放入 manuscript/declarations/cover letter 的最终英文句子。 |
| evidence_source | yes | blank | docs/patient_record_pdf_text_audit.md; results/tables/patient_record_pdf_text_audit.csv | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| patient_record_pdf_text_audit_project_evidence | project_prefill_only | blank | A non-identifying audit of local patient/healthy record-book PDFs inspected 31 PDFs and 277 pages. Extractable text was minimal (434 compact characters total... | Author-approved response required for final insertion. |

### fma_assessors_timing

- Priority: `high`
- Author action: 补充 FMA-UE 评估者资质、盲法状态和治疗前后评估时间点。
- Replacement target: Add assessment-procedure sentence before the proportional-recovery formula.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| assessor_training_or_credentials | yes | blank | none | Author-approved response required for final insertion. |
| assessor_blinding | yes | blank | none | Author-approved response required for final insertion. |
| baseline_assessment_timing | yes | blank | Baseline FMA-UE was assessed before the tACS intervention and before baseline EEG feature extraction. | Author-approved response required for final insertion. |
| post_treatment_assessment_timing | yes | blank | Post-treatment FMA-UE was assessed after the final tACS session according to the current project context. | Author-approved response required for final insertion. |
| evidence_source | yes | blank | F:/CJZFile/EEG_M1/脑卒中患者信息记录表.xlsx; F:/CJZFile/EEG_M1/M1组病历记录表.xlsx; docs/methods_detail_provenance.md; docs/project_context.md; docs/project_context.md; src/... | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| fma_ue_version_or_scoring_reference | project_prefill_only | blank | The current label-generation code and project context use the FMA-UE upper-extremity maximum score of 66 for proportional-recovery calculations. | Author-approved response required for final insertion. |
| source_workbook_scale_fields_project_evidence | project_prefill_only | blank | Clinical source workbooks contain scale fields including MMSE, 治疗前BBT, 治疗前FMA, 治疗前MBI, 治疗后BBT, 治疗后FMA, 治疗后MBI. They do not identify the assessor credentials ... | Author-approved response required for final insertion. |

### funding_competing_acknowledgements

- Priority: `high`
- Author action: 提供基金号、资助方角色、利益冲突和致谢内容。
- Replacement target: Replace declaration templates with author-approved final statements.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| funding_sources_and_grant_numbers | yes | blank | none | Author-approved response required for final insertion. |
| competing_interests_statement | yes | blank | none | Author-approved response required for final insertion. |
| evidence_source | yes | blank | none | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |

### informed_consent

- Priority: `blocking`
- Author action: 说明知情同意路径、签署对象、书面/豁免状态、覆盖 EEG/tACS/临床评估和数据共享的范围。
- Replacement target: Select the correct consent template and adapt data-sharing restrictions accordingly.
- Statement pattern: Example pattern only: Written informed consent was obtained from all participants or their legally authorised representatives before EEG, tACS and clinical assessment procedures.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| consent_route | yes | blank | none | Author-approved response required for final insertion. |
| who_provided_consent | yes | blank | none | Author-approved response required for final insertion. |
| written_or_waived | yes | blank | none | Author-approved response required for final insertion. |
| covered_eeg_tacs_clinical_assessments | yes | blank | none | Author-approved response required for final insertion. |
| covered_data_sharing | yes | blank | none | Author-approved response required for final insertion. |
| ready_to_paste_statement_en | yes | blank | none | 可直接放入 manuscript/declarations/cover letter 的最终英文句子。 |
| evidence_source | yes | blank | docs/patient_record_pdf_text_audit.md; results/tables/patient_record_pdf_text_audit.csv | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| patient_record_pdf_text_audit_project_evidence | project_prefill_only | blank | A non-identifying audit of local patient/healthy record-book PDFs inspected 31 PDFs and 277 pages. Extractable text was minimal (434 compact characters total... | Author-approved response required for final insertion. |

### raw_eeg_preprocessing

- Priority: `blocking`
- Author action: 提供导出 .set/.fdt 前的滤波、陷波、重参考、坏道、ICA/伪迹和导出规则。
- Replacement target: Replace the current missing-protocol caveat with verified upstream preprocessing details while retaining the verified feature-pipeline boundary.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| raw_filtering | yes | blank | none | Author-approved response required for final insertion. |
| notch_filtering | yes | blank | none | Author-approved response required for final insertion. |
| rereference | yes | blank | none | Author-approved response required for final insertion. |
| artifact_rejection | yes | blank | none | Author-approved response required for final insertion. |
| ica_or_eye_muscle_artifact_handling | yes | blank | none | Author-approved response required for final insertion. |
| bad_channel_handling | yes | blank | none | Author-approved response required for final insertion. |
| eeglab_set_fdt_export_rules | yes | blank | Manuscript analyses start from project-provided preprocessed EEGLAB .set/.fdt files. After loading those files, the feature pipeline performs no additional t... | Author-approved response required for final insertion. |
| evidence_source | yes | blank | F:/CJZFile/EEG_M1/Patient_tACS_M1_EEG representative raw CNT headers; docs/methods_detail_provenance.md; docs/eeg_metadata_audit.md; docs/methods_detail_prov... | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| segmentation_or_continuous_export_rules | project_prefill_only | blank | The current indexed EEGLAB files are continuous one-trial recordings loaded as channels by samples. | Author-approved response required for final insertion. |
| raw_cnt_header_boundary_project_evidence | project_prefill_only | blank | Representative raw CNT headers expose Version 3.0 and extended 10-20/10-10 style labels including AF3, AF4, CP3, CPZ, FC3, FCZ, FP1, FP2, FPZ. The inspected ... | Author-approved response required for final insertion. |

### resting_state_instructions

- Priority: `high`
- Author action: 确认睁眼/闭眼顺序、目标时长、注视说明和困倦监测。
- Replacement target: Add task-instruction sentence near the EO/EC file-name state-rule sentence.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| eyes_open_eyes_closed_order | yes | blank | Current project context maps baseline *1.set files to eyes-open resting state and *2.set files to eyes-closed resting state. The actual acquisition instructi... | Author-approved response required for final insertion. |
| target_duration_per_condition | yes | blank | Across the 38 supervised baseline EO/EC files, recording duration averaged 188.4 s and ranged from 101.0 to 247.8 s. | Author-approved response required for final insertion. |
| fixation_or_eye_instruction | yes | blank | none | Author-approved response required for final insertion. |
| evidence_source | yes | blank | docs/project_context.md; docs/eeg_metadata_audit.md; results/tables/eeg_recording_metadata_audit.csv; docs/eeg_metadata_audit.md | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |

### study_site_dates_design

- Priority: `blocking`
- Author action: 提供医院/科室、招募起止日期、末次评估窗口、前瞻/回顾和单/多中心设计。
- Replacement target: Add a concise study-design sentence and site/date window before the cohort-count paragraph.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| hospital_or_department | yes | blank | none | Author-approved response required for final insertion. |
| recruitment_start_date | yes | blank | none | Author-approved response required for final insertion. |
| recruitment_end_date | yes | blank | none | Author-approved response required for final insertion. |
| follow_up_or_last_assessment_window | yes | blank | The project design defines outcome assessment immediately after the final tACS session; no longer-term follow-up window is documented in the current project ... | Author-approved response required for final insertion. |
| prospective_or_retrospective | yes | blank | none | Author-approved response required for final insertion. |
| single_or_multicentre | yes | blank | none | Author-approved response required for final insertion. |
| ready_to_paste_methods_sentence_en | yes | blank | none | 可直接放入 Methods 的最终英文句子。 |
| evidence_source | yes | blank | F:/CJZFile/EEG_M1/Patient_tACS_M1_EEG raw CNT headers and file inventory; tacs_eeg_proportional_recovery_project_design.md; docs/methods_detail_provenance.md | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| patient_raw_cnt_recording_window_project_evidence | project_prefill_only | blank | Raw patient CNT files currently available under Patient_tACS_M1_EEG span 2024-01-12 to 2025-08-14 across 379 files (即时 83, 基线 107, 最终 79, 阶段 78, 随访1 32); par... | Author-approved response required for final insertion. |

### tacs_device_electrodes

- Priority: `blocking`
- Author action: 补充 tACS 设备型号、电极尺寸/材料，并确认当前刺激方案。
- Replacement target: Append missing device/electrode details to the existing tACS protocol sentence.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| device_model | yes | blank | none | Author-approved response required for final insertion. |
| electrode_dimensions | yes | blank | none | Author-approved response required for final insertion. |
| confirmed_target_frequency_intensity_duration_sessions | yes | blank | Project records define a common tACS protocol: stimulation over contralateral M1, C3 for right-hand impairment and C4 for left-hand impairment, 20 Hz, 1000 m... | Author-approved response required for final insertion. |
| evidence_source | yes | blank | docs/methods_detail_provenance.md; tacs_eeg_proportional_recovery_project_design.md | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |

### tacs_safety_adverse_events

- Priority: `blocking`
- Author action: 提供不良事件、耐受性、退出/中止和安全监测方法的汇总。
- Replacement target: Add a short safety/tolerability paragraph or explicitly state that safety data were not available.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| adverse_event_summary | yes | blank | none | Author-approved response required for final insertion. |
| tolerability_summary | yes | blank | none | Author-approved response required for final insertion. |
| withdrawals_or_discontinuations | yes | blank | The M1-group patient-information source workbook records 29 M1-group entries and 9 rows with missing EEG or follow-up data. Recorded reasons include non-trea... | Author-approved response required for final insertion. |
| ready_to_paste_statement_en | yes | blank | none | 可直接放入 manuscript/declarations/cover letter 的最终英文句子。 |
| evidence_source | yes | blank | F:/CJZFile/EEG_M1/脑卒中患者信息记录表.xlsx; docs/patient_record_pdf_text_audit.md; results/tables/patient_record_pdf_text_audit.csv | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| missing_data_note_counts_project_evidence | project_prefill_only | blank | The M1 patient-information workbook contains 29 M1 rows; 9 rows have nonblank missing-data notes and 9 rows have nonblank dropout/reason notes. Non-identifyi... | Author-approved response required for final insertion. |
| patient_record_pdf_text_audit_project_evidence | project_prefill_only | blank | A non-identifying audit of local patient/healthy record-book PDFs inspected 31 PDFs and 277 pages. Extractable text was minimal (434 compact characters total... | Author-approved response required for final insertion. |

### target_journal_reference_style

- Priority: `high`
- Author action: 确认最终投稿期刊、文章类型和参考文献格式。
- Replacement target: Confirm whether JNE remains the first target. Reformat references only after the final journal is selected.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| target_journal | yes | blank | none | Author-approved response required for final insertion. |
| article_type | yes | blank | none | Author-approved response required for final insertion. |
| reference_style | yes | blank | none | Author-approved response required for final insertion. |
| evidence_source | yes | blank | none | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |

### trial_or_study_registration

- Priority: `blocking_if_applicable`
- Author action: 提供注册平台和注册号；如果未注册，提供作者认可的未注册说明。
- Replacement target: Insert registration number or explicit non-registration statement without implying a registered trial if none exists.
- Statement pattern: Example pattern only: This study was registered at [registry] under [registration number]. If not registered, provide an explicit author-approved non-registration statement.

| Metadata key | Required | Current value | Project prefill suggestion | Guidance |
|---|---|---|---|---|
| ready_to_paste_statement_en | yes | blank | none | 可直接放入 manuscript/declarations/cover letter 的最终英文句子。 |
| evidence_source | yes | blank | none | 支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。 |
| non_registration_reason_if_applicable | alternative | blank | none | Author-approved response required for final insertion. |
| registration_number | alternative | blank | none | Author-approved response required for final insertion. |
| registry_name | alternative | blank | none | Author-approved response required for final insertion. |

