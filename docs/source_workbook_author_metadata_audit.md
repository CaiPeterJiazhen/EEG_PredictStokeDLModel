# Source Workbook Author-Metadata Audit

This audit scans the three current M1 source workbooks for non-identifying evidence relevant to author-supplied submission metadata. It reports keyword hits and workbook cell references, but does not expose patient names or individual source values. A hit is treated as evidence only when it can support a submission-ready protocol statement.

## Summary

- keyword_hits_not_submission_ready: 4
- no_source_workbook_evidence: 6
- partial_source_field_present_not_submission_ready: 4

## Field Audit

| Field | Evidence grade | Keyword hits | Matched terms | Interpretation | Author action |
|---|---|---:|---|---|---|
| ethics_approval | no_source_workbook_evidence | 0 | none | No source-workbook keyword hit can provide a submission-ready ethics statement. | Provide the ethics approval document or author-approved ethics statement. |
| informed_consent | no_source_workbook_evidence | 0 | none | No consent-form text or consent scope is available from the source workbooks. | Provide consent wording, waiver route, and whether data sharing is covered. |
| trial_or_study_registration | no_source_workbook_evidence | 0 | none | No registry identifier is available from the source workbooks. | Provide registry information or an author-approved non-registration statement. |
| study_site_dates_design | keyword_hits_not_submission_ready | 1 | 入组 (1) | Source workbooks contain limited timing-like clinical fields but not recruitment dates, site, or final design wording. | Provide hospital/department, recruitment start/end dates, and prospective/retrospective design. |
| eligibility_stroke_timing | partial_source_field_present_not_submission_ready | 3 | 病程 (2); MMSE (1) | Clinical workbooks contain disease-duration and MMSE-like fields, but not protocol-level eligibility or stroke-subtype criteria. | Provide inclusion/exclusion criteria, subtype criteria, lesion/timing rules, and define what disease duration measures. |
| tacs_device_electrodes | keyword_hits_not_submission_ready | 9 | 刺激 (8); M1 (1) | Source workbooks do not contain device model or electrode dimensions. | Provide device model, electrode size/materials, and confirm target/frequency/intensity/session protocol. |
| concurrent_rehabilitation | no_source_workbook_evidence | 0 | none | No source-workbook field verifies concurrent conventional rehabilitation dose or content. | Confirm whether patients received tACS alone or tACS plus standardized rehabilitation, including dose/content. |
| tacs_safety_adverse_events | partial_source_field_present_not_submission_ready | 6 | 依从 (2); 缺少数据 (1); 脱落 (1); 难受 (1); 高血压 (1) | Source workbooks contain missing-data/dropout notes, including discomfort-related terms, but not a formal safety-monitoring dataset. | Provide an adverse-event/tolerability summary or explicitly state that formal safety data were unavailable. |
| fma_assessors_timing | partial_source_field_present_not_submission_ready | 11 | FMA (5); MBI (4); BBT (2) | Source workbooks contain clinical scale columns but not assessor credentials or blinding. | Provide assessor training/credentials, blinding status, and assessment timing language. |
| eeg_hardware_reference_impedance | keyword_hits_not_submission_ready | 11 | 脑电 (10); 帽 (1) | Workbook fields do not verify amplifier, cap, reference, ground, or impedance protocol. | Provide acquisition hardware, cap/montage, reference/ground, impedance, and software details. |
| resting_state_instructions | partial_source_field_present_not_submission_ready | 16 | 眼 (8); 睁眼 (4); 闭眼 (4) | The completeness workbook records EO/EC task availability but not acquisition instructions. | Provide EO/EC order, fixation or eye instruction, target duration, and drowsiness monitoring. |
| raw_eeg_preprocessing | no_source_workbook_evidence | 0 | none | Source workbooks do not document upstream raw EEG preprocessing before the provided EEGLAB files. | Provide raw preprocessing protocol before .set/.fdt export. |
| data_repository_doi_scope | keyword_hits_not_submission_ready | 1 | 数据 (1) | No final repository DOI, licence, or controlled-access route is present in the source workbooks. | Provide repository DOI/accession, public data scope, restricted data scope, licence, and access route. |
| funding_competing_acknowledgements | no_source_workbook_evidence | 0 | none | No funding, competing-interest, or acknowledgement evidence is present in the source workbooks. | Provide author-approved funding, competing-interest, and acknowledgement statements. |

## Interpretation

- Workbook fields can support selected descriptive statements, such as disease-duration availability, clinical-scale availability, EO/EC task completeness, and missing-data/dropout note categories.
- The source workbooks do not contain submission-ready ethics, consent, registration, EEG acquisition hardware, upstream raw preprocessing, conventional-rehabilitation dose, repository DOI/licence, funding, competing-interest, or authorship evidence.
- The final manuscript should keep these items as author-confirmed fields unless the author supplies protocol, ethics, device, rehabilitation, or repository documents.
