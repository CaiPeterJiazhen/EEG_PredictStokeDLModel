from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "results" / "tables" / "author_field_replacement_map.csv"
OUTPUT_MD = ROOT / "docs" / "author_field_replacement_map.md"


ROWS = [
    {
        "field_id": "target_journal_reference_style",
        "author_input_needed": "Final target journal, article type, target submission date, and reference style.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/cover_letter_template.md; docs/jne_submission_checklist.md",
        "manuscript_section": "Title page/submission metadata; References; Cover letter",
        "replacement_action": "Confirm whether JNE remains the first target. Reformat references only after the final journal is selected.",
        "exact_text_or_placeholder_to_replace": "[Paper / Research Paper]; backup-target notes; any journal-specific reference style if changed",
        "final_text_dependency": "Target journal instructions and author decision.",
        "verification_after_replacement": "Run reference metadata audit and regenerate DOCX after final style changes.",
        "priority": "high",
    },
    {
        "field_id": "author_list_affiliations",
        "author_input_needed": "Final author order, full names, affiliations, ORCID IDs, corresponding author address and email.",
        "current_status": "author_required",
        "target_files": "docs/cover_letter_template.md; manuscript title page if added; submission portal fields",
        "manuscript_section": "Title page; Cover letter; Submission portal",
        "replacement_action": "Insert final author metadata consistently across all submission surfaces.",
        "exact_text_or_placeholder_to_replace": "[Name, degree]; [Institution]; [Postal address]; [Email]; [Phone, optional]; [Corresponding author name]",
        "final_text_dependency": "Author-approved names, affiliations, ORCID IDs, and corresponding-author details.",
        "verification_after_replacement": "Check author order, affiliations, ORCID IDs, and corresponding author match across manuscript, cover letter, and portal.",
        "priority": "high",
    },
    {
        "field_id": "author_contributions",
        "author_input_needed": "CRediT or journal-required contribution roles for every author.",
        "current_status": "author_required",
        "target_files": "docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final declarations",
        "manuscript_section": "Author Contributions",
        "replacement_action": "Replace contribution template with author-approved role assignments.",
        "exact_text_or_placeholder_to_replace": "[Initials] conceived the study... All authors approved the final manuscript.",
        "final_text_dependency": "Author-approved CRediT role matrix.",
        "verification_after_replacement": "Confirm every author has at least one contribution and all initials map to the final author list.",
        "priority": "high",
    },
    {
        "field_id": "ethics_approval",
        "author_input_needed": "Ethics committee name, approval number, approval date, applicable site, and Declaration of Helsinki wording.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md",
        "manuscript_section": "Methods, Study cohort; Declarations; Cover letter",
        "replacement_action": "Replace author-query text with a verified ethics approval statement.",
        "exact_text_or_placeholder_to_replace": "Author information required before submission: institutional review board name, approval number, consent language...; [Ethics Committee or Institutional Review Board name, approval number, approval date]",
        "final_text_dependency": "Ethics approval document or institution-approved wording.",
        "verification_after_replacement": "Run author evidence audit and manuscript-source integrity audit; confirm no unverified ethics placeholders remain.",
        "priority": "blocking",
    },
    {
        "field_id": "informed_consent",
        "author_input_needed": "Consent route, who consented, written/waived status, legally authorized representative wording if relevant, and whether consent covered EEG/tACS/clinical/data sharing.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/data_availability_and_fair_audit.md",
        "manuscript_section": "Declarations; Data Availability; Cover letter",
        "replacement_action": "Select the correct consent template and adapt data-sharing restrictions accordingly.",
        "exact_text_or_placeholder_to_replace": "[written informed consent / consent waiver details]; [ethics approval / consent terms / institutional data-use policy]",
        "final_text_dependency": "Consent form wording and institutional data-sharing permission.",
        "verification_after_replacement": "Confirm Data Availability restrictions match consent wording and repository access route.",
        "priority": "blocking",
    },
    {
        "field_id": "trial_or_study_registration",
        "author_input_needed": "Registry name, registration number, registration date, or author-approved reason for non-registration.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md",
        "manuscript_section": "Abstract if required; Methods; Declarations; Cover letter",
        "replacement_action": "Insert registration number or explicit non-registration statement without implying a registered trial if none exists.",
        "exact_text_or_placeholder_to_replace": "[Add clinical trial or study registration number if applicable, or provide a submission-ready statement explaining why the study was not registered.]",
        "final_text_dependency": "Registry record or author-approved non-registration explanation.",
        "verification_after_replacement": "Check JNE abstract/end-of-abstract registration requirements if the study qualifies as a clinical trial.",
        "priority": "blocking_if_applicable",
    },
    {
        "field_id": "study_site_dates_design",
        "author_input_needed": "Hospital/department, recruitment dates, follow-up window, prospective/retrospective status, single/multicentre status, randomization/blinding status.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/author_required_information_form.md",
        "manuscript_section": "Methods, Study cohort",
        "replacement_action": "Add a concise study-design sentence and site/date window before the cohort-count paragraph.",
        "exact_text_or_placeholder_to_replace": "The supervised cohort included 19 stroke patients...; Author information required before submission...",
        "final_text_dependency": "Clinical protocol or author-approved study-design summary.",
        "verification_after_replacement": "Confirm design wording is consistent with registration, ethics approval, and limitations.",
        "priority": "blocking",
    },
    {
        "field_id": "eligibility_stroke_timing",
        "author_input_needed": "Inclusion criteria, exclusion criteria, stroke subtype, lesion side/type, time from stroke to EEG/tACS, and timing statistics.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/tripod_ai_reporting_checklist.md",
        "manuscript_section": "Methods, Study cohort; Limitations if incomplete",
        "replacement_action": "Insert protocol-level eligibility and timing details; preserve limitation if any source detail remains unavailable.",
        "exact_text_or_placeholder_to_replace": "stroke subtype criteria, time from stroke to EEG, inclusion and exclusion criteria",
        "final_text_dependency": "Clinical protocol, case report form, or source workbook summary.",
        "verification_after_replacement": "Update TRIPOD+AI checklist and rerun manuscript integrity audit.",
        "priority": "blocking",
    },
    {
        "field_id": "tacs_device_electrodes",
        "author_input_needed": "tACS device model, manufacturer, electrode size/shape/material, montage details, ramping if used, impedance/safety settings.",
        "current_status": "partly_resolved",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md",
        "manuscript_section": "Methods, Study cohort; Intervention details",
        "replacement_action": "Append missing device/electrode details to the existing tACS protocol sentence.",
        "exact_text_or_placeholder_to_replace": "The project records define a common tACS protocol...; device and electrode dimensions remain missing.",
        "final_text_dependency": "Stimulation device manual, protocol, or treatment log.",
        "verification_after_replacement": "Confirm frequency/intensity/session count still match project design and update safety text if needed.",
        "priority": "blocking",
    },
    {
        "field_id": "concurrent_rehabilitation",
        "author_input_needed": "Whether conventional rehabilitation was delivered, its frequency, session duration, content, and whether it was standardized across participants.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/target_journal_strategy.md",
        "manuscript_section": "Methods; Limitations; Cover letter if needed",
        "replacement_action": "Clarify whether the endpoint reflects tACS alone or tACS plus conventional rehabilitation.",
        "exact_text_or_placeholder_to_replace": "whether any standardized conventional rehabilitation was delivered concurrently with tACS",
        "final_text_dependency": "Clinical rehabilitation protocol or treatment log.",
        "verification_after_replacement": "Check Introduction/Discussion claims do not imply isolated tACS effects if rehabilitation was concurrent.",
        "priority": "blocking",
    },
    {
        "field_id": "tacs_safety_adverse_events",
        "author_input_needed": "Adverse events, tolerability, withdrawals, safety monitoring, and whether any seizures, skin reactions, headache, fatigue, or discomfort occurred.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md",
        "manuscript_section": "Results or Methods safety statement; Declarations if required",
        "replacement_action": "Add a short safety/tolerability paragraph or explicitly state that safety data were not available.",
        "exact_text_or_placeholder_to_replace": "tACS safety | Adverse events, withdrawals, tolerability summary",
        "final_text_dependency": "Adverse-event log or author-approved safety summary.",
        "verification_after_replacement": "For neuromodulation journals, confirm safety statement is present before upload.",
        "priority": "blocking",
    },
    {
        "field_id": "fma_assessors_timing",
        "author_input_needed": "FMA-UE assessor qualifications, blinding, exact assessment timing, and scale version.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md",
        "manuscript_section": "Methods, Outcome definition",
        "replacement_action": "Add assessment-procedure sentence before the proportional-recovery formula.",
        "exact_text_or_placeholder_to_replace": "The target endpoint was proportional recovery status...",
        "final_text_dependency": "Clinical assessment protocol or author-approved outcome assessment summary.",
        "verification_after_replacement": "Confirm no post-treatment data leak into model inputs.",
        "priority": "high",
    },
    {
        "field_id": "eeg_hardware_reference_impedance",
        "author_input_needed": "Amplifier, acquisition software, cap system, original channel count, online reference, ground, and impedance threshold.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/eeg_metadata_audit.md",
        "manuscript_section": "Methods, EEG preprocessing and feature extraction",
        "replacement_action": "Add acquisition paragraph before the sentence stating analyses began from preprocessed EEGLAB files.",
        "exact_text_or_placeholder_to_replace": "The available analysis materials did not include the original acquisition log or preprocessing protocol...",
        "final_text_dependency": "EEG acquisition log, lab protocol, or device/cap records.",
        "verification_after_replacement": "Check acquisition details do not conflict with 250 Hz, 64-channel source design, and 62 retained channels after M1/M2 removal.",
        "priority": "blocking",
    },
    {
        "field_id": "resting_state_instructions",
        "author_input_needed": "EO/EC order, target duration, fixation instruction, vigilance/drowsiness monitoring, and whether rest recordings were before tACS.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md",
        "manuscript_section": "Methods, EEG preprocessing and feature extraction",
        "replacement_action": "Add task-instruction sentence near the EO/EC file-name state-rule sentence.",
        "exact_text_or_placeholder_to_replace": "File-name state rules assigned baseline `*1.set` files to eyes-open recordings and baseline `*2.set` files to eyes-closed recordings.",
        "final_text_dependency": "EEG acquisition protocol or technician instructions.",
        "verification_after_replacement": "Confirm state labels and recording durations remain consistent with metadata audit.",
        "priority": "high",
    },
    {
        "field_id": "raw_eeg_preprocessing",
        "author_input_needed": "Raw filtering, notch, re-reference, bad-channel handling, ICA/artifact rejection, epoching/continuous export, and `.set/.fdt` export rules.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/methods_gap_resolution_from_project_files.md",
        "manuscript_section": "Methods, EEG preprocessing and feature extraction",
        "replacement_action": "Replace the current missing-protocol caveat with verified upstream preprocessing details while retaining the verified feature-pipeline boundary.",
        "exact_text_or_placeholder_to_replace": "acquisition hardware, online reference, filtering, re-reference, artifact rejection, and bad-channel handling before export remain author-confirmed fields",
        "final_text_dependency": "Original EEG preprocessing protocol or analyst log.",
        "verification_after_replacement": "Run methods provenance audit and confirm feature scripts still add no extra filtering/artifact rejection after loading.",
        "priority": "blocking",
    },
    {
        "field_id": "data_repository_doi_scope",
        "author_input_needed": "Repository platform, DOI/accession, version, public data scope, restricted data scope, access committee/contact, data licence.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/data_availability_and_fair_audit.md; docs/repository_readme_for_deposit.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md",
        "manuscript_section": "Data Availability; Cover letter; Repository README",
        "replacement_action": "Replace repository placeholders and select the correct public-vs-controlled Data Availability template.",
        "exact_text_or_placeholder_to_replace": "[repository name and DOI/accession]; [repository name] ([DOI/accession/version]); [data-access committee or corresponding author contact]",
        "final_text_dependency": "Repository record, DOI/accession, licence, and institutional access route.",
        "verification_after_replacement": "Confirm DOI resolves, restricted route is specific, and raw/minimally processed EEG is not over-shared.",
        "priority": "blocking",
    },
    {
        "field_id": "code_repository_license",
        "author_input_needed": "Code repository URL/DOI, version tag or commit hash, code licence, and environment notes.",
        "current_status": "author_required",
        "target_files": "docs/manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md; docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; docs/repository_readme_for_deposit.md",
        "manuscript_section": "Code Availability; Cover letter; Repository README",
        "replacement_action": "Replace code placeholders and add final commit/version information.",
        "exact_text_or_placeholder_to_replace": "[repository name, DOI/accession or URL, version, licence]; [repository name or URL] ([DOI/accession/version]) under [licence]",
        "final_text_dependency": "Public code archive or linked repository, release tag, licence.",
        "verification_after_replacement": "Run package assembly and confirm scripts needed to regenerate tables, figures, workbook, audits, and package are included.",
        "priority": "blocking",
    },
    {
        "field_id": "funding_competing_acknowledgements",
        "author_input_needed": "Funding names/numbers, competing-interest statements for each author, acknowledgements and permissions.",
        "current_status": "author_required",
        "target_files": "docs/jne_final_declaration_templates.md; docs/cover_letter_template.md; final manuscript declarations",
        "manuscript_section": "Funding; Competing Interests; Acknowledgements",
        "replacement_action": "Replace declaration templates with author-approved final statements.",
        "exact_text_or_placeholder_to_replace": "[Insert funder names and grant numbers]; [The authors declare no competing interests / Insert competing-interest statement]; [names or teams]",
        "final_text_dependency": "Author and institutional declaration forms.",
        "verification_after_replacement": "Confirm all authors have approved final declarations before upload.",
        "priority": "high",
    },
    {
        "field_id": "suggested_opposed_reviewers",
        "author_input_needed": "Suggested and opposed reviewer names, institutions, emails, rationale, and conflict reasons.",
        "current_status": "author_required_if_requested",
        "target_files": "docs/cover_letter_template.md; submission portal",
        "manuscript_section": "Cover letter or portal-only field",
        "replacement_action": "Fill only if requested or useful for the target journal; remove template section if not used.",
        "exact_text_or_placeholder_to_replace": "[Reviewer name, institution, email, rationale]; [Reviewer name, institution, reason]",
        "final_text_dependency": "Corresponding author preference and journal policy.",
        "verification_after_replacement": "Check suggested reviewers are not close collaborators and meet journal conflict rules.",
        "priority": "optional",
    },
]


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_csv()
    write_markdown()
    print(OUTPUT_CSV)
    print(OUTPUT_MD)


def write_csv() -> None:
    fieldnames = [
        "field_id",
        "author_input_needed",
        "current_status",
        "target_files",
        "manuscript_section",
        "replacement_action",
        "exact_text_or_placeholder_to_replace",
        "final_text_dependency",
        "verification_after_replacement",
        "priority",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)


def write_markdown() -> None:
    lines = [
        "# Author Field Replacement Map",
        "",
        "This map converts the author-required information form into concrete replacement targets. It should be used after the corresponding author, ethics office, clinical team, or data-governance contact supplies verified facts. Do not infer missing human-participant, consent, device, safety, or repository details from analysis outputs.",
        "",
        "## Summary",
        "",
        f"- Replacement fields: {len(ROWS)}.",
        f"- Blocking or high-priority fields: {sum(1 for row in ROWS if row['priority'] in {'blocking', 'blocking_if_applicable', 'high'})}.",
        "- Use this map together with `docs/author_required_information_form.md`, `docs/author_required_evidence_trace.md`, and `docs/jne_final_declaration_templates.md`.",
        "",
        "## Replacement Targets",
        "",
        "| Field | Priority | Current status | Author input needed | Target section | Replacement action | Verification after replacement |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append(
            "| {field_id} | {priority} | {current_status} | {author_input_needed} | {manuscript_section} | {replacement_action} | {verification_after_replacement} |".format(
                **{key: escape_md(value) for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## Finalization Order",
            "",
            "1. Complete ethics, consent, registration, study-design, intervention, EEG acquisition/preprocessing, and safety fields first.",
            "2. Choose repository scope, DOI/accession, licences, and restricted-data access route.",
            "3. Replace target text in the clean JNE manuscript, declaration templates, cover letter, and repository README.",
            "4. Regenerate DOCX files, source-data workbook, visual QA, manuscript integrity audit, artifact audit, and submission package.",
            "5. Confirm no unverified author-query text remains in the final upload candidate.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def escape_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
