from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "docs" / "author_submission_metadata_template.json"
REPLACEMENT_MAP = ROOT / "results" / "tables" / "author_field_replacement_map.csv"
OUTPUT_MD = ROOT / "docs" / "author_submission_metadata_validation_report.md"


@dataclass(frozen=True)
class FieldRule:
    field_id: str
    priority: str
    required_keys: tuple[str, ...]
    alternative_key_groups: tuple[tuple[str, ...], ...] = ()
    note: str = ""


FIELD_RULES = [
    FieldRule(
        "target_journal_reference_style",
        "high",
        ("target_journal", "article_type", "reference_style"),
    ),
    FieldRule(
        "author_list_affiliations",
        "high",
        ("author_order_full_names", "affiliations", "corresponding_author_name", "corresponding_author_email"),
    ),
    FieldRule("author_contributions", "high", ("credit_roles_by_author", "all_authors_approved_final_manuscript")),
    FieldRule(
        "ethics_approval",
        "blocking",
        (
            "ethics_committee_name",
            "approval_number",
            "approval_date",
            "applicable_site_or_sites",
            "ready_to_paste_statement_en",
        ),
    ),
    FieldRule(
        "informed_consent",
        "blocking",
        (
            "consent_route",
            "who_provided_consent",
            "written_or_waived",
            "covered_eeg_tacs_clinical_assessments",
            "covered_data_sharing",
            "ready_to_paste_statement_en",
        ),
    ),
    FieldRule(
        "trial_or_study_registration",
        "blocking_if_applicable",
        ("ready_to_paste_statement_en",),
        alternative_key_groups=(("registry_name", "registration_number"), ("non_registration_reason_if_applicable",)),
        note="Provide either registry details or an author-approved non-registration explanation.",
    ),
    FieldRule(
        "study_site_dates_design",
        "blocking",
        (
            "hospital_or_department",
            "recruitment_start_date",
            "recruitment_end_date",
            "follow_up_or_last_assessment_window",
            "prospective_or_retrospective",
            "single_or_multicentre",
            "ready_to_paste_methods_sentence_en",
        ),
    ),
    FieldRule(
        "eligibility_stroke_timing",
        "blocking",
        ("inclusion_criteria", "exclusion_criteria", "stroke_subtype_criteria", "time_since_stroke_to_eeg"),
    ),
    FieldRule(
        "tacs_device_electrodes",
        "blocking",
        ("device_model", "electrode_dimensions", "confirmed_target_frequency_intensity_duration_sessions"),
    ),
    FieldRule(
        "concurrent_rehabilitation",
        "blocking",
        (
            "was_conventional_rehabilitation_delivered",
            "frequency",
            "duration_per_session",
            "main_training_content",
            "ready_to_paste_methods_sentence_en",
        ),
    ),
    FieldRule(
        "tacs_safety_adverse_events",
        "blocking",
        ("adverse_event_summary", "tolerability_summary", "withdrawals_or_discontinuations", "ready_to_paste_statement_en"),
    ),
    FieldRule(
        "fma_assessors_timing",
        "high",
        ("assessor_training_or_credentials", "assessor_blinding", "baseline_assessment_timing", "post_treatment_assessment_timing"),
    ),
    FieldRule(
        "eeg_hardware_reference_impedance",
        "blocking",
        (
            "amplifier_model",
            "acquisition_software",
            "cap_system",
            "original_channel_count",
            "online_reference",
            "ground",
            "impedance_threshold",
        ),
    ),
    FieldRule(
        "resting_state_instructions",
        "high",
        ("eyes_open_eyes_closed_order", "target_duration_per_condition", "fixation_or_eye_instruction"),
    ),
    FieldRule(
        "raw_eeg_preprocessing",
        "blocking",
        (
            "raw_filtering",
            "notch_filtering",
            "rereference",
            "artifact_rejection",
            "ica_or_eye_muscle_artifact_handling",
            "bad_channel_handling",
            "eeglab_set_fdt_export_rules",
        ),
    ),
    FieldRule(
        "data_repository_doi_scope",
        "blocking",
        (
            "repository_name",
            "doi_or_accession",
            "version",
            "data_license",
            "public_data_scope",
            "restricted_data_scope",
            "controlled_access_contact_or_committee",
            "request_review_requirements",
            "ready_to_paste_data_availability_en",
        ),
    ),
    FieldRule(
        "code_repository_license",
        "blocking",
        ("code_repository_url_or_doi", "version_or_commit_hash", "code_license", "ready_to_paste_code_availability_en"),
    ),
    FieldRule(
        "funding_competing_acknowledgements",
        "high",
        ("funding_sources_and_grant_numbers", "competing_interests_statement"),
    ),
    FieldRule("suggested_opposed_reviewers", "optional", ()),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate author-supplied submission metadata.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_TEMPLATE, help="Path to author metadata JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if blocking or high-priority fields are incomplete.")
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    metadata = load_metadata(metadata_path)
    replacement_targets = load_replacement_targets()
    rows = validate(metadata, replacement_targets)
    OUTPUT_MD.write_text(build_report(metadata_path, rows), encoding="utf-8")

    missing_blocking = sum(1 for row in rows if row["priority"].startswith("blocking") and row["status"] != "complete")
    missing_high = sum(1 for row in rows if row["priority"] == "high" and row["status"] != "complete")
    complete = sum(1 for row in rows if row["status"] == "complete")

    print(OUTPUT_MD)
    print(f"metadata_fields={len(rows)}")
    print(f"complete_fields={complete}")
    print(f"incomplete_blocking_fields={missing_blocking}")
    print(f"incomplete_high_priority_fields={missing_high}")

    if args.strict and (missing_blocking or missing_high):
        raise SystemExit(1)


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("fields"), dict):
        raise ValueError("Metadata JSON must contain a top-level 'fields' object.")
    return data["fields"]


def load_replacement_targets() -> dict[str, dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    if not REPLACEMENT_MAP.exists():
        return targets
    with REPLACEMENT_MAP.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            targets[row["field_id"]] = row
    return targets


def validate(metadata: dict[str, Any], replacement_targets: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rule in FIELD_RULES:
        values = metadata.get(rule.field_id, {})
        if not isinstance(values, dict):
            values = {}
        target = replacement_targets.get(rule.field_id, {})
        effective_priority = target.get("priority") or rule.priority
        missing = [key for key in rule.required_keys if is_blank(values.get(key))]
        if effective_priority != "optional" and is_blank(values.get("evidence_source")):
            missing.append("evidence_source")
        alternative_missing = missing_alternative_groups(values, rule.alternative_key_groups)
        status = "complete" if not missing and not alternative_missing else "incomplete"
        rows.append(
            {
                "field_id": rule.field_id,
                "priority": effective_priority,
                "status": status,
                "missing_required_keys": ", ".join(missing) if missing else "none",
                "missing_alternative_requirement": alternative_missing or "none",
                "target_files": target.get("target_files", "not mapped"),
                "replacement_action": target.get("replacement_action", "not mapped"),
                "note": rule.note,
            }
        )
    missing_fields = sorted(set(metadata) - {rule.field_id for rule in FIELD_RULES})
    for field_id in missing_fields:
        rows.append(
            {
                "field_id": field_id,
                "priority": "unknown",
                "status": "unmapped",
                "missing_required_keys": "not checked",
                "missing_alternative_requirement": "not checked",
                "target_files": "not mapped",
                "replacement_action": "not mapped",
                "note": "This field is present in the metadata JSON but is not in the validator rules.",
            }
        )
    return rows

def missing_alternative_groups(values: dict[str, Any], groups: tuple[tuple[str, ...], ...]) -> str:
    if not groups:
        return ""
    for group in groups:
        if all(not is_blank(values.get(key)) for key in group):
            return ""
    return "provide one of: " + " OR ".join(" + ".join(group) for group in groups)


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return not stripped or stripped.lower() in {"tbd", "todo", "unknown", "not sure", "n/a?"}
    if isinstance(value, list | tuple | dict):
        return not value
    return False


def build_report(metadata_path: Path, rows: list[dict[str, str]]) -> str:
    complete = sum(1 for row in rows if row["status"] == "complete")
    incomplete_blocking = [row for row in rows if row["priority"].startswith("blocking") and row["status"] != "complete"]
    incomplete_high = [row for row in rows if row["priority"] == "high" and row["status"] != "complete"]
    lines = [
        "# Author Submission Metadata Validation Report",
        "",
        f"Metadata file: `{metadata_path}`",
        "",
        "This report validates whether author-approved metadata are ready for final insertion into the manuscript, declarations, cover letter, repository README, and submission checklist. It does not verify the truth of supplied values; ethics, consent, registration, safety, and repository statements still require author and institutional confirmation.",
        "",
        "## Summary",
        "",
        f"- Fields checked: {len(rows)}",
        f"- Complete fields: {complete}",
        f"- Incomplete blocking or blocking-if-applicable fields: {len(incomplete_blocking)}",
        f"- Incomplete high-priority fields: {len(incomplete_high)}",
        f"- Final insertion readiness: {'yes' if not incomplete_blocking and not incomplete_high else 'no'}",
        "",
        "## Blocking Fields",
        "",
    ]
    lines.extend(table_lines(incomplete_blocking))
    lines.extend(["", "## High-Priority Fields", ""])
    lines.extend(table_lines(incomplete_high))
    lines.extend(["", "## Full Field Map", ""])
    lines.append("| Field | Priority | Status | Missing required keys | Alternative requirement | Target files |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "| {field_id} | {priority} | {status} | {missing_required_keys} | {missing_alternative_requirement} | {target_files} |".format(
                **{key: md_escape(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Use After Completion",
            "",
            "1. Fill `docs/author_submission_metadata_template.json` with verified author-approved information.",
            "2. Run `python scripts/63_validate_author_submission_metadata.py --strict`.",
            "3. If strict validation passes, use `results/tables/author_field_replacement_map.csv` to replace the corresponding manuscript, declaration, cover-letter, and repository fields.",
            "4. Regenerate DOCX files, source-data workbook, visual QA, artifact audit, and the submission package.",
            "",
        ]
    )
    return "\n".join(lines)


def table_lines(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["No incomplete fields in this category."]
    lines = ["| Field | Missing required keys | Alternative requirement | Replacement action |", "|---|---|---|---|"]
    for row in rows:
        lines.append(
            "| {field_id} | {missing_required_keys} | {missing_alternative_requirement} | {replacement_action} |".format(
                **{key: md_escape(value) for key, value in row.items()}
            )
        )
    return lines


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
