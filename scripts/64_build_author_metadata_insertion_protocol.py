from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "docs" / "author_submission_metadata_template.json"
REPLACEMENT_MAP = ROOT / "results" / "tables" / "author_field_replacement_map.csv"
VALIDATOR = ROOT / "scripts" / "63_validate_author_submission_metadata.py"
OUTPUT_MD = ROOT / "docs" / "author_metadata_insertion_protocol.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a dry-run final-insertion protocol from author submission metadata."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Author metadata JSON path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if strict metadata validation fails. No manuscript files are edited.",
    )
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    metadata = load_metadata(metadata_path)
    replacement_rows = load_replacement_rows()
    validation_code, validation_output = run_strict_validator(metadata_path)
    protocol_rows = build_protocol_rows(metadata, replacement_rows)
    OUTPUT_MD.write_text(build_markdown(metadata_path, validation_code, validation_output, protocol_rows), encoding="utf-8")

    ready_rows = sum(1 for row in protocol_rows if row["status"] == "ready")
    blocked_rows = sum(1 for row in protocol_rows if row["status"] == "blocked")
    print(OUTPUT_MD)
    print(f"insertion_fields={len(protocol_rows)}")
    print(f"ready_fields={ready_rows}")
    print(f"blocked_fields={blocked_rows}")
    print(f"strict_validation_exit={validation_code}")

    if args.strict and validation_code != 0:
        raise SystemExit(validation_code)


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("Metadata JSON must contain a top-level 'fields' object.")
    return {key: value if isinstance(value, dict) else {} for key, value in fields.items()}


def load_replacement_rows() -> list[dict[str, str]]:
    if not REPLACEMENT_MAP.exists():
        raise FileNotFoundError(REPLACEMENT_MAP)
    with REPLACEMENT_MAP.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_strict_validator(metadata_path: Path) -> tuple[int, str]:
    command = [sys.executable, str(VALIDATOR), "--metadata", str(metadata_path), "--strict"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    return completed.returncode, output


def build_protocol_rows(metadata: dict[str, dict[str, Any]], replacement_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in replacement_rows:
        field_id = row["field_id"]
        field_values = metadata.get(field_id, {})
        non_empty_keys = [key for key, value in field_values.items() if key != "evidence_source" and not is_blank(value)]
        evidence_present = not is_blank(field_values.get("evidence_source"))
        status = "ready" if non_empty_keys and evidence_present else "blocked"
        source_keys = ", ".join(non_empty_keys) if non_empty_keys else "none supplied"
        preview = preview_values(field_values, non_empty_keys)
        blocking_reason = ""
        if not non_empty_keys:
            blocking_reason = "No field-specific author metadata supplied."
        elif not evidence_present:
            blocking_reason = "Evidence source is missing."
        rows.append(
            {
                "field_id": field_id,
                "priority": row.get("priority", ""),
                "status": status,
                "target_files": row.get("target_files", ""),
                "manuscript_section": row.get("manuscript_section", ""),
                "replacement_action": row.get("replacement_action", ""),
                "source_metadata_keys": source_keys,
                "metadata_preview": preview,
                "verification_after_replacement": row.get("verification_after_replacement", ""),
                "blocking_reason": blocking_reason,
            }
        )
    return rows


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return not value
    return False


def preview_values(values: dict[str, Any], keys: list[str]) -> str:
    if not keys:
        return "none"
    parts: list[str] = []
    for key in keys[:4]:
        value = str(values.get(key, "")).strip().replace("\n", " ")
        if len(value) > 90:
            value = value[:87] + "..."
        parts.append(f"{key}={value}")
    if len(keys) > 4:
        parts.append(f"+{len(keys) - 4} more")
    return "; ".join(parts)


def build_markdown(
    metadata_path: Path,
    validation_code: int,
    validation_output: str,
    rows: list[dict[str, str]],
) -> str:
    ready = [row for row in rows if row["status"] == "ready"]
    blocked = [row for row in rows if row["status"] != "ready"]
    strict_ready = validation_code == 0
    lines = [
        "# Author Metadata Final-Insertion Protocol",
        "",
        f"Metadata file: `{metadata_path}`",
        "",
        "This dry-run protocol maps author-approved metadata to final manuscript, declaration, cover-letter, repository, and submission-checklist replacements. It deliberately does not edit manuscript files. Final replacement should be performed only after strict metadata validation passes and the author confirms the supplied statements.",
        "",
        "## Readiness",
        "",
        f"- Strict validation exit code: {validation_code}",
        f"- Strict validation ready: {'yes' if strict_ready else 'no'}",
        f"- Fields with any supplied metadata and evidence source: {len(ready)}",
        f"- Fields blocked from insertion: {len(blocked)}",
        "",
        "## Validator Output",
        "",
        "```text",
        validation_output or "No validator output captured.",
        "```",
        "",
        "## Blocked Insertions",
        "",
    ]
    lines.extend(protocol_table(blocked))
    lines.extend(["", "## Ready Insertions", ""])
    lines.extend(protocol_table(ready))
    lines.extend(
        [
            "",
            "## Final Replacement Workflow",
            "",
            "1. Fill `docs/author_submission_metadata_template.json` with author-approved values and evidence sources.",
            "2. Run `python scripts/63_validate_author_submission_metadata.py --strict`.",
            "3. Run `python scripts/64_build_author_metadata_insertion_protocol.py --strict` to produce a ready-only insertion plan.",
            "4. Replace only the fields marked ready, using the target files and replacement actions listed below.",
            "5. Regenerate JNE and clean variants, DOCX files, source-data workbook, visual QA, manuscript integrity audit, artifact audit, and submission package.",
            "6. Confirm no unverified placeholders remain in clean submission files before upload.",
            "",
            "## Full Insertion Map",
            "",
            "| Field | Priority | Status | Target files | Source metadata keys | Replacement action | Verification |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {field_id} | {priority} | {status} | {target_files} | {source_metadata_keys} | {replacement_action} | {verification_after_replacement} |".format(
                **{key: md_escape(value) for key, value in row.items()}
            )
        )
    lines.append("")
    return "\n".join(lines)


def protocol_table(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["None."]
    lines = ["| Field | Priority | Reason or preview | Target files |", "|---|---|---|---|"]
    for row in rows:
        reason = row["blocking_reason"] or row["metadata_preview"]
        lines.append(
            "| {field_id} | {priority} | {reason} | {target_files} |".format(
                field_id=md_escape(row["field_id"]),
                priority=md_escape(row["priority"]),
                reason=md_escape(reason),
                target_files=md_escape(row["target_files"]),
            )
        )
    return lines


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
