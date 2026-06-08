from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD = ROOT / "docs" / "author_required_evidence_trace.md"
OUTPUT_CSV = ROOT / "results" / "tables" / "author_required_evidence_trace.csv"


FIELD_DEFINITIONS = [
    {
        "field": "ethics_approval",
        "required_for_submission": "Ethics committee name, approval number, approval date, and study-site applicability.",
        "patterns": [
            r"\bethic",
            r"\bIRB\b",
            r"review board",
            r"Declaration of Helsinki",
            r"伦理",
            r"批件",
            r"伦理委员会",
            r"审批",
        ],
        "manual_status": "author_required",
        "source_summary": "Only workbook-structure and generated-audit mentions were found; no ethics committee name, approval number, or approval date is verifiable from project-source evidence.",
        "reason": "No verifiable ethics committee name, approval number, or approval date is present in project-source evidence.",
    },
    {
        "field": "informed_consent",
        "required_for_submission": "Consent route, who consented, whether consent covered EEG/tACS/clinical assessments, and data sharing.",
        "patterns": [
            r"consent",
            r"informed",
            r"知情",
            r"同意",
            r"授权",
        ],
        "manual_status": "author_required",
        "source_summary": "Only generic consent/data-sharing mentions were found; no consent form text, waiver, legally authorized representative wording, or data-sharing permission scope is available.",
        "reason": "Project files contain no consent form text or consent scope for secondary data sharing.",
    },
    {
        "field": "trial_or_study_registration",
        "required_for_submission": "Registry name and number, or an author-approved reason for non-registration.",
        "patterns": [
            r"\btrial",
            r"registr",
            r"\bNCT\d+",
            r"ChiCTR",
            r"clinicaltrials",
            r"注册",
            r"登记",
        ],
        "manual_status": "author_required",
        "source_summary": "Keyword hits are dominated by model trials/training notes; no NCT, ChiCTR, clinicaltrials.gov, or other registry identifier is present.",
        "reason": "No registry identifier was found in project-source evidence.",
    },
    {
        "field": "recruitment_site_and_dates",
        "required_for_submission": "Hospital/department, recruitment start and end dates, follow-up date window.",
        "patterns": [
            r"recruit",
            r"enrol",
            r"enroll",
            r"site",
            r"hospital",
            r"department",
            r"招募",
            r"入组",
            r"医院",
            r"科室",
            r"中心",
        ],
        "manual_status": "author_required",
        "source_summary": "Cohort files provide patient counts and clinical summaries, but no final hospital/department name, recruitment start/end date, or assessment window.",
        "reason": "Project-source evidence does not provide final study site and recruitment dates.",
    },
    {
        "field": "eligibility_criteria",
        "required_for_submission": "Inclusion criteria, exclusion criteria, stroke subtype criteria, and lesion/timing criteria.",
        "patterns": [
            r"inclusion",
            r"exclusion",
            r"eligib",
            r"criterion",
            r"criteria",
            r"纳入",
            r"排除",
            r"标准",
            r"卒中亚型",
            r"病灶",
        ],
        "manual_status": "author_required",
        "source_summary": "The design file defines 19 labelled patients and notes two ceiling-effect exclusions, but it does not provide protocol-level inclusion/exclusion, stroke subtype, lesion, or timing criteria.",
        "reason": "Current source evidence does not contain a complete protocol-level eligibility section.",
    },
    {
        "field": "tacs_protocol",
        "required_for_submission": "Target, montage, side rule, frequency, intensity, duration, sessions, device, and electrode size.",
        "patterns": [
            r"\btACS\b",
            r"transcranial alternating",
            r"stimulation",
            r"\bM1\b",
            r"\bC3\b",
            r"\bC4\b",
            r"20 Hz",
            r"1000",
            r"microampere",
            r"电刺激",
            r"经颅",
        ],
        "manual_status": "partly_resolved",
        "source_summary": "Resolved from project design: contralateral M1; C3 for right-hand impairment; C4 for left-hand impairment; 20 Hz; 1000 microampere; 20 min per session; 14 daily sessions over 2 weeks; post-treatment assessment immediately after final tACS. Device model and electrode dimensions remain missing.",
        "reason": "Project design resolves target-side rule, frequency, intensity, duration, and session count, but device and electrode dimensions remain author-required.",
    },
    {
        "field": "concurrent_rehabilitation",
        "required_for_submission": "Whether conventional rehabilitation was delivered concurrently, with dose and content.",
        "patterns": [
            r"rehabilitation",
            r"therapy",
            r"training",
            r"conventional",
            r"康复",
            r"训练",
            r"常规",
            r"治疗",
        ],
        "manual_status": "author_required",
        "source_summary": "The project is framed as tACS-context rehabilitation, but no source file verifies whether standardized conventional rehabilitation was delivered concurrently or its dose/content.",
        "reason": "The project design mentions tACS context but does not verify concurrent conventional rehabilitation dose/content.",
    },
    {
        "field": "tacs_safety_adverse_events",
        "required_for_submission": "Adverse events, tolerability, withdrawals, and safety monitoring.",
        "patterns": [
            r"adverse",
            r"safety",
            r"tolerab",
            r"withdraw",
            r"headache",
            r"seizure",
            r"skin",
            r"不良",
            r"安全",
            r"退出",
            r"头痛",
            r"癫痫",
            r"皮肤",
        ],
        "manual_status": "author_required",
        "source_summary": "No complete adverse-event, tolerability, withdrawal, or safety-monitoring table was found. Keyword hits are nonspecific and do not support a manuscript safety statement.",
        "reason": "No complete safety or adverse-event summary was found in project-source evidence.",
    },
    {
        "field": "eeg_acquisition_hardware",
        "required_for_submission": "Amplifier, acquisition software, cap system, original channel count, online reference, ground, impedance.",
        "patterns": [
            r"amplifier",
            r"cap",
            r"montage",
            r"reference",
            r"ground",
            r"impedance",
            r"10-20",
            r"10-10",
            r"Neuroscan",
            r"BrainProducts",
            r"BioSemi",
            r"电极帽",
            r"阻抗",
            r"参考",
            r"地线",
        ],
        "manual_status": "author_required",
        "source_summary": "Resolved: design file says 64-channel EEG and 250 Hz; metadata audit verifies 250 Hz and 62 retained analysis channels after M1/M2 removal. Missing: amplifier, cap, acquisition software, online reference, ground, and impedance.",
        "reason": "Available evidence verifies 250 Hz and 62 retained channels after M1/M2 removal, but not hardware/reference/impedance.",
    },
    {
        "field": "raw_eeg_preprocessing",
        "required_for_submission": "Raw filtering, notch, re-reference, artifact rejection, ICA, bad-channel handling, and export rules before `.set/.fdt`.",
        "patterns": [
            r"filter",
            r"notch",
            r"ICA",
            r"artifact",
            r"bad channel",
            r"interpolation",
            r"re-reference",
            r"preprocess",
            r"滤波",
            r"陷波",
            r"伪迹",
            r"坏道",
            r"重参考",
        ],
        "manual_status": "author_required",
        "source_summary": "Resolved for manuscript pipeline: analyses start from preprocessed EEGLAB `.set/.fdt` and feature scripts do not add filtering/artifact rejection/interpolation after loading. Missing: upstream raw filtering, notch, re-reference, ICA/artifact rejection, bad-channel handling, and export rules.",
        "reason": "The manuscript feature pipeline has no additional filtering/artifact rejection after loading, but upstream raw preprocessing remains unverified.",
    },
    {
        "field": "data_code_repository",
        "required_for_submission": "Repository name, DOI/accession, version, data licence, code licence, and controlled-access route.",
        "patterns": [
            r"repository",
            r"Zenodo",
            r"OSF",
            r"Figshare",
            r"DOI",
            r"accession",
            r"licen[cs]e",
            r"data-use",
            r"controlled access",
            r"仓储",
            r"数据共享",
            r"许可证",
        ],
        "manual_status": "author_required",
        "source_summary": "Repository README, source-data workbook, data dictionary, and reproducibility scripts are prepared locally, but no public repository DOI/accession, version, data licence, code licence, or controlled-access committee/contact is finalized.",
        "reason": "Repository README and data availability drafts exist, but final DOI/licence/access route is not author-confirmed.",
    },
]

SOURCE_INCLUDE_EXTENSIONS = {".md", ".txt", ".csv", ".tsv", ".yaml", ".yml", ".json", ".py", ".cjs", ".toml"}
SCAN_ROOTS = [
    ROOT,
    ROOT / "docs",
    ROOT / "configs",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "tests",
    ROOT / "results" / "tables",
    ROOT / "results" / "statistics",
]
SOURCE_EXCLUDE_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "experiment_exports",
    "outputs",
    "output",
    "runs",
    "submission_package_20260601",
}
GENERATED_DOC_HINTS = {
    "author_required_information_form",
    "author_required_evidence_trace",
    "jne_final_declaration_templates",
    "jne_submission_checklist",
    "submission_readiness_audit",
    "submission_artifact_quality_audit",
    "data_availability_and_fair_audit",
    "methods_gap_resolution_from_project_files",
    "methods_detail_provenance",
    "manuscript_residual_aware",
}
PROJECT_SOURCE_DOCS = {
    "tacs_eeg_proportional_recovery_project_design.md",
    "docs/project_context.md",
    "docs/cohort_characteristics.md",
    "docs/eeg_metadata_audit.md",
    "results/tables/eeg_recording_metadata_audit.csv",
    "results/tables/eeg_recording_summary.csv",
    "results/tables/clinical_workbook_structure_audit.csv",
    "results/tables/table1_cohort_characteristics.csv",
    "results/tables/patient_characteristics_table.csv",
}


def main() -> None:
    source_files = list(iter_source_files())
    records: list[dict[str, str]] = []
    for definition in FIELD_DEFINITIONS:
        hits = collect_hits(definition, source_files)
        source_hits = [hit for hit in hits if hit["evidence_class"] == "project_source"]
        generated_hits = [hit for hit in hits if hit["evidence_class"] == "generated_audit_or_template"]
        best_hits = source_hits[:5] if source_hits else generated_hits[:5]
        records.append(
            {
                "field": definition["field"],
                "submission_requirement": definition["required_for_submission"],
                "status": definition["manual_status"],
                "project_source_hits": str(len(source_hits)),
                "generated_or_template_hits": str(len(generated_hits)),
                "best_evidence": " | ".join(format_hit(hit) for hit in best_hits) if best_hits else "No matching project text found.",
                "verified_source_summary": str(definition["source_summary"]),
                "decision": definition["reason"],
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "field",
                "submission_requirement",
                "status",
                "project_source_hits",
                "generated_or_template_hits",
                "best_evidence",
                "verified_source_summary",
                "decision",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    OUTPUT_MD.write_text(build_markdown(records), encoding="utf-8")
    print(OUTPUT_MD)
    print(OUTPUT_CSV)


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue
        candidates = scan_root.glob("*") if scan_root == ROOT else scan_root.rglob("*")
        for path in candidates:
            if path in seen or not path.is_file() or path.suffix.lower() not in SOURCE_INCLUDE_EXTENSIONS:
                continue
            rel_parts = set(path.relative_to(ROOT).parts)
            if rel_parts.intersection(SOURCE_EXCLUDE_PARTS):
                continue
            if path.stat().st_size > 500_000:
                continue
            seen.add(path)
            files.append(path)
    return sorted(files)


def collect_hits(definition: dict[str, object], source_files: list[Path]) -> list[dict[str, str]]:
    patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in definition["patterns"]]  # type: ignore[index]
    hits: list[dict[str, str]] = []
    per_file_counts: defaultdict[str, int] = defaultdict(int)
    for path in source_files:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for line_number, line in enumerate(lines, start=1):
            if not any(pattern.search(line) for pattern in patterns):
                continue
            if per_file_counts[rel] >= 3:
                continue
            per_file_counts[rel] += 1
            hits.append(
                {
                    "path": rel,
                    "line": str(line_number),
                    "snippet": compact(line),
                    "evidence_class": classify_path(path),
                }
            )
    return sorted(
        hits,
        key=lambda hit: (
            hit["evidence_class"] != "project_source",
            source_priority(hit["path"]),
            hit["path"],
            int(hit["line"]),
        ),
    )


def classify_path(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix().lower()
    if rel in {item.lower() for item in PROJECT_SOURCE_DOCS}:
        return "project_source"
    if rel.startswith("configs/") or rel.startswith("src/"):
        return "project_source"
    if any(hint in rel for hint in GENERATED_DOC_HINTS):
        return "generated_audit_or_template"
    if rel.startswith("docs/"):
        return "generated_audit_or_template"
    if rel.startswith("results/"):
        return "generated_audit_or_template"
    if "audit" in rel or "checklist" in rel or "template" in rel:
        return "generated_audit_or_template"
    return "generated_audit_or_template"


def source_priority(rel_path: str) -> int:
    rel = rel_path.lower()
    if rel == "tacs_eeg_proportional_recovery_project_design.md":
        return 0
    if rel == "docs/project_context.md":
        return 1
    if rel in {"docs/cohort_characteristics.md", "docs/eeg_metadata_audit.md"}:
        return 2
    if rel.startswith("results/tables/"):
        return 3
    if rel.startswith("configs/"):
        return 4
    if rel.startswith("src/"):
        return 5
    return 9


def compact(line: str) -> str:
    text = " ".join(line.strip().split())
    if len(text) > 180:
        return text[:177] + "..."
    return text


def format_hit(hit: dict[str, str]) -> str:
    return f"{hit['path']}:{hit['line']} - {hit['snippet']}"


def build_markdown(records: list[dict[str, str]]) -> str:
    status_counts = {status: sum(1 for row in records if row["status"] == status) for status in sorted({row["status"] for row in records})}
    lines = [
        "# Author-Required Evidence Trace",
        "",
        "This audit records a reproducible keyword sweep for author-supplied submission fields that cannot be safely invented from the analysis outputs. It separates project-source hits from generated audit/template hits so that placeholder text is not mistaken for evidence.",
        "",
        "## Summary",
        "",
        f"- Fields audited: {len(records)}.",
        f"- Status counts: {', '.join(f'{key}={value}' for key, value in status_counts.items())}.",
        "- `partly_resolved` means the current project supports part of the manuscript wording, but final author confirmation is still needed.",
        "- `author_required` means the project does not contain enough verifiable source evidence to write a submission-ready statement.",
        "",
        "## Field Trace",
        "",
        "| Field | Status | Project-source hits | Generated/template hits | Submission requirement | Verified source summary | Best evidence | Decision |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in records:
        lines.append(
            "| {field} | {status} | {project_source_hits} | {generated_or_template_hits} | {submission_requirement} | {verified_source_summary} | {best_evidence} | {decision} |".format(
                **{key: md_escape(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Practical Use",
            "",
            "Use this file together with `docs/author_required_information_form.md`. Fields marked `author_required` should be answered by the author, ethics office, clinical team, or data-governance contact before the clean JNE manuscript is treated as submission-ready.",
            "",
        ]
    )
    return "\n".join(lines)


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
