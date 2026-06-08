from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "presubmission_editorial_readiness_audit.md"

MANUSCRIPTS = {
    "nature_working": ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_polished.md",
    "nature_clean": ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md",
    "jne_working": ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_jne_structured.md",
    "jne_clean": ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md",
}

REFERENCE_AUDIT = ROOT / "docs" / "reference_metadata_audit.md"
CITATION_MAP = ROOT / "docs" / "citation_artifacts" / "manuscript_citation_map.md"
FINAL_GATE = ROOT / "docs" / "final_submission_gate_report.md"
ARTIFACT_QA = ROOT / "docs" / "submission_artifact_quality_audit.md"
DOCX_QA = ROOT / "docs" / "docx_visual_qa_report.md"


@dataclass(frozen=True)
class Check:
    category: str
    item: str
    observed: str
    target: str
    status: str
    action: str


def main() -> None:
    checks = build_checks()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_report(checks), encoding="utf-8")
    print(OUTPUT)
    print(format_counts(checks))


def build_checks() -> list[Check]:
    checks: list[Check] = []
    manuscripts = {name: read_text(path) for name, path in MANUSCRIPTS.items()}

    for name, text in manuscripts.items():
        title = first_title(text)
        abstract = section_text(text, "Abstract")
        abstract_words = word_count(abstract)
        body_words = word_count(strip_references(text))
        author_queries = text.count("Author information required before submission")

        checks.append(
            Check(
                "title",
                name,
                f"{word_count(title)} words; {len(title)} characters",
                "<=20 words; searchable and bounded",
                "PASS" if word_count(title) <= 20 else "WARN",
                "Keep title concise; revise only after the final target journal is chosen.",
            )
        )
        checks.append(
            Check(
                "abstract",
                name,
                f"{abstract_words} words",
                "<=250 words preferred for broad journal portability",
                "PASS" if abstract_words <= 250 else "WARN",
                "Shorten context, comparator detail, or exploratory caveats if the target journal enforces a tighter limit.",
            )
        )
        checks.append(
            Check(
                "main_text",
                name,
                f"{body_words} words before references",
                "<=5000 words as a practical pre-submission target",
                "PASS" if body_words <= 5000 else "WARN",
                "Keep Methods concise; target-journal limits should be checked before final upload.",
            )
        )
        expected_clean = "clean" in name
        checks.append(
            Check(
                "author_queries",
                name,
                f"{author_queries} explicit author-query lines",
                "0 in clean variants; working variants may retain queries",
                "PASS" if (author_queries == 0) == expected_clean else "WARN",
                "Use clean variants for upload after author metadata are inserted.",
            )
        )

    nature = manuscripts["nature_working"]
    jne = manuscripts["jne_working"]
    checks.extend(
        [
            Check(
                "structured_abstract",
                "jne_working",
                ", ".join(label for label in ["Objective", "Approach", "Main results", "Significance"] if f"### {label}" in jne),
                "Objective, Approach, Main results, and Significance present",
                "PASS"
                if all(f"### {label}" in jne for label in ["Objective", "Approach", "Main results", "Significance"])
                else "WARN",
                "Maintain structured headings for JNE-style submission.",
            ),
            Check(
                "availability",
                "nature_working",
                f"Data availability={'## Data availability' in nature}; Code availability={'## Code availability' in nature}",
                "Both sections present",
                "PASS" if "## Data availability" in nature and "## Code availability" in nature else "WARN",
                "Replace repository DOI/licence placeholders only after author-approved metadata are available.",
            ),
            Check(
                "references",
                "metadata_audit",
                compact_status(REFERENCE_AUDIT, ["Lookup status:", "Match status:"]),
                "Lookup and match checks pass",
                "PASS" if contains_all(REFERENCE_AUDIT, ["Lookup status: PASS=25", "Match status: PASS=25"]) else "WARN",
                "Re-run reference metadata audit after target-journal style edits or added references.",
            ),
            Check(
                "citations",
                "claim_map",
                "citation map present" if CITATION_MAP.exists() else "missing",
                "Main citable claim areas mapped",
                "PASS" if CITATION_MAP.exists() else "WARN",
                "Do not add citations unless a manuscript sentence requires them directly.",
            ),
            Check(
                "docx_visual_qa",
                "current_package",
                compact_status(DOCX_QA, ["Page checks passed:", "Page checks failed:"]),
                "0 failed rendered pages",
                "PASS" if contains_all(DOCX_QA, ["Page checks failed: 0"]) else "WARN",
                "Inspect final PDF previews manually before upload.",
            ),
            Check(
                "artifact_quality",
                "current_package",
                compact_status(ARTIFACT_QA, ["Manifest rows:", "Zip entries:", "Status: PASS"]),
                "Package integrity passes",
                "PASS" if contains_all(ARTIFACT_QA, ["Status: PASS", "Missing package paths: 0"]) else "WARN",
                "Warnings from working manuscripts retaining author-query text are expected until final insertion.",
            ),
            Check(
                "final_gate",
                "current_package",
                compact_status(FINAL_GATE, ["Overall status:", "PASS:", "WARN:", "BLOCKED:", "FAIL:"]),
                "No non-author FAIL; author metadata may remain blocked",
                "BLOCKED" if contains_all(FINAL_GATE, ["Overall status: **BLOCKED_BY_AUTHOR_METADATA**"]) else "PASS",
                "Complete author metadata before final upload; do not mark the manuscript ready while this remains blocked.",
            ),
        ]
    )
    return checks


def render_report(checks: list[Check]) -> str:
    counts = status_counts(checks)
    lines = [
        "# Pre-submission Editorial Readiness Audit",
        "",
        "This audit checks editorial preflight risks that are not fully captured by structural package gates: title length, abstract length, clean manuscript variants, author-query boundaries, data/code availability, citation coverage, and final submission gate status.",
        "",
        "## Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARN: {counts.get('WARN', 0)}",
        f"- BLOCKED: {counts.get('BLOCKED', 0)}",
        "",
        "## Checks",
        "",
        "| Category | Item | Observed | Target | Status | Action |",
        "|---|---|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            "| "
            + " | ".join(
                md_escape(value)
                for value in [
                    check.category,
                    check.item,
                    check.observed,
                    check.target,
                    check.status,
                    check.action,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The abstract-length target is a conservative portability threshold, not a substitute for the final journal's instructions.",
            "- Clean variants are the upload candidates after author metadata are inserted; working variants intentionally retain author-query lines.",
            "- `BLOCKED` indicates a submission-critical author metadata dependency rather than a failure of the manuscript package itself.",
            "",
        ]
    )
    return "\n".join(lines)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def first_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def section_text(text: str, heading: str) -> str:
    pattern = rf"## {re.escape(heading)}\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else ""


def strip_references(text: str) -> str:
    return re.sub(r"\n## References\n.*", "", text, flags=re.S)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def compact_status(path: Path, labels: list[str]) -> str:
    text = read_text(path)
    found: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(label in stripped for label in labels):
            found.append(stripped)
    return "; ".join(found) if found else "not found"


def contains_all(path: Path, snippets: list[str]) -> bool:
    text = read_text(path)
    return all(snippet in text for snippet in snippets)


def status_counts(checks: list[Check]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def format_counts(checks: list[Check]) -> str:
    counts = status_counts(checks)
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
