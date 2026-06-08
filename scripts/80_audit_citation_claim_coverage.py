from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_polished.md"
SELECTED_REFERENCES = ROOT / "docs" / "citation_artifacts" / "selected_references.md"
OUTPUT_MD = ROOT / "docs" / "citation_claim_coverage_audit.md"


@dataclass(frozen=True)
class ClaimRequirement:
    area: str
    anchor_pattern: str
    expected_refs: tuple[int, ...]
    support_boundary: str


@dataclass(frozen=True)
class ClaimResult:
    area: str
    status: str
    expected_refs: tuple[int, ...]
    found_refs: tuple[int, ...]
    support_boundary: str
    paragraph_preview: str


CLAIM_REQUIREMENTS: tuple[ClaimRequirement, ...] = (
    ClaimRequirement(
        "Proportional recovery and PREP2 context",
        r"proportional-recovery framework",
        (1, 2, 3),
        "Supports recovery-rule framing; does not validate the cohort-median residual threshold.",
    ),
    ClaimRequirement(
        "tACS stroke neuromodulation context",
        r"tACS is an emerging",
        (22,),
        "Supports frequency-specific tACS context only; not a claim of tACS treatment efficacy in this dataset.",
    ),
    ClaimRequirement(
        "EEG biomarkers and connectivity prognosis",
        r"Prior work has linked EEG biomarkers",
        (7, 8, 9, 16),
        "Supports EEG biomarker plausibility; not an external validation of this model.",
    ),
    ClaimRequirement(
        "WPLI method rationale",
        r"WPLI reduces zero-lag",
        (10,),
        "Supports WPLI choice for phase synchronization in scalp EEG.",
    ),
    ClaimRequirement(
        "Stroke recovery ML and deep learning",
        r"Recent stroke-prognosis studies",
        (4, 5, 6, 23, 24),
        "Supports broader ML/DL prognosis context; not a direct comparator claim.",
    ),
    ClaimRequirement(
        "EEG deep learning feasibility",
        r"stroke-severity assessment",
        (25,),
        "Supports feasibility of EEG-derived neural signatures, with treatment-response prognosis left unresolved.",
    ),
    ClaimRequirement(
        "Prediction-model reporting guidance",
        r"prediction-model guidance",
        (13, 14, 15),
        "Supports TRIPOD/TRIPOD+AI/PROBAST reporting and risk-of-bias framing.",
    ),
    ClaimRequirement(
        "Explanation alignment with prior physiology",
        r"These findings align with prior reports",
        (7, 8, 9, 16),
        "Supports biological plausibility, not causal interpretation of saliency or WPLI edges.",
    ),
    ClaimRequirement(
        "Clinical predictor discussion boundary",
        r"baseline clinical variables were highly predictive",
        (1, 2, 3, 4, 5, 6, 23, 24),
        "Supports clinical-predictor context while preserving the no-incremental-EEG-value boundary.",
    ),
    ClaimRequirement(
        "Methods WPLI justification",
        r"WPLI was selected",
        (10,),
        "Supports WPLI method selection and volume-conduction/sample-bias rationale.",
    ),
    ClaimRequirement(
        "MNE-Python EEG processing and visualization",
        r"MNE-Python where applicable",
        (11, 12),
        "Supports software provenance for EEG processing and topographic visualization.",
    ),
    ClaimRequirement(
        "Self-supervised Barlow Twins pretraining",
        r"Barlow Twins self-supervised learning",
        (17,),
        "Supports redundancy-reduction SSL method provenance.",
    ),
    ClaimRequirement(
        "Attribution methods",
        r"Model explanation used SmoothGrad-smoothed integrated gradients",
        (18, 19),
        "Supports attribution-method provenance; not a claim that saliency is causal.",
    ),
    ClaimRequirement(
        "Software stack",
        r"Analyses used Python, PyTorch, scikit-learn, MNE-Python",
        (11, 12, 20, 21),
        "Supports named software packages used in the analysis workflow.",
    ),
)

RISK_PATTERNS: tuple[str, ...] = (
    "clinical deployment",
    "clinical use",
    "incremental value",
    "causal",
    "validated biomarker",
    "superiority",
    "external validation",
)

GUARD_TERMS: tuple[str, ...] = (
    "not",
    "before",
    "require",
    "requires",
    "unresolved",
    "exploratory",
    "hypothesis-generating",
    "rather than",
    "do not",
    "does not",
    "no external validation",
    "insufficient",
    "unsupported clinical",
    "should be tested",
    "should be viewed",
)


def main() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    references_text = SELECTED_REFERENCES.read_text(encoding="utf-8")

    reference_count = count_selected_references(references_text)
    all_cited_refs = tuple(sorted(extract_citations(manuscript)))
    invalid_citations = tuple(ref for ref in all_cited_refs if ref < 1 or ref > reference_count)
    uncited_references = tuple(ref for ref in range(1, reference_count + 1) if ref not in set(all_cited_refs))

    claim_results = [evaluate_requirement(manuscript, requirement) for requirement in CLAIM_REQUIREMENTS]
    risk_results = evaluate_overclaim_boundaries(manuscript)

    warnings = [
        result
        for result in claim_results
        if result.status != "PASS"
    ]
    risk_warnings = [row for row in risk_results if row["status"] != "PASS"]
    overall = "PASS" if not warnings and not invalid_citations and not risk_warnings else "WARN"

    OUTPUT_MD.write_text(
        build_report(
            overall=overall,
            reference_count=reference_count,
            all_cited_refs=all_cited_refs,
            invalid_citations=invalid_citations,
            uncited_references=uncited_references,
            claim_results=claim_results,
            risk_results=risk_results,
        ),
        encoding="utf-8",
    )

    print(OUTPUT_MD)
    print(f"overall_status={overall}")
    print(f"claim_requirements={len(claim_results)}")
    print(f"invalid_citations={len(invalid_citations)}")
    print(f"overclaim_warnings={len(risk_warnings)}")


def count_selected_references(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\d+\.\s", line))


def extract_citations(text: str) -> set[int]:
    citations: set[int] = set()
    for match in re.finditer(r"\[([0-9,\-\s]+)\]", text):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = [piece.strip() for piece in part.split("-", 1)]
                if start_text.isdigit() and end_text.isdigit():
                    start = int(start_text)
                    end = int(end_text)
                    if start <= end:
                        citations.update(range(start, end + 1))
                    else:
                        citations.update((start, end))
            elif part.isdigit():
                citations.add(int(part))
    return citations


def evaluate_requirement(text: str, requirement: ClaimRequirement) -> ClaimResult:
    paragraph = find_paragraph(text, requirement.anchor_pattern)
    if not paragraph:
        return ClaimResult(
            area=requirement.area,
            status="WARN",
            expected_refs=requirement.expected_refs,
            found_refs=(),
            support_boundary=requirement.support_boundary,
            paragraph_preview="Anchor not found.",
        )

    found_refs = tuple(sorted(extract_citations(paragraph)))
    missing_refs = set(requirement.expected_refs) - set(found_refs)
    status = "PASS" if not missing_refs else "WARN"
    return ClaimResult(
        area=requirement.area,
        status=status,
        expected_refs=requirement.expected_refs,
        found_refs=found_refs,
        support_boundary=requirement.support_boundary,
        paragraph_preview=compact(paragraph),
    )


def find_paragraph(text: str, pattern: str) -> str | None:
    for paragraph in re.split(r"\n\s*\n", text):
        if re.search(pattern, paragraph, flags=re.IGNORECASE):
            return paragraph.strip()
    return None


def evaluate_overclaim_boundaries(text: str) -> list[dict[str, str]]:
    body = text.split("## References", 1)[0]
    rows: list[dict[str, str]] = []
    for sentence in split_sentences(body):
        sentence_lower = sentence.lower()
        for pattern in RISK_PATTERNS:
            if pattern in sentence_lower:
                guarded = any(term in sentence_lower for term in GUARD_TERMS)
                rows.append(
                    {
                        "phrase": pattern,
                        "status": "PASS" if guarded else "WARN",
                        "sentence": compact(sentence, limit=180),
                    }
                )
    return rows


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text)
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", cleaned) if piece.strip()]


def compact(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def format_refs(refs: tuple[int, ...]) -> str:
    return ", ".join(str(ref) for ref in refs) if refs else "none"


def markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_cell(cell) for cell in row) + " |")
    return lines


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def build_report(
    overall: str,
    reference_count: int,
    all_cited_refs: tuple[int, ...],
    invalid_citations: tuple[int, ...],
    uncited_references: tuple[int, ...],
    claim_results: list[ClaimResult],
    risk_results: list[dict[str, str]],
) -> str:
    passed_claims = sum(1 for result in claim_results if result.status == "PASS")
    lines = [
        "# Citation Claim Coverage Audit",
        "",
        f"Overall status: **{overall}**",
        "",
        "This audit checks whether the main manuscript's key literature-dependent claims are supported by the selected reference set and whether high-risk interpretive phrases remain properly bounded.",
        "",
        "## Summary",
        "",
        f"- Selected references: {reference_count}",
        f"- Cited references in manuscript: {len(all_cited_refs)} ({format_refs(all_cited_refs)})",
        f"- Invalid citation numbers: {format_refs(invalid_citations)}",
        f"- Uncited selected references: {format_refs(uncited_references)}",
        f"- Claim requirements passing: {passed_claims}/{len(claim_results)}",
        f"- Overclaim boundary warnings: {sum(1 for row in risk_results if row['status'] != 'PASS')}",
        "",
        "## Claim-Level Coverage",
        "",
    ]
    lines.extend(
        markdown_table(
            ("Area", "Status", "Expected refs", "Found refs", "Support boundary"),
            [
                (
                    result.area,
                    result.status,
                    format_refs(result.expected_refs),
                    format_refs(result.found_refs),
                    result.support_boundary,
                )
                for result in claim_results
            ],
        )
    )
    lines.extend(["", "## Anchor Evidence", ""])
    for result in claim_results:
        lines.extend(
            [
                f"### {result.area}",
                "",
                f"- Status: {result.status}",
                f"- Paragraph anchor: {result.paragraph_preview}",
                "",
            ]
        )

    lines.extend(["## Overclaim Boundary Scan", ""])
    if risk_results:
        lines.extend(
            markdown_table(
                ("Phrase", "Status", "Sentence"),
                [(row["phrase"], row["status"], row["sentence"]) for row in risk_results],
            )
        )
    else:
        lines.append("No configured high-risk interpretive phrase was detected in the manuscript body.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current citation layer supports the major background, method, reporting-guideline, software, and interpretation claims checked here.",
            "- The audit does not replace target-journal reference-style formatting or author approval of the reference list.",
            "- If new acquisition, preprocessing, intervention, or clinical deployment claims are added later, this audit should be rerun and the selected references should be updated.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
