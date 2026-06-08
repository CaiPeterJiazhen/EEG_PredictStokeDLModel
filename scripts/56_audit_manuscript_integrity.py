from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATURE_MD = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_polished.md"
NATURE_CLEAN_MD = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_clean_placeholder.md"
JNE_MD = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_jne_structured.md"
JNE_CLEAN_MD = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md"
OUTPUT_MD = ROOT / "docs" / "manuscript_integrity_audit.md"
OUTPUT_CSV = ROOT / "results" / "tables" / "manuscript_integrity_audit.csv"


@dataclass
class AuditRow:
    category: str
    artifact: str
    check: str
    observed: str
    expected: str
    status: str
    notes: str = ""


def main() -> None:
    rows: list[AuditRow] = []
    text = NATURE_MD.read_text(encoding="utf-8")

    audit_citations(text, rows)
    audit_tables_and_figures(text, rows)
    audit_key_claims(text, rows)
    audit_manuscript_variants(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "artifact", "check", "observed", "expected", "status", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    OUTPUT_MD.write_text(build_markdown(rows), encoding="utf-8")
    print(OUTPUT_MD)
    print(OUTPUT_CSV)

    if any(row.status == "FAIL" for row in rows):
        sys.exit(1)


def audit_citations(text: str, rows: list[AuditRow]) -> None:
    reference_text = section(text, "## References", "## Author queries before journal submission")
    manuscript_text = text.split("## References", maxsplit=1)[0]
    reference_numbers = [int(match.group(1)) for match in re.finditer(r"^(\d+)\.\s", reference_text, re.M)]
    expected_numbers = list(range(1, max(reference_numbers) + 1)) if reference_numbers else []
    add(
        rows,
        "citation",
        "References",
        "numbering_contiguous",
        observed=range_text(reference_numbers),
        expected=range_text(expected_numbers),
        status="PASS" if reference_numbers == expected_numbers else "FAIL",
        notes="Reference numbering should remain contiguous after citation edits.",
    )

    cited_numbers = sorted(expand_citations(manuscript_text))
    missing_refs = sorted(set(cited_numbers) - set(reference_numbers))
    uncited_refs = sorted(set(reference_numbers) - set(cited_numbers))
    add(
        rows,
        "citation",
        "Main text",
        "all_numeric_citations_have_reference_entries",
        observed=", ".join(map(str, missing_refs)) if missing_refs else "none missing",
        expected="none missing",
        status="PASS" if not missing_refs else "FAIL",
    )
    add(
        rows,
        "citation",
        "References",
        "uncited_reference_entries",
        observed=", ".join(map(str, uncited_refs)) if uncited_refs else "none",
        expected="none preferred",
        status="PASS" if not uncited_refs else "WARN",
        notes="Uncited references are not fatal for a draft but should be removed before journal upload.",
    )

    refs_without_id = []
    for line in reference_text.splitlines():
        if re.match(r"^\d+\.\s", line) and "doi:" not in line.lower() and "http" not in line.lower():
            refs_without_id.append(line.split(".", maxsplit=1)[0])
    add(
        rows,
        "citation",
        "References",
        "doi_or_url_present",
        observed=", ".join(refs_without_id) if refs_without_id else "all reference entries include DOI or URL",
        expected="all reference entries include DOI or URL",
        status="PASS" if not refs_without_id else "WARN",
    )


def audit_tables_and_figures(text: str, rows: list[AuditRow]) -> None:
    table_paths = re.findall(r"Source file: `([^`]+)`", section(text, "## Tables", "## Figure legends"))
    for source in table_paths:
        path = ROOT / source
        add(
            rows,
            "table_source",
            source,
            "table_source_exists",
            observed=str(path.exists()),
            expected="True",
            status="PASS" if path.exists() else "FAIL",
        )

    figure_paths = re.findall(r"Figure file: `([^`]+)`", section(text, "## Figure legends", "## References"))
    for source in figure_paths:
        path = ROOT / source
        add(
            rows,
            "figure_source",
            source,
            "figure_file_exists",
            observed=str(path.exists()),
            expected="True",
            status="PASS" if path.exists() else "FAIL",
        )
        if path.exists() and path.parent == ROOT / "results" / "figures" / "nature":
            missing = []
            for suffix in [".png", ".svg", ".pdf", ".tiff"]:
                sibling = path.with_suffix(suffix)
                if not sibling.exists():
                    missing.append(suffix)
            add(
                rows,
                "figure_export",
                source,
                "journal_export_set",
                observed=", ".join(missing) if missing else "png, svg, pdf, tiff present",
                expected="png, svg, pdf, tiff present",
                status="PASS" if not missing else "WARN",
            )

    connectivity_manifest = ROOT / "results" / "figures" / "explainability" / "mne_wpli_connectivity" / "mne_wpli_connectivity_manifest.csv"
    if connectivity_manifest.exists():
        manifest_rows = read_csv(connectivity_manifest)
        svg_count = sum(1 for row in manifest_rows if Path(row.get("svg", "")).exists())
        pdf_count = sum(1 for row in manifest_rows if Path(row.get("pdf", "")).exists())
        png_count = sum(1 for row in manifest_rows if Path(row.get("png", "")).exists())
        expected = len(manifest_rows)
        status = "PASS" if expected >= 12 and svg_count == pdf_count == png_count == expected else "WARN"
        add(
            rows,
            "figure_export",
            connectivity_manifest.relative_to(ROOT).as_posix(),
            "mne_connectivity_panel_exports",
            observed=f"{expected} rows; png={png_count}, svg={svg_count}, pdf={pdf_count}",
            expected=">=12 rows with png, svg, and pdf",
            status=status,
        )
    else:
        add(
            rows,
            "figure_export",
            connectivity_manifest.relative_to(ROOT).as_posix(),
            "mne_connectivity_panel_exports",
            observed="missing",
            expected="present",
            status="FAIL",
        )


def audit_key_claims(text: str, rows: list[AuditRow]) -> None:
    patient_rows = {row["variable"]: row for row in read_csv(ROOT / "results" / "tables" / "patient_characteristics_table.csv")}
    model_rows = {row["source_model"]: row for row in read_csv(ROOT / "results" / "tables" / "model_performance_main_table.csv")}
    table2_rows = {row["model_name"]: row for row in read_csv(ROOT / "results" / "tables" / "table2_main_model_performance.csv")}
    ablation_rows = read_csv(ROOT / "results" / "tables" / "table3_ablation.csv")
    permutation_rows = read_csv(ROOT / "results" / "statistics" / "model_permutation_tests.csv")

    check_text_value(rows, text, "cohort", "supervised_n", "19", "19 patients")
    check_text_value(rows, text, "cohort", "all_patient_eeg_pool_n", "28", "28 patients")
    age_mean, age_sd = rounded_mean_sd_parts(patient_rows["age"]["all"])
    fma_pre_mean, fma_pre_sd = rounded_mean_sd_parts(patient_rows["FMA_pre"]["all"])
    fma_post_mean, fma_post_sd = rounded_mean_sd_parts(patient_rows["FMA_post"]["all"])
    delta_mean, delta_sd = rounded_mean_sd_parts(patient_rows["Delta_FMA_obs"]["all"])
    check_text_value(rows, text, "cohort", "age_mean_sd", patient_rows["age"]["all"], f"Mean age was {age_mean} years (SD {age_sd})")
    check_text_value(rows, text, "cohort", "baseline_fma_mean_sd", patient_rows["FMA_pre"]["all"], f"Baseline FMA was {fma_pre_mean} (SD {fma_pre_sd})")
    check_text_value(rows, text, "cohort", "followup_fma_mean_sd", patient_rows["FMA_post"]["all"], f"follow-up FMA was {fma_post_mean} (SD {fma_post_sd})")
    check_text_value(rows, text, "cohort", "observed_delta_mean_sd", patient_rows["Delta_FMA_obs"]["all"], f"observed FMA improvement was {delta_mean} points (SD {delta_sd})")
    check_text_value(rows, text, "cohort", "female_count", patient_rows["sex=女"]["all"], f"{patient_rows['sex=女']['all']} patients were female")
    check_text_value(rows, text, "cohort", "left_affected_count", patient_rows["affected_hand=左"]["all"], f"{patient_rows['affected_hand=左']['all']} had left-sided affected upper limbs")
    check_text_value(rows, text, "cohort", "label_distribution", "10/9", "10 proportional-recovery and 9 poor-recovery")

    final_model = model_rows["residual_aware_SSL_CNN_seedmean10"]
    no_ssl = model_rows["no_SSL_CNN_updated_sub05_sub28_seedensemble10"]
    logistic = model_rows["ML_EEG_updated_no_selector_logistic_l1"]
    for model_name, source_row in [
        ("final_residual_aware_ssl_cnn", final_model),
        ("no_ssl_cnn", no_ssl),
        ("logistic_eeg_baseline", logistic),
    ]:
        for metric in ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]:
            value = round_float(source_row[metric], 3)
            check_text_value(rows, text, "model_metric", f"{model_name}_{metric}", value, value)

    clinical = table2_rows["clinical_only_best_available_clinical_only_logistic_l2"]
    for metric in ["roc_auc", "pr_auc", "brier_score"]:
        value = round_float(clinical[metric], 3)
        check_text_value(rows, text, "clinical_metric", f"clinical_only_logistic_l2_{metric}", value, value)

    for metric, expected_text in [
        ("accuracy", "p = 0.004"),
        ("balanced_accuracy", "p = 0.003"),
        ("roc_auc", "p = 0.005"),
        ("pr_auc", "p = 0.015"),
    ]:
        source = next(
            row
            for row in permutation_rows
            if row["model_name"] == "residual_aware_SSL_CNN_seedmean10" and row["metric"] == metric
        )
        add(
            rows,
            "statistics",
            "model_permutation_tests.csv",
            f"final_model_{metric}_permutation_p_reported",
            observed=expected_text if expected_text in text else "not found",
            expected=f"{round_float(source['p_value'], 3)} rounded and reported",
            status="PASS" if expected_text in text else "FAIL",
        )

    ablation_expectations = [
        ("d_no_ssl_cnn_residual_heads", "seedmean10", ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]),
        ("e_patient_barlow_ssl_residual_heads", "seedmean10", ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]),
        ("c_patient_barlow_ssl_no_residual_heads", "ensemble10", ["roc_auc", "pr_auc"]),
        ("psd_only", "", ["roc_auc"]),
    ]
    for ablation_name, row_type, metrics in ablation_expectations:
        row = find_ablation(ablation_rows, ablation_name, row_type)
        for metric in metrics:
            value = round_float(row[metric], 3)
            check_text_value(rows, text, "ablation", f"{ablation_name}_{row_type or 'row'}_{metric}", value, value)


def audit_manuscript_variants(rows: list[AuditRow]) -> None:
    for path in [NATURE_MD, JNE_MD, NATURE_CLEAN_MD, JNE_CLEAN_MD]:
        text = path.read_text(encoding="utf-8")
        add(
            rows,
            "manuscript_variant",
            path.relative_to(ROOT).as_posix(),
            "old_overclaim_phrase_removed",
            observed=str("can improve patient-level prediction" in text),
            expected="False",
            status="PASS" if "can improve patient-level prediction" not in text else "FAIL",
        )
        has_pretraining_caveat = "self-supervised pretraining" in text and (
            "not established" in text or "do not establish" in text or "independent benefit" in text
        )
        add(
            rows,
            "manuscript_variant",
            path.relative_to(ROOT).as_posix(),
            "ssl_pretraining_caveat_present",
            observed=str(has_pretraining_caveat),
            expected="True",
            status="PASS" if has_pretraining_caveat else "WARN",
        )

    for path in [NATURE_CLEAN_MD, JNE_CLEAN_MD]:
        text = path.read_text(encoding="utf-8")
        has_author_query = "Author information required before submission" in text or "## Author queries before journal submission" in text
        add(
            rows,
            "manuscript_variant",
            path.relative_to(ROOT).as_posix(),
            "clean_variant_author_query_removed",
            observed=str(has_author_query),
            expected="False",
            status="PASS" if not has_author_query else "FAIL",
        )


def check_text_value(
    rows: list[AuditRow],
    text: str,
    category: str,
    check: str,
    source_value: str,
    expected_phrase: str,
) -> None:
    add(
        rows,
        category,
        NATURE_MD.relative_to(ROOT).as_posix(),
        check,
        observed="found" if expected_phrase in text else "not found",
        expected=f"{expected_phrase} from source value {source_value}",
        status="PASS" if expected_phrase in text else "FAIL",
    )


def find_ablation(rows: list[dict[str, str]], ablation_name: str, row_type: str) -> dict[str, str]:
    for row in rows:
        if row["ablation_name"] == ablation_name and (not row_type or row["row_type"] == row_type):
            return row
    raise KeyError((ablation_name, row_type))


def build_markdown(rows: list[AuditRow]) -> str:
    counts = Counter(row.status for row in rows)
    lines = [
        "# Manuscript Integrity Audit",
        "",
        "This audit checks manuscript-source consistency for the current polished Nature manuscript and its submission variants. It covers citation numbering, reference identifiers, table and figure source paths, key numeric claims, manuscript-variant safeguards, and export availability. It does not verify author-supplied ethics, consent, EEG acquisition, raw preprocessing, or repository DOI fields.",
        "",
        "## Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARN: {counts.get('WARN', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        "",
        "## Checks",
        "",
        "| Category | Artifact | Check | Observed | Expected | Status | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                sanitize_md(value)
                for value in [
                    row.category,
                    row.artifact,
                    row.check,
                    row.observed,
                    row.expected,
                    row.status,
                    row.notes,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `FAIL` indicates a manuscript-source inconsistency that should be fixed before submission.",
            "- `WARN` indicates a draft-level issue that may be acceptable temporarily but should be checked before journal upload.",
            "- The author-query text in the working Nature and JNE manuscripts is intentional; clean placeholder variants are checked separately to ensure those lines are removed.",
            "",
        ]
    )
    return "\n".join(lines)


def section(text: str, start: str, end: str | None = None) -> str:
    if start not in text:
        return ""
    content = text.split(start, maxsplit=1)[1]
    if end and end in content:
        content = content.split(end, maxsplit=1)[0]
    return content


def expand_citations(text: str) -> set[int]:
    cited: set[int] = set()
    for bracket in re.findall(r"\[([0-9,\-\s]+)\]", text):
        for part in bracket.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, finish = [int(value) for value in part.split("-", maxsplit=1)]
                cited.update(range(start, finish + 1))
            else:
                cited.add(int(part))
    return cited


def rounded_mean_sd_parts(value: str) -> tuple[str, str]:
    match = re.match(r"([0-9.]+)\s+\(([0-9.]+)\)", value)
    if not match:
        return value, ""
    return f"{float(match.group(1)):.1f}", f"{float(match.group(2)):.1f}"


def round_float(value: str, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def range_text(values: list[int]) -> str:
    if not values:
        return "none"
    return f"{min(values)}-{max(values)} ({len(values)} entries)"


def sanitize_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def add(
    rows: list[AuditRow],
    category: str,
    artifact: str,
    check: str,
    observed: str,
    expected: str,
    status: str,
    notes: str = "",
) -> None:
    rows.append(AuditRow(category, artifact, check, observed, expected, status, notes))


if __name__ == "__main__":
    main()
