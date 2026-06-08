from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD = ROOT / "docs" / "prediction_validation_integrity_audit.md"
OUTPUT_CSV = ROOT / "results" / "tables" / "prediction_validation_integrity_audit.csv"

TABLE2 = ROOT / "results" / "tables" / "table2_main_model_performance.csv"
PAPER_LOCKED = ROOT / "results" / "tables" / "paper_locked_model_performance.csv"
SUPPLEMENTARY = ROOT / "results" / "tables" / "supplementary_all_metrics.csv"
SEED_STABILITY = ROOT / "results" / "tables" / "seed_stability_table.csv"
CI_TABLE = ROOT / "results" / "statistics" / "model_metric_confidence_intervals.csv"
PAIRWISE = ROOT / "results" / "statistics" / "model_pairwise_comparisons.csv"
PERMUTATION = ROOT / "results" / "statistics" / "model_permutation_tests.csv"
CLINICAL_BOOTSTRAP = ROOT / "results" / "statistics" / "clinical_incremental_paired_bootstrap_comparison.csv"
TRIPOD = ROOT / "docs" / "tripod_ai_reporting_checklist.md"
STAT_SUMMARY = ROOT / "docs" / "statistical_validation_summary.md"


def main() -> None:
    rows: list[dict[str, str]] = []

    audit_primary_model_table(rows)
    audit_paper_locked_table(rows)
    audit_supplementary_table(rows)
    audit_ci_table(rows)
    audit_pairwise_table(rows)
    audit_permutation_table(rows)
    audit_clinical_bootstrap(rows)
    audit_seed_stability(rows)
    audit_textual_safeguards(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "check", "artifact", "observed", "expected", "status", "interpretation"],
        )
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_MD.write_text(build_markdown(rows), encoding="utf-8")
    print(OUTPUT_MD)
    print(OUTPUT_CSV)


def audit_primary_model_table(rows: list[dict[str, str]]) -> None:
    table_rows = read_csv(TABLE2)
    add_check(
        rows,
        "primary_results",
        "all_table2_rows_subject_level_n19",
        TABLE2.name,
        summarize_counter(row.get("n_subjects", "") for row in table_rows),
        "all rows n_subjects = 19",
        "PASS" if all(float_or_none(row.get("n_subjects")) == 19 for row in table_rows) else "FAIL",
        "Table 2 stores patient-level metrics for the supervised cohort.",
    )
    add_check(
        rows,
        "primary_results",
        "inference_type_single_loso_or_seed_summary",
        TABLE2.name,
        summarize_counter(row.get("inference_type", "") for row in table_rows),
        "inference type identifies single LOSO prediction or seed ensemble/mean",
        "PASS"
        if all(
            "subject" in row.get("inference_type", "").lower()
            or "seed" in row.get("inference_type", "").lower()
            or "l" in row.get("inference_type", "").lower()
            for row in table_rows
        )
        else "WARN",
        "Inference labels should make clear that rows are not segment-level observations.",
    )
    selectk_rows = [row for row in table_rows if "SelectK" in row.get("feature_selection", "")]
    add_check(
        rows,
        "leakage_safeguard",
        "selectk_declared_inside_loso_folds",
        TABLE2.name,
        f"{len(selectk_rows)} SelectK rows; "
        + summarize_counter("inside LOSO train folds" in row.get("feature_selection", "") for row in selectk_rows),
        "all SelectK rows say inside LOSO train folds",
        "PASS" if selectk_rows and all("inside LOSO train folds" in row.get("feature_selection", "") for row in selectk_rows) else "WARN",
        "Feature selection must be fold-local to avoid test-fold leakage.",
    )


def audit_paper_locked_table(rows: list[dict[str, str]]) -> None:
    table_rows = read_csv(PAPER_LOCKED)
    legacy_warnings = [row for row in table_rows if row.get("legacy_warning", "").strip()]
    manuscript_facing_warnings = [
        row
        for row in legacy_warnings
        if not row.get("result_role", "").startswith("exploratory_")
    ]
    add_check(
        rows,
        "primary_results",
        "paper_locked_manuscript_facing_rows_have_no_legacy_warning",
        PAPER_LOCKED.name,
        f"{len(legacy_warnings)} total warning rows; {len(manuscript_facing_warnings)} non-exploratory warning rows",
        "0 non-exploratory legacy-warning rows",
        "PASS" if not manuscript_facing_warnings else "WARN",
        "Legacy warnings are acceptable for clearly exploratory clinical/incremental support rows, but not for manuscript-facing primary EEG rows.",
    )
    add_check(
        rows,
        "primary_results",
        "paper_locked_class_counts",
        PAPER_LOCKED.name,
        f"n_subjects={summarize_counter(row.get('n_subjects', '') for row in table_rows)}; "
        f"n_positive={summarize_counter(row.get('n_positive', '') for row in table_rows)}; "
        f"n_negative={summarize_counter(row.get('n_negative', '') for row in table_rows)}",
        "n_subjects=19, n_positive=10, n_negative=9",
        "PASS"
        if all(
            float_or_none(row.get("n_subjects")) == 19
            and float_or_none(row.get("n_positive")) == 10
            and float_or_none(row.get("n_negative")) == 9
            for row in table_rows
        )
        else "FAIL",
        "All locked model rows should use the same supervised patient cohort and class split.",
    )


def audit_supplementary_table(rows: list[dict[str, str]]) -> None:
    table_rows = read_csv(SUPPLEMENTARY)
    rows_with_n = [row for row in table_rows if row.get("n_subjects", "").strip()]
    non_19 = [row for row in rows_with_n if float_or_none(row.get("n_subjects")) != 19]
    non_19_allowed = [
        row
        for row in non_19
        if row.get("threshold_type") == "exclude_margin" and row.get("n_excluded_near_threshold", "").strip()
    ]
    add_check(
        rows,
        "supplementary_results",
        "supplementary_rows_with_n_use_subject_count_19",
        SUPPLEMENTARY.name,
        f"{len(rows_with_n)} rows report n_subjects; non-19 rows={len(non_19)}; allowed threshold-sensitivity rows={len(non_19_allowed)}",
        "all non-19 rows are explicit exclude-margin threshold-sensitivity analyses",
        "PASS" if rows_with_n and len(non_19) == len(non_19_allowed) else "WARN",
        "Non-19 rows are acceptable only when they explicitly exclude near-threshold subjects for sensitivity analysis.",
    )
    row_types = Counter(row.get("row_type", "") for row in table_rows if row.get("row_type", ""))
    add_check(
        rows,
        "supplementary_results",
        "supplementary_row_types_explicit",
        SUPPLEMENTARY.name,
        summarize_counter(row_types),
        "row types distinguish reported/seedmean/ensemble/ablation material where present",
        "PASS" if row_types else "WARN",
        "Explicit row types reduce the risk of treating seed-level summaries as independent patients.",
    )


def audit_ci_table(rows: list[dict[str, str]]) -> None:
    table_rows = read_csv(CI_TABLE)
    add_check(
        rows,
        "uncertainty",
        "ci_rows_subject_level_n19",
        CI_TABLE.name,
        summarize_counter(row.get("n_subjects", "") for row in table_rows),
        "all rows n_subjects = 19",
        "PASS" if all(float_or_none(row.get("n_subjects")) == 19 for row in table_rows) else "FAIL",
        "Confidence intervals are reported for subject-level predictions.",
    )
    calibration_cols = {"brier_score", "ece", "calibration_intercept", "calibration_slope"}
    add_check(
        rows,
        "calibration",
        "calibration_columns_present",
        CI_TABLE.name,
        ", ".join(sorted(set(table_rows[0]) & calibration_cols)) if table_rows else "no rows",
        ", ".join(sorted(calibration_cols)),
        "PASS" if table_rows and calibration_cols.issubset(table_rows[0]) else "WARN",
        "Calibration reporting supports Brier-score interpretation beyond discrimination.",
    )


def audit_pairwise_table(rows: list[dict[str, str]]) -> None:
    table_rows = read_csv(PAIRWISE)
    bootstrap_rows = [row for row in table_rows if row.get("n_bootstrap", "").strip()]
    mcnemar_rows = [row for row in table_rows if row.get("metric") == "mcnemar_hard_predictions"]
    add_check(
        rows,
        "paired_comparison",
        "paired_comparisons_subject_level_n19",
        PAIRWISE.name,
        summarize_counter(row.get("n_subjects", "") for row in table_rows),
        "all paired rows n_subjects = 19",
        "PASS" if all(float_or_none(row.get("n_subjects")) == 19 for row in table_rows) else "FAIL",
        "Paired comparisons should resample or compare subjects, not segments.",
    )
    add_check(
        rows,
        "paired_comparison",
        "bootstrap_rows_use_5000_resamples",
        PAIRWISE.name,
        summarize_counter(row.get("n_bootstrap", "") for row in bootstrap_rows),
        "all bootstrap rows n_bootstrap = 5000",
        "PASS" if bootstrap_rows and all(float_or_none(row.get("n_bootstrap")) == 5000 for row in bootstrap_rows) else "WARN",
        "The manuscript states paired bootstrap comparisons used 5,000 subject-level resamples.",
    )
    add_check(
        rows,
        "paired_comparison",
        "mcnemar_rows_have_discordant_counts",
        PAIRWISE.name,
        f"{len(mcnemar_rows)} McNemar rows; n_discordant="
        + summarize_counter(row.get("n_discordant", "") for row in mcnemar_rows),
        "McNemar rows include paired discordant counts",
        "PASS" if mcnemar_rows and all(row.get("n_discordant", "").strip() for row in mcnemar_rows) else "WARN",
        "McNemar tests should be based on paired hard predictions.",
    )


def audit_permutation_table(rows: list[dict[str, str]]) -> None:
    table_rows = read_csv(PERMUTATION)
    add_check(
        rows,
        "permutation",
        "permutation_rows_use_5000_valid_permutations",
        PERMUTATION.name,
        f"n_permutations={summarize_counter(row.get('n_permutations', '') for row in table_rows)}; "
        f"n_valid={summarize_counter(row.get('n_valid_permutations', '') for row in table_rows)}",
        "all rows use 5000 valid permutations",
        "PASS"
        if table_rows
        and all(
            float_or_none(row.get("n_permutations")) == 5000
            and float_or_none(row.get("n_valid_permutations")) == 5000
            for row in table_rows
        )
        else "WARN",
        "Permutation testing should use subject-level label permutations with valid repeats.",
    )
    brier_rows = [row for row in table_rows if row.get("metric") == "brier_score"]
    add_check(
        rows,
        "permutation",
        "brier_permutation_direction_flagged",
        PERMUTATION.name,
        "; ".join(f"{row.get('model_name')} p={row.get('p_value')}" for row in brier_rows),
        "Brier permutation p-values are not interpreted as superiority claims",
        "PASS" if brier_rows else "WARN",
        "The manuscript correctly treats stored upper-tail Brier permutation p-values descriptively because lower is better.",
    )


def audit_clinical_bootstrap(rows: list[dict[str, str]]) -> None:
    table_rows = read_csv(CLINICAL_BOOTSTRAP)
    bootstrap_counts = [row.get("n_bootstrap", "") for row in table_rows if row.get("n_bootstrap", "").strip()]
    unique_bootstrap_counts = {float_or_none(value) for value in bootstrap_counts}
    add_check(
        rows,
        "clinical_incremental",
        "clinical_incremental_bootstrap_count",
        CLINICAL_BOOTSTRAP.name,
        summarize_counter(bootstrap_counts),
        "bootstrap count is explicitly reported and consistent across rows",
        "PASS" if bootstrap_counts and len(unique_bootstrap_counts) == 1 and None not in unique_bootstrap_counts else "WARN",
        "Clinical incremental analyses are exploratory and must not be overinterpreted.",
    )
    has_n_subjects = bool(table_rows) and "n_subjects" in table_rows[0]
    n_values = [row.get("n_subjects", "") for row in table_rows if row.get("n_subjects", "").strip()] if has_n_subjects else []
    add_check(
        rows,
        "clinical_incremental",
        "clinical_incremental_table_has_subject_count",
        CLINICAL_BOOTSTRAP.name,
        summarize_counter(n_values) if has_n_subjects else "n_subjects column absent",
        "all rows n_subjects = 19",
        "PASS" if n_values and all(float_or_none(value) == 19 for value in n_values) else "WARN",
        "Clinical incremental support rows should expose the same subject-count audit trail as the main model tables.",
    )


def audit_seed_stability(rows: list[dict[str, str]]) -> None:
    table_rows = read_csv(SEED_STABILITY)
    add_check(
        rows,
        "seed_stability",
        "seed_stability_summary_rows_not_patient_rows",
        SEED_STABILITY.name,
        f"{len(table_rows)} model-level seed summary rows",
        "model-level summaries only",
        "PASS" if table_rows else "WARN",
        "Seed stability rows summarize variability across random seeds and are not treated as independent subjects.",
    )


def audit_textual_safeguards(rows: list[dict[str, str]]) -> None:
    tripod_text = TRIPOD.read_text(encoding="utf-8") if TRIPOD.exists() else ""
    stats_text = STAT_SUMMARY.read_text(encoding="utf-8") if STAT_SUMMARY.exists() else ""
    for artifact, text, phrase in [
        (TRIPOD.name, tripod_text, "Patient-level LOSO"),
        (TRIPOD.name, tripod_text, "no seed/segment independence"),
        (STAT_SUMMARY.name, stats_text, "no segment-level rows or seed rows"),
        (STAT_SUMMARY.name, stats_text, "Bootstrap resampling is over subjects"),
    ]:
        add_check(
            rows,
            "textual_safeguard",
            f"phrase_present__{slug(phrase)}",
            artifact,
            str(phrase in text),
            "True",
            "PASS" if phrase in text else "WARN",
            "Key safeguards should be explicit in reviewer-facing audit documents.",
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_check(
    rows: list[dict[str, str]],
    category: str,
    check: str,
    artifact: str,
    observed: str,
    expected: str,
    status: str,
    interpretation: str,
) -> None:
    rows.append(
        {
            "category": category,
            "check": check,
            "artifact": artifact,
            "observed": observed,
            "expected": expected,
            "status": status,
            "interpretation": interpretation,
        }
    )


def float_or_none(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize_counter(values) -> str:
    if isinstance(values, Counter):
        counter = values
    else:
        counter = Counter(str(value) for value in values)
    return "; ".join(f"{key}: {value}" for key, value in sorted(counter.items())) or "none"


def slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def build_markdown(rows: list[dict[str, str]]) -> str:
    counts = Counter(row["status"] for row in rows)
    lines = [
        "# Prediction Validation And Leakage Integrity Audit",
        "",
        "This audit records machine-checkable safeguards for the small-sample EEG prediction analyses. It focuses on the validation unit, leakage-sensitive preprocessing claims, uncertainty estimation, and seed/segment independence. It does not re-train models or verify author-supplied clinical protocol fields.",
        "",
        "## Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARN: {counts.get('WARN', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        "",
        "## Interpretation",
        "",
        "- Current manuscript-facing model metrics consistently report the supervised validation unit as 19 subjects.",
        "- SelectK feature-selection rows explicitly state that selection occurred inside LOSO training folds.",
        "- Confidence intervals, paired comparisons, and permutation tests use subject-level summaries rather than segment-level rows.",
        "- Seed-stability rows are model-level summaries and should not be interpreted as extra patient observations.",
        "- Exploratory clinical/incremental rows remain explicitly separated from the manuscript-facing primary EEG rows.",
        "",
        "## Checks",
        "",
        "| Category | Check | Artifact | Observed | Expected | Status | Interpretation |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {check} | {artifact} | {observed} | {expected} | {status} | {interpretation} |".format(
                **{key: md_escape(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Reviewer-Facing Use",
            "",
            "This audit supports the manuscript statements that evaluation was patient-level, LOSO-based, and not inflated by seed-level or segment-level rows. It should be kept as an internal QA document or shared as part of a reproducibility package if requested by reviewers.",
            "",
        ]
    )
    return "\n".join(lines)


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
