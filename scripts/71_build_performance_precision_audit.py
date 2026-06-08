from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "results" / "tables" / "performance_precision_audit.csv"
OUTPUT_MD = ROOT / "docs" / "performance_precision_audit.md"

FINAL_MODEL = "residual_aware_SSL_CNN_seedmean10"
NO_SSL = "no_SSL_CNN_updated_sub05_sub28_seedensemble10"
LOGISTIC = "ML_EEG_updated_no_selector_logistic_l1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt(value: float) -> str:
    return f"{value:.3f}"


def find(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def add_row(
    rows: list[dict[str, str]],
    category: str,
    item: str,
    observed: str,
    uncertainty: str,
    interpretation: str,
    evidence_source: str,
) -> None:
    rows.append(
        {
            "category": category,
            "item": item,
            "observed_value": observed,
            "uncertainty_or_resolution": uncertainty,
            "interpretation": interpretation,
            "evidence_source": evidence_source,
        }
    )


def main() -> None:
    ci_rows = read_csv(ROOT / "results" / "statistics" / "model_metric_confidence_intervals.csv")
    pair_rows = read_csv(ROOT / "results" / "statistics" / "model_pairwise_comparisons.csv")
    perm_rows = read_csv(ROOT / "results" / "statistics" / "model_permutation_tests.csv")

    final_ci = find(ci_rows, model_name=FINAL_MODEL)
    n_subjects = int(float(final_ci["n_subjects"]))
    n_proportional = 10
    n_poor = 9

    rows: list[dict[str, str]] = []

    add_row(
        rows,
        "validation_resolution",
        "Patient-level validation cohort",
        f"n={n_subjects}; proportional recovery={n_proportional}; poor recovery={n_poor}",
        "No external validation cohort available",
        "Internal LOSO estimates should be interpreted as pilot precision-limited estimates.",
        "results/statistics/model_metric_confidence_intervals.csv; results/tables/table1_cohort_characteristics.csv",
    )
    add_row(
        rows,
        "validation_resolution",
        "Accuracy one-subject resolution",
        fmt(1 / n_subjects),
        "One changed hard prediction moves accuracy by 5.3 percentage points",
        "Small changes in subject-level predictions can materially change reported accuracy.",
        "results/statistics/model_metric_confidence_intervals.csv",
    )
    add_row(
        rows,
        "validation_resolution",
        "Sensitivity one-case resolution",
        fmt(1 / n_proportional),
        "One changed proportional-recovery case moves sensitivity by 10.0 percentage points",
        "Class-specific operating characteristics are coarse in the current cohort.",
        "results/tables/table1_cohort_characteristics.csv",
    )
    add_row(
        rows,
        "validation_resolution",
        "Specificity one-case resolution",
        fmt(1 / n_poor),
        "One changed poor-recovery case moves specificity by 11.1 percentage points",
        "Class-specific operating characteristics are coarse in the current cohort.",
        "results/tables/table1_cohort_characteristics.csv",
    )

    for metric in ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]:
        value = as_float(final_ci, metric)
        low = as_float(final_ci, f"{metric}_low_ci")
        high = as_float(final_ci, f"{metric}_high_ci")
        width = high - low
        add_row(
            rows,
            "final_model_precision",
            f"Final model {metric}",
            fmt(value),
            f"bootstrap 95% interval {fmt(low)} to {fmt(high)}; width={fmt(width)}",
            "Estimate is compatible with a broad performance range and should not be framed as definitive clinical performance.",
            "results/statistics/model_metric_confidence_intervals.csv",
        )

    for reference, label in [(NO_SSL, "no-SSL CNN"), (LOGISTIC, "logistic EEG baseline")]:
        for metric in ["accuracy", "roc_auc", "pr_auc", "brier_score"]:
            pair = find(pair_rows, model_a=reference, model_b=FINAL_MODEL, metric=metric)
            diff = as_float(pair, "difference")
            low = as_float(pair, "ci_low")
            high = as_float(pair, "ci_high")
            p_value = as_float(pair, "p_value_two_sided")
            crosses_null = (low <= 0 <= high)
            null_text = "crosses the null" if crosses_null else "does not cross the null"
            add_row(
                rows,
                "paired_comparison_precision",
                f"Final model versus {label}: {metric}",
                f"difference={fmt(diff)}",
                f"bootstrap 95% interval {fmt(low)} to {fmt(high)}; {null_text}; p={fmt(p_value)}",
                "Paired evidence is insufficient for a definitive superiority claim in this pilot cohort.",
                "results/statistics/model_pairwise_comparisons.csv",
            )

    for metric in ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc"]:
        perm = find(perm_rows, model_name=FINAL_MODEL, metric=metric)
        add_row(
            rows,
            "above_chance_testing",
            f"Final model permutation test: {metric}",
            fmt(as_float(perm, "observed")),
            f"p={fmt(as_float(perm, 'p_value'))}; permutations={perm['n_valid_permutations']}",
            "Permutation testing supports above-chance internal discrimination, while paired comparisons do not establish model superiority.",
            "results/statistics/model_permutation_tests.csv",
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_report(rows)
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print(f"precision_audit_rows={len(rows)}")


def write_report(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Performance Precision Audit",
        "",
        "This audit summarizes the precision limits of the current 19-patient LOSO validation analysis. It is intended to make the manuscript's exploratory framing explicit and to prevent overinterpretation of internally validated performance estimates.",
        "",
        "| Category | Item | Observed value | Uncertainty or resolution | Interpretation | Evidence source |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {item} | {observed_value} | {uncertainty_or_resolution} | {interpretation} | {evidence_source} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "- Patient-level internal validation is appropriate for leakage control, but n=19 yields coarse operating-characteristic resolution.",
            "- The final model's permutation tests support above-chance discrimination in the current cohort.",
            "- Bootstrap paired comparisons cross the null for the principal model comparisons, so superiority and clinical utility claims remain unsupported.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
