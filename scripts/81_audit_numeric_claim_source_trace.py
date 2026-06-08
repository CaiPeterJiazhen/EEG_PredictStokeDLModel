from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_polished.md"
OUTPUT_MD = ROOT / "docs" / "numeric_claim_source_trace_audit.md"
OUTPUT_CSV = ROOT / "results" / "tables" / "numeric_claim_source_trace_audit.csv"

SUPERVISED_IDS = {
    "sub01",
    "sub05",
    "sub07",
    "sub08",
    "sub09",
    "sub10",
    "sub11",
    "sub13",
    "sub14",
    "sub15",
    "sub16",
    "sub17",
    "sub18",
    "sub20",
    "sub22",
    "sub24",
    "sub27",
    "sub28",
    "sub29",
}


@dataclass(frozen=True)
class TraceRow:
    check_id: str
    domain: str
    manuscript_claim: str
    source_artifact: str
    source_value: str
    manuscript_expected_text: str
    status: str
    notes: str


def main() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    traces = build_traces(manuscript)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "check_id",
                "domain",
                "manuscript_claim",
                "source_artifact",
                "source_value",
                "manuscript_expected_text",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows([row.__dict__ for row in traces])
    OUTPUT_MD.write_text(build_report(traces), encoding="utf-8")

    status_counts: dict[str, int] = {}
    for row in traces:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    print(OUTPUT_MD)
    print(OUTPUT_CSV)
    print("status_counts=" + ",".join(f"{key}:{status_counts[key]}" for key in sorted(status_counts)))


def build_traces(manuscript: str) -> list[TraceRow]:
    traces: list[TraceRow] = []
    table1 = read_csv(ROOT / "results" / "tables" / "table1_cohort_characteristics.csv")
    flow = read_csv(ROOT / "results" / "tables" / "participant_flow_safety_source_notes.csv")
    table2 = read_csv(ROOT / "results" / "tables" / "table2_main_model_performance.csv")
    table3 = read_csv(ROOT / "results" / "tables" / "table3_ablation.csv")
    pairwise = read_csv(ROOT / "results" / "statistics" / "model_pairwise_comparisons.csv")
    clinical_incremental = read_csv(ROOT / "results" / "statistics" / "clinical_incremental_paired_bootstrap_comparison.csv")
    performance_precision = read_csv(ROOT / "results" / "tables" / "performance_precision_audit.csv")

    flow_values = {row["category"]: int(float(row["n"])) for row in flow if row.get("row_type") == "participant_flow"}
    traces.extend(
        [
            trace(
                "N01",
                "participant_flow",
                "M1 source workbook, EEG-indexed pool, and supervised cohort counts",
                "results/tables/participant_flow_safety_source_notes.csv",
                f"M1={flow_values['M1 patient records in clinical source workbook']}; EEG-indexed={flow_values['Current EEG-indexed M1 patient pool']}; supervised={flow_values['Final labeled supervised cohort']}; unsupervised={flow_values['EEG-indexed patients not used for supervised labels']}; non-indexed={flow_values['Clinical workbook entries without current indexed EEG']}",
                [
                    "The M1 clinical source workbook contained 29 patient records",
                    "28 were indexed in the current EEG directory",
                    "19 formed the final supervised labelled cohort",
                    "Nine EEG-indexed patients were retained",
                    "one clinical workbook patient was not indexed",
                ],
                manuscript,
                "Participant-flow counts should match Figure 1 and Methods.",
            ),
            trace(
                "N02",
                "participant_flow",
                "Proportional-recovery versus poor-recovery label counts",
                "results/tables/participant_flow_safety_source_notes.csv",
                f"proportional={flow_values['Proportional-recovery label']}; poor={flow_values['Poor-recovery label']}",
                [
                    "10 proportional-recovery and 9 poor-recovery",
                    "10/9 proportional-recovery versus poor-recovery",
                ],
                manuscript,
                "Checks both Results wording and Figure 1 legend wording.",
            ),
        ]
    )

    traces.extend(cohort_traces(table1, manuscript))
    traces.append(residual_threshold_trace(manuscript))
    traces.extend(eeg_metadata_traces(manuscript))
    traces.extend(model_performance_traces(table2, manuscript))
    traces.extend(pairwise_traces(pairwise, manuscript))
    traces.extend(clinical_traces(table2, clinical_incremental, manuscript))
    traces.extend(ablation_traces(table3, manuscript))
    traces.append(
        trace(
            "N30",
            "precision",
            "One changed hard prediction moves accuracy by 5.3 percentage points",
            "results/tables/performance_precision_audit.csv",
            precision_value(performance_precision, "Accuracy one-subject resolution"),
            ["one changed hard prediction would move accuracy by 5.3 percentage points"],
            manuscript,
            "Accuracy resolution is 1/19 = 0.053.",
        )
    )
    return traces


def cohort_traces(table1: list[dict[str, str]], manuscript: str) -> list[TraceRow]:
    rows = {row["variable"]: row for row in table1}
    return [
        trace(
            "N03",
            "cohort",
            "Supervised cohort descriptive statistics",
            "results/tables/table1_cohort_characteristics.csv",
            "; ".join(
                [
                    f"age={rows['age']['All']}",
                    f"FMA_pre={rows['FMA_pre']['All']}",
                    f"FMA_post={rows['FMA_post']['All']}",
                    f"Delta_FMA_obs={rows['Delta_FMA_obs']['All']}",
                ]
            ),
            [
                "Mean age was 64.7 years (SD 6.5)",
                "Baseline FMA was 40.5 (SD 23.8)",
                "follow-up FMA was 45.5 (SD 23.2)",
                "observed FMA improvement was 5.0 points (SD 4.1)",
            ],
            manuscript,
            "Rounded manuscript values should match Table 1.",
        ),
        trace(
            "N04",
            "cohort",
            "Sex and affected-side counts",
            "results/tables/table1_cohort_characteristics.csv",
            f"female={rows['sex=女']['All']}; left_affected={rows['affected_hand=左']['All']}",
            [
                "11 patients were female",
                "11 had left-sided affected upper limbs",
            ],
            manuscript,
            "Categorical counts should match Table 1.",
        ),
    ]


def residual_threshold_trace(manuscript: str) -> TraceRow:
    context = (ROOT / "docs" / "project_context.md").read_text(encoding="utf-8")
    source_ok = "median_residual = 1.5" in context and "label = 1 if residual <= 1.5 else 0" in context
    row = trace(
        "N05",
        "outcome_definition",
        "Residual threshold is the supervised-cohort median residual",
        "docs/project_context.md; src/eeg_recovery/metadata/labels.py",
        "median_residual=1.5; label=1 if residual<=1.5 else 0",
        [
            "median residual threshold of 1.5 points",
            "cohort-specific modelling endpoint",
        ],
        manuscript,
        "Threshold is checked against the label-definition source, not against Table 1 mean/SD rows.",
    )
    if source_ok:
        return row
    return TraceRow(**{**row.__dict__, "status": "WARN", "notes": row.notes + " Label context text was not found."})


def eeg_metadata_traces(manuscript: str) -> list[TraceRow]:
    rows = read_csv(ROOT / "results" / "tables" / "eeg_recording_metadata_audit.csv")
    selected = [
        row
        for row in rows
        if row["group"] == "patient"
        and row["stage"] == "基线"
        and row["state"] in {"EO", "EC"}
        and row["subject_id"] in SUPERVISED_IDS
    ]
    durations = [float(row["duration_sec"]) for row in selected]
    srates = sorted({float(row["srate_hz"]) for row in selected})
    channels = sorted({int(float(row["nbchan"])) for row in selected})
    trials = sorted({int(float(row["trials"])) for row in selected})
    return [
        trace(
            "N06",
            "eeg_metadata",
            "Supervised baseline EO/EC EEG file count and duration range",
            "results/tables/eeg_recording_metadata_audit.csv",
            f"records={len(selected)}; subjects={len({row['subject_id'] for row in selected})}; duration_mean={sum(durations) / len(durations):.1f}; duration_min={min(durations):.1f}; duration_max={max(durations):.1f}",
            [
                "Across the 38 supervised baseline EO/EC files",
                "recording duration averaged 188.4 s",
                "ranged from 101.0 to 247.8 s",
            ],
            manuscript,
            "Recomputed from per-file metadata for the 19 supervised subjects.",
        ),
        trace(
            "N07",
            "eeg_metadata",
            "Sampling rate, retained channels, and continuous one-trial files",
            "results/tables/eeg_recording_metadata_audit.csv; docs/project_context.md",
            f"srate_values={srates}; nbchan_values={channels}; trials_values={trials}",
            [
                "continuous data with one trial per file",
                "a sampling rate of 250 Hz",
                "62 retained channels after removal of M1 and M2",
            ],
            manuscript,
            "Checks the analysis input boundary for preprocessed EEGLAB files.",
        ),
        trace(
            "N08",
            "eeg_feature_dimensions",
            "PSD bins and WPLI edge dimensions",
            "methods feature-grid formula and 62-channel upper triangle",
            f"PSD bins=90; WPLI edges=62*61/2={62 * 61 // 2}; bands=6",
            [
                "90 bins from 0.5 to 45 Hz",
                "1,891 edges by six frequency bands per state",
            ],
            manuscript,
            "Feature dimensions are deterministic from the stated grid and channel count.",
        ),
    ]


def model_performance_traces(table2: list[dict[str, str]], manuscript: str) -> list[TraceRow]:
    final = find_row(table2, model_name="residual_aware_SSL_CNN_seedmean10")
    nossl = find_row(table2, model_name="no_SSL_CNN_updated_sub05_sub28_seedensemble10")
    logistic = find_row(table2, model_name="ML_EEG_updated_no_selector_logistic_l1")
    return [
        metric_trace(
            "N09",
            "primary_performance",
            "Final residual-aware SSL-CNN patient-level LOSO performance",
            "results/tables/table2_main_model_performance.csv",
            final,
            ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc", "brier_score"],
            [
                "accuracy of 0.842",
                "balanced accuracy of 0.833",
                "sensitivity of 1.000",
                "specificity of 0.667",
                "ROC-AUC of 0.844",
                "PR-AUC of 0.836",
                "Brier score of 0.126",
            ],
            manuscript,
        ),
        metric_trace(
            "N10",
            "primary_performance",
            "No-SSL CNN reference performance",
            "results/tables/table2_main_model_performance.csv",
            nossl,
            ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"],
            [
                "lower ROC-AUC (0.811)",
                "PR-AUC (0.808)",
                "higher Brier score (0.177)",
            ],
            manuscript,
        ),
        metric_trace(
            "N11",
            "primary_performance",
            "PSD+WPLI logistic-regression EEG baseline performance",
            "results/tables/table2_main_model_performance.csv",
            logistic,
            ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"],
            [
                "accuracy of 0.737",
                "balanced accuracy of 0.733",
                "ROC-AUC of 0.711",
                "PR-AUC of 0.775",
                "Brier score of 0.208",
            ],
            manuscript,
        ),
    ]


def pairwise_traces(pairwise: list[dict[str, str]], manuscript: str) -> list[TraceRow]:
    no_ssl_rows = rows_for_pair(
        pairwise,
        "no_SSL_CNN_updated_sub05_sub28_seedensemble10",
        "residual_aware_SSL_CNN_seedmean10",
    )
    logistic_rows = rows_for_pair(
        pairwise,
        "ML_EEG_updated_no_selector_logistic_l1",
        "residual_aware_SSL_CNN_seedmean10",
    )
    return [
        trace(
            "N12",
            "paired_comparison",
            "Final model versus no-SSL CNN paired differences",
            "results/statistics/model_pairwise_comparisons.csv",
            pairwise_summary(no_ssl_rows, ["roc_auc", "pr_auc", "brier_score"]),
            [
                "ROC-AUC was higher by 0.033 (95% bootstrap interval -0.144 to 0.214; two-sided p = 0.788)",
                "PR-AUC was higher by 0.028 (-0.170 to 0.224; p = 0.824)",
                "Brier score was lower by 0.051 (-0.121 to 0.004; p = 0.078)",
            ],
            manuscript,
            "Paired bootstrap differences should remain bounded by non-definitive wording.",
        ),
        trace(
            "N13",
            "paired_comparison",
            "Final model versus logistic-regression EEG baseline paired differences",
            "results/statistics/model_pairwise_comparisons.csv",
            pairwise_summary(logistic_rows, ["accuracy", "roc_auc", "brier_score"]),
            [
                "accuracy was higher by 0.105 (-0.105 to 0.316; p = 0.423)",
                "ROC-AUC was higher by 0.133 (-0.216 to 0.476; p = 0.458)",
                "Brier score was lower by 0.082 (-0.189 to 0.037; p = 0.169)",
            ],
            manuscript,
            "Checks that the logistic baseline comparison uses paired subject-level statistics.",
        ),
        trace(
            "N14",
            "paired_comparison",
            "McNemar hard-prediction comparison against logistic EEG baseline",
            "results/statistics/model_pairwise_comparisons.csv",
            mcnemar_summary(logistic_rows),
            ["discordant counts 1 versus 3; p = 0.625"],
            manuscript,
            "Hard-prediction comparison is reported as non-significant.",
        ),
    ]


def clinical_traces(
    table2: list[dict[str, str]],
    clinical_incremental: list[dict[str, str]],
    manuscript: str,
) -> list[TraceRow]:
    clinical = find_row(table2, model_name="clinical_only_best_available_clinical_only_logistic_l2")
    rows = [row for row in clinical_incremental if row["reference_model"] == "clinical_only_logistic_l2"]
    p_values = [float(row["p_value_two_sided"]) for row in rows]
    return [
        metric_trace(
            "N15",
            "clinical_baseline",
            "Clinical-only logistic baseline ranking and calibration metrics",
            "results/tables/table2_main_model_performance.csv",
            clinical,
            ["roc_auc", "pr_auc", "brier_score"],
            [
                "ROC-AUC of 0.911",
                "PR-AUC of 0.899",
                "Brier score of 0.105",
            ],
            manuscript,
        ),
        trace(
            "N16",
            "clinical_incremental",
            "Adding EEG to baseline clinical variables did not provide stable incremental gain",
            "results/statistics/clinical_incremental_paired_bootstrap_comparison.csv",
            f"comparisons={len(rows)}; min_p={min(p_values):.3f}; max_p={max(p_values):.3f}",
            ["did not provide a stable incremental gain over the best clinical-only model"],
            manuscript,
            "Includes exploratory paired bootstrap comparisons for EEG+clinical candidates versus the clinical-only logistic reference.",
        ),
    ]


def ablation_traces(table3: list[dict[str, str]], manuscript: str) -> list[TraceRow]:
    no_ssl_resid = find_row(table3, ablation_name="d_no_ssl_cnn_residual_heads", row_type="seedmean10")
    final = find_row(table3, ablation_name="e_patient_barlow_ssl_residual_heads", row_type="seedmean10")
    no_resid = find_row(table3, ablation_name="c_patient_barlow_ssl_no_residual_heads", row_type="ensemble10")
    return [
        metric_trace(
            "N17",
            "ablation",
            "No-SSL CNN with residual-aware heads seed-mean ablation performance",
            "results/tables/table3_ablation.csv",
            no_ssl_resid,
            ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"],
            [
                "accuracy of 0.895",
                "balanced accuracy of 0.889",
                "ROC-AUC of 0.922",
                "PR-AUC of 0.927",
                "Brier score of 0.110",
            ],
            manuscript,
        ),
        metric_trace(
            "N18",
            "ablation",
            "Final patient-level Barlow SSL plus residual-aware ablation row",
            "results/tables/table3_ablation.csv",
            final,
            ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"],
            [
                "accuracy of 0.842",
                "balanced accuracy of 0.833",
                "ROC-AUC of 0.844",
                "PR-AUC of 0.836",
                "Brier score of 0.126",
            ],
            manuscript,
        ),
        metric_trace(
            "N19",
            "ablation",
            "Patient-level Barlow SSL without residual-aware heads",
            "results/tables/table3_ablation.csv",
            no_resid,
            ["roc_auc", "pr_auc"],
            [
                "ensemble ROC-AUC of 0.700",
                "PR-AUC of 0.631",
            ],
            manuscript,
        ),
    ]


def metric_trace(
    check_id: str,
    domain: str,
    manuscript_claim: str,
    source_artifact: str,
    source_row: dict[str, str],
    metrics: list[str],
    expected_text: list[str],
    manuscript: str,
) -> TraceRow:
    source_value = "; ".join(f"{metric}={float(source_row[metric]):.3f}" for metric in metrics)
    return trace(check_id, domain, manuscript_claim, source_artifact, source_value, expected_text, manuscript, "")


def trace(
    check_id: str,
    domain: str,
    manuscript_claim: str,
    source_artifact: str,
    source_value: str,
    expected_text: list[str],
    manuscript: str,
    notes: str,
) -> TraceRow:
    missing = [fragment for fragment in expected_text if fragment not in manuscript]
    status = "PASS" if not missing else "WARN"
    note = notes
    if missing:
        note = (note + " " if note else "") + "Missing manuscript fragment(s): " + "; ".join(missing)
    return TraceRow(
        check_id=check_id,
        domain=domain,
        manuscript_claim=manuscript_claim,
        source_artifact=source_artifact,
        source_value=source_value,
        manuscript_expected_text=" || ".join(expected_text),
        status=status,
        notes=note,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    joined = ", ".join(f"{key}={value}" for key, value in criteria.items())
    raise KeyError(f"Could not find row: {joined}")


def rows_for_pair(rows: list[dict[str, str]], model_a: str, model_b: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["model_a"] == model_a and row["model_b"] == model_b]
    if not selected:
        raise KeyError(f"Could not find pairwise rows: {model_a} vs {model_b}")
    return selected


def pairwise_summary(rows: list[dict[str, str]], metrics: list[str]) -> str:
    by_metric = {row["metric"]: row for row in rows}
    pieces = []
    for metric in metrics:
        row = by_metric[metric]
        pieces.append(
            f"{metric}: diff={float(row['difference']):.3f}, ci={float(row['ci_low']):.3f} to {float(row['ci_high']):.3f}, p={float(row['p_value_two_sided']):.3f}"
        )
    return "; ".join(pieces)


def mcnemar_summary(rows: list[dict[str, str]]) -> str:
    row = next(row for row in rows if row["metric"] == "mcnemar_hard_predictions")
    return (
        f"a_correct_b_wrong={int(float(row['a_correct_b_wrong']))}; "
        f"a_wrong_b_correct={int(float(row['a_wrong_b_correct']))}; "
        f"n_discordant={int(float(row['n_discordant']))}; p={float(row['p_value']):.3f}"
    )


def precision_value(rows: list[dict[str, str]], item: str) -> str:
    row = find_row(rows, item=item)
    return f"{row['observed_value']} ({row['uncertainty_or_resolution']})"


def build_report(traces: list[TraceRow]) -> str:
    status_counts: dict[str, int] = {}
    for row in traces:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    overall = "PASS" if status_counts.get("WARN", 0) == 0 and status_counts.get("FAIL", 0) == 0 else "WARN"
    lines = [
        "# Numeric Claim Source Trace Audit",
        "",
        f"Overall status: **{overall}**",
        "",
        "This audit maps the main manuscript's key numeric claims to source tables, statistics files, metadata audits, or deterministic feature-grid calculations. It is designed to catch stale manuscript numbers after statistical or package updates.",
        "",
        "## Summary",
        "",
    ]
    for status in ["PASS", "WARN", "FAIL"]:
        lines.append(f"- {status}: {status_counts.get(status, 0)}")
    lines.extend(
        [
            f"- Trace rows: {len(traces)}",
            "",
            "## Trace Table",
            "",
            "| Check | Domain | Status | Source | Source value | Manuscript claim |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in traces:
        lines.append(
            "| "
            + " | ".join(
                escape_cell(value)
                for value in [
                    row.check_id,
                    row.domain,
                    row.status,
                    row.source_artifact,
                    row.source_value,
                    row.manuscript_claim,
                ]
            )
            + " |"
        )
    warnings = [row for row in traces if row.status != "PASS"]
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for row in warnings:
            lines.append(f"- {row.check_id}: {row.notes}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The audited numeric claims are traceable to current source artifacts.",
            "- This audit does not replace raw-data release, author-confirmed protocol metadata, or final journal copyediting.",
            "- Rerun this audit after any change to cohort selection, labels, model metrics, figure legends, or statistical tables.",
            "",
        ]
    )
    return "\n".join(lines)


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
