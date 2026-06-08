from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "results" / "tables" / "probast_tripod_ai_risk_audit.csv"
OUTPUT_MD = ROOT / "docs" / "probast_tripod_ai_risk_audit.md"


ROWS = [
    {
        "domain": "Participants",
        "assessment_item": "Source population, eligibility, and representativeness",
        "current_evidence": (
            "The manuscript reports an M1 clinical source frame of 29 patient records, a 28-patient EEG-indexed "
            "pool, and a 19-patient final supervised LOSO cohort. Inclusion/exclusion criteria, recruitment dates, "
            "stroke subtype criteria, and study site are still author-required."
        ),
        "risk_of_bias": "High",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": (
            "Keep conclusions exploratory; obtain protocol-level eligibility, site, recruitment window, and stroke "
            "subtype details before submission."
        ),
        "evidence_source": (
            "docs/cohort_characteristics.md; results/tables/participant_flow_safety_source_notes.csv; "
            "docs/author_submission_metadata_validation_report.md"
        ),
    },
    {
        "domain": "Predictors",
        "assessment_item": "Predictor availability before outcome and preprocessing transparency",
        "current_evidence": (
            "Baseline EO/EC PSD and WPLI features are derived before post-treatment outcome assessment. Feature "
            "scripts perform no additional filtering or artifact rejection after loading project-provided EEGLAB "
            ".set/.fdt files. Upstream acquisition hardware, reference, impedance, filtering, and artifact handling "
            "remain author-required."
        ),
        "risk_of_bias": "Some concerns",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": (
            "Retain the current pipeline-boundary statement and add verified upstream EEG acquisition/preprocessing "
            "details before final submission."
        ),
        "evidence_source": (
            "docs/methods_gap_resolution_from_project_files.md; docs/eeg_metadata_audit.md; "
            "src/eeg_recovery/features/psd.py; src/eeg_recovery/features/connectivity.py"
        ),
    },
    {
        "domain": "Outcome",
        "assessment_item": "Outcome definition, timing, and threshold derivation",
        "current_evidence": (
            "The binary endpoint is derived from a proportional-recovery residual with threshold equal to the "
            "supervised-cohort median residual of 1.5 points. FMA-UE timing is partially supported, but assessor "
            "training, blinding, and final scoring-reference details remain author-required."
        ),
        "risk_of_bias": "Some concerns",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": (
            "Present the endpoint as cohort-specific and exploratory; pre-specify the threshold in future cohorts "
            "and obtain assessor/timing details."
        ),
        "evidence_source": "docs/project_context.md; docs/cohort_characteristics.md; docs/author_metadata_project_prefill_report.md",
    },
    {
        "domain": "Sample size",
        "assessment_item": "Events per predictor and precision of performance estimates",
        "current_evidence": (
            "The supervised validation cohort contains 19 patients with 10 proportional-recovery and 9 poor-recovery "
            "labels. No external validation cohort is available."
        ),
        "risk_of_bias": "High",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": (
            "Avoid definitive superiority or clinical-readiness claims; emphasize bootstrap intervals, permutation "
            "testing, and the need for larger external validation."
        ),
        "evidence_source": "docs/cohort_characteristics.md; results/tables/table2_main_model_performance.csv",
    },
    {
        "domain": "Analysis",
        "assessment_item": "Validation unit and leakage control",
        "current_evidence": (
            "All reported primary validation uses patient-level LOSO predictions; seed-level and segment-level rows "
            "are not treated as independent patients. Integrity audit checks subject-level prediction rows and "
            "leakage safeguards."
        ),
        "risk_of_bias": "Low",
        "applicability_concern": "Low",
        "mitigation_or_required_action": "Maintain patient-level reporting and keep fold-local preprocessing/feature-selection safeguards in deposited code.",
        "evidence_source": "docs/prediction_validation_integrity_audit.md; results/tables/prediction_validation_integrity_audit.csv",
    },
    {
        "domain": "Analysis",
        "assessment_item": "Performance metrics, uncertainty, and calibration",
        "current_evidence": (
            "The manuscript reports discrimination, hard-label metrics, PR-AUC, Brier score, reliability curves, "
            "bootstrap intervals, paired comparisons, and permutation tests."
        ),
        "risk_of_bias": "Low",
        "applicability_concern": "Low",
        "mitigation_or_required_action": "Continue to report calibration-oriented metrics and uncertainty; avoid relying only on accuracy.",
        "evidence_source": (
            "results/tables/table2_main_model_performance.csv; "
            "results/statistics/model_metric_confidence_intervals.csv; results/figures/nature/figure2_performance_calibration.png"
        ),
    },
    {
        "domain": "Analysis",
        "assessment_item": "Model comparison and clinical incremental value",
        "current_evidence": (
            "The final EEG model has favorable numerical scores versus EEG baselines, but paired differences are not "
            "definitive. Clinical-only modelling is strong in this cohort, and EEG-plus-clinical analyses do not "
            "establish stable incremental value."
        ),
        "risk_of_bias": "Some concerns",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": (
            "Keep clinical-only and EEG-incremental analyses exploratory; do not claim robust clinical incremental "
            "utility."
        ),
        "evidence_source": "docs/clinical_incremental_value_results.md; results/statistics/clinical_incremental_paired_bootstrap_comparison.csv",
    },
    {
        "domain": "Missing data",
        "assessment_item": "Participant flow, unavailable follow-up, and discontinuation notes",
        "current_evidence": (
            "De-identified participant-flow source notes list the 29-record source frame, the 28-patient EEG-indexed "
            "pool, the 19-patient supervised cohort, and source-note categories for non-supervised or non-indexed "
            "entries. These notes do not constitute a complete adverse-event monitoring dataset."
        ),
        "risk_of_bias": "Some concerns",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": (
            "Use Supplementary Table 7 for transparent participant-flow reporting; obtain formal missing-data and "
            "adverse-event monitoring details from the author team."
        ),
        "evidence_source": "results/tables/participant_flow_safety_source_notes.csv; docs/participant_flow_and_safety_source_notes.md",
    },
    {
        "domain": "Explainability",
        "assessment_item": "Interpretability and biomarker claims",
        "current_evidence": (
            "Integrated gradients, SmoothGrad, occlusion, topographic maps, and WPLI summaries are provided. "
            "Edge- and channel-level findings are exploratory and not independently validated."
        ),
        "risk_of_bias": "Some concerns",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": (
            "Frame explainability as hypothesis-generating; avoid definitive channel-level or network biomarker "
            "claims until replicated."
        ),
        "evidence_source": "docs/manuscript_residual_aware_ssl_cnn_nature_polished.md; results/tables/table4_explainability_biomarkers.csv",
    },
    {
        "domain": "Reproducibility",
        "assessment_item": "Code, source data, and package integrity",
        "current_evidence": (
            "The manuscript package includes reproducibility scripts, source-data workbook, validation audits, figure "
            "exports, package manifest, and final submission gate. Final repository DOI, licence, and commit/version "
            "identifier remain author-required."
        ),
        "risk_of_bias": "Low",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": "Archive the final code/data package with DOI, licence, commit hash, and controlled-access route.",
        "evidence_source": "outputs/submission_package_20260601/submission_package_manifest.csv; docs/final_submission_gate_report.md",
    },
    {
        "domain": "Overall judgement",
        "assessment_item": "Overall risk-of-bias and applicability summary",
        "current_evidence": (
            "The model is internally validated with strong leakage safeguards and extensive reporting, but the "
            "supervised cohort is small, lacks external validation, has a cohort-derived threshold, and still needs "
            "author-confirmed protocol and ethics metadata."
        ),
        "risk_of_bias": "High",
        "applicability_concern": "Some concerns",
        "mitigation_or_required_action": (
            "Report as a pilot, exploratory, internally validated model; require larger prospective external "
            "validation before clinical deployment."
        ),
        "evidence_source": "docs/final_submission_gate_report.md; docs/tripod_ai_reporting_checklist.md",
    },
]


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "domain",
                "assessment_item",
                "current_evidence",
                "risk_of_bias",
                "applicability_concern",
                "mitigation_or_required_action",
                "evidence_source",
            ],
        )
        writer.writeheader()
        writer.writerows(ROWS)

    OUTPUT_MD.write_text(build_markdown(ROWS), encoding="utf-8")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print(f"risk_audit_rows={len(ROWS)}")


def build_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# PROBAST/TRIPOD+AI-Oriented Prediction Model Risk Audit",
        "",
        "This table is a conservative internal reporting audit for the current pilot prediction-model manuscript. It is not a substitute for a formal journal-specific checklist or independent PROBAST assessment.",
        "",
        "| Domain | Assessment item | Risk of bias | Applicability concern | Mitigation or required action | Evidence source |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {assessment_item} | {risk_of_bias} | {applicability_concern} | {mitigation_or_required_action} | {evidence_source} |".format(
                **{key: escape(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "- The strongest current safeguards are patient-level LOSO validation, explicit leakage controls, calibration-oriented metrics, bootstrap/permutation statistics, and package-level reproducibility audits.",
            "- The main remaining scientific risks are the small labelled cohort, lack of external validation, cohort-derived residual threshold, strong clinical-only exploratory baseline, and incomplete author-confirmed protocol metadata.",
            "- The manuscript should continue to frame the model as exploratory and internally validated, not clinically deployable.",
            "",
        ]
    )
    return "\n".join(lines)


def escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
