from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "results" / "tables" / "model_reporting_card.csv"
OUTPUT_MD = ROOT / "docs" / "model_reporting_card.md"


ROWS = [
    {
        "card_section": "Intended use",
        "reporting_item": "Prediction task",
        "current_value": "Patient-level prediction of upper-limb proportional-recovery status after stroke using baseline resting-state EEG.",
        "evidence_source": "docs/manuscript_residual_aware_ssl_cnn_nature_polished.md; results/tables/table2_main_model_performance.csv",
        "reviewer_risk": "The model is internally validated only and should not be framed as deployable.",
        "manuscript_action": "Report as an exploratory prognostic model requiring external validation.",
    },
    {
        "card_section": "Intended use",
        "reporting_item": "Clinical decision context",
        "current_value": "Candidate stratification support for rehabilitation-response research, not an approved clinical decision tool.",
        "evidence_source": "docs/claim_strength_audit.md; docs/final_submission_gate_report.md",
        "reviewer_risk": "Overstating clinical readiness would exceed the evidence.",
        "manuscript_action": "Retain the current prospective-validation and no-deployment wording.",
    },
    {
        "card_section": "Population",
        "reporting_item": "Supervised validation cohort",
        "current_value": "19 stroke patients with baseline EEG and complete baseline/follow-up FMA-UE.",
        "evidence_source": "results/tables/table1_cohort_characteristics.csv; docs/cohort_characteristics.md",
        "reviewer_risk": "Small sample size limits precision, calibration, and generalizability.",
        "manuscript_action": "State n=19 wherever performance is reported and avoid population-level claims.",
    },
    {
        "card_section": "Population",
        "reporting_item": "Unlabelled or non-supervised EEG pool",
        "current_value": "28 EEG-indexed patients were available, with 9 outside supervised outcome modelling.",
        "evidence_source": "results/tables/participant_flow_safety_source_notes.csv; docs/participant_flow_and_safety_source_notes.md",
        "reviewer_risk": "The additional EEG pool cannot substitute for labelled external validation.",
        "manuscript_action": "Describe as self-supervised/descriptive support only.",
    },
    {
        "card_section": "Outcome",
        "reporting_item": "Endpoint definition",
        "current_value": "Label derived from residual = 0.7 x (66 - baseline FMA-UE) - observed FMA-UE change; threshold was cohort median residual 1.5.",
        "evidence_source": "docs/project_context.md; results/tables/table1_cohort_characteristics.csv",
        "reviewer_risk": "Median-derived endpoint may not transport to future cohorts.",
        "manuscript_action": "Call the threshold cohort-specific and require prospective pre-specification.",
    },
    {
        "card_section": "Predictors",
        "reporting_item": "Model inputs included",
        "current_value": "Baseline eyes-open and eyes-closed PSD plus WPLI features from preprocessed EEGLAB files.",
        "evidence_source": "docs/methods_gap_resolution_from_project_files.md; docs/eeg_metadata_audit.md",
        "reviewer_risk": "Upstream acquisition and preprocessing details remain author-confirmed.",
        "manuscript_action": "Keep the current feature-pipeline boundary explicit.",
    },
    {
        "card_section": "Predictors",
        "reporting_item": "Inputs excluded from EEG model",
        "current_value": "Post-treatment outcomes, observed improvement, predicted improvement, residuals, and labels were excluded from model inputs.",
        "evidence_source": "docs/manuscript_residual_aware_ssl_cnn_nature_polished.md; docs/prediction_validation_integrity_audit.md",
        "reviewer_risk": "Outcome leakage would invalidate the model if not clearly excluded.",
        "manuscript_action": "Retain explicit exclusion language in Methods.",
    },
    {
        "card_section": "Training",
        "reporting_item": "Architecture",
        "current_value": "Multimodal CNN with PSD and WPLI branches, gated 32-dimensional fusion, patient-level Barlow Twins pretraining, and residual-aware auxiliary heads.",
        "evidence_source": "docs/manuscript_residual_aware_ssl_cnn_nature_polished.md; scripts/45_make_nature_manuscript_figures.py",
        "reviewer_risk": "Architecture claims must match reproducibility scripts.",
        "manuscript_action": "Package code and exact commit hash before final submission.",
    },
    {
        "card_section": "Training",
        "reporting_item": "Residual-aware supervision",
        "current_value": "Fine-tuning combined binary classification with signed residual-distance and pairwise recovery-ranking objectives.",
        "evidence_source": "docs/manuscript_residual_aware_ssl_cnn_nature_polished.md; results/tables/table3_ablation.csv",
        "reviewer_risk": "Ablations support this as exploratory training-signal evidence, not a universal mechanism.",
        "manuscript_action": "Use 'ablation-supported in this dataset' rather than 'proved'.",
    },
    {
        "card_section": "Validation",
        "reporting_item": "Validation unit",
        "current_value": "Patient-level leave-one-subject-out validation; no segment-level or seed-level row was treated as an independent patient.",
        "evidence_source": "docs/prediction_validation_integrity_audit.md; results/tables/prediction_validation_integrity_audit.csv",
        "reviewer_risk": "Segment-level leakage is a common EEG prognosis concern.",
        "manuscript_action": "Keep patient-level validation safeguards prominent.",
    },
    {
        "card_section": "Validation",
        "reporting_item": "Uncertainty and comparisons",
        "current_value": "Subject-level bootstrap intervals, paired bootstrap comparisons, McNemar tests, permutation tests, and one-case resolution audit.",
        "evidence_source": "results/statistics/model_metric_confidence_intervals.csv; results/statistics/model_pairwise_comparisons.csv; results/tables/performance_precision_audit.csv",
        "reviewer_risk": "Intervals remain wide and paired superiority was not established.",
        "manuscript_action": "Preserve conservative wording for favorable numerical differences.",
    },
    {
        "card_section": "Performance",
        "reporting_item": "Final internal performance",
        "current_value": "Accuracy 0.842, balanced accuracy 0.833, ROC-AUC 0.844, PR-AUC 0.836, and Brier score 0.126.",
        "evidence_source": "results/tables/table2_main_model_performance.csv; docs/manuscript_integrity_audit.md",
        "reviewer_risk": "Point estimates can look stronger than warranted by n=19.",
        "manuscript_action": "Report together with bootstrap intervals, precision audit, and no-external-validation caveat.",
    },
    {
        "card_section": "Performance",
        "reporting_item": "Clinical-only comparator boundary",
        "current_value": "Clinical-only baseline was strong in exploratory analyses; EEG incremental value over clinical variables was not established.",
        "evidence_source": "docs/clinical_incremental_value_results.md; results/statistics/clinical_incremental_paired_bootstrap_comparison.csv",
        "reviewer_risk": "Claiming EEG clinical utility would be unsupported.",
        "manuscript_action": "Frame EEG as candidate neurophysiological marker for prospective testing.",
    },
    {
        "card_section": "Explainability",
        "reporting_item": "Attribution interpretation",
        "current_value": "SmoothGrad-integrated-gradient, occlusion, topographic, and WPLI connectome summaries are model-dependent and hypothesis-generating.",
        "evidence_source": "results/tables/table4_explainability_biomarkers.csv; results/figures/nature/figure4_explainability_neurophysiology.png",
        "reviewer_risk": "Channel, frequency, or edge findings could be overread as causal biomarkers.",
        "manuscript_action": "Retain model-dependent and associative interpretation.",
    },
    {
        "card_section": "Limitations",
        "reporting_item": "External validation",
        "current_value": "No independent external validation cohort is available in the current package.",
        "evidence_source": "docs/probast_tripod_ai_risk_audit.md; docs/final_submission_gate_report.md",
        "reviewer_risk": "External performance, calibration, and transportability are unknown.",
        "manuscript_action": "Make prospective external validation the primary next step.",
    },
    {
        "card_section": "Limitations",
        "reporting_item": "Author-confirmed protocol metadata",
        "current_value": "Ethics, consent, recruitment, EEG acquisition, upstream preprocessing, tACS device/electrode/safety, rehabilitation co-intervention, and repository metadata remain incomplete.",
        "evidence_source": "docs/author_required_evidence_trace.md; docs/author_submission_metadata_validation_report.md",
        "reviewer_risk": "Final submission cannot be considered complete until these fields are author-approved.",
        "manuscript_action": "Use the author metadata workbook before final insertion.",
    },
    {
        "card_section": "Availability",
        "reporting_item": "Shareable derived evidence",
        "current_value": "Derived tables, locked predictions/statistics, figure exports, audit tables, source-data workbook, and scripts can be deposited after author approval.",
        "evidence_source": "docs/data_availability_and_fair_audit.md; outputs/manuscript_package/ResidualAware_SSL_CNN_Source_Data.xlsx",
        "reviewer_risk": "No repository DOI or licence has been assigned yet.",
        "manuscript_action": "Deposit final derived-data and code package with DOI before journal upload or publication.",
    },
    {
        "card_section": "Availability",
        "reporting_item": "Restricted source evidence",
        "current_value": "Raw EEG, minimally processed EEG, and clinical source records require ethics, consent, privacy, and data-use review.",
        "evidence_source": "docs/data_availability_and_fair_audit.md; docs/final_submission_gate_report.md",
        "reviewer_risk": "A vague 'available upon request' statement would be weak without restriction rationale and access route.",
        "manuscript_action": "Specify controlled-access route, contact, review criteria, and data-use agreement once author information is available.",
    },
    {
        "card_section": "Use restrictions",
        "reporting_item": "Unsupported uses",
        "current_value": "Do not use the model for individual treatment allocation, standalone prognosis, or clinical deployment without external validation and governance approval.",
        "evidence_source": "docs/claim_strength_audit.md; docs/probast_tripod_ai_risk_audit.md",
        "reviewer_risk": "Clinical-deployment language would conflict with the current evidence.",
        "manuscript_action": "Keep the final conclusion limited to feasibility and prospective validation.",
    },
]


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ROWS)

    lines = [
        "# Model Reporting Card",
        "",
        "This table summarizes intended use, evidence boundaries, validation safeguards, deployment restrictions, and reproducibility actions for the current residual-aware SSL-CNN manuscript package. It is included as Supplementary Table 11 in the compiled Supplementary Information.",
        "",
        "| Card section | Reporting item | Current value | Reviewer risk | Manuscript action |",
        "|---|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append(
            "| {card_section} | {reporting_item} | {current_value} | {reviewer_risk} | {manuscript_action} |".format(
                **{key: value.replace("|", "/") for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "- The model is ready for transparent exploratory reporting, not clinical deployment.",
            "- The strongest current safeguards are patient-level LOSO validation, fold-local feature handling, subject-level statistical comparisons, and explicit claim-strength boundaries.",
            "- Remaining blockers are author-approved ethics, consent, recruitment, EEG acquisition/preprocessing, intervention/safety, data-access, and repository metadata.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()
