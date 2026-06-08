from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "results" / "tables" / "claim_strength_audit.csv"
OUTPUT_MD = ROOT / "docs" / "claim_strength_audit.md"


ROWS = [
    {
        "claim_id": "C01",
        "manuscript_section": "Cohort and outcome definition",
        "manuscript_claim": "The supervised labelled cohort contains 19 patients, with 10 proportional-recovery and 9 poor-recovery labels.",
        "evidence_strength": "verified project result",
        "allowed_wording": "Report as an observed cohort count.",
        "overclaim_to_avoid": "Do not imply population representativeness or external generalizability.",
        "evidence_source": "results/tables/table1_cohort_characteristics.csv; docs/cohort_characteristics.md",
        "verification_status": "supported",
    },
    {
        "claim_id": "C02",
        "manuscript_section": "Outcome definition",
        "manuscript_claim": "The residual threshold of 1.5 points is the supervised-cohort median residual.",
        "evidence_strength": "verified project result",
        "allowed_wording": "Describe as a cohort-specific modelling endpoint.",
        "overclaim_to_avoid": "Do not present the threshold as a validated clinical cut-off.",
        "evidence_source": "results/tables/table1_cohort_characteristics.csv; docs/performance_precision_audit.md",
        "verification_status": "supported",
    },
    {
        "claim_id": "C03",
        "manuscript_section": "Residual-aware SSL-CNN performance",
        "manuscript_claim": "The final model achieved accuracy 0.842, balanced accuracy 0.833, ROC-AUC 0.844, PR-AUC 0.836, and Brier score 0.126.",
        "evidence_strength": "verified project result",
        "allowed_wording": "Report as internal patient-level LOSO performance.",
        "overclaim_to_avoid": "Do not describe as externally validated clinical performance.",
        "evidence_source": "results/tables/table2_main_model_performance.csv; results/statistics/model_metric_confidence_intervals.csv",
        "verification_status": "supported",
    },
    {
        "claim_id": "C04",
        "manuscript_section": "Residual-aware SSL-CNN performance",
        "manuscript_claim": "Permutation tests support above-chance internal discrimination for the final model.",
        "evidence_strength": "internally supported",
        "allowed_wording": "State that subject-level permutation tests supported above-chance discrimination in this cohort.",
        "overclaim_to_avoid": "Do not state that permutation tests establish superiority over comparator models.",
        "evidence_source": "results/statistics/model_permutation_tests.csv; docs/performance_precision_audit.md",
        "verification_status": "supported with boundary",
    },
    {
        "claim_id": "C05",
        "manuscript_section": "Residual-aware SSL-CNN performance",
        "manuscript_claim": "The final model showed numerically favorable ROC-AUC, PR-AUC, and Brier-score differences versus the no-SSL CNN, but paired intervals crossed the null.",
        "evidence_strength": "internally supported",
        "allowed_wording": "Use 'numerically favorable' or 'moved in the favorable direction'.",
        "overclaim_to_avoid": "Do not claim statistically definitive superiority over the no-SSL CNN.",
        "evidence_source": "results/statistics/model_pairwise_comparisons.csv; docs/performance_precision_audit.md",
        "verification_status": "supported with boundary",
    },
    {
        "claim_id": "C06",
        "manuscript_section": "Ablation and robustness analyses",
        "manuscript_claim": "Residual-aware auxiliary supervision is the most consistent training signal in the current ablation evidence.",
        "evidence_strength": "exploratory support",
        "allowed_wording": "Frame as an ablation-supported interpretation in the current dataset.",
        "overclaim_to_avoid": "Do not infer mechanism or general superiority without external validation.",
        "evidence_source": "results/tables/table3_ablation.csv; docs/probast_tripod_ai_risk_audit.md",
        "verification_status": "supported as exploratory",
    },
    {
        "claim_id": "C07",
        "manuscript_section": "Ablation and robustness analyses",
        "manuscript_claim": "The independent benefit of Barlow-style self-supervised pretraining is unresolved in this cohort.",
        "evidence_strength": "limitation supported",
        "allowed_wording": "State that SSL benefit was not established and requires larger EEG pools.",
        "overclaim_to_avoid": "Do not claim that self-supervised pretraining alone improved performance.",
        "evidence_source": "results/tables/table3_ablation.csv; results/tables/performance_precision_audit.csv",
        "verification_status": "supported with boundary",
    },
    {
        "claim_id": "C08",
        "manuscript_section": "Exploratory clinical and incremental analyses",
        "manuscript_claim": "Baseline clinical variables were highly informative, and EEG incremental value over clinical variables was not established.",
        "evidence_strength": "exploratory support",
        "allowed_wording": "Report as exploratory clinical support analysis.",
        "overclaim_to_avoid": "Do not claim robust EEG incremental clinical utility.",
        "evidence_source": "docs/clinical_incremental_value_results.md; results/statistics/clinical_incremental_paired_bootstrap_comparison.csv",
        "verification_status": "supported as exploratory",
    },
    {
        "claim_id": "C09",
        "manuscript_section": "Model explanation and EEG biomarker localization",
        "manuscript_claim": "Model explanations localize contributions to state-dependent PSD patterns and beta-band connectivity.",
        "evidence_strength": "hypothesis-generating support",
        "allowed_wording": "Describe as model-dependent, associative, and hypothesis-generating.",
        "overclaim_to_avoid": "Do not claim causal biomarkers or definitive channel/edge-level biomarkers.",
        "evidence_source": "results/tables/table4_explainability_biomarkers.csv; results/figures/nature/figure4_explainability_neurophysiology.png",
        "verification_status": "supported as exploratory",
    },
    {
        "claim_id": "C10",
        "manuscript_section": "Cross-validation and statistical analysis",
        "manuscript_claim": "Evaluation was performed at the patient level, with fold-local safeguards against segment-level or feature-selection leakage.",
        "evidence_strength": "verified audit result",
        "allowed_wording": "State patient-level LOSO and fold-local safeguards.",
        "overclaim_to_avoid": "Do not claim that internal validation replaces external validation.",
        "evidence_source": "docs/prediction_validation_integrity_audit.md; results/tables/prediction_validation_integrity_audit.csv",
        "verification_status": "supported",
    },
    {
        "claim_id": "C11",
        "manuscript_section": "Study cohort",
        "manuscript_claim": "The core tACS protocol was contralateral M1 stimulation at 20 Hz, 1000 microampere, 20 min daily for 14 sessions.",
        "evidence_strength": "project-record supported",
        "allowed_wording": "Report as project-record protocol and retain author-confirmation query for device, electrode, safety, and concurrent rehabilitation details.",
        "overclaim_to_avoid": "Do not present unverified device, electrode, safety, or conventional-rehabilitation details as complete.",
        "evidence_source": "docs/methods_detail_provenance.md; docs/author_required_evidence_trace.md",
        "verification_status": "partly supported; author confirmation required",
    },
    {
        "claim_id": "C12",
        "manuscript_section": "EEG preprocessing and feature extraction",
        "manuscript_claim": "The manuscript feature pipeline starts from preprocessed EEGLAB files and does not add further filtering or artifact rejection after loading.",
        "evidence_strength": "verified project boundary",
        "allowed_wording": "State the pipeline boundary and identify upstream acquisition/preprocessing as author-confirmed fields.",
        "overclaim_to_avoid": "Do not invent raw EEG hardware, reference, impedance, or artifact-rejection procedures.",
        "evidence_source": "docs/methods_gap_resolution_from_project_files.md; docs/eeg_metadata_audit.md",
        "verification_status": "supported with author-confirmation boundary",
    },
    {
        "claim_id": "C13",
        "manuscript_section": "Data availability",
        "manuscript_claim": "Derived data and code can be deposited, while raw EEG and clinical source records require ethics and consent confirmation.",
        "evidence_strength": "data-governance support",
        "allowed_wording": "Describe public derived data and controlled or unresolved raw-data access routes.",
        "overclaim_to_avoid": "Do not invent repository DOI, licence, access committee, or consent permissions.",
        "evidence_source": "docs/data_availability_and_fair_audit.md; docs/final_submission_gate_report.md",
        "verification_status": "supported; author metadata required",
    },
    {
        "claim_id": "C14",
        "manuscript_section": "Discussion",
        "manuscript_claim": "The model is exploratory and requires larger prospective external validation before clinical deployment.",
        "evidence_strength": "limitation supported",
        "allowed_wording": "Use as a central conclusion and limitation.",
        "overclaim_to_avoid": "Do not claim clinical deployment readiness.",
        "evidence_source": "docs/probast_tripod_ai_risk_audit.md; docs/performance_precision_audit.md; docs/final_submission_gate_report.md",
        "verification_status": "supported",
    },
]


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)
    write_report()
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print(f"claim_strength_rows={len(ROWS)}")


def write_report() -> None:
    lines = [
        "# Claim Strength Audit",
        "",
        "This audit maps central manuscript claims to their current evidence strength, acceptable wording, and overclaims to avoid. It is included as Supplementary Table 10 in the compiled Supplementary Information and is intended to keep the manuscript's language aligned with the evidence actually available in the project package.",
        "",
        "| Claim ID | Manuscript section | Manuscript claim | Evidence strength | Allowed wording | Overclaim to avoid | Evidence source | Verification status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append(
            "| {claim_id} | {manuscript_section} | {manuscript_claim} | {evidence_strength} | {allowed_wording} | {overclaim_to_avoid} | {evidence_source} | {verification_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "- Counts, locked internal performance, validation-unit safeguards, and feature-pipeline boundaries are supported by project evidence.",
            "- Model superiority, EEG incremental clinical utility, self-supervised pretraining benefit, and channel/edge biomarker claims must remain exploratory.",
            "- Ethics, consent, raw-data sharing, acquisition hardware, upstream EEG preprocessing, tACS device/electrode/safety details, and repository identifiers remain author-confirmed fields.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
