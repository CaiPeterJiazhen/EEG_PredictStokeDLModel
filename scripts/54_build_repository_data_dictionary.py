from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "results" / "tables" / "source_data_dictionary.csv"
OUTPUT_README = ROOT / "docs" / "repository_readme_for_deposit.md"

FILE_ROLES = {
    "author_field_replacement_map.csv": "Replacement targets for author-supplied final submission fields.",
    "author_required_evidence_trace.csv": "Author-required field evidence trace and current submission-blocker status.",
    "participant_flow_safety_source_notes.csv": "De-identified participant-flow counts and source-workbook note categories for Figure 1 and author safety review.",
    "probast_tripod_ai_risk_audit.csv": "PROBAST/TRIPOD+AI-oriented prediction-model risk-of-bias and applicability audit.",
    "performance_precision_audit.csv": "Performance precision and validation-boundary audit for the 19-patient LOSO analysis.",
    "claim_strength_audit.csv": "Claim-strength audit mapping central manuscript claims to evidence, allowed wording, and overclaims to avoid.",
    "model_reporting_card.csv": "AI model reporting card summarizing intended use, validation safeguards, reproducibility actions, deployment boundaries, and unsupported uses.",
    "table1_cohort_characteristics.csv": "Main Table 1 cohort summaries by outcome group.",
    "patient_characteristics_table.csv": "Compact cohort-characteristics table embedded in DOCX draft.",
    "table2_main_model_performance.csv": "Main model performance table for all locked candidate/reference rows.",
    "model_performance_main_table.csv": "Compact main-performance table with confidence intervals for manuscript display.",
    "paper_locked_model_performance.csv": "Locked model-performance archive used to define current manuscript-facing rows.",
    "table3_ablation.csv": "Core ablation, modality/state/band ablation, and interpretive notes.",
    "table4_explainability_biomarkers.csv": "Long-form explainability biomarker table for PSD and WPLI attributions.",
    "explainability_key_findings_table.csv": "Compact explainability findings used for manuscript Table 4.",
    "seed_stability_table.csv": "Seed-level stability summaries by model.",
    "error_subject_clinical_eeg_summary.csv": "Exploratory subject-level repeated-error/QC table.",
    "eeg_recording_metadata_audit.csv": "Non-identifying EEGLAB file-level metadata audit.",
    "eeg_recording_summary.csv": "Grouped summary of EEGLAB metadata audit.",
    "clinical_workbook_structure_audit.csv": "Non-identifying source-workbook structure audit.",
    "docx_visual_qa_metrics.csv": "Page-level visual QA metrics from DOCX-to-PDF rendering.",
    "local_reference_positioning_matrix.csv": "Conservative positioning audit for locally supplied reference PDFs.",
    "manuscript_integrity_audit.csv": "Automated manuscript-source consistency checks.",
    "prediction_validation_integrity_audit.csv": "Patient-level validation and leakage-safeguard audit.",
    "reference_metadata_audit.csv": "Reference DOI/URL metadata verification audit.",
    "source_workbook_author_metadata_audit.csv": "Non-identifying source-workbook scan for author-required protocol and submission metadata evidence.",
    "patient_record_pdf_text_audit.csv": "Non-identifying text-layer and image-coverage audit for local patient/healthy record-book PDFs.",
    "supplementary_all_metrics.csv": "Supplementary archive of model metrics and ablation metadata.",
    "submission_artifact_quality_audit.csv": "Structural QA audit for DOCX, figures, and package integrity.",
    "model_metric_confidence_intervals.csv": "Subject-level model metric estimates with bootstrap intervals and calibration summaries.",
    "model_pairwise_comparisons.csv": "Paired subject-level bootstrap and McNemar comparisons.",
    "model_permutation_tests.csv": "Subject-level label-permutation tests.",
    "clinical_incremental_paired_bootstrap_comparison.csv": "Exploratory clinical-only and clinical-plus-EEG paired bootstrap comparisons.",
    "figure_manifest.csv": "Main figure export manifest.",
    "mne_wpli_connectivity_manifest.csv": "Supplementary MNE WPLI connectivity figure manifest.",
}

COLUMN_DESCRIPTIONS = {
    "subject_id": ("Pseudonymous project subject identifier.", "identifier"),
    "group": ("Data group or cohort grouping.", "category"),
    "stage": ("Recording stage in the project workflow.", "category"),
    "state": ("Resting-state EEG condition, EO or EC.", "category"),
    "is_supervised_subject": ("Whether the subject belongs to the final supervised 19-patient cohort.", "boolean"),
    "nbchan": ("Number of retained EEG channels in an EEGLAB file.", "count"),
    "srate_hz": ("EEG sampling rate.", "Hz"),
    "trials": ("Number of EEGLAB trials in the file.", "count"),
    "points_per_trial": ("Number of sample points per trial.", "count"),
    "duration_sec": ("Recording duration estimated from sample points and sampling rate.", "seconds"),
    "duration_sec_mean": ("Mean recording duration in a grouped EEG metadata summary.", "seconds"),
    "duration_sec_min": ("Minimum recording duration in a grouped EEG metadata summary.", "seconds"),
    "duration_sec_max": ("Maximum recording duration in a grouped EEG metadata summary.", "seconds"),
    "xmin": ("EEGLAB recording start time metadata.", "seconds"),
    "xmax": ("EEGLAB recording end time metadata.", "seconds"),
    "channel_count": ("Number of channel labels read from the EEGLAB metadata.", "count"),
    "first_channel": ("First channel label in the retained channel order.", "label"),
    "last_channel": ("Last channel label in the retained channel order.", "label"),
    "resolved_fdt_exists": ("Whether the resolved companion .fdt file exists.", "boolean"),
    "variable": ("Cohort variable or row label.", "text"),
    "summary": ("Summary statistic descriptor.", "text"),
    "type": ("Variable type or row type descriptor.", "category"),
    "all": ("Overall cohort summary.", "summary"),
    "All": ("Overall cohort summary.", "summary"),
    "proportional_label1": ("Summary for the proportional-recovery label-1 group.", "summary"),
    "Proportional recovery": ("Summary for the proportional-recovery group.", "summary"),
    "poor_recovery_label0": ("Summary for the poor-recovery label-0 group.", "summary"),
    "Poor recovery": ("Summary for the poor-recovery group.", "summary"),
    "model_family": ("Model family grouping used in manuscript analyses.", "category"),
    "model_name": ("Exact model or result-row name.", "identifier"),
    "model": ("Display model label or model key.", "text"),
    "source_model": ("Internal source model name used to select manuscript rows.", "identifier"),
    "input_features": ("Feature families used by a model.", "text"),
    "feature_selection": ("Feature-selection method or fold-local selection description.", "text"),
    "inference_type": ("Inference aggregation strategy, such as seed mean or ensemble.", "category"),
    "result_role": ("Role of the result row in the manuscript or supplementary archive.", "category"),
    "legacy_warning": ("Warning flag for rows not intended as current primary results.", "text"),
    "n_subjects": ("Number of subjects used for the metric.", "count"),
    "n": ("Count for a participant-flow, source-note, or summary row.", "count"),
    "domain": ("Reporting, risk-audit, or model-evaluation domain.", "category"),
    "assessment_item": ("Specific risk-audit or reporting item being assessed.", "text"),
    "current_evidence": ("Conservative summary of current manuscript/package evidence for an audit item.", "text"),
    "risk_of_bias": ("Risk-of-bias judgement for a prediction-model audit item.", "category"),
    "applicability_concern": ("Applicability concern judgement for a prediction-model audit item.", "category"),
    "mitigation_or_required_action": ("Required action or mitigation for a risk-audit item.", "text"),
    "evidence_source": ("File, protocol, audit, or output supporting the row.", "path list"),
    "observed_value": ("Observed validation estimate, cohort count, or comparison value used in the performance precision audit.", "metric or text"),
    "uncertainty_or_resolution": ("Bootstrap interval, permutation count, or one-case resolution relevant to a precision-audit item.", "metric or text"),
    "claim_id": ("Stable identifier for a central manuscript claim in the claim-strength audit.", "identifier"),
    "manuscript_claim": ("Central manuscript claim being checked against project evidence.", "text"),
    "evidence_strength": ("Conservative judgement of the evidence strength supporting the manuscript claim.", "category"),
    "allowed_wording": ("Recommended evidence-aligned wording strength for the claim.", "text"),
    "overclaim_to_avoid": ("Manuscript wording or interpretation that would exceed the current evidence.", "text"),
    "verification_status": ("Support status and remaining confirmation boundary for a claim.", "category"),
    "card_section": ("Model-card section grouping, such as intended use, population, validation, or availability.", "category"),
    "reporting_item": ("Specific AI model reporting-card item.", "text"),
    "current_value": ("Current project-supported value or boundary for a model-card item.", "text"),
    "reviewer_risk": ("Likely reviewer concern or overclaim risk for a model-card item.", "text"),
    "manuscript_action": ("Manuscript or package action required to keep the item evidence-aligned.", "text"),
    "n_positive": ("Number of positive label-1 subjects.", "count"),
    "n_negative": ("Number of negative label-0 subjects.", "count"),
    "accuracy": ("Hard-label accuracy at the subject level.", "proportion"),
    "balanced_accuracy": ("Mean of sensitivity and specificity.", "proportion"),
    "sensitivity": ("True-positive rate for proportional recovery.", "proportion"),
    "specificity": ("True-negative rate for poor recovery.", "proportion"),
    "precision": ("Positive predictive value.", "proportion"),
    "f1": ("F1 score.", "proportion"),
    "roc_auc": ("Area under the receiver-operating-characteristic curve.", "unitless"),
    "pr_auc": ("Area under the precision-recall curve.", "unitless"),
    "brier_score": ("Brier score; lower values indicate better probabilistic accuracy.", "unitless"),
    "ece": ("Expected calibration error.", "unitless"),
    "calibration_intercept": ("Calibration intercept estimated from subject-level predictions.", "unitless"),
    "calibration_slope": ("Calibration slope estimated from subject-level predictions.", "unitless"),
    "p_value": ("Statistical p-value.", "probability"),
    "p_value_two_sided": ("Two-sided p-value.", "probability"),
    "binomial_accuracy_p": ("Binomial test p-value for accuracy.", "probability"),
    "n_bootstrap": ("Number of bootstrap resamples.", "count"),
    "n_permutations": ("Number of label permutations requested.", "count"),
    "n_valid_permutations": ("Number of valid label permutations used.", "count"),
    "ci_low": ("Lower confidence-interval or bootstrap-interval bound.", "metric-specific"),
    "ci_high": ("Upper confidence-interval or bootstrap-interval bound.", "metric-specific"),
    "difference": ("Candidate-minus-reference metric difference.", "metric-specific"),
    "metric": ("Metric name.", "text"),
    "metric_a": ("Metric value for model A or reference model.", "metric-specific"),
    "metric_b": ("Metric value for model B or candidate model.", "metric-specific"),
    "model_a": ("First model in a paired comparison.", "identifier"),
    "model_b": ("Second model in a paired comparison.", "identifier"),
    "reference_model": ("Reference model in an incremental or paired comparison.", "identifier"),
    "candidate_model": ("Candidate model in an incremental or paired comparison.", "identifier"),
    "observed": ("Observed statistic or QA value.", "metric-specific"),
    "ablation_block": ("Ablation-analysis block.", "category"),
    "ablation_name": ("Ablation setting name.", "category"),
    "model_key": ("Internal model key.", "identifier"),
    "row_type": ("Row classification for reported, seed-mean, ensemble, or supplementary rows.", "category"),
    "n_input_columns": ("Number of model input columns/features for an ablation row.", "count"),
    "interpretation": ("Conservative manuscript interpretation for a result row.", "text"),
    "explainability_alignment": ("Whether an ablation result aligns with explainability findings.", "text"),
    "finding_type": ("Explainability finding category.", "category"),
    "finding": ("Plain-language explainability finding.", "text"),
    "value": ("Reported value for a finding.", "metric-specific"),
    "stability_or_validation": ("Stability or validation note for an explainability finding.", "text"),
    "channel": ("EEG channel label.", "label"),
    "channel_index": ("Zero-based channel index in the fixed 62-channel order.", "index"),
    "frequency_hz": ("Frequency bin for PSD attribution.", "Hz"),
    "frequency_bin": ("Frequency-bin index.", "index"),
    "band": ("Frequency band label.", "category"),
    "band_index": ("Frequency-band index.", "index"),
    "mean_signed_attribution": ("Mean signed feature attribution.", "model-attribution units"),
    "mean_abs_attribution": ("Mean absolute feature attribution.", "model-attribution units"),
    "std_abs_attribution": ("Standard deviation of absolute feature attribution.", "model-attribution units"),
    "n_samples": ("Number of attribution samples/subjects contributing to a summary.", "count"),
    "positive_mean_abs_attribution": ("Mean absolute attribution in label-1 subjects.", "model-attribution units"),
    "negative_mean_abs_attribution": ("Mean absolute attribution in label-0 subjects.", "model-attribution units"),
    "correct_only_mean_abs_attribution": ("Mean absolute attribution among correctly classified subjects.", "model-attribution units"),
    "spearman_residual_r": ("Spearman correlation with residual.", "correlation"),
    "spearman_residual_p": ("P-value for Spearman correlation with residual.", "probability"),
    "spearman_signed_distance_r": ("Spearman correlation with signed distance from threshold.", "correlation"),
    "spearman_signed_distance_p": ("P-value for Spearman signed-distance correlation.", "probability"),
    "spearman_signed_distance_fdr_p": ("False-discovery-adjusted p-value for signed-distance correlation.", "probability"),
    "label_group_permutation_p": ("Permutation-test p-value for label-group attribution difference.", "probability"),
    "feature_family": ("PSD or WPLI/explainability feature family.", "category"),
    "edge_index": ("Connectivity edge index in deterministic upper-triangle list.", "index"),
    "channel_i": ("First EEG channel in a connectivity edge.", "label"),
    "channel_j": ("Second EEG channel in a connectivity edge.", "label"),
    "channel_i_index": ("Index of the first EEG channel in a connectivity edge.", "index"),
    "channel_j_index": ("Index of the second EEG channel in a connectivity edge.", "index"),
    "network_group": ("Network/topographic grouping assigned to an EEG feature.", "category"),
    "interhemispheric": ("Whether a connectivity edge crosses hemispheres.", "boolean"),
    "FMA_pre": ("Baseline Fugl-Meyer Assessment upper-extremity score.", "points"),
    "FMA_post": ("Post-treatment Fugl-Meyer Assessment upper-extremity score.", "points"),
    "MBI_pre": ("Baseline Modified Barthel Index.", "points"),
    "MBI_post": ("Post-treatment Modified Barthel Index.", "points"),
    "Residual": ("Predicted minus observed FMA-UE improvement.", "points"),
    "signed_distance": ("1.5 minus residual; positive values indicate label-1 side of the endpoint.", "points"),
    "distance_to_threshold": ("Absolute distance to the residual threshold.", "points"),
    "age": ("Age.", "years"),
    "sex": ("Sex as encoded in the clinical workbook.", "category"),
    "duration": ("Disease duration as encoded in the clinical workbook.", "source units"),
    "affected_hand": ("Affected hand side.", "category"),
    "figure": ("Figure identifier.", "identifier"),
    "conclusion": ("One-sentence conclusion or purpose of a figure.", "text"),
    "source_data": ("Source data used to generate a figure.", "text"),
    "png": ("Path to PNG figure export.", "path"),
    "svg": ("Path to SVG figure export.", "path"),
    "pdf": ("Path to PDF figure export.", "path"),
    "tiff": ("Path to TIFF figure export.", "path"),
    "paths": ("Figure export paths.", "path list"),
    "artifact_type": ("Type of artifact checked by QA.", "category"),
    "artifact": ("Artifact filename or relative path.", "path or label"),
    "check": ("QA check name.", "text"),
    "expected": ("Expected value or condition for QA.", "text"),
    "status": ("Result status, such as PASS, WARN, or FAIL.", "category"),
}


COLUMN_DESCRIPTIONS.update(
    {
        "category": ("Audit category, result grouping, or source-evidence category.", "category"),
        "field": ("Author-required or audit-tracked field name.", "identifier"),
    "field_id": ("Stable identifier for an author-supplied field or replacement target.", "identifier"),
    "required_evidence": ("Evidence required to resolve an author-supplied submission metadata field.", "text"),
    "source_workbooks_inspected": ("Source workbooks included in the author-metadata audit.", "path list"),
    "record_group": ("Non-identifying source-record group for PDF text-layer audit.", "category"),
    "record_id": ("Stable pseudonymous record identifier used for non-identifying PDF source audits.", "identifier"),
    "page_count": ("Number of pages in a PDF record book.", "count"),
    "pages_sampled_for_images": ("Number of PDF pages sampled for image-coverage estimation.", "count"),
    "extracted_text_chars": ("Number of compact extractable text characters detected by PyMuPDF.", "count"),
    "pages_with_text": ("Number of PDF pages with any extractable text layer.", "count"),
    "embedded_image_count": ("Number of embedded images detected in a PDF.", "count"),
    "mean_image_coverage_sampled_pages": ("Mean estimated embedded-image coverage across sampled pages.", "proportion"),
    "text_layer_status": ("PDF text-layer audit status.", "category"),
    "metadata_utility": ("Whether the PDF can support automatic metadata prefill or requires OCR/manual review.", "category"),
    "keyword_hit_count": ("Number of non-identifying keyword hits found in source workbooks for an audit item.", "count"),
    "matched_terms": ("Keyword terms matched during the source-workbook audit, summarized without patient identifiers.", "text"),
    "nonidentifying_cell_refs": ("Workbook, sheet, and cell references for keyword hits without exposing patient-level values.", "cell reference list"),
    "evidence_grade": ("Evidence grade assigned after source-workbook keyword audit.", "category"),
    "author_input_needed": ("Exact author, ethics, clinical, or data-governance input needed before replacement.", "text"),
    "current_status": ("Current completion status for an author-required replacement field.", "category"),
    "row_type": ("Participant-flow or source-note row type.", "category"),
    "denominator": ("Denominator used for the count in a participant-flow or source-note row.", "count"),
    "source_basis": ("Source files or project evidence used to derive the row.", "text"),
    "manuscript_use": ("How the row supports the manuscript or author review workflow.", "text"),
    "target_files": ("Files where the author-supplied field should be inserted or checked.", "path list"),
        "manuscript_section": ("Manuscript, declaration, cover-letter, or portal section affected by a field.", "text"),
        "replacement_action": ("Concrete edit to perform once the verified author input is available.", "text"),
        "exact_text_or_placeholder_to_replace": ("Placeholder text, author-query text, or section cue to replace.", "text"),
        "final_text_dependency": ("Evidence or author decision required before a replacement can be finalized.", "text"),
        "verification_after_replacement": ("Verification step to run after inserting the author-supplied field.", "text"),
        "priority": ("Priority level for finalization, such as blocking, high, or optional.", "category"),
        "submission_requirement": ("Submission requirement that the field is intended to satisfy.", "text"),
        "project_source_hits": ("Number of hits found in project-source files during evidence tracing.", "count"),
        "generated_or_template_hits": ("Number of hits found only in generated audit, template, or draft files.", "count"),
        "best_evidence": ("Most relevant source excerpts located for an audit field.", "text"),
        "verified_source_summary": ("Conservative summary of what the project-source evidence verifies.", "text"),
        "decision": ("Audit decision or author-action recommendation.", "text"),
        "workbook": ("Clinical or source workbook filename included in structure audit.", "path or label"),
        "sheet_name": ("Worksheet name in a source-workbook structure audit.", "label"),
        "max_row": ("Maximum populated row index detected in a workbook sheet.", "count"),
        "max_column": ("Maximum populated column index detected in a workbook sheet.", "count"),
        "first_non_empty_row": ("First non-empty row detected in a workbook sheet.", "index"),
        "candidate_header_cells": ("Candidate header cells detected during source-workbook structure audit.", "text"),
        "date_or_timing": ("Whether date or timing terms were detected in a workbook sheet.", "text"),
        "stroke_or_diagnosis": ("Whether stroke or diagnosis terms were detected in a workbook sheet.", "text"),
        "ethics_or_consent": ("Whether ethics or consent terms were detected in a workbook sheet.", "text"),
        "eeg_or_protocol": ("Whether EEG or protocol terms were detected in a workbook sheet.", "text"),
        "document": ("Document identifier used in DOCX visual QA.", "identifier"),
        "page": ("One-based page index in rendered DOCX visual QA.", "index"),
        "width_px": ("Rendered page or figure width.", "pixels"),
        "height_px": ("Rendered page or figure height.", "pixels"),
        "mean_rgb": ("Mean rendered RGB pixel values.", "RGB tuple"),
        "std_rgb": ("Standard deviation of rendered RGB pixel values.", "RGB tuple"),
        "dark_fraction": ("Fraction of rendered page pixels below the dark-pixel threshold.", "proportion"),
        "nonwhite_fraction": ("Fraction of rendered page pixels not classified as near-white.", "proportion"),
        "n_records": ("Number of EEG records contributing to a grouped metadata summary.", "count"),
        "srate_values_hz": ("Distinct EEG sampling-rate values in a grouped metadata summary.", "Hz list"),
        "nbchan_values": ("Distinct retained-channel counts in a grouped metadata summary.", "count list"),
        "trials_values": ("Distinct EEGLAB trial counts in a grouped metadata summary.", "count list"),
        "points_min": ("Minimum sample-point count in a grouped EEG metadata summary.", "count"),
        "points_max": ("Maximum sample-point count in a grouped EEG metadata summary.", "count"),
        "skip_reason": ("Reason why a supplementary result row was skipped or not treated as current primary evidence.", "text"),
        "contrast": ("Result contrast or comparison label used in a supplementary analysis.", "text"),
        "reproduce_command": ("Command or script path recorded for reproducing a result row.", "command"),
        "availability": ("Availability status of an output, checkpoint, or result row.", "text"),
        "ablation_description": ("Plain-language description of an ablation setting.", "text"),
        "label_definition": ("Outcome-label definition used for a result row.", "text"),
        "threshold_type": ("Type of threshold used to define labels or sensitivity analyses.", "category"),
        "residual_threshold": ("Residual threshold used for proportional-recovery label assignment.", "points"),
        "margin": ("Margin around the residual threshold used in sensitivity analyses.", "points"),
        "n_excluded_near_threshold": ("Number of near-threshold subjects excluded in a sensitivity analysis.", "count"),
        "positive_mean_value": ("Mean explainability value in proportional-recovery label-1 subjects.", "model-attribution units"),
        "negative_mean_value": ("Mean explainability value in poor-recovery label-0 subjects.", "model-attribution units"),
        "summary_name": ("Name of a summarized explainability quantity.", "text"),
        "mean_abs_value": ("Mean absolute explainability value.", "model-attribution units"),
        "mean_signed_value": ("Mean signed explainability value.", "model-attribution units"),
        "effect_direction": ("Direction of a reported explainability effect or group contrast.", "text"),
        "spearman_signed_attribution_distance_r": ("Spearman correlation between signed attribution and signed-distance measure.", "correlation"),
        "y_true": ("True binary proportional-recovery label for a subject-level error analysis row.", "binary label"),
        "error_rate": ("Proportion of model variants or seeds that misclassified a subject.", "proportion"),
        "near_threshold_margin_0.5": ("Whether the subject was within 0.5 FMA points of the residual threshold.", "boolean"),
        "near_threshold_margin_1.0": ("Whether the subject was within 1.0 FMA points of the residual threshold.", "boolean"),
        "possible_clinical_heterogeneity": ("Flag or note indicating possible clinical heterogeneity in repeated-error review.", "text"),
        "max_abs_feature_z": ("Maximum absolute z-score among audited clinical or EEG features.", "z-score"),
        "n_abs_z_ge_2": ("Number of audited features with absolute z-score at least 2.", "count"),
        "n_abs_z_ge_3": ("Number of audited features with absolute z-score at least 3.", "count"),
        "top_outlier_features": ("Highest-ranked outlier features for a subject-level error analysis row.", "text"),
        "top_outlier_z": ("Z-scores corresponding to top outlier features.", "z-score list"),
        "attribution_explained_error_rate": ("Whether attribution summaries were sufficient to explain repeated-error status.", "text"),
        "notes": ("Free-text audit or reuse notes.", "text"),
        "number": ("Reference number in the manuscript bibliography.", "index"),
        "identifier_type": ("Type of reference identifier, such as DOI or URL.", "category"),
        "identifier": ("Reference DOI, URL, or stable identifier.", "identifier"),
        "local_year": ("Publication year recorded in the manuscript reference entry.", "year"),
        "metadata_year": ("Publication year returned by reference metadata lookup.", "year"),
        "local_title": ("Title recorded in the manuscript reference entry.", "text"),
        "metadata_title": ("Title returned by DOI or URL metadata lookup.", "text"),
        "metadata_container": ("Journal, proceedings, repository, or container title returned by metadata lookup.", "text"),
        "lookup_status": ("Status of DOI or URL metadata retrieval.", "PASS/WARN/FAIL"),
        "match_status": ("Status of manuscript-reference and metadata consistency check.", "PASS/WARN/FAIL"),
        "source_url": ("URL used to verify reference metadata.", "URL"),
        "local_file": ("Local reference PDF filename inspected for positioning.", "path or label"),
        "citation_key": ("Short citation key assigned during local-reference positioning audit.", "text"),
        "title": ("Reference or local-paper title.", "text"),
        "doi": ("Digital Object Identifier extracted or assigned for a reference.", "DOI"),
        "pdf_doi_probe": ("DOI-like string detected directly from local PDF text.", "DOI or text"),
        "domain": ("Scientific or clinical domain assigned during local-reference audit.", "category"),
        "modality": ("Data modality or method family assigned during local-reference audit.", "category"),
        "outcome": ("Outcome or target endpoint assigned during local-reference audit.", "text"),
        "current_status": ("Current inclusion status of a local reference in the manuscript.", "category"),
        "support_grade": ("Strength of support that a local reference provides for the manuscript claim.", "category"),
        "positioning_decision": ("Citation-positioning recommendation for a local reference.", "text"),
        "reason": ("Rationale for a local-reference positioning decision.", "text"),
        "pdf_text_probe": ("Short extracted text snippet used to audit a local PDF reference.", "text"),
        "mean_brier": ("Mean Brier score across seeds or repeated model runs.", "unitless"),
        "a_correct_b_wrong": ("McNemar discordant count where model A is correct and model B is wrong.", "count"),
        "a_wrong_b_correct": ("McNemar discordant count where model A is wrong and model B is correct.", "count"),
        "n_discordant": ("Total discordant paired hard predictions in a model comparison.", "count"),
        "top_n": ("Number of top-ranked WPLI edges rendered in a connectivity figure.", "count"),
        "value_column": ("Column used to determine absolute edge magnitude for a connectivity render.", "column name"),
        "signed_column": ("Column used to determine signed edge color or direction for a connectivity render.", "column name"),
    }
)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dictionary_rows()
    write_csv(rows)
    write_repository_readme(rows)
    print(OUTPUT_CSV)
    print(OUTPUT_README)


def build_dictionary_rows() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for relative in source_files():
        path = ROOT / relative
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            example_row = next(reader, [])
        examples = dict(zip(header, example_row))
        for column in header:
            description, units = describe_column(column)
            output.append(
                {
                    "source_file": relative,
                    "table_role": FILE_ROLES.get(path.name, "Project output table."),
                    "column_name": column,
                    "inferred_type": infer_type(examples.get(column, "")),
                    "description": description,
                    "unit_or_scale": units,
                    "access_level": access_level(relative),
                    "notes": notes_for_column(column),
                }
            )
    return output


def source_files() -> list[str]:
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "results" / "tables").glob("*.csv")))
    paths.extend(sorted((ROOT / "results" / "statistics").glob("*.csv")))
    paths.append(ROOT / "results" / "figures" / "nature" / "figure_manifest.csv")
    paths.append(
        ROOT
        / "results"
        / "figures"
        / "explainability"
        / "mne_wpli_connectivity"
        / "mne_wpli_connectivity_manifest.csv"
    )
    unique_paths = []
    seen: set[Path] = set()
    for path in paths:
        if path.name == OUTPUT_CSV.name:
            continue
        if path in seen or not path.exists():
            continue
        seen.add(path)
        unique_paths.append(path)
    return [path.relative_to(ROOT).as_posix() for path in unique_paths]


def describe_column(column: str) -> tuple[str, str]:
    if column in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[column]
    lower = column.lower()
    if lower.endswith("_low_ci") or lower.endswith("_high_ci") or lower.endswith("_95ci"):
        return ("Confidence-interval field for the corresponding metric.", "metric-specific")
    if "p_value" in lower or lower.endswith("_p"):
        return ("Statistical p-value.", "probability")
    if "accuracy" in lower or lower.endswith("_auc") or "score" in lower:
        return ("Model-performance metric.", "unitless")
    if "path" in lower or "file" in lower:
        return ("File path or source-file descriptor.", "path")
    cleaned = column.replace("_", " ")
    return (f"Field `{column}` from a project output table; inspect source table context before reuse.", "not specified")


def infer_type(value: str) -> str:
    stripped = str(value).strip()
    if stripped == "":
        return "empty/example unavailable"
    lower = stripped.lower()
    if lower in {"true", "false"}:
        return "boolean"
    try:
        int(stripped)
        return "integer"
    except ValueError:
        pass
    try:
        float(stripped)
        return "number"
    except ValueError:
        pass
    return "text"


def access_level(relative_path: str) -> str:
    lower = relative_path.lower()
    if "eeg_recording_metadata_audit" in lower or "clinical_workbook_structure_audit" in lower or "patient_record_pdf_text_audit" in lower:
        return "public non-identifying metadata after author review"
    if "table" in lower or "statistics" in lower or "figure" in lower:
        return "public derived source data after author approval"
    return "project output after author review"


def notes_for_column(column: str) -> str:
    if column in {"subject_id", "model_name", "source_model", "model_key"}:
        return "Identifier is project-specific and should not be treated as a clinical identifier."
    if column in {"Residual", "signed_distance"}:
        return "Derived from the cohort-specific residual threshold; not externally validated."
    if column in {"accuracy", "roc_auc", "pr_auc", "brier_score"}:
        return "Computed at subject level; seed and segment rows are not independent patients."
    if "attribution" in column:
        return "Model-explanation quantity; hypothesis-generating rather than causal."
    return ""


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_file",
        "table_role",
        "column_name",
        "inferred_type",
        "description",
        "unit_or_scale",
        "access_level",
        "notes",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_repository_readme(rows: list[dict[str, str]]) -> None:
    n_files = len({row["source_file"] for row in rows})
    text = f"""# Repository README For Deposit

This README is a repository-facing draft for depositing the derived data and reproducibility materials supporting the residual-aware SSL-CNN stroke EEG manuscript.

## Recommended Public Deposit Contents

- `ResidualAware_SSL_CNN_Source_Data.xlsx`: curated source-data workbook containing manuscript tables, statistical outputs, EEG metadata summaries, figure manifests, and the data dictionary.
- `results/tables/*.csv`: machine-readable source tables and quality-audit tables.
- `results/statistics/*.csv`: subject-level bootstrap, paired-comparison, permutation, and clinical-incremental analyses.
- `results/figures/nature/`: PNG, SVG, PDF, and TIFF exports for main manuscript figures and the error-subject supplementary figure.
- `results/figures/explainability/mne_wpli_connectivity/`: supplementary MNE WPLI connectivity maps and manifest.
- `docs/*.md`: manuscript drafts, citation map, data-availability audit, methods provenance, reporting checklist, target-journal strategy, and artifact QA.
- `scripts/` and `src/`: analysis, figure, manuscript, workbook, audit, and packaging scripts needed to regenerate the deposited derived outputs.

## Data Dictionary

The machine-readable data dictionary is `results/tables/source_data_dictionary.csv`. It currently describes {len(rows)} fields across {n_files} derived CSV or figure-manifest files. The dictionary includes source file, table role, column name, inferred type, description, unit/scale, access route, and reuse notes.

## Data Not Included In A Public Deposit Without Additional Approval

Raw EEG, minimally processed EEG, individual-level clinical source workbooks, and any directly identifiable or linkable participant records should not be deposited publicly until the ethics approval, consent language, institutional access route, de-identification plan, and data-use agreement are confirmed.

## Suggested Data Availability Text

De-identified subject-level derived tables, locked model predictions, statistical output tables, figure source summaries, figure exports, and analysis code are available in this repository. Raw EEG recordings and individual-level clinical source records are not publicly available in this draft because they contain human-participant data and may be subject to institutional review board, consent, and data-use restrictions. Access to restricted raw or minimally processed data should be reviewed by the responsible institution. Qualified researchers may request access after ethics approval and completion of a data-use agreement.

## Reproducibility Notes

- All performance metrics in the manuscript are based on subject-level LOSO predictions.
- Segment-level rows and repeated seed rows are not independent patient observations.
- The residual threshold is derived from the supervised cohort median residual and should not be treated as an externally validated clinical cut-off.
- Explainability outputs are model-dependent and should be interpreted as hypothesis-generating.

## Required Before Public Release

- Repository DOI or accession number.
- Code and data licence.
- Final commit hash.
- Ethics approval institution and approval number.
- Consent/data-sharing permission and access route for restricted data.
- Confirmation of which derived EEG features may be public.
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
