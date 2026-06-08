from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD = ROOT / "docs" / "figure_submission_readiness_audit.md"
NATURE_MANUSCRIPT = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_nature_polished.md"
JNE_CLEAN_MANUSCRIPT = ROOT / "docs" / "manuscript_residual_aware_ssl_cnn_jne_clean_placeholder.md"
FIGURE_MANIFEST = ROOT / "results" / "figures" / "nature" / "figure_manifest.csv"
MNE_MANIFEST = (
    ROOT
    / "results"
    / "figures"
    / "explainability"
    / "mne_wpli_connectivity"
    / "mne_wpli_connectivity_manifest.csv"
)
MNE_CONTACT_SHEET = (
    ROOT
    / "results"
    / "figures"
    / "explainability"
    / "mne_wpli_connectivity"
    / "mne_wpli_connectivity_contact_sheet.png"
)
MNE_TOPOMAP_MANIFEST = (
    ROOT
    / "results"
    / "figures"
    / "explainability"
    / "mne_topomaps"
    / "mne_topomap_manifest.csv"
)
MNE_TOPOMAP_CONTACT_SHEET = (
    ROOT
    / "results"
    / "figures"
    / "explainability"
    / "mne_topomaps"
    / "mne_topomap_contact_sheet.png"
)


@dataclass(frozen=True)
class FigureContract:
    label: str
    stem: str
    archetype: str
    core_conclusion: str
    evidence_chain: str
    source_paths: tuple[str, ...]
    review_risk: str
    placement: str


@dataclass(frozen=True)
class TableContract:
    label: str
    source_path: str
    manuscript_role: str


FIGURES = (
    FigureContract(
        "Figure 1",
        "figure1_study_design_model",
        "schematic-led composite",
        "The study uses a locked participant-flow and residual-aware EEG modelling workflow.",
        "participant flow counts -> label definition -> modality/state branches -> classification inference",
        ("results/tables/participant_flow_safety_source_notes.csv", "results/tables/table1_cohort_characteristics.csv"),
        "Author confirmation is still needed for final ethics, recruitment dates, intervention device details, and safety wording.",
        "early Methods overview",
    ),
    FigureContract(
        "Figure 2",
        "figure2_performance_calibration",
        "quantitative grid",
        "The final residual-aware SSL-CNN has internally validated ranking and calibration advantages over EEG-only comparators.",
        "locked LOSO predictions -> bootstrap intervals -> paired comparisons -> calibration plot",
        (
            "results/tables/table2_main_model_performance.csv",
            "results/statistics/model_metric_confidence_intervals.csv",
            "results/statistics/model_pairwise_comparisons.csv",
            "results/predictions/paper_locked_model_predictions.csv",
        ),
        "Clinical-only baseline remains strong; the figure should not be used to claim independent EEG incremental utility.",
        "Results performance section",
    ),
    FigureContract(
        "Figure 3",
        "figure3_robustness_ablation",
        "quantitative grid",
        "Residual-aware auxiliary supervision is the most consistent training signal in this cohort.",
        "ablation table -> seed stability -> threshold sensitivity -> feature-family comparisons",
        (
            "results/tables/table3_ablation.csv",
            "results/tables/seed_stability_table.csv",
            "results/tables/performance_precision_audit.csv",
        ),
        "Self-supervised pretraining alone should remain framed as non-definitive in this cohort.",
        "Results ablation and robustness section",
    ),
    FigureContract(
        "Figure 4",
        "figure4_explainability_neurophysiology",
        "asymmetric mixed-modality figure",
        "Model explanations localize state- and frequency-dependent PSD and WPLI patterns as hypothesis-generating EEG biomarkers.",
        "integrated gradients -> occlusion -> topomaps -> WPLI network summaries",
        (
            "results/tables/table4_explainability_biomarkers.csv",
            "results/tables/explainability_key_findings_table.csv",
            "results/figures/explainability/mne_topomaps/mne_topomap_manifest.csv",
            "results/figures/explainability/mne_topomaps/mne_topomap_contact_sheet.png",
            "results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_manifest.csv",
        ),
        "Interpret as model-dependent associations, not independently validated neural mechanisms.",
        "Results model explanation section",
    ),
    FigureContract(
        "Supplementary Figure 1",
        "supplementary_error_subjects",
        "quantitative grid",
        "Repeated-error subjects identify review targets for future cohorts rather than label corrections.",
        "subject-level error summaries -> model/seed recurrence -> post-hoc review boundary",
        ("results/tables/error_subject_clinical_eeg_summary.csv",),
        "Post-hoc exploratory figure; do not revise labels based on this analysis.",
        "Supplementary Information",
    ),
    FigureContract(
        "Supplementary Figure 2",
        "mne_wpli_connectivity_contact_sheet",
        "asymmetric mixed-modality figure",
        "WPLI attribution maps summarize top signed connectivity edges across states and frequency bands.",
        "edge attribution table -> MNE scalp layout -> 12 state-band panels -> contact sheet",
        (
            "results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_manifest.csv",
            "results/tables/table4_explainability_biomarkers.csv",
        ),
        "Exploratory visualization; individual connectivity exports provide vector/PDF versions but the contact sheet is PNG-only.",
        "Supplementary Information",
    ),
    FigureContract(
        "Supplementary Figure 3",
        "supplementary_performance_precision",
        "quantitative grid",
        "The small 19-patient LOSO validation has coarse metric resolution and imprecise paired-comparison evidence.",
        "bootstrap intervals -> paired differences -> one-case sensitivity/specificity movement",
        (
            "results/statistics/model_metric_confidence_intervals.csv",
            "results/statistics/model_pairwise_comparisons.csv",
            "results/tables/performance_precision_audit.csv",
        ),
        "Use to constrain claims and avoid over-interpreting superiority.",
        "Supplementary Information",
    ),
    FigureContract(
        "Supplementary Figure 4",
        "supplementary_clinical_incremental_value",
        "quantitative grid",
        "Clinical-only variables are strong and EEG-plus-clinical candidates do not show stable incremental gain.",
        "clinical-only baseline -> EEG-plus-clinical paired bootstrap differences -> directional candidate screen",
        (
            "results/tables/table2_main_model_performance.csv",
            "results/statistics/clinical_incremental_paired_bootstrap_comparison.csv",
        ),
        "Use to constrain EEG incremental-value claims; the exploratory clinical comparison is not a definitive clinical model selection analysis.",
        "Supplementary Information",
    ),
)

TABLES = (
    TableContract("Table 1", "results/tables/table1_cohort_characteristics.csv", "cohort characteristics"),
    TableContract("Table 2", "results/tables/table2_main_model_performance.csv", "main performance"),
    TableContract("Table 3", "results/tables/table3_ablation.csv", "ablation analysis"),
    TableContract("Table 4", "results/tables/table4_explainability_biomarkers.csv", "explainability biomarkers"),
)


def main() -> None:
    OUTPUT_MD.write_text(build_report(), encoding="utf-8")
    print(OUTPUT_MD)


def build_report() -> str:
    nature_text = NATURE_MANUSCRIPT.read_text(encoding="utf-8")
    jne_text = JNE_CLEAN_MANUSCRIPT.read_text(encoding="utf-8")
    manifest = read_manifest(FIGURE_MANIFEST)
    mne_rows = read_csv(MNE_MANIFEST)
    topomap_rows = read_csv(MNE_TOPOMAP_MANIFEST)

    figure_rows = [audit_figure(contract, nature_text, jne_text, manifest, mne_rows) for contract in FIGURES]
    table_rows = [audit_table(contract, nature_text, jne_text) for contract in TABLES]
    topomap_status = audit_topomap_sheet(topomap_rows)
    overall_status = "PASS" if all(row["status"] == "PASS" for row in figure_rows + table_rows) and topomap_status.startswith("PASS") else "WARN"

    lines = [
        "# Figure And Table Submission Readiness Audit",
        "",
        "This audit applies a Nature-style figure contract to the current manuscript figures and main tables. It checks figure logic, source-data traceability, caption presence, export coverage, raster nonblankness, and review-risk boundaries without changing any figure content.",
        "",
        f"Overall status: **{overall_status}**",
        "",
        "## Figure Contract Summary",
        "",
        "| Figure | Archetype | Core conclusion | Evidence chain | Caption status | Export/visual status | Source status | Review risk | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in figure_rows:
        lines.append(
            "| {label} | {archetype} | {core_conclusion} | {evidence_chain} | {caption_status} | {export_status} | {source_status} | {review_risk} | {status} |".format(
                **{key: escape(str(value)) for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## Main Table Source Trace",
            "",
            "| Table | Role | Caption status | Source file | Source status | Status |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in table_rows:
        lines.append(
            "| {label} | {role} | {caption_status} | `{source_path}` | {source_status} | {status} |".format(
                **{key: escape(str(value)) for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## Connectivity Export Detail",
            "",
            f"- MNE connectivity state-band rows: {len(mne_rows)}.",
            f"- Rows with PNG/SVG/PDF exports present: {count_mne_complete(mne_rows)}.",
            f"- Contact sheet present: {MNE_CONTACT_SHEET.exists()}.",
            "",
            "## Topomap Export Detail",
            "",
            f"- MNE topomap rows: {len(topomap_rows)}.",
            f"- Rows with PNG/SVG exports present: {count_topomap_complete(topomap_rows)}.",
            f"- Contact sheet status: {topomap_status}.",
            "",
            "## Interpretation",
            "",
            "- Figures 1-4 are ready as manuscript-level visual arguments provided the unresolved author metadata are not inserted into figure text without confirmation.",
            "- Figure 4 source visualizations include MNE-Python topomap exports and a non-blank topomap contact sheet.",
            "- Supplementary Figure 2 is appropriate as a contact-sheet overview; the underlying 12 MNE panels preserve separate PNG/SVG/PDF exports for editorial requests.",
            "- Figure 4 and Supplementary Figure 2 must remain explicitly hypothesis-generating because the explanations are model-dependent and the supervised cohort is small.",
            "- Supplementary Figure 4 documents the clinical-only baseline boundary and should be retained if the manuscript discusses EEG incremental value.",
            "- The figure package should be regenerated after any author-confirmed changes to cohort flow, intervention details, or safety reporting.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_figure(
    contract: FigureContract,
    nature_text: str,
    jne_text: str,
    manifest: dict[str, dict[str, str]],
    mne_rows: list[dict[str, str]],
) -> dict[str, str]:
    caption_status = caption_check(contract.label, nature_text, jne_text)
    source_missing = [path for path in contract.source_paths if not (ROOT / path).exists()]
    source_status = "all present" if not source_missing else "missing: " + ", ".join(source_missing)

    if contract.stem == "mne_wpli_connectivity_contact_sheet":
        export_status = audit_contact_sheet(mne_rows)
    else:
        row = manifest.get(contract.stem)
        export_status = audit_manifest_row(row) if row else "missing from figure_manifest.csv"

    status = "PASS" if caption_status == "present in Nature and JNE clean" and source_status == "all present" and export_status.startswith("PASS") else "WARN"
    return {
        "label": contract.label,
        "archetype": contract.archetype,
        "core_conclusion": contract.core_conclusion,
        "evidence_chain": contract.evidence_chain,
        "caption_status": caption_status,
        "export_status": export_status,
        "source_status": source_status,
        "review_risk": contract.review_risk,
        "status": status,
    }


def audit_table(contract: TableContract, nature_text: str, jne_text: str) -> dict[str, str]:
    source = ROOT / contract.source_path
    caption_status = caption_check(contract.label, nature_text, jne_text)
    source_status = "present" if source.exists() and source.stat().st_size > 0 else "missing_or_empty"
    status = "PASS" if caption_status == "present in Nature and JNE clean" and source_status == "present" else "WARN"
    return {
        "label": contract.label,
        "role": contract.manuscript_role,
        "caption_status": caption_status,
        "source_path": contract.source_path,
        "source_status": source_status,
        "status": status,
    }


def caption_check(label: str, nature_text: str, jne_text: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}\.", re.MULTILINE)
    nature = bool(pattern.search(nature_text))
    jne = bool(pattern.search(jne_text))
    if nature and jne:
        return "present in Nature and JNE clean"
    if nature:
        return "Nature only"
    if jne:
        return "JNE clean only"
    return "missing"


def audit_manifest_row(row: dict[str, str]) -> str:
    formats = ["png", "svg", "pdf", "tiff"]
    missing = [fmt for fmt in formats if not path_exists(row.get(fmt, ""))]
    if missing:
        return "WARN missing exports: " + ", ".join(missing)
    size_checks = [export_file_check(Path(row[fmt]), fmt) for fmt in formats]
    failed_size = [status for status in size_checks if not status.startswith("PASS")]
    if failed_size:
        return "WARN " + "; ".join(failed_size)
    png_visual = raster_check(Path(row["png"]), "png")
    tiff_visual = raster_check(Path(row["tiff"]), "tiff")
    failed_visual = [status for status in [png_visual, tiff_visual] if not status.startswith("PASS")]
    if failed_visual:
        return "WARN " + "; ".join(failed_visual)
    png_text = png_visual.removeprefix("PASS ")
    tiff_text = tiff_visual.removeprefix("PASS ")
    return "PASS " + png_text + "; " + tiff_text + "; svg/pdf/tiff file sizes ok"


def audit_contact_sheet(mne_rows: list[dict[str, str]]) -> str:
    if not MNE_CONTACT_SHEET.exists():
        return "WARN contact sheet missing"
    complete = count_mne_complete(mne_rows)
    visual = raster_check(MNE_CONTACT_SHEET)
    if complete != 12:
        return f"WARN {complete}/12 MNE state-band rows have PNG/SVG/PDF exports"
    if not visual.startswith("PASS"):
        return visual
    return "PASS contact sheet and 12 MNE rows with PNG/SVG/PDF"


def audit_topomap_sheet(rows: list[dict[str, str]]) -> str:
    if not MNE_TOPOMAP_CONTACT_SHEET.exists():
        return "WARN topomap contact sheet missing"
    complete = count_topomap_complete(rows)
    visual = raster_check(MNE_TOPOMAP_CONTACT_SHEET, "topomap contact sheet")
    if complete != 16:
        return f"WARN {complete}/16 MNE topomap rows have PNG/SVG exports"
    if not visual.startswith("PASS"):
        return visual
    return "PASS topomap contact sheet and 16 MNE rows with PNG/SVG"


def export_file_check(path: Path, label: str) -> str:
    if not path.exists():
        return f"{label} missing"
    if path.stat().st_size <= 1024:
        return f"{label} too small ({path.stat().st_size} bytes)"
    return f"PASS {label} {path.stat().st_size} bytes"


def raster_check(path: Path, label: str = "raster") -> str:
    if not path.exists():
        return f"WARN {label} missing"
    image = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(image)
    min_dimension = min(image.size)
    min_std = min(stat.stddev)
    if min_dimension < 500:
        return f"WARN {label} dimension {image.size[0]}x{image.size[1]} below 500 px"
    if min_std < 10:
        return f"WARN {label} low variance {min_std:.2f}"
    return f"PASS {label} {image.size[0]}x{image.size[1]}, min RGB std {min_std:.1f}"


def count_mne_complete(rows: list[dict[str, str]]) -> int:
    complete = 0
    for row in rows:
        paths = [Path(path) for path in row.get("paths", "").split(";") if path]
        suffixes = {path.suffix.lower() for path in paths if path.exists()}
        if {".png", ".svg", ".pdf"}.issubset(suffixes):
            complete += 1
    return complete


def count_topomap_complete(rows: list[dict[str, str]]) -> int:
    complete = 0
    for row in rows:
        paths = [Path(path) for path in row.get("paths", "").split(";") if path]
        suffixes = {path.suffix.lower() for path in paths if path.exists()}
        if {".png", ".svg"}.issubset(suffixes):
            complete += 1
    return complete


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {row["figure"]: row for row in rows}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def path_exists(value: str) -> bool:
    return bool(value) and Path(value).exists()


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
