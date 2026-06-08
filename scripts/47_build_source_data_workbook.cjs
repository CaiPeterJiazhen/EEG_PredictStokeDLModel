const fs = require("fs");
const path = require("path");
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const ROOT = path.resolve(__dirname, "..");
const OUTPUT_DIR = path.join(ROOT, "outputs", "manuscript_package");
const OUTPUT_XLSX = path.join(OUTPUT_DIR, "ResidualAware_SSL_CNN_Source_Data.xlsx");
const PREVIEW_PNG = path.join(OUTPUT_DIR, "source_data_readme_preview.png");

const csvSheets = [
  ["Data_Dictionary", "results/tables/source_data_dictionary.csv"],
  ["FigureManifest", "results/figures/nature/figure_manifest.csv"],
  ["Manuscript_Audit", "results/tables/manuscript_integrity_audit.csv"],
  ["Reference_Metadata", "results/tables/reference_metadata_audit.csv"],
  ["Local_Refs", "results/tables/local_reference_positioning_matrix.csv"],
  ["Artifact_QA", "results/tables/submission_artifact_quality_audit.csv"],
  ["Docx_Visual_QA", "results/tables/docx_visual_qa_metrics.csv"],
  ["Participant_Flow", "results/tables/participant_flow_safety_source_notes.csv"],
  ["Risk_Audit", "results/tables/probast_tripod_ai_risk_audit.csv"],
  ["Precision_Audit", "results/tables/performance_precision_audit.csv"],
  ["Claim_Audit", "results/tables/claim_strength_audit.csv"],
  ["Model_Card", "results/tables/model_reporting_card.csv"],
  ["Table1_Cohort", "results/tables/table1_cohort_characteristics.csv"],
  ["Table2_Main", "results/tables/table2_main_model_performance.csv"],
  ["Table2_Compact", "results/tables/model_performance_main_table.csv"],
  ["Table3_Ablation", "results/tables/table3_ablation.csv"],
  ["Table4_Explain", "results/tables/table4_explainability_biomarkers.csv"],
  ["Stats_CI", "results/statistics/model_metric_confidence_intervals.csv"],
  ["Stats_Pairwise", "results/statistics/model_pairwise_comparisons.csv"],
  ["Stats_Permutation", "results/statistics/model_permutation_tests.csv"],
  ["Clinical_Bootstrap", "results/statistics/clinical_incremental_paired_bootstrap_comparison.csv"],
  ["Seed_Stability", "results/tables/seed_stability_table.csv"],
  ["Error_Subjects", "results/tables/error_subject_clinical_eeg_summary.csv"],
  ["Explain_Key", "results/tables/explainability_key_findings_table.csv"],
  ["EEG_Metadata", "results/tables/eeg_recording_metadata_audit.csv"],
  ["EEG_Summary", "results/tables/eeg_recording_summary.csv"],
  ["Workbook_Audit", "results/tables/clinical_workbook_structure_audit.csv"],
  ["Validation_Audit", "results/tables/prediction_validation_integrity_audit.csv"],
  ["Source_Metadata_Audit", "results/tables/source_workbook_author_metadata_audit.csv"],
  ["PatientPDF_Audit", "results/tables/patient_record_pdf_text_audit.csv"],
  ["Author_Evidence", "results/tables/author_required_evidence_trace.csv"],
  ["Replacement_Map", "results/tables/author_field_replacement_map.csv"],
  ["MNE_WPLI_Figures", "results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_manifest.csv"],
];

function padRows(rows) {
  const width = Math.max(...rows.map((row) => row.length));
  return rows.map((row) => row.concat(Array(width - row.length).fill("")));
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;
  const input = text.replace(/^\uFEFF/, "");

  for (let i = 0; i < input.length; i += 1) {
    const char = input[i];
    const next = input[i + 1];
    if (inQuotes) {
      if (char === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        cell += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (char !== "\r") {
      cell += char;
    }
  }
  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((values) => values.some((value) => value !== ""));
}

function writeMatrix(sheet, rows) {
  const matrix = padRows(rows);
  sheet.getRangeByIndexes(0, 0, matrix.length, matrix[0].length).values = matrix;
  try {
    const used = sheet.getUsedRange();
    used.format.autofitColumns();
    used.format.autofitRows();
    const header = sheet.getRangeByIndexes(0, 0, 1, matrix[0].length);
    header.format.fill.color = "#D9E2F3";
    header.format.font.bold = true;
    header.format.wrapText = true;
    sheet.freezePanes.freezeRows(1);
    sheet.showGridLines = false;
  } catch (error) {
    console.warn(`Formatting skipped on ${sheet.name}: ${error.message}`);
  }
}

async function main() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const workbook = Workbook.create();
  const dictionaryStats = getDictionaryStats();

  const readme = workbook.worksheets.add("README");
  writeMatrix(readme, [
    ["Field", "Value"],
    ["Workbook purpose", "Source data and statistical outputs for the residual-aware SSL-CNN stroke EEG manuscript"],
    ["Generated from", "Current project analysis outputs"],
    ["Primary validation unit", "Subject-level LOSO predictions; seed/segment rows are not independent patients"],
    ["Main manuscript", "docs/manuscript_residual_aware_ssl_cnn_nature_polished.md"],
    ["Compiled manuscript DOCX", "output/doc/ResidualAware_SSL_CNN_Nature_Manuscript.docx"],
    ["Compiled supplementary DOCX", "output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx"],
    ["Figure manifest", "results/figures/nature/figure_manifest.csv"],
    ["Data dictionary", "results/tables/source_data_dictionary.csv"],
    ["Data dictionary coverage", `${dictionaryStats.fields} fields across ${dictionaryStats.files} derived CSV or figure-manifest files`],
    ["Repository README draft", "docs/repository_readme_for_deposit.md"],
    ["Data availability status", "Derived tables and code can be shared; raw EEG/clinical data require author/ethics confirmation"],
    ["Reviewer risk", "Small n=19 supervised cohort; no external validation; clinical-only baseline is exploratory but strong"],
  ]);

  const audit = workbook.worksheets.add("Data_Audit");
  writeMatrix(audit, [
    ["Dataset", "File or location", "Access route", "Supports", "Submission action"],
    ["Cohort characteristics", "results/tables/table1_cohort_characteristics.csv", "Derived table; depositable", "Table 1; cohort description", "Deposit with source data"],
    ["Participant flow and safety source notes", "results/tables/participant_flow_safety_source_notes.csv", "Derived de-identified count table; depositable after author review", "Figure 1 participant-flow panel; safety-source review", "Deposit with source data and keep final safety wording author-confirmed"],
    ["PROBAST/TRIPOD+AI risk audit", "results/tables/probast_tripod_ai_risk_audit.csv", "Reporting audit; depositable after author review", "Supplementary Table 8; prediction-model bias and applicability transparency", "Deposit with source data and reporting checklist"],
    ["Performance precision audit", "results/tables/performance_precision_audit.csv", "Reporting audit; depositable after author review", "Supplementary Table 9; validation precision and superiority-claim boundary", "Deposit with source data and statistical audit tables"],
    ["Claim strength audit", "results/tables/claim_strength_audit.csv", "Reporting audit; depositable after author review", "Supplementary Table 10; evidence-to-wording alignment for central manuscript claims", "Deposit with source data and reporting audit tables"],
    ["AI model reporting card", "results/tables/model_reporting_card.csv", "Reporting audit; depositable after author review", "Supplementary Table 11; intended use, validation safeguards, reproducibility actions, and unsupported uses", "Deposit with source data and reporting audit tables"],
    ["Main model performance", "results/tables/table2_main_model_performance.csv", "Derived table; depositable", "Table 2; Figure 2", "Deposit with source data"],
    ["Model confidence intervals", "results/statistics/model_metric_confidence_intervals.csv", "Derived table; depositable", "Results uncertainty", "Deposit with statistics"],
    ["Pairwise comparisons", "results/statistics/model_pairwise_comparisons.csv", "Derived table; depositable", "Paired bootstrap and McNemar results", "Deposit with statistics"],
    ["Permutation tests", "results/statistics/model_permutation_tests.csv", "Derived table; depositable", "Above-chance model testing", "Deposit with statistics"],
    ["Ablations", "results/tables/table3_ablation.csv", "Derived table; depositable", "Table 3; Figure 3", "Deposit with source data"],
    ["Explainability biomarkers", "results/tables/table4_explainability_biomarkers.csv", "Derived table; depositable", "Table 4; Figure 4", "Deposit with source data"],
    ["Generated figures", "results/figures/nature/", "Figure exports; depositable", "Main and supplementary figures", "Deposit TIFF/PDF/SVG as required"],
    ["MNE WPLI connectivity figures", "results/figures/explainability/mne_wpli_connectivity/", "Figure exports; depositable", "Supplementary Figure 2", "Deposit PNG/SVG/PDF as required"],
    ["Manuscript integrity audit", "results/tables/manuscript_integrity_audit.csv", "Quality audit; depositable", "Citation, numeric-claim, table-source, figure-export, and clean-variant checks", "Deposit with audit tables"],
    ["Reference metadata audit", "results/tables/reference_metadata_audit.csv", "Quality audit; depositable", "DOI/URL verification for the manuscript reference list", "Deposit with audit tables"],
    ["Local reference positioning matrix", "results/tables/local_reference_positioning_matrix.csv", "Citation audit; depositable after author review", "Conservative use of the provided local reference PDFs", "Deposit with reference audit"],
    ["DOCX visual QA metrics", "results/tables/docx_visual_qa_metrics.csv", "Quality audit; depositable", "Word COM PDF export and rendered-page visual checks", "Deposit with audit tables"],
    ["Submission artifact QA", "results/tables/submission_artifact_quality_audit.csv", "Quality audit; depositable", "DOCX, workbook, figure, reference, visual QA, and package-integrity checks", "Deposit with audit tables"],
    ["Prediction-validation integrity audit", "results/tables/prediction_validation_integrity_audit.csv", "Quality audit; depositable", "Patient-level validation and leakage-safeguard checks", "Deposit with audit tables"],
    ["Source workbook author-metadata audit", "results/tables/source_workbook_author_metadata_audit.csv", "Submission-support metadata; depositable after author review", "Non-identifying Excel-source scan for author-required protocol metadata", "Use to keep unresolved author fields evidence-based"],
    ["Author-required evidence trace", "results/tables/author_required_evidence_trace.csv", "Submission-support metadata; depositable after author review", "Fields requiring author/ethics/clinical-team confirmation", "Deposit with author query materials"],
    ["Author field replacement map", "results/tables/author_field_replacement_map.csv", "Submission-support metadata; depositable after author review", "Replacement targets for author-supplied fields across manuscript and declaration files", "Use during final author-field insertion"],
    ["Raw EEG", "data directory, raw source not inventoried here", "Restricted human-participant data", "EEG preprocessing and features", "Confirm ethics/data-use route"],
    ["Clinical records", "clinical workbook/source records", "Restricted human-participant data", "Labels and clinical covariates", "Confirm consent and access committee"],
    ["Analysis code", "scripts/ and src/", "Code repository; depositable after cleanup", "Reproducibility", "Archive with commit hash and environment"],
    ["Data dictionary", "results/tables/source_data_dictionary.csv", "Derived metadata; depositable", "Field-level source-data reuse", "Deposit with source data"],
    ["Repository README draft", "docs/repository_readme_for_deposit.md", "Repository metadata; depositable after author review", "Repository-level reuse instructions", "Complete DOI/licence/access fields"],
    ["Supplementary Information", "output/doc/ResidualAware_SSL_CNN_Supplementary_Information.docx", "Compiled document; depositable", "Supplementary figures and tables", "Submit with manuscript after author approval"],
  ]);

  const methods = workbook.worksheets.add("Methods_Provenance");
  writeMatrix(methods, [
    ["Methods detail", "Evidence file", "Evidence status"],
    ["Baseline EEG predicts post-tACS FMA-UE proportional recovery", "tacs_eeg_proportional_recovery_project_design.md; docs/project_context.md", "Project design and context"],
    ["tACS protocol: contralateral M1, C3/C4 by affected hand, 20 Hz, 1000 microampere, 20 min, 14 daily sessions", "tacs_eeg_proportional_recovery_project_design.md", "Project design"],
    ["Supervised cohort n=19", "docs/project_context.md; docs/cohort_characteristics.md; results/tables/table1_cohort_characteristics.csv", "Confirmed current output"],
    ["All-patient EEG cohort n=28", "docs/cohort_characteristics.md", "Confirmed current output"],
    ["FMA-UE predicted improvement = 0.7 x (66 - baseline FMA-UE)", "docs/project_context.md; src/eeg_recovery/metadata/labels.py", "Confirmed in design and code"],
    ["Residual threshold = supervised-cohort median, 1.5", "docs/project_context.md; src/eeg_recovery/metadata/labels.py", "Confirmed in design and code"],
    ["Retained EEG channels = 62, excluding M1 and M2", "docs/project_context.md; src/eeg_recovery/channels/mapping.py; configs/channel_mapping.yaml", "Confirmed in context, code, config"],
    ["Sampling rate = 250 Hz", "docs/project_context.md", "Confirmed in project context"],
    ["Baseline *1.set = EO, *2.set = EC", "docs/project_context.md", "Confirmed in project context"],
    ["Supervised baseline recording duration = mean 188.4 s, range 101.0-247.8 s across 38 EO/EC files", "results/tables/eeg_recording_metadata_audit.csv; docs/eeg_metadata_audit.md", "Confirmed from EEGLAB metadata audit"],
    ["PSD: Welch, Hann, 0.5 Hz resolution, 50% overlap, 0.5-45 Hz", "src/eeg_recovery/features/psd.py", "Confirmed in code"],
    ["Connectivity: STFT, 2 s Hann windows, 50% overlap, WPLI/imcoh, six bands", "src/eeg_recovery/features/connectivity.py", "Confirmed in code"],
    ["Affected-side channel alignment before PSD and FC", "src/eeg_recovery/channels/hemisphere_flip.py; configs/channel_mapping.yaml", "Confirmed in code and config"],
    ["Patient-level LOSO; no segment/seed independence", "docs/statistical_validation_summary.md; docs/project_context.md", "Confirmed in statistical summary and context"],
  ]);

  for (const [sheetName, relPath] of csvSheets) {
    const filePath = path.join(ROOT, relPath);
    if (!fs.existsSync(filePath)) {
      console.warn(`Missing source file: ${relPath}`);
      continue;
    }
    const csvText = fs.readFileSync(filePath, "utf8");
    const sheet = workbook.worksheets.add(sheetName);
    const parsedRows = parseCsv(csvText);
    if (parsedRows.length === 0) {
      continue;
    }
    writeMatrix(sheet, parsedRows);
    try {
      const used = sheet.getUsedRange();
      used.format.autofitColumns();
      used.format.autofitRows();
      const rowCount = used.rowCount || 1;
      const columnCount = used.columnCount || 1;
      const header = sheet.getRangeByIndexes(0, 0, 1, columnCount);
      header.format.fill.color = "#D9E2F3";
      header.format.font.bold = true;
      header.format.wrapText = true;
      sheet.freezePanes.freezeRows(1);
      sheet.showGridLines = false;
      if (rowCount > 1 && columnCount > 1) {
        sheet.getRangeByIndexes(1, 0, rowCount - 1, columnCount).format.wrapText = true;
      }
    } catch (error) {
      console.warn(`Formatting skipped on ${sheetName}: ${error.message}`);
    }
  }

  const inspect = await workbook.inspect({
    kind: "workbook,sheet",
    include: "name,usedRange",
  });
  console.log(inspect.ndjson || inspect);

  try {
    const preview = await workbook.render({ sheetName: "README", autoCrop: "all", scale: 1, format: "png" });
    const previewBytes = new Uint8Array(await preview.arrayBuffer());
    fs.writeFileSync(PREVIEW_PNG, previewBytes);
  } catch (error) {
    console.warn(`Preview render skipped: ${error.message}`);
  }

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(OUTPUT_XLSX);
  console.log(OUTPUT_XLSX);
}

function getDictionaryStats() {
  const dictionaryPath = path.join(ROOT, "results/tables/source_data_dictionary.csv");
  if (!fs.existsSync(dictionaryPath)) {
    return { fields: "not generated", files: "not generated" };
  }
  const rows = parseCsv(fs.readFileSync(dictionaryPath, "utf8"));
  if (rows.length <= 1) {
    return { fields: 0, files: 0 };
  }
  const header = rows[0];
  const sourceFileIndex = header.indexOf("source_file");
  const sourceFiles = new Set();
  for (const row of rows.slice(1)) {
    if (sourceFileIndex >= 0 && row[sourceFileIndex]) {
      sourceFiles.add(row[sourceFileIndex]);
    }
  }
  return { fields: rows.length - 1, files: sourceFiles.size };
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
