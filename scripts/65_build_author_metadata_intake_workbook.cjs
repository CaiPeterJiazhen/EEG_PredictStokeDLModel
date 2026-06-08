const fs = require("fs");
const path = require("path");
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const ROOT = path.resolve(__dirname, "..");
const METADATA_JSON = path.join(ROOT, "docs", "author_submission_metadata_template.json");
const REPLACEMENT_MAP = path.join(ROOT, "results", "tables", "author_field_replacement_map.csv");
const SOURCE_WORKBOOK_AUDIT = path.join(ROOT, "results", "tables", "source_workbook_author_metadata_audit.csv");
const MINIMAL_COMPLETION_CSV = path.join(ROOT, "results", "tables", "author_minimal_completion_pack.csv");
const OUTPUT_DIR = path.join(ROOT, "outputs", "manuscript_package");
const PROJECT_PREFILL_JSON = path.join(OUTPUT_DIR, "author_submission_metadata_project_prefill.json");
const OUTPUT_XLSX = path.join(OUTPUT_DIR, "Author_Submission_Metadata_Intake.xlsx");
const PREVIEW_PNG = path.join(OUTPUT_DIR, "author_metadata_intake_preview.png");

const REQUIRED_KEYS = {
  target_journal_reference_style: ["target_journal", "article_type", "reference_style", "evidence_source"],
  author_list_affiliations: [
    "author_order_full_names",
    "affiliations",
    "corresponding_author_name",
    "corresponding_author_email",
    "evidence_source",
  ],
  author_contributions: ["credit_roles_by_author", "all_authors_approved_final_manuscript", "evidence_source"],
  ethics_approval: [
    "ethics_committee_name",
    "approval_number",
    "approval_date",
    "applicable_site_or_sites",
    "ready_to_paste_statement_en",
    "evidence_source",
  ],
  informed_consent: [
    "consent_route",
    "who_provided_consent",
    "written_or_waived",
    "covered_eeg_tacs_clinical_assessments",
    "covered_data_sharing",
    "ready_to_paste_statement_en",
    "evidence_source",
  ],
  trial_or_study_registration: ["ready_to_paste_statement_en", "evidence_source"],
  study_site_dates_design: [
    "hospital_or_department",
    "recruitment_start_date",
    "recruitment_end_date",
    "follow_up_or_last_assessment_window",
    "prospective_or_retrospective",
    "single_or_multicentre",
    "ready_to_paste_methods_sentence_en",
    "evidence_source",
  ],
  eligibility_stroke_timing: [
    "inclusion_criteria",
    "exclusion_criteria",
    "stroke_subtype_criteria",
    "time_since_stroke_to_eeg",
    "evidence_source",
  ],
  tacs_device_electrodes: [
    "device_model",
    "electrode_dimensions",
    "confirmed_target_frequency_intensity_duration_sessions",
    "evidence_source",
  ],
  concurrent_rehabilitation: [
    "was_conventional_rehabilitation_delivered",
    "frequency",
    "duration_per_session",
    "main_training_content",
    "ready_to_paste_methods_sentence_en",
    "evidence_source",
  ],
  tacs_safety_adverse_events: [
    "adverse_event_summary",
    "tolerability_summary",
    "withdrawals_or_discontinuations",
    "ready_to_paste_statement_en",
    "evidence_source",
  ],
  fma_assessors_timing: [
    "assessor_training_or_credentials",
    "assessor_blinding",
    "baseline_assessment_timing",
    "post_treatment_assessment_timing",
    "evidence_source",
  ],
  eeg_hardware_reference_impedance: [
    "amplifier_model",
    "acquisition_software",
    "cap_system",
    "original_channel_count",
    "online_reference",
    "ground",
    "impedance_threshold",
    "evidence_source",
  ],
  resting_state_instructions: [
    "eyes_open_eyes_closed_order",
    "target_duration_per_condition",
    "fixation_or_eye_instruction",
    "evidence_source",
  ],
  raw_eeg_preprocessing: [
    "raw_filtering",
    "notch_filtering",
    "rereference",
    "artifact_rejection",
    "ica_or_eye_muscle_artifact_handling",
    "bad_channel_handling",
    "eeglab_set_fdt_export_rules",
    "evidence_source",
  ],
  data_repository_doi_scope: [
    "repository_name",
    "doi_or_accession",
    "version",
    "data_license",
    "public_data_scope",
    "restricted_data_scope",
    "controlled_access_contact_or_committee",
    "request_review_requirements",
    "ready_to_paste_data_availability_en",
    "evidence_source",
  ],
  code_repository_license: [
    "code_repository_url_or_doi",
    "version_or_commit_hash",
    "code_license",
    "ready_to_paste_code_availability_en",
    "evidence_source",
  ],
  funding_competing_acknowledgements: [
    "funding_sources_and_grant_numbers",
    "competing_interests_statement",
    "evidence_source",
  ],
  suggested_opposed_reviewers: [],
};

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

function csvRowsToObjects(rows) {
  const [header, ...body] = rows;
  return body.map((row) => Object.fromEntries(header.map((key, index) => [key, row[index] || ""])));
}

function padRows(rows) {
  const width = Math.max(...rows.map((row) => row.length));
  return rows.map((row) => row.concat(Array(width - row.length).fill("")));
}

function writeMatrix(sheet, rows, options = {}) {
  const matrix = padRows(rows);
  sheet.getRangeByIndexes(0, 0, matrix.length, matrix[0].length).values = matrix;
  const used = sheet.getUsedRange();
  used.format.autofitColumns();
  used.format.autofitRows();
  used.format.wrapText = true;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const header = sheet.getRangeByIndexes(0, 0, 1, matrix[0].length);
  header.format.fill.color = options.headerColor || "#D9EAF7";
  header.format.font.bold = true;
  header.format.wrapText = true;
  if (options.authorInputColumn != null && matrix.length > 1) {
    sheet.getRangeByIndexes(1, options.authorInputColumn, matrix.length - 1, 1).format.fill.color = "#FFF8E8";
  }
  if (options.authorInputColumns && matrix.length > 1) {
    for (const columnIndex of options.authorInputColumns) {
      sheet.getRangeByIndexes(1, columnIndex, matrix.length - 1, 1).format.fill.color = "#FFF8E8";
    }
  }
  if (options.columnWidthsPx) {
    setColumnWidths(sheet, matrix.length, options.columnWidthsPx);
    used.format.autofitRows();
  }
}

function setColumnWidths(sheet, rowCount, widthsPx) {
  for (const [columnIndexText, width] of Object.entries(widthsPx)) {
    const columnIndex = Number(columnIndexText);
    try {
      sheet.getRangeByIndexes(0, columnIndex, Math.max(rowCount, 1), 1).format.columnWidthPx = width;
    } catch (error) {
      console.warn(`Column width skipped on ${sheet.name} column ${columnIndex + 1}: ${error.message}`);
    }
  }
}

function keyHelp(key) {
  const labels = {
    evidence_source: "Document, protocol, repository record, or author-approved source supporting this answer.",
    ready_to_paste_statement_en: "Final English statement that can be pasted into the manuscript/declarations.",
    ready_to_paste_methods_sentence_en: "Final English methods sentence that can be pasted into the manuscript.",
    ready_to_paste_data_availability_en: "Final English Data Availability wording with DOI/access and restrictions.",
    ready_to_paste_code_availability_en: "Final English Code Availability wording with repository/version/licence.",
  };
  return labels[key] || "Author response required for final manuscript insertion.";
}

async function main() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const metadata = JSON.parse(fs.readFileSync(METADATA_JSON, "utf8")).fields;
  const projectPrefill = fs.existsSync(PROJECT_PREFILL_JSON)
    ? JSON.parse(fs.readFileSync(PROJECT_PREFILL_JSON, "utf8")).fields
    : {};
  const replacementRows = csvRowsToObjects(parseCsv(fs.readFileSync(REPLACEMENT_MAP, "utf8")));
  const sourceAuditRows = fs.existsSync(SOURCE_WORKBOOK_AUDIT)
    ? csvRowsToObjects(parseCsv(fs.readFileSync(SOURCE_WORKBOOK_AUDIT, "utf8")))
    : [];
  const replacementByField = new Map(replacementRows.map((row) => [row.field_id, row]));
  const sourceAuditByField = new Map(sourceAuditRows.map((row) => [row.field_id, row]));
  const workbook = Workbook.create();

  const readme = workbook.worksheets.add("README");
  writeMatrix(readme, [
    ["Item", "Instruction"],
    ["Purpose", "Author-facing workbook for final submission metadata needed before clean manuscript upload."],
    ["How to fill", "Complete the yellow Author response cells and provide an evidence source for every non-optional field."],
    ["Do not infer", "Ethics, consent, registration, EEG acquisition, raw preprocessing, tACS safety, and repository access cannot be guessed from model scripts."],
    ["JSON mirror", "This workbook mirrors docs/author_submission_metadata_template.json."],
    ["Strict validation", "After responses are transferred back to JSON, run python scripts/63_validate_author_submission_metadata.py --strict."],
    ["Workbook import", "Run python scripts/66_import_author_metadata_intake_workbook.py to convert completed workbook responses into JSON."],
    ["Insertion protocol", "Then run python scripts/64_build_author_metadata_insertion_protocol.py --strict before editing final manuscript files."],
    ["Fastest path", "Fill Minimal_Completion.author_response and Minimal_Completion.evidence_source first. The importer reads this sheet and then falls back to Author_Input if a value is blank."],
    ["Status columns", "Minimal_Completion columns N:P are formula-based helpers that show row completion, field completion, and missing required counts. Do not edit them manually; the importer reads only field_id, metadata_key, author_response, and evidence_source."],
    ["Project prefill", "Review the Project_Prefill sheet for project-supported values. Copy only author-approved values into Author_Input.author_response."],
    ["Source-workbook audit", "Review Source_Workbook_Audit and the evidence-grade columns in Author_Input before inferring any protocol, ethics, EEG, rehabilitation, or repository detail."],
    ["Current status", "The template is intentionally blank; final manuscript replacement is blocked until author-approved evidence is supplied."],
  ]);

  const minimalRows = buildMinimalCompletionRows();
  const minimal = workbook.worksheets.add("Minimal_Completion");
  writeMatrix(minimal, minimalRows, {
    headerColor: "#FCE4D6",
    authorInputColumns: [5, 6],
    columnWidthsPx: {
      0: 210,
      1: 85,
      2: 245,
      3: 110,
      4: 105,
      5: 300,
      6: 260,
      7: 430,
      8: 360,
      9: 360,
      10: 360,
      11: 420,
      12: 360,
      13: 170,
      14: 190,
      15: 115,
    },
  });
  addMinimalCompletionStatus(minimal, minimalRows.length);

  const authorInputRows = [
    [
      "field_id",
      "priority",
      "metadata_key",
      "required_for_strict_validation",
      "source_workbook_evidence_grade",
      "author_response",
      "evidence_source",
      "guidance",
      "source_workbook_author_action",
      "target_files",
      "replacement_action",
    ],
  ];
  const summaryRows = [
    [
      "field_id",
      "priority",
      "required_key_count",
      "metadata_key_count",
      "current_template_status",
      "source_workbook_evidence_grade",
      "manuscript_section",
      "target_files",
      "verification_after_replacement",
    ],
  ];

  for (const [fieldId, fieldValues] of Object.entries(metadata)) {
    const replacement = replacementByField.get(fieldId) || {};
    const sourceAudit = sourceAuditByField.get(fieldId) || {};
    const keys = Object.keys(fieldValues);
    const required = new Set(REQUIRED_KEYS[fieldId] || []);
    summaryRows.push([
      fieldId,
      replacement.priority || "not_mapped",
      String(required.size),
      String(keys.length),
      keys.some((key) => String(fieldValues[key] || "").trim()) ? "partly supplied" : "blank",
      sourceAudit.evidence_grade || "not_audited",
      replacement.manuscript_section || "",
      replacement.target_files || "",
      replacement.verification_after_replacement || "",
    ]);
    for (const key of keys) {
      authorInputRows.push([
        fieldId,
        replacement.priority || "not_mapped",
        key,
        required.has(key) ? "yes" : "no",
        sourceAudit.evidence_grade || "not_audited",
        fieldValues[key] || "",
        key === "evidence_source" ? fieldValues[key] || "" : "",
        keyHelp(key),
        sourceAudit.author_action || "",
        replacement.target_files || "",
        replacement.replacement_action || "",
      ]);
    }
  }

  const input = workbook.worksheets.add("Author_Input");
  writeMatrix(input, authorInputRows, {
    headerColor: "#D9EAF7",
    authorInputColumn: 5,
    columnWidthsPx: {
      0: 190,
      1: 85,
      2: 240,
      3: 120,
      4: 190,
      5: 260,
      6: 260,
      7: 330,
      8: 360,
      9: 420,
      10: 360,
    },
  });

  const summary = workbook.worksheets.add("Field_Summary");
  writeMatrix(summary, summaryRows, {
    headerColor: "#E2F0D9",
    columnWidthsPx: { 0: 210, 1: 90, 2: 120, 3: 120, 4: 135, 5: 190, 6: 220, 7: 420, 8: 360 },
  });

  const replacement = workbook.worksheets.add("Replacement_Map");
  writeMatrix(replacement, parseCsv(fs.readFileSync(REPLACEMENT_MAP, "utf8")), {
    headerColor: "#EADCF8",
    columnWidthsPx: { 0: 210, 1: 260, 2: 120, 3: 420, 4: 220, 5: 360, 6: 360, 7: 320, 8: 340, 9: 95 },
  });

  const prefillRows = buildProjectPrefillRows(projectPrefill, replacementByField);
  const prefill = workbook.worksheets.add("Project_Prefill");
  writeMatrix(prefill, prefillRows, {
    headerColor: "#E2F0D9",
    columnWidthsPx: { 0: 210, 1: 240, 2: 430, 3: 360, 4: 125, 5: 420, 6: 360 },
  });

  const sourceAudit = workbook.worksheets.add("Source_Workbook_Audit");
  writeMatrix(sourceAudit, buildSourceWorkbookAuditRows(sourceAuditRows), {
    headerColor: "#FCE4D6",
    columnWidthsPx: { 0: 210, 1: 330, 2: 260, 3: 95, 4: 190, 5: 330, 6: 220, 7: 430, 8: 360 },
  });

  const guide = workbook.worksheets.add("Validation_Guide");
  writeMatrix(guide, [
    ["Step", "Action"],
    ["1", "Fill Author_Input.author_response for each required metadata_key."],
    ["2", "For each field_id, fill evidence_source with the ethics document, consent form, protocol, device record, repository DOI, or author-approved source."],
    ["3", "Review Project_Prefill. If a prefilled value is correct, copy it into the matching Author_Input.author_response cell and keep or adapt its evidence source."],
    ["4", "For the shortest path, fill Minimal_Completion.author_response and Minimal_Completion.evidence_source. Columns N:P are formula-based status helpers, should not be edited manually, and are not imported."],
    ["5", "Review Source_Workbook_Audit. Treat no_source_workbook_evidence and keyword_hits_not_submission_ready as requiring protocol, ethics, device, repository, or author-confirmed evidence before final insertion."],
    ["6", "Run python scripts/66_import_author_metadata_intake_workbook.py to generate outputs/manuscript_package/author_submission_metadata_from_intake.json."],
    ["7", "Run python scripts/63_validate_author_submission_metadata.py --metadata outputs/manuscript_package/author_submission_metadata_from_intake.json --strict."],
    ["8", "Run python scripts/64_build_author_metadata_insertion_protocol.py --metadata outputs/manuscript_package/author_submission_metadata_from_intake.json --strict."],
    ["9", "Only after both strict checks pass, replace ready fields in manuscript/declarations/cover letter/repository README."],
    ["10", "Regenerate DOCX files, source-data workbook, visual QA, manuscript integrity audit, artifact audit, and the submission package."],
  ], { headerColor: "#FCE4D6" });

  const inspect = await workbook.inspect({
    kind: "workbook,sheet",
    include: "name,usedRange",
  });
  console.log(inspect.ndjson || inspect);

  const preview = await workbook.render({ sheetName: "Minimal_Completion", range: "A1:P28", scale: 1, format: "png" });
  fs.writeFileSync(PREVIEW_PNG, new Uint8Array(await preview.arrayBuffer()));

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson || errors);

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(OUTPUT_XLSX);
  console.log(OUTPUT_XLSX);
}

function buildMinimalCompletionRows() {
  const header = [
    "field_id",
    "priority",
    "metadata_key",
    "strict_required",
    "missing_now",
    "author_response",
    "evidence_source",
    "project_prefill_suggestion",
    "author_action",
    "key_guidance",
    "alternative_requirement",
    "target_files",
    "replacement_action",
  ];
  if (!fs.existsSync(MINIMAL_COMPLETION_CSV)) {
    return [
      header,
      [
        "missing",
        "blocking",
        "missing_source_csv",
        "yes",
        "yes",
        "",
        "",
        "",
        "Run scripts/76_build_author_minimal_completion_pack.py before building this workbook.",
        "The minimal completion source CSV is missing.",
        "none",
        "",
        "",
      ],
    ];
  }
  const sourceRows = csvRowsToObjects(parseCsv(fs.readFileSync(MINIMAL_COMPLETION_CSV, "utf8")));
  if (!sourceRows.length) {
    return [header, ["none", "none", "none", "no", "no", "", "", "", "No incomplete fields.", "", "none", "", ""]];
  }
  return [
    header,
    ...sourceRows.map((row) => [
      row.field_id || "",
      row.priority || "",
      row.metadata_key || "",
      row.strict_required || "",
      row.missing_now || "",
      row.current_author_response || "",
      "",
      row.project_prefill_suggestion || "",
      row.author_action || "",
      row.key_guidance || "",
      row.alternative_requirement || "",
      row.target_files || "",
      row.replacement_action || "",
    ]),
  ];
}

function addMinimalCompletionStatus(sheet, rowCount) {
  if (rowCount < 2) {
    return;
  }
  sheet.getRange("N1:P1").values = [["row_completion_status", "field_completion_status", "field_missing_required_count"]];
  sheet.getRange("N1:P1").format.fill.color = "#D9EAD3";
  sheet.getRange("N1:P1").format.font.bold = true;
  sheet.getRangeByIndexes(1, 13, rowCount - 1, 3).format.fill.color = "#F3F7EF";
  const lastRow = rowCount;
  const rowStatus = [];
  const fieldStatus = [];
  const missingCount = [];
  for (let row = 2; row <= rowCount; row += 1) {
    const fieldMissingFormula =
      `=SUMPRODUCT(($A$2:$A$${lastRow}=$A${row})*($D$2:$D$${lastRow}="yes")*($C$2:$C$${lastRow}<>"evidence_source")*($F$2:$F$${lastRow}=""))+` +
      `SUMPRODUCT(($A$2:$A$${lastRow}=$A${row})*($D$2:$D$${lastRow}="yes")*($C$2:$C$${lastRow}="evidence_source")*($F$2:$F$${lastRow}="")*($G$2:$G$${lastRow}=""))`;
    rowStatus.push([
      `=IF($A${row}="","",IF($D${row}="project_prefill_only","prefill suggestion only",IF($C${row}="evidence_source",IF(OR(LEN(TRIM($F${row}))>0,LEN(TRIM($G${row}))>0),"filled","needs evidence source"),IF(LEN(TRIM($F${row}))>0,"filled","needs author response"))))`,
    ]);
    fieldStatus.push([
      `=IF($A${row}="","",IF($P${row}=0,"field ready for import","field incomplete"))`,
    ]);
    missingCount.push([fieldMissingFormula]);
  }
  sheet.getRange(`N2:N${rowCount}`).formulas = rowStatus;
  sheet.getRange(`O2:O${rowCount}`).formulas = fieldStatus;
  sheet.getRange(`P2:P${rowCount}`).formulas = missingCount;
  sheet.getRange(`P2:P${rowCount}`).format.numberFormat = "0";
}

function buildProjectPrefillRows(projectPrefill, replacementByField) {
  const rows = [
    [
      "field_id",
      "metadata_key",
      "project_prefill_value",
      "evidence_source",
      "author_action",
      "target_files",
      "replacement_action",
    ],
  ];
  for (const [fieldId, fieldValues] of Object.entries(projectPrefill || {})) {
    const replacement = replacementByField.get(fieldId) || {};
    const fieldEvidence = fieldValues.evidence_source || "";
    for (const [key, value] of Object.entries(fieldValues)) {
      if (key === "evidence_source" || !String(value || "").trim()) {
        continue;
      }
      rows.push([
        fieldId,
        key,
        value,
        fieldEvidence,
        "Review; copy to Author_Input only if author-approved.",
        replacement.target_files || "",
        replacement.replacement_action || "",
      ]);
    }
  }
  if (rows.length === 1) {
    rows.push(["none", "none", "No project prefill values found.", "", "No action.", "", ""]);
  }
  return rows;
}

function buildSourceWorkbookAuditRows(sourceAuditRows) {
  const header = [
    "field_id",
    "required_evidence",
    "source_workbooks_inspected",
    "keyword_hit_count",
    "matched_terms",
    "nonidentifying_cell_refs",
    "evidence_grade",
    "current_interpretation",
    "author_action",
  ];
  if (!sourceAuditRows.length) {
    return [
      header,
      [
        "none",
        "none",
        "No source-workbook author metadata audit CSV was found.",
        "0",
        "none",
        "none",
        "not_audited",
        "Run scripts/74_audit_source_workbook_author_metadata.py before using this sheet.",
        "Do not infer submission metadata from source workbooks until the audit is available.",
      ],
    ];
  }
  return [header, ...sourceAuditRows.map((row) => header.map((key) => row[key] || ""))];
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
