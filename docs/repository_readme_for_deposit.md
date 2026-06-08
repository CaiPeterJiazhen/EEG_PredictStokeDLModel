# Repository README For Deposit

This README is a repository-facing draft for depositing the derived data and reproducibility materials supporting the manuscript "Residual-aware EEG learning for post-stroke proportional recovery". It is intended for a public or controlled-access data repository record after author and institutional confirmation.

## Dataset Record Metadata To Complete Before Deposit

| Repository field | Draft value or author-required input |
|---|---|
| Dataset title | Source data and reproducibility package for "Residual-aware EEG learning for post-stroke proportional recovery" |
| Creators | [Full author names, affiliations, and ORCID IDs to be inserted after author confirmation] |
| Related article | [Journal article title, journal, DOI, and manuscript identifier after submission or publication] |
| Description | Derived subject-level analysis tables, locked LOSO predictions, bootstrap, permutation and paired-comparison outputs, participant-flow source notes, figure source summaries, validation and artifact-quality audit tables, figure exports, source-data workbook, data dictionary, and scripts needed to regenerate the deposited derived outputs. |
| Keywords | stroke; electroencephalography; proportional recovery; upper-limb recovery; self-supervised learning; convolutional neural network; residual-aware learning; leave-one-subject-out validation; tACS; PSD; WPLI |
| Public data scope | De-identified derived tables, statistical outputs, locked prediction files, figure source summaries, figure exports, audit tables, source-data workbook, source-data dictionary, and reproducibility scripts after author approval. |
| Restricted data scope | Raw EEG recordings, minimally processed EEG files, individual-level clinical source records, and directly linkable participant-level source files. |
| Restricted access route | [Data-access committee or corresponding-author contact, request documents, ethics approval requirement, data-use agreement, review timeline, and allowed reuse scope to be supplied by the responsible institution] |
| Repository platform | [Zenodo, OSF, institutional repository, or other repository to be selected] |
| DOI or accession | [Repository DOI or accession number to be inserted after deposit] |
| Version | [Dataset version, release date, and final commit hash to be inserted after package freeze] |
| Data licence | [Data licence or controlled-access terms to be confirmed; do not assign a licence before ethics and consent scope are checked] |
| Code licence | [Software licence to be confirmed before code release] |
| Citation template | [Authors]. Source data and reproducibility package for "Residual-aware EEG learning for post-stroke proportional recovery". [Repository]. [Version]. [Year]. [DOI/accession]. |

## Recommended Public Deposit Contents

- `ResidualAware_SSL_CNN_Source_Data.xlsx`: curated source-data workbook containing manuscript tables, statistical outputs, EEG metadata summaries, figure manifests, and the data dictionary.
- `results/tables/*.csv`: machine-readable source tables and quality-audit tables.
- `results/statistics/*.csv`: subject-level bootstrap, paired-comparison, permutation, and clinical-incremental analyses.
- `results/figures/nature/`: PNG, SVG, PDF, and TIFF exports for main manuscript figures and the error-subject supplementary figure.
- `results/figures/explainability/mne_wpli_connectivity/`: supplementary MNE WPLI connectivity maps and manifest.
- `docs/*.md`: manuscript drafts, citation map, data-availability audit, methods provenance, reporting checklist, target-journal strategy, and artifact QA.
- `scripts/` and `src/`: analysis, figure, manuscript, workbook, audit, and packaging scripts needed to regenerate the deposited derived outputs.

## Data Dictionary

The machine-readable data dictionary is `results/tables/source_data_dictionary.csv`. It currently describes 446 fields across 37 derived CSV or figure-manifest files. The dictionary includes source file, table role, column name, inferred type, description, unit/scale, access route, and reuse notes.

## Data Not Included In A Public Deposit Without Additional Approval

Raw EEG, minimally processed EEG, individual-level clinical source workbooks, and any directly identifiable or linkable participant records should not be deposited publicly until the ethics approval, consent language, institutional access route, de-identification plan, and data-use agreement are confirmed.

## Suggested Data Availability Text

De-identified subject-level derived tables, locked model predictions, statistical output tables, figure source summaries, figure exports, source-data workbook, source-data dictionary, and analysis code are available in this repository [repository name, DOI/accession, version, and licence to be inserted after deposit]. Raw EEG recordings, minimally processed EEG files, individual-level clinical source records, and directly linkable participant-level source files are not publicly available in this draft because they contain human-participant data and may be subject to institutional review board, consent, and data-use restrictions. Access to restricted raw or minimally processed data should be reviewed by the responsible institution. Qualified researchers may request access through [data-access committee or corresponding-author contact] after ethics approval, project review, and completion of a data-use agreement.

## Reproducibility Notes

- All performance metrics in the manuscript are based on subject-level LOSO predictions.
- Segment-level rows and repeated seed rows are not independent patient observations.
- The residual threshold is derived from the supervised cohort median residual and should not be treated as an externally validated clinical cut-off.
- Explainability outputs are model-dependent and should be interpreted as hypothesis-generating.
- The deposited model outputs support retrospective reproducibility and method evaluation only; they do not support direct clinical deployment or patient-level treatment decisions.

## Required Before Public Release

- Repository DOI or accession number.
- Code licence, data licence, and any controlled-access terms.
- Final commit hash.
- Ethics approval institution and approval number.
- Consent/data-sharing permission and access route for restricted data.
- Confirmation of which derived EEG features may be public.
- Repository record metadata, including creators, ORCID IDs, related article DOI, dataset version, release date, and citation text.
