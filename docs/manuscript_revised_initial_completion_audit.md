# Revised Initial Manuscript Completion Audit

Audit date: 2026-06-02

Manuscript audited: `docs/manuscript_revised_initial_draft.md`

## Requirement Status

| Requirement | Evidence | Status |
|:--|:--|:--|
| Follow the user-approved revised outline | `docs/manuscript_outline_direction_revised_zh.md`; manuscript sections match Introduction, Materials and Methods, Results, Discussion, Conclusion, Data/Code Availability, Tables and Figure Legends | Complete for initial draft |
| Use Nature-style writing structure | Manuscript has evidence-led abstract, 898-word Introduction, concise Methods subsections, claim-first Results and bounded Discussion | Complete for initial draft |
| Nature-style polishing | Draft avoids clinical overclaiming, keeps SSL claims bounded, uses conservative interpretation for small n and explainability | Complete for initial draft |
| Complete citation coverage | 25 references in `docs/manuscript_revised_initial_draft.md`; all are cited; no missing citation targets; metadata audit `results/tables/reference_metadata_audit.csv` has 25 PASS rows | Complete for initial draft |
| Data/statistical analysis from project data | Cohort counts, Table 1, ML results, ten-seed deep-model summaries, ablations, locked confusion matrix and explainability features come from `results/tables`, `results/metrics`, `results/statistics`, `results/predictions` and `results/explainability` | Complete for initial draft |
| Nature-ready Data Availability | Data Availability section distinguishes derived data, raw/minimally processed EEG, clinical source records, controlled access route and missing repository fields | Complete as draft; repository fields pending |
| Complete main figures | Figure 1-6 generated under `results/figures/revised_initial` with PNG, SVG, PDF and TIFF exports | Complete |
| MNE topomap/connectivity | Figure 6 uses existing MNE-rendered topomap and connectivity panels; MNE tests passed | Complete |
| Main tables | Table 1-5 are embedded in the manuscript, with source files listed | Complete |
| Introduction length | Rechecked at 898 English words | Complete |
| No Introduction figure | No figure callout or image appears inside Introduction | Complete |
| Separate methods figures | Figure 1 in Study Design, Figure 2 in Self-supervised Learning, Figure 3 in CNN and Residual-Aware Learning | Complete |
| Methods split into ML, SSL, CNN/residual-aware | Three separate short Methods subsections are present | Complete |
| Label definition in Methods with formulas | Expected improvement, observed improvement, residual, median threshold, binary label and signed distance are defined in Methods | Complete |
| Explainability Methods has methods only | Explainability methods list IG/SmoothGrad, occlusion, topomap, connectivity and stability without reporting results | Complete |
| Traditional ML Results include multiple models | Logistic L1, Logistic L2 and SVM RBF + SelectK=100 are reported in Results and Table 2 | Complete |
| Final model section includes confusion matrix | Figure 4 and Results report TP=10, FN=0, FP=3, TN=6 | Complete |
| Avoid disallowed mainline analyses | Search confirmed absence of `clinical-only`, `clinical + EEG`, `seedmean10`, `incremental` and `3.9` | Complete |

## Verification Commands

```text
python -m py_compile scripts\45_make_mne_explainability_topomaps.py scripts\46_make_mne_wpli_connectivity.py scripts\83_make_revised_initial_manuscript_figures.py
python -m pytest tests\test_mne_topomap_coordinates.py tests\test_mne_wpli_connectivity.py -q
```

Observed result: MNE tests passed (`2 passed`), and the three plotting scripts compiled.

## Remaining Author Fields

These are intentionally retained as formal placeholders because the user requested placeholder wording where details are not yet available:

| Field | Manuscript placeholder |
|:--|:--|
| Ethics approval | `[IRB name and approval number to be inserted]` |
| Assessor qualification and blinding | `[assessor qualification and blinding status to be inserted]` |
| Repository DOI/licence/version | `Repository DOI, licence, access committee name and final dataset version remain to be inserted.` |
| Code repository/version | Public repository or linked code archive, exact commit hash or version tag to be inserted before final submission |

## Current Deliverables

| Artifact | Path |
|:--|:--|
| Revised initial manuscript | `docs/manuscript_revised_initial_draft.md` |
| Revised outline | `docs/manuscript_outline_direction_revised_zh.md` |
| Main figure directory | `results/figures/revised_initial` |
| Figure manifest | `results/figures/revised_initial/figure_manifest.csv` |
| Revised figure-generation script | `scripts/83_make_revised_initial_manuscript_figures.py` |
