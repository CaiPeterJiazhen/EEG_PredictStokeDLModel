# Explainability Paper Validation Summary

Updated: 2026-06-01

`scripts/31_explain_residual_aware_ssl_cnn.py` was updated so prediction CSVs are resolved from `path_config.output_root`, the residual threshold is configurable via `--residual-threshold`, and sanity checks default to all processed seed/fold samples with `--max-sanity-samples 0`.

Existing explainability CSVs were normalized into the requested paper-facing outputs:

- `results/explainability/full_sanity_check_summary.csv`
- `results/explainability/ig_smoothgrad_occlusion_consistency.csv`
- `results/explainability/biomarker_group_difference_validation.csv`
- `results/explainability/error_subject_explainability_summary.csv`
- `results/figures/explainability/paper_*.png`

The current alias files are based on existing explainability outputs. A full regeneration over all 10 seeds x 19 folds should be run locally with checkpoints/GPU before making strong biomarker claims:

```powershell
python -B scripts\31_explain_residual_aware_ssl_cnn.py --config configs\paths.example.yaml --device cuda --residual-threshold 1.5 --max-sanity-samples 0
```

If standard 1005 coordinates are unavailable or incomplete, report channel barplots/connectomes rather than claiming standard scalp topomap validation.
