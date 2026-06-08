from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config


METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score", "sensitivity", "specificity")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect core 10-seed ablation matrix without launching long retraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    rows = collect_core_ablation_rows(output_root)

    metrics_dir = output_root / "results" / "metrics"
    tables_dir = output_root / "results" / "tables"
    docs_dir = output_root / "docs"
    for directory in (metrics_dir, tables_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows.to_csv(metrics_dir / "core_ablation_10seed_summary.csv", index=False)
    (tables_dir / "core_ablation_10seed_summary.md").write_text(_to_markdown(rows), encoding="utf-8")
    write_core_ablation_doc(docs_dir / "core_ablation_results.md", rows)
    print(f"Wrote {metrics_dir / 'core_ablation_10seed_summary.csv'}")


def collect_core_ablation_rows(output_root: Path) -> pd.DataFrame:
    specs = [
        {
            "contrast": "a_ml_psd_wpli_baseline",
            "model_key": "ML PSD+WPLI Logistic L1",
            "source": output_root / "results" / "tables" / "paper_locked_model_performance.csv",
            "row_filter": ("model_name", "ML_EEG_updated_no_selector_logistic_l1"),
            "command": "python -B scripts/04_train_ml_baselines.py --feature-set psd-fc-wpli --disable-selector --output-tag updated_sub05_sub28_psdfcwpli_no_selector",
            "interpretation": "Traditional PSD+WPLI baseline; no CNN, no SSL, no residual-aware objectives.",
        },
        {
            "contrast": "b_no_ssl_cnn_same_arch_10seed",
            "model_key": "no-SSL CNN same architecture",
            "source": output_root / "results" / "metrics" / "updated_sub05_sub28_10seed_no_ssl_barlow_cnn_summary.csv",
            "row_filter": ("method", "no_ssl_schemeA"),
            "command": "for seed in 0 1 2 3 4 5 6 7 8 13: python -B scripts/05_train_supervised_loso.py --architecture multimodal --feature-kind psd-fc-wpli --fusion gated --encoder cnn --seed <seed>",
            "interpretation": "CNN architecture contribution over traditional ML, using the same paired 10-seed set as the Barlow SSL no-residual-head comparison.",
        },
        {
            "contrast": "c_patient_barlow_ssl_no_residual_heads",
            "model_key": "Patient-level Barlow SSL without residual-aware heads",
            "source": output_root / "results" / "metrics" / "updated_sub05_sub28_10seed_no_ssl_barlow_cnn_summary.csv",
            "row_filter": ("method", "barlow_ssl"),
            "command": "python -B scripts/29_train_patient_barlow_stabilized.py --seeds 0 1 2 3 4 5 6 7 8 13",
            "interpretation": "Patient-level SSL contribution before residual-aware auxiliary training.",
        },
        {
            "contrast": "d_no_ssl_cnn_residual_heads",
            "model_key": "no-SSL CNN + residual-aware auxiliary heads",
            "source": output_root / "results" / "metrics" / "no_ssl_residualaware_highrank_swa_clsalpha1_10seed_summary.csv",
            "command": "python -B scripts/30_train_residual_aware_patient_barlow.py --no-ssl --model-group no_ssl_residualaware_highrank_swa_clsalpha1 --seeds 0 1 2 3 4 5 7 13 21 42",
            "interpretation": "Residual-aware objective contribution without SSL pretraining.",
        },
        {
            "contrast": "e_patient_barlow_ssl_residual_heads",
            "model_key": "Patient-level Barlow SSL + residual-aware auxiliary heads",
            "source": output_root / "results" / "metrics" / "patient_barlow_residualaware_highrank_swa_clsalpha1_10seed_summary.csv",
            "command": "python -B scripts/30_train_residual_aware_patient_barlow.py --model-group residualaware_highrank_swa_clsalpha1 --seeds 0 1 2 3 4 5 7 13 21 42",
            "interpretation": "Final combined SSL plus residual-aware training contribution.",
        },
        {
            "contrast": "f_final_without_swa",
            "model_key": "Residual-aware SSL-CNN without SWA",
            "source": output_root / "results" / "metrics" / "patient_barlow_residualaware_highrank_10seed_summary.csv",
            "command": "python -B scripts/30_train_residual_aware_patient_barlow.py --disable-swa --model-group residualaware_highrank --seeds 0 1 2 3 4 5 7 13 21 42",
            "interpretation": "SWA contribution.",
        },
        {
            "contrast": "g_without_residual_regression_loss",
            "model_key": "Residual-aware SSL-CNN without residual regression loss",
            "source": output_root / "results" / "metrics" / "patient_barlow_residualaware_variant_10seed_comparison.csv",
            "row_filter_contains": ("model_group", "no_reg"),
            "command": "python -B scripts/30_train_residual_aware_patient_barlow.py --residual-regression-weight 0 --seeds 0 1 2 3 4 5 7 13 21 42",
            "interpretation": "Residual regression auxiliary loss contribution.",
        },
        {
            "contrast": "h_without_ranking_loss",
            "model_key": "Residual-aware SSL-CNN without ranking loss",
            "source": output_root / "results" / "metrics" / "patient_barlow_residualaware_variant_10seed_comparison.csv",
            "row_filter_contains": ("model_group", "no_rank"),
            "command": "python -B scripts/30_train_residual_aware_patient_barlow.py --ranking-weight 0 --seeds 0 1 2 3 4 5 7 13 21 42",
            "interpretation": "Pairwise residual ranking auxiliary loss contribution.",
        },
        {
            "contrast": "i_without_soft_label_loss",
            "model_key": "Residual-aware SSL-CNN without soft-label loss",
            "source": output_root / "results" / "metrics" / "patient_barlow_residualaware_variant_10seed_comparison.csv",
            "row_filter_contains": ("model_group", "no_soft"),
            "command": "python -B scripts/30_train_residual_aware_patient_barlow.py --soft-label-weight 0 --seeds 0 1 2 3 4 5 7 13 21 42",
            "interpretation": "Soft-label auxiliary loss contribution.",
        },
        {
            "contrast": "j_classification_head_vs_residual_fusion",
            "model_key": "Classification-head inference vs residual-probability fusion",
            "source": output_root / "results" / "metrics" / "patient_barlow_residualaware_variant_10seed_comparison.csv",
            "row_filter_contains": ("model_group", "fusion"),
            "command": "python -B scripts/30_train_residual_aware_patient_barlow.py --compare-inference-heads --seeds 0 1 2 3 4 5 7 13 21 42",
            "interpretation": "Inference-head choice contribution.",
        },
    ]
    rows = []
    for spec in specs:
        rows.extend(_rows_from_spec(spec))
    return pd.DataFrame(rows)


def _rows_from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(spec["source"])
    base = {
        "contrast": spec["contrast"],
        "model_key": spec["model_key"],
        "source_file": path.name,
        "interpretation": spec["interpretation"],
        "reproduce_command": spec["command"],
    }
    if not path.exists():
        return [{**base, "availability": "missing", "row_type": "missing"}]
    frame = pd.read_csv(path)
    if "row_filter" in spec:
        column, value = spec["row_filter"]
        if column not in frame.columns:
            return [{**base, "availability": "missing_row", "row_type": "missing"}]
        frame = frame[frame[column].astype(str) == str(value)].copy()
    if "row_filter_contains" in spec:
        column, token = spec["row_filter_contains"]
        if column not in frame.columns:
            return [{**base, "availability": "missing_row", "row_type": "missing"}]
        frame = frame[frame[column].astype(str).str.contains(str(token), case=False, regex=False)].copy()
    if frame.empty:
        return [{**base, "availability": "missing_row", "row_type": "missing"}]

    row_type_column = "row_type" if "row_type" in frame.columns else None
    rows = []
    if row_type_column:
        selected = frame[frame[row_type_column].isin(["mean", "std", "min", "max", "seedmean10", "ensemble10"])]
        if selected.empty:
            selected = frame.head(1).assign(row_type="reported")
        for _, row in selected.iterrows():
            rows.append(_metric_row(base, row, str(row.get(row_type_column, "reported")), availability="available"))
    else:
        rows.append(_metric_row(base, frame.iloc[0], "reported", availability="available"))
    return rows


def _metric_row(base: dict[str, Any], row: pd.Series, row_type: str, *, availability: str) -> dict[str, Any]:
    output = {**base, "availability": availability, "row_type": row_type}
    for metric in METRICS:
        output[metric] = row.get(metric, pd.NA)
    return output


def write_core_ablation_doc(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    available = rows[rows["availability"] == "available"]
    missing = rows[rows["availability"] != "available"]
    lines = [
        "# Core Ablation Results",
        "",
        "This script only aggregates existing LOSO/10-seed outputs. Missing ablations are listed with reproducibility commands and are not silently retrained.",
        "",
        "## Available Summary Rows",
        "",
        _to_markdown(available),
        "",
        "## Missing Or Not-yet-run Ablations",
        "",
        _to_markdown(missing[["contrast", "model_key", "availability", "reproduce_command"]]) if not missing.empty else "_None._",
        "",
        "## Interpretation Guide",
        "",
        "- CNN architecture contribution is estimated by comparing the updated ML PSD+WPLI baseline with no-SSL CNN.",
        "- SSL contribution is estimated by comparing no-SSL CNN against Patient-level Barlow SSL without residual-aware heads.",
        "- Residual-aware auxiliary training is isolated by no-SSL + residual heads and SSL + residual heads rows.",
        "- SWA and inference-head choice are support ablations; improvements there should not be described as SSL biology.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
