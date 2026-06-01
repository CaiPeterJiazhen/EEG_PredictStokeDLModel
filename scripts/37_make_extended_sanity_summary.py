from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CHECK_ORDER = [
    "classifier_only_randomization",
    "classifier_final_projection_randomization",
    "full_psd_encoder_randomization",
    "full_wpli_encoder_randomization",
    "input_permutation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize extended explainability sanity checks.")
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    explain_root = output_root / "results" / "explainability"
    source = pd.read_csv(explain_root / "sanity_check_summary.csv")

    normalized = _normalize_sanity_rows(source)
    rows = []
    for check in CHECK_ORDER:
        subset = normalized[normalized["sanity_check"] == check]
        if subset.empty:
            rows.append(
                {
                    "sanity_check": check,
                    "status": "implemented_for_rerun_not_recomputed_in_committed_summary",
                    "n_samples": 0,
                    "psd_eo_signed_spearman_mean": np.nan,
                    "wpli_eo_signed_spearman_mean": np.nan,
                    "psd_eo_abs_spearman_mean": np.nan,
                    "wpli_eo_abs_spearman_mean": np.nan,
                    "note": "scripts/31_explain_residual_aware_ssl_cnn.py now implements this check; full recomputation was not run to avoid overwriting locked attribution artifacts.",
                }
            )
            continue
        rows.append(
            {
                "sanity_check": check,
                "status": "available",
                "n_samples": int(len(subset)),
                "psd_eo_signed_spearman_mean": float(subset["psd_eo_signed_correlation"].mean()),
                "wpli_eo_signed_spearman_mean": float(subset["wpli_eo_signed_correlation"].mean()),
                "psd_eo_abs_spearman_mean": float(subset["psd_eo_abs_correlation"].mean()),
                "wpli_eo_abs_spearman_mean": float(subset["wpli_eo_abs_correlation"].mean()),
                "note": subset["note"].iloc[0],
            }
        )

    output = explain_root / "sanity_check_extended_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    _write_doc(output_root / "docs" / "explainability_sanity_check_extension.md", pd.DataFrame(rows))
    print(f"Wrote {output}")


def _normalize_sanity_rows(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    if "sanity_check" not in frame.columns:
        raise ValueError("sanity_check_summary.csv is missing sanity_check column")
    frame["sanity_check"] = frame["sanity_check"].replace(
        {"classifier_randomization": "classifier_only_randomization"}
    )
    if "psd_eo_signed_correlation" not in frame.columns:
        frame["psd_eo_signed_correlation"] = frame["psd_eo_correlation"]
        frame["wpli_eo_signed_correlation"] = frame["wpli_eo_correlation"]
        frame["psd_eo_abs_correlation"] = frame["psd_eo_correlation"].abs()
        frame["wpli_eo_abs_correlation"] = frame["wpli_eo_correlation"].abs()
        frame["note"] = "Legacy summary: signed correlations available; abs correlations are absolute-value proxies and should be interpreted conservatively."
    else:
        frame["note"] = "Extended summary with signed and absolute attribution correlations."
    return frame


def _write_doc(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Explainability Sanity Check Extension",
        "",
        "The explainability script now implements classifier-only randomization, classifier plus final projection randomization, full PSD encoder randomization, full WPLI encoder randomization, and input permutation checks.",
        "",
        "The committed extended summary is conservative: it reuses the existing locked sanity CSV for available rows and marks newly implemented checks that require a full attribution rerun. A full rerun was not launched here because it would overwrite the locked attribution summary artifacts.",
        "",
        "| sanity check | status | n | PSD signed rho | WPLI signed rho | PSD abs rho | WPLI abs rho |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['sanity_check']} | {row['status']} | {int(row['n_samples'])} | "
            f"{_fmt(row['psd_eo_signed_spearman_mean'])} | {_fmt(row['wpli_eo_signed_spearman_mean'])} | "
            f"{_fmt(row['psd_eo_abs_spearman_mean'])} | {_fmt(row['wpli_eo_abs_spearman_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation remains conservative. The prior classifier-randomization result was mixed, especially for absolute attribution structure, so final biomarker claims should rely on convergent evidence from stability, occlusion, and network-level validation rather than a single attribution table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
