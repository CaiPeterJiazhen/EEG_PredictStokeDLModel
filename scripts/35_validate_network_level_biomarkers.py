from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.config import load_path_config
from eeg_recovery.visualization.topomap import normalize_channel_name


BETA_BANDS = {"Beta Low", "Beta Medium", "Beta High"}
MOTOR_CHANNELS = {"C3", "C4", "C5", "C6", "FC3", "FC4", "CP3", "CP4", "FC5", "FC6", "CP5", "CP6"}
FRONTAL_CHANNELS = {"FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"}
FRONTO_CENTRAL_CHANNELS = FRONTAL_CHANNELS | {"FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "C5", "C3", "C1", "CZ", "C2", "C4", "C6"}
TEMPORAL_CHANNELS = {"FT7", "FT8", "T7", "T8", "TP7", "TP8"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Network/ROI-level biomarker validation for explainability summaries.")
    parser.add_argument("--config", default="configs/paths.example.yaml")
    parser.add_argument("--n-permutations", type=int, default=5000)
    args = parser.parse_args()

    config = load_path_config(args.config)
    output_root = config.output_root
    explain_root = output_root / "results" / "explainability"

    psd_long = pd.read_csv(explain_root / "psd_attribution_long.csv")
    wpli_long = pd.read_csv(explain_root / "wpli_edge_attribution_long.csv")

    summaries = []
    summaries.extend(_summarize_wpli_networks(wpli_long))
    summaries.extend(_summarize_psd_rois(psd_long))

    rows = [
        _validate_summary(frame, n_permutations=args.n_permutations, random_state=17)
        for frame in summaries
    ]
    result = pd.DataFrame(rows)
    result["spearman_signed_distance_fdr_p"] = _bh_fdr(
        result["spearman_signed_distance_p"].to_numpy(dtype=float)
    )
    result["group_permutation_fdr_p"] = _bh_fdr(
        result["label_group_permutation_p"].to_numpy(dtype=float)
    )

    output_path = explain_root / "network_level_biomarker_validation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    _write_doc(output_root / "docs" / "network_level_biomarker_validation.md", result)
    print(f"Wrote {output_path}")


def _summarize_wpli_networks(frame: pd.DataFrame) -> list[pd.DataFrame]:
    beta = frame[frame["band"].isin(BETA_BANDS)].copy()
    beta["channel_i_norm"] = beta["channel_i"].map(normalize_channel_name)
    beta["channel_j_norm"] = beta["channel_j"].map(normalize_channel_name)
    beta["network_group_norm"] = [
        _edge_network_group(left, right)
        for left, right in zip(beta["channel_i_norm"], beta["channel_j_norm"], strict=True)
    ]
    if "interhemispheric" not in beta.columns:
        beta["interhemispheric"] = [
            _is_interhemispheric(left, right)
            for left, right in zip(beta["channel_i_norm"], beta["channel_j_norm"], strict=True)
        ]
    targets = {
        "wpli_beta_frontal_central": beta["network_group_norm"].eq("central|frontal"),
        "wpli_beta_frontal_parietal": beta["network_group_norm"].eq("frontal|parietal"),
        "wpli_beta_central_parietal": beta["network_group_norm"].eq("central|parietal"),
        "wpli_beta_interhemispheric": beta["interhemispheric"].astype(bool),
        "wpli_beta_motor_adjacent": beta["channel_i_norm"].isin(MOTOR_CHANNELS)
        | beta["channel_j_norm"].isin(MOTOR_CHANNELS),
    }
    return [
        _patient_summary(beta.loc[mask], feature_family="WPLI", summary_name=name)
        for name, mask in targets.items()
        if beta.loc[mask].shape[0] > 0
    ]


def _summarize_psd_rois(frame: pd.DataFrame) -> list[pd.DataFrame]:
    data = frame.copy()
    data["channel_norm"] = data["channel"].map(normalize_channel_name)
    targets = {
        "psd_frontal_delta": data["band"].eq("Delta") & data["channel_norm"].isin(FRONTAL_CHANNELS),
        "psd_fronto_central_beta": data["band"].isin(BETA_BANDS)
        & data["channel_norm"].isin(FRONTO_CENTRAL_CHANNELS),
        "psd_temporal_delta": data["band"].eq("Delta") & data["channel_norm"].isin(TEMPORAL_CHANNELS),
        "psd_motor_beta_medium": data["band"].eq("Beta Medium")
        & data["channel_norm"].isin(MOTOR_CHANNELS),
    }
    return [
        _patient_summary(data.loc[mask], feature_family="PSD", summary_name=name)
        for name, mask in targets.items()
        if data.loc[mask].shape[0] > 0
    ]


def _patient_summary(frame: pd.DataFrame, *, feature_family: str, summary_name: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["subject_id", "y_true"], as_index=False)
        .agg(
            residual=("residual", "mean"),
            signed_distance=("signed_distance", "mean"),
            mean_abs_value=("abs_attribution", "mean"),
            mean_signed_value=("signed_attribution", "mean"),
        )
        .assign(feature_family=feature_family, summary_name=summary_name)
    )
    return grouped


def _validate_summary(frame: pd.DataFrame, *, n_permutations: int, random_state: int) -> dict[str, float | str | int]:
    value = frame["mean_abs_value"].to_numpy(dtype=float)
    signed_value = frame["mean_signed_value"].to_numpy(dtype=float)
    residual = frame["residual"].to_numpy(dtype=float)
    signed_distance = frame["signed_distance"].to_numpy(dtype=float)
    labels = frame["y_true"].to_numpy(dtype=int)

    corr_residual = spearmanr(value, residual)
    corr_distance = spearmanr(value, signed_distance)
    corr_signed_distance = spearmanr(signed_value, signed_distance)
    positive = value[labels == 1]
    negative = value[labels == 0]
    perm_p = _permutation_pvalue(value, labels, n_permutations=n_permutations, random_state=random_state)
    direction = "positive-class association" if positive.mean() > negative.mean() else "poor-recovery association"
    return {
        "feature_family": frame["feature_family"].iloc[0],
        "summary_name": frame["summary_name"].iloc[0],
        "n_subjects": int(frame["subject_id"].nunique()),
        "mean_abs_value": float(np.mean(value)),
        "mean_signed_value": float(np.mean(signed_value)),
        "positive_mean_value": float(positive.mean()),
        "negative_mean_value": float(negative.mean()),
        "effect_direction": direction,
        "spearman_residual_r": float(corr_residual.statistic),
        "spearman_residual_p": float(corr_residual.pvalue),
        "spearman_signed_distance_r": float(corr_distance.statistic),
        "spearman_signed_distance_p": float(corr_distance.pvalue),
        "spearman_signed_attribution_distance_r": float(corr_signed_distance.statistic),
        "spearman_signed_attribution_distance_p": float(corr_signed_distance.pvalue),
        "label_group_permutation_p": float(perm_p),
    }


def _permutation_pvalue(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int,
    random_state: int,
) -> float:
    rng = np.random.default_rng(random_state)
    observed = abs(values[labels == 1].mean() - values[labels == 0].mean())
    count = 1
    for _ in range(n_permutations):
        permuted = rng.permutation(labels)
        diff = abs(values[permuted == 1].mean() - values[permuted == 0].mean())
        if diff >= observed:
            count += 1
    return count / (n_permutations + 1)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    n = len(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        value = min(prev, ranked[i] * n / (i + 1))
        adjusted[i] = value
        prev = value
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def _normalize_network_group(value: str) -> str:
    parts = sorted(item.strip().lower() for item in value.split("|") if item.strip())
    return "|".join(parts)


def _edge_network_group(left: str, right: str) -> str:
    return "|".join(sorted({_region(left), _region(right)}))


def _region(channel: str) -> str:
    if channel.startswith(("FP", "AF", "F")) and not channel.startswith(("FC", "FT")):
        return "frontal"
    if channel.startswith(("FC", "C")):
        return "central"
    if channel.startswith(("P", "PO", "CP")):
        return "parietal"
    if channel.startswith(("T", "FT", "TP")):
        return "temporal"
    if channel.startswith(("O", "CB")):
        return "occipital"
    return "other"


def _is_interhemispheric(left: str, right: str) -> bool:
    return _hemisphere(left) != _hemisphere(right) and "midline" not in {_hemisphere(left), _hemisphere(right)}


def _hemisphere(channel: str) -> str:
    digits = [char for char in channel if char.isdigit()]
    if not digits:
        return "midline"
    number = int(digits[-1])
    return "left" if number % 2 == 1 else "right"


def _write_doc(path: Path, result: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = result.sort_values("spearman_signed_distance_fdr_p").head(6)
    lines = [
        "# Network-Level Biomarker Validation",
        "",
        "This analysis validates network/ROI-level attribution summaries from the locked final model. It does not promote any single WPLI edge as a confirmed biomarker.",
        "",
        "Patient-level summaries average the 10 seed/fold attribution rows within each subject before correlation tests, so seeds are not treated as independent clinical samples.",
        "",
        "## Strongest Network/ROI Associations",
        "",
        "| feature family | summary | rho vs signed distance | p | FDR q | direction |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['feature_family']} | {row['summary_name']} | "
            f"{row['spearman_signed_distance_r']:.3f} | {row['spearman_signed_distance_p']:.4f} | "
            f"{row['spearman_signed_distance_fdr_p']:.4f} | {row['effect_direction']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These results are manuscript support for network-level patterns. They should be described as exploratory and hypothesis-generating because the cohort has 19 patients and lacks external validation.",
            "",
            "Reported groups include WPLI beta frontal-central, frontal-parietal, central-parietal, interhemispheric, motor-adjacent summaries and PSD frontal delta, fronto-central beta, temporal delta, and motor beta-medium summaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
