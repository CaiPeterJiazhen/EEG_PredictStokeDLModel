from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def plot_baseline_health_distance_group_boxplot(
    distances: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot baseline distance-to-health grouped by recovery group."""

    output = _prepare_output(output_path)
    required = {"group", "timepoint", "level", "recovery_group", "distance_to_health"}
    if distances.empty or not required.issubset(distances.columns):
        return _placeholder(output, "Baseline health distance by recovery group\nNo plottable rows")
    frame = distances[
        distances["group"].eq("patient")
        & distances["timepoint"].eq("baseline")
        & distances["level"].eq("overall")
    ].copy()
    if frame.empty:
        return _placeholder(output, "Baseline health distance by recovery group\nNo plottable rows")
    labels = []
    data = []
    for group_name, group_frame in frame.groupby("recovery_group"):
        labels.append(str(group_name))
        data.append(group_frame["distance_to_health"].astype(float).to_numpy())
    plt.figure(figsize=(7.2, 4.8))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("Baseline RMS z-distance to healthy")
    plt.title("Baseline PSD/wPLI health distance: group-level association")
    plt.figtext(
        0.5,
        0.01,
        "Group-level association only; not an individual prediction tool.",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def plot_health_distance_slopeplot(normalization: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot paired pre/post health-distance trajectories."""

    output = _prepare_output(output_path)
    if normalization.empty or not {"level", "D_pre", "D_post"}.issubset(normalization.columns):
        return _placeholder(output, "D_pre to D_post slopeplot\nNo paired rows")
    frame = normalization[normalization["level"].eq("overall")].copy()
    if frame.empty:
        return _placeholder(output, "D_pre to D_post slopeplot\nNo paired rows")
    plt.figure(figsize=(7.2, 5.2))
    color_map = {"proportional_recovery": "#1f77b4", "poor_recovery": "#d62728"}
    for _, row in frame.iterrows():
        color = color_map.get(str(row.get("recovery_group", "")), "0.4")
        plt.plot([0, 1], [row["D_pre"], row["D_post"]], marker="o", alpha=0.55, color=color)
    plt.xticks([0, 1], ["Baseline", "Post14"])
    plt.ylabel("RMS z-distance to healthy")
    plt.title("Patient PSD/wPLI distance to healthy template")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def plot_normalization_scatter(
    normalization: pd.DataFrame,
    output_path: str | Path,
    *,
    outcome_column: str,
    y_label: str,
) -> Path:
    """Scatter NI against a functional outcome with Spearman annotation."""

    output = _prepare_output(output_path)
    if normalization.empty or not {"level", "NI", outcome_column}.issubset(normalization.columns):
        return _placeholder(output, f"NI vs {y_label}\nNo plottable rows")
    frame = normalization[
        normalization["level"].eq("overall")
        & normalization["NI"].notna()
        & normalization[outcome_column].notna()
    ].copy()
    if frame.empty:
        return _placeholder(output, f"NI vs {y_label}\nNo plottable rows")
    x = frame["NI"].astype(float).to_numpy()
    y = frame[outcome_column].astype(float).to_numpy()
    rho, p_value = (np.nan, np.nan) if len(frame) < 3 else spearmanr(x, y)
    plt.figure(figsize=(6.4, 5.0))
    plt.scatter(x, y, c="#2f6f8f", edgecolor="white", linewidth=0.8, s=54)
    if len(frame) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
        plt.plot(xs, slope * xs + intercept, color="#444444", linewidth=1.2)
    plt.xlabel("Normalization index")
    plt.ylabel(y_label)
    plt.title(f"Normalization vs {y_label}")
    plt.text(
        0.03,
        0.97,
        f"Spearman rho={rho:.3f}, p={p_value:.3g}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def plot_psd_topomap_normalization(feature_normalization: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot PSD channel-band normalization as a compact heatmap."""

    output = _prepare_output(output_path)
    if feature_normalization.empty or "modality" not in feature_normalization.columns:
        return _placeholder(output, "PSD topomap normalization\nNo PSD feature rows")
    frame = feature_normalization[feature_normalization["modality"].eq("psd")].copy()
    if frame.empty or not {"channel", "band", "normalization_change"}.issubset(frame.columns):
        return _placeholder(output, "PSD topomap normalization\nNo PSD feature rows")
    pivot = frame.groupby(["channel", "band"], as_index=False)["normalization_change"].mean()
    matrix = pivot.pivot(index="channel", columns="band", values="normalization_change").fillna(0.0)
    plt.figure(figsize=(8.0, max(4.0, 0.24 * len(matrix))))
    plt.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="|z_pre| - |z_post|")
    plt.yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    plt.title("PSD normalization by channel and band")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def plot_wpli_connectome_normalization(feature_normalization: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot normalized wPLI edges in a simple circular connectome."""

    output = _prepare_output(output_path)
    if feature_normalization.empty or "modality" not in feature_normalization.columns:
        return _placeholder(output, "wPLI connectome normalization\nNo wPLI edge rows")
    frame = feature_normalization[feature_normalization["modality"].eq("wpli")].copy()
    if frame.empty or not {"channel_i", "channel_j", "normalization_change"}.issubset(frame.columns):
        return _placeholder(output, "wPLI connectome normalization\nNo wPLI edge rows")
    beta = frame[frame.get("band", "").astype(str).eq("Beta High")]
    plot_frame = beta if not beta.empty else frame
    plot_frame = plot_frame.sort_values("normalization_change", ascending=False).head(40)
    channels = sorted(set(plot_frame["channel_i"].astype(str)) | set(plot_frame["channel_j"].astype(str)))
    angles = np.linspace(0, 2 * np.pi, len(channels), endpoint=False)
    coords = {channel: (np.cos(angle), np.sin(angle)) for channel, angle in zip(channels, angles)}
    max_abs = max(float(plot_frame["normalization_change"].abs().max()), 1e-12)
    plt.figure(figsize=(7.0, 7.0))
    ax = plt.gca()
    for channel, (x, y) in coords.items():
        ax.scatter([x], [y], color="black", s=18)
        ax.text(x * 1.10, y * 1.10, channel, fontsize=7, ha="center", va="center")
    for _, row in plot_frame.iterrows():
        x1, y1 = coords[str(row["channel_i"])]
        x2, y2 = coords[str(row["channel_j"])]
        value = float(row["normalization_change"])
        color = "#b2182b" if value >= 0 else "#2166ac"
        width = 0.5 + 4.0 * abs(value) / max_abs
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.55)
    ax.set_title("wPLI connectome normalization")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def plot_feature_evidence_overlap(overlap: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot overlap counts between model attribution and evidence sets."""

    output = _prepare_output(output_path)
    if overlap.empty or "overlap_count" not in overlap.columns:
        return _placeholder(output, "Feature evidence overlap\nNo overlap rows")
    frame = overlap.sort_values("overlap_count", ascending=False)
    plt.figure(figsize=(8.0, 4.8))
    plt.bar(frame["evidence_set"].astype(str), frame["overlap_count"].astype(float), color="#4c78a8")
    plt.ylabel("Overlap count")
    plt.title("Model high-attribution features overlapping statistical evidence")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def make_biomarker_validity_summary_figure(image_paths: Iterable[str | Path], output_path: str | Path) -> Path:
    """Combine generated biomarker-validity panels into a single summary figure."""

    output = _prepare_output(output_path)
    paths = [Path(path) for path in image_paths if Path(path).exists()]
    if not paths:
        return _placeholder(
            output,
            "PSD/wPLI feature validity validation\nNo component figures available",
        )
    n = len(paths)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, max(5, rows * 4.2)))
    axes_array = np.asarray(axes).reshape(-1)
    for ax, path in zip(axes_array, paths):
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.set_title(path.stem.replace("_", " "), fontsize=9)
        ax.axis("off")
    for ax in axes_array[len(paths) :]:
        ax.axis("off")
    fig.suptitle("PSD/wPLI feature validity validation: abnormalities, normalization, and model explanation consistency")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _placeholder(path: Path, text: str) -> Path:
    plt.figure(figsize=(7.0, 4.2))
    plt.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path
