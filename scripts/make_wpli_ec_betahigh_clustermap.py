from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


DEFAULT_CAPTION = (
    "EC 状态下 Beta High 频段 wPLI 聚类热图显示，卒中患者治疗前功能连接模式"
    "相较健康模板存在整体偏离。该结果提示 wPLI 能够刻画卒中后脑网络异常，"
    "为后续深度学习模型的功能连接输入提供了特征有效性依据。"
)


@dataclass(frozen=True)
class WPLIBandMean:
    matrix: np.ndarray
    channels: tuple[str, ...]
    n_subjects: int
    band: str
    state: str
    group_label: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw EC Beta High wPLI stroke-minus-healthy clustered difference heatmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features-dir",
        default=Path("data") / "features_all_stages",
        type=Path,
        help="Feature root containing feature_manifest.csv and fc/.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("results") / "biomarker_validity" / "figures",
        type=Path,
    )
    parser.add_argument("--state", default="EC")
    parser.add_argument("--band", default="Beta High")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--skip-supplementary",
        action="store_true",
        help="Only write the main EC Beta High clustermap.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    manifest_path = args.features_dir / "feature_manifest.csv"
    healthy = load_mean_wpli_matrix(
        manifest_path=manifest_path,
        group="health",
        stage="health",
        state=args.state,
        band=args.band,
        group_label="Healthy reference",
    )
    stroke = load_mean_wpli_matrix(
        manifest_path=manifest_path,
        group="patient",
        stage="基线",
        state=args.state,
        band=args.band,
        group_label="Stroke baseline",
    )
    if healthy.channels != stroke.channels:
        raise ValueError("Healthy and stroke wPLI matrices use different channel orders.")

    difference = stroke.matrix - healthy.matrix
    np.fill_diagonal(difference, 0.0)
    order = cluster_order(difference)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_stem = args.out_dir / "wpli_ec_betahigh_clustermap"
    draw_clustermap(
        difference=difference,
        channels=healthy.channels,
        order=order,
        output_stem=main_stem,
        dpi=args.dpi,
        title="EC Beta High wPLI Difference Matrix",
        subtitle=(
            f"Stroke baseline vs Healthy reference "
            f"(n={stroke.n_subjects} vs {healthy.n_subjects})"
        ),
    )
    (args.out_dir / "wpli_ec_betahigh_clustermap_caption.md").write_text(
        DEFAULT_CAPTION + "\n",
        encoding="utf-8",
    )

    if not args.skip_supplementary:
        draw_supplementary_six_band_heatmap(
            manifest_path=manifest_path,
            channels=healthy.channels,
            order=order,
            state=args.state,
            out_dir=args.out_dir,
            dpi=args.dpi,
        )


def configure_matplotlib() -> None:
    """Use restrained publication-style matplotlib defaults."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 7,
        }
    )


def load_mean_wpli_matrix(
    *,
    manifest_path: Path,
    group: str,
    stage: str,
    state: str,
    band: str,
    group_label: str,
) -> WPLIBandMean:
    """Load one group/stage/state/band and return its mean symmetric matrix."""

    manifest = pd.read_csv(manifest_path)
    required = {"group", "stage", "state", "modality", "feature_path", "status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Feature manifest is missing required columns: {sorted(missing)}")
    selected = manifest[
        manifest["group"].astype(str).str.strip().eq(group)
        & manifest["stage"].astype(str).str.strip().eq(stage)
        & manifest["state"].astype(str).str.strip().eq(state)
        & manifest["modality"].astype(str).str.strip().eq("wpli")
        & manifest["status"].astype(str).str.strip().eq("written")
    ].copy()
    if selected.empty:
        raise ValueError(f"No wPLI rows found for group={group}, stage={stage}, state={state}.")

    matrices: list[np.ndarray] = []
    channel_order: tuple[str, ...] | None = None
    resolved_band = band
    for path_text in selected["feature_path"]:
        path = Path(str(path_text).strip())
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            wpli = np.asarray(payload["wpli"], dtype=float)
            band_names = tuple(str(item) for item in payload["band_names"].tolist())
            band_index = _band_index(band_names, band)
            resolved_band = band_names[band_index]
            edge_list = tuple(tuple(str(item) for item in edge) for edge in payload["edge_list"].tolist())
            channels = _channel_order_from_payload(payload, edge_list)
            if channel_order is None:
                channel_order = channels
            elif channel_order != channels:
                raise ValueError(f"Inconsistent channel order in {path}.")
            matrices.append(edge_values_to_symmetric_matrix(wpli[:, band_index], edge_list, channels))

    if not matrices or channel_order is None:
        raise ValueError(f"No readable wPLI feature files for {group_label}.")
    mean_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
    np.fill_diagonal(mean_matrix, 0.0)
    return WPLIBandMean(
        matrix=mean_matrix,
        channels=channel_order,
        n_subjects=len(matrices),
        band=resolved_band,
        state=state,
        group_label=group_label,
    )


def edge_values_to_symmetric_matrix(
    edge_values: Sequence[float],
    edge_list: Sequence[tuple[str, str]],
    channels: Sequence[str],
) -> np.ndarray:
    """Restore an upper-triangle edge vector into a symmetric channel matrix."""

    values = np.asarray(edge_values, dtype=float).reshape(-1)
    if len(values) != len(edge_list):
        raise ValueError("edge_values length must match edge_list length.")
    channel_index = {channel: index for index, channel in enumerate(channels)}
    matrix = np.zeros((len(channels), len(channels)), dtype=float)
    for value, (first, second) in zip(values, edge_list, strict=True):
        if first not in channel_index or second not in channel_index:
            raise ValueError(f"Edge contains channel not present in channel order: {first}-{second}")
        i = channel_index[first]
        j = channel_index[second]
        matrix[i, j] = float(value)
        matrix[j, i] = float(value)
    return matrix


def cluster_order(matrix: np.ndarray) -> np.ndarray:
    """Return one hierarchical clustering order to apply to rows and columns."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Clustered wPLI matrix must be square.")
    condensed = pdist(values, metric="euclidean")
    row_linkage = linkage(condensed, method="average", optimal_ordering=True)
    return leaves_list(row_linkage)


def draw_clustermap(
    *,
    difference: np.ndarray,
    channels: Sequence[str],
    order: np.ndarray,
    output_stem: Path,
    dpi: int,
    title: str,
    subtitle: str,
) -> None:
    """Draw the main clustered difference heatmap without visible dendrogram lines."""

    ordered = difference[np.ix_(order, order)]
    ordered_channels = [channels[index] for index in order]
    color_limit = float(np.nanmax(np.abs(ordered)))
    color_limit = max(color_limit, 1e-6)

    fig = plt.figure(figsize=(7.8, 8.1), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=(6.0, 0.30),
        left=0.12,
        right=0.90,
        bottom=0.10,
        top=0.88,
        wspace=0.035,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_cbar = fig.add_subplot(grid[0, 1])

    image = ax_heat.imshow(
        ordered,
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        aspect="equal",
    )
    ax_heat.set_xticks(np.arange(len(ordered_channels)))
    ax_heat.set_yticks(np.arange(len(ordered_channels)))
    label_size = 4.8 if len(ordered_channels) > 55 else 5.5
    ax_heat.set_xticklabels(ordered_channels, rotation=90, fontsize=label_size)
    ax_heat.set_yticklabels(ordered_channels, fontsize=label_size)
    ax_heat.tick_params(length=0, pad=1.5)
    for spine in ax_heat.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#444444")

    cbar = fig.colorbar(image, cax=ax_cbar)
    cbar.set_label("Δ wPLI (Stroke baseline - Healthy)", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(title, y=0.965, fontsize=12, fontweight="bold")
    fig.text(0.50, 0.925, subtitle, ha="center", va="center", fontsize=8, color="#333333")

    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_supplementary_six_band_heatmap(
    *,
    manifest_path: Path,
    channels: Sequence[str],
    order: np.ndarray,
    state: str,
    out_dir: Path,
    dpi: int,
) -> None:
    """Save a supplementary 2x3 overview of all EC bands using the same order."""

    bands = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High")
    matrices = []
    for band in bands:
        healthy = load_mean_wpli_matrix(
            manifest_path=manifest_path,
            group="health",
            stage="health",
            state=state,
            band=band,
            group_label="Healthy reference",
        )
        stroke = load_mean_wpli_matrix(
            manifest_path=manifest_path,
            group="patient",
            stage="基线",
            state=state,
            band=band,
            group_label="Stroke baseline",
        )
        if healthy.channels != tuple(channels) or stroke.channels != tuple(channels):
            raise ValueError("Supplementary matrices use inconsistent channel orders.")
        diff = stroke.matrix - healthy.matrix
        np.fill_diagonal(diff, 0.0)
        matrices.append(diff[np.ix_(order, order)])

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4), constrained_layout=False)
    for ax, band, matrix in zip(axes.flat, bands, matrices, strict=True):
        vmin, vmax = robust_centered_limits(matrix)
        image = ax.imshow(
            matrix,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(band, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#555555")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.0%", pad=0.035)
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=5.5, length=1.6, width=0.45)
        cbar.outline.set_linewidth(0.4)

    fig.subplots_adjust(left=0.045, right=0.965, bottom=0.045, top=0.965, wspace=0.25, hspace=0.18)
    stem = out_dir / "wpli_ec_allbands_difference_heatmap_supplementary"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def robust_centered_limits(matrix: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    """Return robust color limits that preserve zero as the white center."""

    values = off_diagonal_values(matrix)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1e-6, 1e-6

    raw_min = float(np.nanmin(values))
    raw_max = float(np.nanmax(values))
    vmin = float(np.nanpercentile(values, lower))
    vmax = float(np.nanpercentile(values, upper))
    if vmin >= 0.0:
        vmin = raw_min if raw_min < 0.0 else -max(abs(vmax), 1e-6)
    if vmax <= 0.0:
        vmax = raw_max if raw_max > 0.0 else max(abs(vmin), 1e-6)
    if not vmin < 0.0:
        vmin = -1e-6
    if not vmax > 0.0:
        vmax = 1e-6
    return vmin, vmax


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return values.reshape(-1)
    return values[np.triu_indices_from(values, k=1)]


def _band_index(band_names: Sequence[str], target_band: str) -> int:
    normalized = [name.strip().lower() for name in band_names]
    target = target_band.strip().lower()
    if target not in normalized:
        raise ValueError(f"Band {target_band!r} is not present in {band_names}.")
    return normalized.index(target)


def _channel_order_from_payload(
    payload: np.lib.npyio.NpzFile,
    edge_list: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    if "channel_names_after_alignment" in payload.files:
        return tuple(str(channel) for channel in payload["channel_names_after_alignment"].tolist())
    channels: list[str] = []
    for first, second in edge_list:
        if first not in channels:
            channels.append(first)
        if second not in channels:
            channels.append(second)
    return tuple(channels)

if __name__ == "__main__":
    main()
