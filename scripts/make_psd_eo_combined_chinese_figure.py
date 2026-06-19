from __future__ import annotations

import argparse
import shutil
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


BAND_LABELS: tuple[tuple[str, float, float], ...] = (
    ("Delta", 1.0, 3.0),
    ("Theta", 4.0, 7.0),
    ("Alpha", 8.0, 13.0),
    ("Beta Low", 13.0, 18.0),
    ("Beta Medium", 18.0, 21.0),
    ("Beta High", 21.0, 30.0),
)


@dataclass(frozen=True)
class DistanceStats:
    q_value: float
    effect_size: float


@dataclass(frozen=True)
class PSDGroupStats:
    mean: np.ndarray
    std: np.ndarray
    channels: tuple[str, ...]
    frequencies: tuple[float, ...]
    n_subjects: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw a Chinese two-panel EO PSD biomarker validity figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features-dir",
        default=Path("data") / "features_all_stages",
        type=Path,
        help="Feature root containing feature_manifest.csv and psd/.",
    )
    parser.add_argument(
        "--results-dir",
        default=Path("results") / "biomarker_validity",
        type=Path,
        help="Directory containing biomarker validity CSV results.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("results") / "biomarker_validity" / "figures",
        type=Path,
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--copy-dir",
        type=Path,
        default=None,
        help="Optional directory to receive a copy of the exported PNG/PDF.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    distance_frame = extract_psd_eo_overall_distances(
        pd.read_csv(args.results_dir / "healthy_reference_distances.csv")
    )
    distance_stats = extract_psd_eo_overall_stats(
        pd.read_csv(args.results_dir / "baseline_abnormality_stats.csv")
    )

    manifest_path = args.features_dir / "feature_manifest.csv"
    healthy = load_psd_group_stats(
        manifest_path=manifest_path,
        group="health",
        stage="health",
        state="EO",
    )
    stroke = load_psd_group_stats(
        manifest_path=manifest_path,
        group="patient",
        stage="基线",
        state="EO",
    )
    if healthy.channels != stroke.channels:
        raise ValueError("Healthy and stroke PSD files use different channel orders.")
    if not np.allclose(healthy.frequencies, stroke.frequencies):
        raise ValueError("Healthy and stroke PSD files use different frequency bins.")
    z_matrix = compute_z_difference(stroke.mean, healthy.mean, healthy.std)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.out_dir / "psd_eo_combined_chinese_figure"
    draw_combined_figure(
        distance_frame=distance_frame,
        distance_stats=distance_stats,
        z_matrix=z_matrix,
        channels=healthy.channels,
        frequencies=healthy.frequencies,
        output_stem=output_stem,
        dpi=args.dpi,
    )
    if args.copy_dir is not None:
        args.copy_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            shutil.copy2(output_stem.with_suffix(suffix), args.copy_dir / output_stem.with_suffix(suffix).name)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
        }
    )


def extract_psd_eo_overall_distances(distances: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "group", "timepoint", "modality", "state", "level", "distance_to_health"}
    missing = required - set(distances.columns)
    if missing:
        raise ValueError(f"Distance table is missing required columns: {sorted(missing)}")
    frame = distances[
        distances["timepoint"].astype(str).str.strip().eq("baseline")
        & distances["modality"].astype(str).str.strip().eq("psd")
        & distances["state"].astype(str).str.strip().eq("EO")
        & distances["level"].astype(str).str.strip().eq("overall")
        & distances["group"].astype(str).str.strip().isin(["healthy", "patient"])
    ].copy()
    if frame.empty:
        raise ValueError("No Healthy/Stroke baseline EO overall PSD distance rows were found.")
    frame["distance_to_health"] = pd.to_numeric(frame["distance_to_health"], errors="coerce")
    frame = frame.dropna(subset=["distance_to_health"])
    frame["group_label"] = frame["group"].map({"healthy": "Healthy", "patient": "Stroke baseline"})
    counts = frame.groupby("group_label")["distance_to_health"].size().to_dict()
    if counts.get("Healthy", 0) == 0 or counts.get("Stroke baseline", 0) == 0:
        raise ValueError(f"Both groups are required, got counts={counts}.")
    return frame[["subject_id", "group", "group_label", "distance_to_health"]].reset_index(drop=True)


def extract_psd_eo_overall_stats(stats: pd.DataFrame) -> DistanceStats:
    required = {"feature_id", "q_value", "effect_size"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Statistics table is missing required columns: {sorted(missing)}")
    matches = stats[stats["feature_id"].astype(str).str.strip().eq("distance|psd|EO|overall")]
    if matches.empty:
        raise ValueError("Statistics row distance|psd|EO|overall was not found.")
    row = matches.iloc[0]
    return DistanceStats(q_value=float(row["q_value"]), effect_size=float(row["effect_size"]))


def load_psd_group_stats(
    *,
    manifest_path: Path,
    group: str,
    stage: str,
    state: str,
) -> PSDGroupStats:
    manifest = pd.read_csv(manifest_path)
    required = {"group", "stage", "state", "modality", "feature_path", "status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Feature manifest is missing required columns: {sorted(missing)}")
    selected = manifest[
        manifest["group"].astype(str).str.strip().eq(group)
        & manifest["stage"].astype(str).str.strip().eq(stage)
        & manifest["state"].astype(str).str.strip().eq(state)
        & manifest["modality"].astype(str).str.strip().eq("psd")
        & manifest["status"].astype(str).str.strip().eq("written")
    ].copy()
    if selected.empty:
        raise ValueError(f"No PSD rows found for group={group}, stage={stage}, state={state}.")

    matrices: list[np.ndarray] = []
    channel_order: tuple[str, ...] | None = None
    frequency_bins: tuple[float, ...] | None = None
    for path_text in selected["feature_path"]:
        path = Path(str(path_text).strip())
        if not path.exists():
            continue
        matrix, channels, frequencies = read_psd_payload(path)
        if channel_order is None:
            channel_order = channels
        elif channel_order != channels:
            raise ValueError(f"Inconsistent PSD channel order in {path}.")
        if frequency_bins is None:
            frequency_bins = frequencies
        elif not np.allclose(frequency_bins, frequencies):
            raise ValueError(f"Inconsistent PSD frequency bins in {path}.")
        matrices.append(matrix)

    if not matrices or channel_order is None or frequency_bins is None:
        raise ValueError(f"No readable PSD feature files for group={group}, stage={stage}.")
    stack = np.stack(matrices, axis=0).astype(float)
    ddof = 1 if stack.shape[0] > 1 else 0
    return PSDGroupStats(
        mean=np.mean(stack, axis=0),
        std=np.std(stack, axis=0, ddof=ddof),
        channels=channel_order,
        frequencies=frequency_bins,
        n_subjects=stack.shape[0],
    )


def read_psd_payload(path: Path) -> tuple[np.ndarray, tuple[str, ...], tuple[float, ...]]:
    with np.load(path, allow_pickle=False) as payload:
        if "psd" not in payload.files:
            raise KeyError(f"PSD file {path} is missing required key 'psd'.")
        matrix = np.asarray(payload["psd"], dtype=float)
        if matrix.shape != (62, 90):
            raise ValueError(f"Expected PSD shape 62 x 90, got {matrix.shape} in {path}.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"PSD matrix contains non-finite values in {path}.")
        channels = _payload_strings(payload, "channel_names_after_alignment", matrix.shape[0])
        frequencies = _payload_frequencies(payload, matrix.shape[1])
    return matrix, channels, frequencies


def compute_z_difference(stroke_mean: np.ndarray, healthy_mean: np.ndarray, healthy_std: np.ndarray) -> np.ndarray:
    scale = np.asarray(healthy_std, dtype=float)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (np.asarray(stroke_mean, dtype=float) - np.asarray(healthy_mean, dtype=float)) / scale


def draw_combined_figure(
    *,
    distance_frame: pd.DataFrame,
    distance_stats: DistanceStats,
    z_matrix: np.ndarray,
    channels: Sequence[str],
    frequencies: Sequence[float],
    output_stem: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(13.8, 6.0), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(0.95, 1.88, 0.08),
        left=0.055,
        right=0.955,
        bottom=0.13,
        top=0.91,
        wspace=0.22,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_cbar = fig.add_subplot(grid[0, 2])

    draw_distance_panel(ax_a, distance_frame, distance_stats)
    draw_heatmap_panel(ax_b, ax_cbar, z_matrix, channels, frequencies)

    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_distance_panel(ax: plt.Axes, frame: pd.DataFrame, stats: DistanceStats) -> None:
    order = ("Healthy", "Stroke baseline")
    data = [frame.loc[frame["group_label"].eq(label), "distance_to_health"].to_numpy(dtype=float) for label in order]
    colors = ("#8DAEC2", "#D98D68")
    edge_colors = ("#506F80", "#9A573A")
    positions = np.arange(1, len(order) + 1)

    violin = ax.violinplot(data, positions=positions, widths=0.78, showmeans=False, showmedians=False, showextrema=False)
    for body, fill, edge in zip(violin["bodies"], colors, edge_colors, strict=True):
        body.set_facecolor(fill)
        body.set_edgecolor(edge)
        body.set_alpha(0.45)
        body.set_linewidth(0.8)
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.9},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.92)

    rng = np.random.default_rng(20260614)
    for index, (label, fill, edge) in enumerate(zip(order, colors, edge_colors, strict=True), start=1):
        values = frame.loc[frame["group_label"].eq(label), "distance_to_health"].to_numpy(dtype=float)
        jitter = np.clip(rng.normal(loc=0.0, scale=0.045, size=len(values)), -0.10, 0.10)
        ax.scatter(
            np.full(len(values), index, dtype=float) + jitter,
            values,
            s=20,
            color=fill,
            edgecolor=edge,
            linewidth=0.42,
            alpha=0.88,
            zorder=4,
        )

    y_max = max(float(np.nanmax(values)) for values in data)
    y_min = min(float(np.nanmin(values)) for values in data)
    y_range = max(y_max - y_min, 1.0)
    bracket_y = y_max + 0.12 * y_range
    text_y = bracket_y + 0.04 * y_range
    draw_significance_bracket(ax, positions[0], positions[1], bracket_y, 0.035 * y_range)
    ax.text(
        np.mean(positions),
        text_y,
        f"q = {format_p_value(stats.q_value)}\neffect size = {stats.effect_size:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#222222",
        linespacing=1.25,
    )

    ax.set_xlabel("Group", fontsize=9)
    ax.set_ylabel("PSD deviation from healthy reference", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlim(0.45, 2.55)
    ax.set_ylim(0, text_y + 0.16 * y_range)
    ax.yaxis.grid(True, color="#E1E1E1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=3, width=0.7, color="#333333")
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.75)


def draw_heatmap_panel(
    ax: plt.Axes,
    ax_cbar: plt.Axes,
    matrix: np.ndarray,
    channels: Sequence[str],
    frequencies: Sequence[float],
) -> None:
    values = np.asarray(matrix, dtype=float)
    freqs = np.asarray(frequencies, dtype=float)
    if values.shape != (len(channels), len(freqs)):
        raise ValueError("PSD heatmap matrix shape must match channels x frequencies.")

    freq_step = float(np.median(np.diff(freqs)))
    x_min = float(freqs[0] - freq_step / 2.0)
    x_max = float(freqs[-1] + freq_step / 2.0)
    color_limit = robust_symmetric_limit(values, percentile=98.0)
    image = ax.imshow(
        values,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
        interpolation="nearest",
        aspect="auto",
        extent=(x_min, x_max, len(channels) - 0.5, -0.5),
    )
    ax.set_xlim(0.5, 45.0)
    ax.set_ylim(len(channels) - 0.5, -0.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("EEG channels", fontsize=9)
    x_ticks = [0.5, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([format_frequency_tick(tick) for tick in x_ticks], fontsize=7)
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(channels, fontsize=4.8)
    ax.tick_params(length=0, pad=1.4)

    for boundary in sorted({edge for _, low, high in BAND_LABELS for edge in (low, high)}):
        ax.axvline(boundary, color="#B8B8B8", linewidth=0.55, linestyle=(0, (2.2, 2.2)), alpha=0.85, zorder=2)
    for band, low, high in BAND_LABELS:
        ax.text(
            (low + high) / 2.0,
            1.018,
            band,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
            clip_on=False,
        )
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#444444")

    cbar = ax.figure.colorbar(image, cax=ax_cbar)
    cbar.set_label("Δ PSD z-score (Stroke baseline - Healthy)", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.5)


def draw_significance_bracket(ax: plt.Axes, x1: float, x2: float, y: float, height: float) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#333333", linewidth=0.8, clip_on=False)


def robust_symmetric_limit(matrix: np.ndarray, percentile: float = 98.0) -> float:
    values = np.asarray(matrix, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    return max(float(np.nanpercentile(np.abs(values), percentile)), 1e-6)


def _payload_strings(payload: np.lib.npyio.NpzFile, key: str, length: int) -> tuple[str, ...]:
    if key not in payload.files:
        return tuple(f"item{index:03d}" for index in range(length))
    values = np.asarray(payload[key]).astype(str).reshape(-1)
    if len(values) != length:
        return tuple(f"item{index:03d}" for index in range(length))
    return tuple(str(value) for value in values)


def _payload_frequencies(payload: np.lib.npyio.NpzFile, length: int) -> tuple[float, ...]:
    if "frequency_bins" not in payload.files:
        return tuple(0.5 * (index + 1) for index in range(length))
    values = np.asarray(payload["frequency_bins"], dtype=float).reshape(-1)
    if len(values) != length:
        raise ValueError("frequency_bins length does not match PSD feature width.")
    return tuple(float(value) for value in values)


def format_frequency_tick(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def format_p_value(value: float) -> str:
    number = float(value)
    if number < 0.001:
        return f"{number:.1e}"
    if number < 0.01:
        return f"{number:.4f}"
    return f"{number:.3f}"


if __name__ == "__main__":
    main()
