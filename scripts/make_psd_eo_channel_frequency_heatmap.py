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


DEFAULT_CAPTION = (
    "EO 状态下 PSD 通道—频率热图显示，卒中患者治疗前在多个通道和频率范围内"
    "相较健康模板存在系统性偏离。该结果提示 PSD 能够反映卒中后局部脑电振荡异常，"
    "并为深度学习模型提供具有生理意义的频谱输入表征。"
)

BAND_LABELS: tuple[tuple[str, float, float], ...] = (
    ("Delta", 1.0, 3.0),
    ("Theta", 4.0, 7.0),
    ("Alpha", 8.0, 13.0),
    ("Beta Low", 13.0, 18.0),
    ("Beta Medium", 18.0, 21.0),
    ("Beta High", 21.0, 30.0),
)


@dataclass(frozen=True)
class PSDGroupStats:
    mean: np.ndarray
    std: np.ndarray
    channels: tuple[str, ...]
    frequencies: tuple[float, ...]
    n_subjects: int
    group_label: str
    state: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw EO PSD channel-frequency stroke-minus-healthy heatmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features-dir",
        default=Path("data") / "features_all_stages",
        type=Path,
        help="Feature root containing feature_manifest.csv and psd/.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("results") / "biomarker_validity" / "figures",
        type=Path,
    )
    parser.add_argument("--state", default="EO")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--measure",
        choices=("z", "raw"),
        default="z",
        help="Use healthy-standardized z-score difference or raw PSD difference.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    manifest_path = args.features_dir / "feature_manifest.csv"
    healthy = load_psd_group_stats(
        manifest_path=manifest_path,
        group="health",
        stage="health",
        state=args.state,
        group_label="Healthy reference",
    )
    stroke = load_psd_group_stats(
        manifest_path=manifest_path,
        group="patient",
        stage="基线",
        state=args.state,
        group_label="Stroke baseline",
    )
    if healthy.channels != stroke.channels:
        raise ValueError("Healthy and stroke PSD files use different channel orders.")
    if not np.allclose(healthy.frequencies, stroke.frequencies):
        raise ValueError("Healthy and stroke PSD files use different frequency bins.")

    raw_difference = stroke.mean - healthy.mean
    if args.measure == "z":
        display_matrix = compute_z_difference(stroke.mean, healthy.mean, healthy.std)
        colorbar_label = "Δ PSD z-score (Stroke baseline - Healthy)"
    else:
        display_matrix = raw_difference
        colorbar_label = "Δ PSD (Stroke baseline - Healthy)"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.out_dir / "psd_eo_channel_frequency_heatmap"
    draw_psd_heatmap(
        matrix=display_matrix,
        channels=healthy.channels,
        frequencies=healthy.frequencies,
        output_stem=output_stem,
        dpi=args.dpi,
        title="EO PSD Channel-Frequency Difference Map",
        subtitle=(
            f"Stroke baseline vs Healthy reference "
            f"(n={stroke.n_subjects} vs {healthy.n_subjects})"
        ),
        colorbar_label=colorbar_label,
    )
    (args.out_dir / "psd_eo_channel_frequency_heatmap_caption.md").write_text(
        DEFAULT_CAPTION + "\n",
        encoding="utf-8",
    )


def configure_matplotlib() -> None:
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


def load_psd_group_stats(
    *,
    manifest_path: Path,
    group: str,
    stage: str,
    state: str,
    group_label: str,
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
        raise ValueError(f"No readable PSD feature files for {group_label}.")
    stack = np.stack(matrices, axis=0).astype(float)
    ddof = 1 if stack.shape[0] > 1 else 0
    return PSDGroupStats(
        mean=np.mean(stack, axis=0),
        std=np.std(stack, axis=0, ddof=ddof),
        channels=channel_order,
        frequencies=frequency_bins,
        n_subjects=stack.shape[0],
        group_label=group_label,
        state=state,
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


def compute_z_difference(
    stroke_mean: np.ndarray,
    healthy_mean: np.ndarray,
    healthy_std: np.ndarray,
) -> np.ndarray:
    scale = np.asarray(healthy_std, dtype=float)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (np.asarray(stroke_mean, dtype=float) - np.asarray(healthy_mean, dtype=float)) / scale


def robust_symmetric_limit(matrix: np.ndarray, percentile: float = 98.0) -> float:
    values = np.asarray(matrix, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    limit = float(np.nanpercentile(np.abs(values), percentile))
    return max(limit, 1e-6)


def draw_psd_heatmap(
    *,
    matrix: np.ndarray,
    channels: Sequence[str],
    frequencies: Sequence[float],
    output_stem: Path,
    dpi: int,
    title: str,
    subtitle: str,
    colorbar_label: str,
) -> None:
    values = np.asarray(matrix, dtype=float)
    freqs = np.asarray(frequencies, dtype=float)
    if values.shape != (len(channels), len(freqs)):
        raise ValueError("PSD heatmap matrix shape must match channels x frequencies.")
    if len(freqs) < 2:
        raise ValueError("At least two frequency bins are required.")

    freq_step = float(np.median(np.diff(freqs)))
    x_min = float(freqs[0] - freq_step / 2.0)
    x_max = float(freqs[-1] + freq_step / 2.0)
    color_limit = robust_symmetric_limit(values, percentile=98.0)

    fig = plt.figure(figsize=(9.2, 7.4), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=(7.8, 0.34),
        left=0.10,
        right=0.91,
        bottom=0.12,
        top=0.84,
        wspace=0.045,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_cbar = fig.add_subplot(grid[0, 1])

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
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("EEG channels", fontsize=8)

    x_ticks = [0.5, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([format_frequency_tick(tick) for tick in x_ticks], fontsize=7)
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(channels, fontsize=5.2)
    ax.tick_params(length=0, pad=1.8)

    for boundary in sorted({edge for _, low, high in BAND_LABELS for edge in (low, high)}):
        ax.axvline(
            boundary,
            color="#B8B8B8",
            linewidth=0.6,
            linestyle=(0, (2.2, 2.2)),
            alpha=0.85,
            zorder=2,
        )
    for band, low, high in BAND_LABELS:
        ax.text(
            (low + high) / 2.0,
            1.015,
            band,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.1,
            color="#333333",
            clip_on=False,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#444444")

    cbar = fig.colorbar(image, cax=ax_cbar)
    cbar.set_label(colorbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(title, y=0.960, fontsize=12, fontweight="bold")
    fig.text(0.50, 0.920, subtitle, ha="center", va="center", fontsize=8, color="#333333")

    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _payload_strings(payload: np.lib.npyio.NpzFile, key: str, length: int) -> tuple[str, ...]:
    if key not in payload.files:
        return tuple(f"item{index:03d}" for index in range(length))
    values = np.asarray(payload[key]).astype(str).reshape(-1)
    if len(values) != length:
        return tuple(f"item{index:03d}" for index in range(length))
    return tuple(str(value) for value in values)


def format_frequency_tick(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _payload_frequencies(payload: np.lib.npyio.NpzFile, length: int) -> tuple[float, ...]:
    if "frequency_bins" not in payload.files:
        return tuple(0.5 * (index + 1) for index in range(length))
    values = np.asarray(payload["frequency_bins"], dtype=float).reshape(-1)
    if len(values) != length:
        raise ValueError("frequency_bins length does not match PSD feature width.")
    return tuple(float(value) for value in values)


if __name__ == "__main__":
    main()
