from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CAPTION = (
    "EC 状态下 wPLI 网络层面统计结果显示，卒中患者治疗前功能连接异常主要集中于"
    "中心相关网络及前后部耦合网络。该结果提示 wPLI 能够反映卒中后脑网络重组异常，"
    "并为深度学习模型的功能连接输入提供统计依据。"
)

PREFERRED_NETWORKS: tuple[str, ...] = (
    "central",
    "central|frontal",
    "central|parietal",
    "central|temporal",
    "frontal|occipital",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw EC wPLI network-level abnormality lollipop plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default=Path("results") / "biomarker_validity",
        type=Path,
        help="Directory containing baseline_abnormality_stats.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("results") / "biomarker_validity" / "figures",
        type=Path,
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    configure_matplotlib()
    stats = pd.read_csv(args.results_dir / "baseline_abnormality_stats.csv")
    selected = select_wpli_ec_network_rows(stats)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.out_dir / "wpli_ec_network_lollipop"
    draw_lollipop(selected, output_stem=output_stem, dpi=args.dpi)
    (args.out_dir / "wpli_ec_network_lollipop_caption.md").write_text(
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
            "axes.linewidth": 0.75,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
        }
    )


def select_wpli_ec_network_rows(stats: pd.DataFrame, max_rows: int = 8) -> pd.DataFrame:
    required = {
        "modality",
        "state",
        "level",
        "network_group",
        "q_value",
        "effect_size",
        "effect_size_name",
    }
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Statistics table is missing required columns: {sorted(missing)}")

    frame = stats[
        stats["modality"].astype(str).str.strip().eq("wpli")
        & stats["state"].astype(str).str.strip().eq("EC")
        & stats["level"].astype(str).str.strip().eq("network")
    ].copy()
    if frame.empty:
        raise ValueError("No EC wPLI network-level rows were found.")
    frame["q_value"] = pd.to_numeric(frame["q_value"], errors="coerce")
    frame["effect_size"] = pd.to_numeric(frame["effect_size"], errors="coerce")
    frame = frame.dropna(subset=["network_group", "q_value", "effect_size"])

    preferred = frame[frame["network_group"].isin(PREFERRED_NETWORKS)].copy()
    preferred_names = set(preferred["network_group"].astype(str))
    if all(network in preferred_names for network in PREFERRED_NETWORKS):
        selected = preferred[preferred["network_group"].isin(PREFERRED_NETWORKS)].copy()
        selected["_network_rank"] = selected["network_group"].map(
            {network: rank for rank, network in enumerate(PREFERRED_NETWORKS)}
        )
        selected = selected.sort_values("_network_rank").drop(columns="_network_rank")
    else:
        significant = frame[frame["q_value"] < 0.05].copy()
        if significant.empty:
            raise ValueError("No FDR-significant EC wPLI network-level rows were found.")
        selected = pd.concat(
            [
                preferred[preferred["q_value"] < 0.05],
                significant[~significant["network_group"].isin(preferred_names)].sort_values(
                    "effect_size",
                    ascending=False,
                ),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["network_group"], keep="first")
        selected = selected.head(max_rows)

    selected = selected.copy()
    selected["display_label"] = selected["network_group"].map(format_network_label)
    selected["neg_log10_q"] = -np.log10(np.clip(selected["q_value"].to_numpy(dtype=float), 1e-300, None))
    return selected.sort_values("effect_size", ascending=True).reset_index(drop=True)


def draw_lollipop(frame: pd.DataFrame, *, output_stem: Path, dpi: int) -> None:
    if frame.empty:
        raise ValueError("No rows were selected for plotting.")
    y = np.arange(len(frame))
    effect = frame["effect_size"].to_numpy(dtype=float)
    neg_log_q = frame["neg_log10_q"].to_numpy(dtype=float)
    x_min = max(0.0, float(np.nanmin(effect)) - 0.055)
    x_max = min(1.0, float(np.nanmax(effect)) + 0.040)
    if x_max - x_min < 0.12:
        x_min = max(0.0, x_max - 0.12)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=False)
    for yi, value in zip(y, effect, strict=True):
        ax.hlines(yi, x_min, value, color="#B9B9B9", linewidth=1.6, zorder=1)

    size = point_sizes(neg_log_q)
    color = "#D8845D"
    edge = "#8F4A34"
    ax.scatter(
        effect,
        y,
        s=size,
        color=color,
        edgecolor=edge,
        linewidth=0.8,
        alpha=0.92,
        zorder=3,
    )

    label_offset = 0.010
    for yi, value, q_value in zip(y, effect, frame["q_value"].to_numpy(dtype=float), strict=True):
        ax.text(
            value + label_offset,
            yi,
            f"q = {format_q_value(q_value)}",
            ha="left",
            va="center",
            fontsize=7.5,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(frame["display_label"].tolist(), fontsize=8)
    ax.set_xlabel("Effect size (rank-biserial)", fontsize=8.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.xaxis.grid(True, color="#E3E3E3", linewidth=0.6)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=3, width=0.7, color="#333333")
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.75)

    ax.set_title("EC wPLI Network-Level Abnormalities", fontsize=12, fontweight="bold", pad=16)
    ax.text(
        0.5,
        1.015,
        "Stroke baseline vs Healthy reference",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    ax.text(
        0.99,
        -0.18,
        "Dot size reflects -log10(q value)",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def point_sizes(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    if np.nanmax(array) - np.nanmin(array) < 1e-12:
        return np.full_like(array, 115.0, dtype=float)
    scaled = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
    return 85.0 + 75.0 * scaled


def format_network_label(value: object) -> str:
    return str(value).replace("|", "\u2013")


def format_q_value(value: float) -> str:
    number = float(value)
    if number < 0.001:
        return f"{number:.1e}"
    if number < 0.01:
        return f"{number:.4f}"
    return f"{number:.3f}"


if __name__ == "__main__":
    main()
