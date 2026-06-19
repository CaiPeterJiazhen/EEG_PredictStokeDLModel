from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CAPTION = (
    "与健康对照相比，卒中患者治疗前 EO 状态下 PSD 健康距离升高，"
    "提示基线功率谱特征存在明确异常。该结果从统计层面支持 PSD "
    "作为后续疗效预测模型输入的合理性。"
)


@dataclass(frozen=True)
class DistanceStats:
    q_value: float
    p_value: float
    effect_size: float
    effect_size_name: str
    test: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw Healthy vs Stroke baseline EO PSD distance violin/boxplot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default=Path("results") / "biomarker_validity",
        type=Path,
        help="Directory containing healthy_reference_distances.csv and baseline_abnormality_stats.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("results") / "biomarker_validity" / "figures",
        type=Path,
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    configure_matplotlib()
    distances = pd.read_csv(args.results_dir / "healthy_reference_distances.csv")
    stats = pd.read_csv(args.results_dir / "baseline_abnormality_stats.csv")
    plot_frame = extract_psd_eo_overall_distances(distances)
    stat = extract_psd_eo_overall_stats(stats)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.out_dir / "psd_eo_distance_violin_boxplot"
    draw_distance_plot(plot_frame, stat, output_stem=output_stem, dpi=args.dpi)
    (args.out_dir / "psd_eo_distance_violin_boxplot_caption.md").write_text(
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
    frame["group_label"] = frame["group"].map(
        {
            "healthy": "Healthy",
            "patient": "Stroke baseline",
        }
    )
    counts = frame.groupby("group_label")["distance_to_health"].size().to_dict()
    if counts.get("Healthy", 0) == 0 or counts.get("Stroke baseline", 0) == 0:
        raise ValueError(f"Both Healthy and Stroke baseline groups are required, got counts={counts}.")
    return frame[["subject_id", "group", "group_label", "distance_to_health"]].reset_index(drop=True)


def extract_psd_eo_overall_stats(stats: pd.DataFrame) -> DistanceStats:
    required = {"feature_id", "p_value", "q_value", "effect_size", "effect_size_name", "test"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Statistics table is missing required columns: {sorted(missing)}")
    matches = stats[stats["feature_id"].astype(str).str.strip().eq("distance|psd|EO|overall")]
    if matches.empty:
        raise ValueError("Statistics row distance|psd|EO|overall was not found.")
    row = matches.iloc[0]
    return DistanceStats(
        q_value=float(row["q_value"]),
        p_value=float(row["p_value"]),
        effect_size=float(row["effect_size"]),
        effect_size_name=str(row["effect_size_name"]),
        test=str(row["test"]),
    )


def draw_distance_plot(
    frame: pd.DataFrame,
    stats: DistanceStats,
    *,
    output_stem: Path,
    dpi: int,
) -> None:
    order = ("Healthy", "Stroke baseline")
    data = [frame.loc[frame["group_label"].eq(label), "distance_to_health"].to_numpy(dtype=float) for label in order]
    colors = ("#8DAEC2", "#D98D68")
    edge_colors = ("#506F80", "#9A573A")

    fig, ax = plt.subplots(figsize=(4.8, 5.3), constrained_layout=False)
    positions = np.arange(1, len(order) + 1)
    violin = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
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
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
        jitter = np.clip(jitter, -0.10, 0.10)
        ax.scatter(
            np.full(len(values), index, dtype=float) + jitter,
            values,
            s=28,
            color=fill,
            edgecolor=edge,
            linewidth=0.45,
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

    ax.set_title("Baseline EO PSD Distance to Healthy Reference", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("PSD distance to healthy reference", fontsize=9)
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

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_significance_bracket(ax: plt.Axes, x1: float, x2: float, y: float, height: float) -> None:
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        color="#333333",
        linewidth=0.8,
        clip_on=False,
    )


def format_p_value(value: float) -> str:
    number = float(value)
    if number < 0.001:
        return f"{number:.1e}"
    if number < 0.01:
        return f"{number:.4f}"
    return f"{number:.3f}"


if __name__ == "__main__":
    main()
