"""Create Figure 6D: WPLI network-group attribution heatmap."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


LOGGER = logging.getLogger("figure6d_wpli_network_group_heatmap")

PROJECT_ROOT = Path("F:/CJZProjectFile/EEG_PredictStokeDLModel")
RESULTS_ROOT = (
    PROJECT_ROOT / "final_model_ablation_explainability_results_20260610" / "results"
)

INPUT_PATH = RESULTS_ROOT / "wpli_network_group_importance.csv"
FALLBACK_INPUT_PATHS = [
    PROJECT_ROOT
    / "rerun_ablation_explainability_secondary_20260609"
    / "results"
    / "explainability"
    / "wpli_network_group_importance.csv",
]

OUTPUT_BASE = (
    RESULTS_ROOT / "figures" / "paper_panels" / "figure6d_wpli_network_group_heatmap"
)

STATE_BAND_ORDER = [
    ("EO", "Delta"),
    ("EO", "Theta"),
    ("EO", "Alpha"),
    ("EO", "Beta Low"),
    ("EO", "Beta Medium"),
    ("EO", "Beta High"),
    ("EC", "Delta"),
    ("EC", "Theta"),
    ("EC", "Alpha"),
    ("EC", "Beta Low"),
    ("EC", "Beta Medium"),
    ("EC", "Beta High"),
]

STATE_BAND_LABELS = {
    ("EO", "Delta"): "EO\nDelta",
    ("EO", "Theta"): "EO\nTheta",
    ("EO", "Alpha"): "EO\nAlpha",
    ("EO", "Beta Low"): "EO\nBeta\nLow",
    ("EO", "Beta Medium"): "EO\nBeta\nMed",
    ("EO", "Beta High"): "EO\nBeta\nHigh",
    ("EC", "Delta"): "EC\nDelta",
    ("EC", "Theta"): "EC\nTheta",
    ("EC", "Alpha"): "EC\nAlpha",
    ("EC", "Beta Low"): "EC\nBeta\nLow",
    ("EC", "Beta Medium"): "EC\nBeta\nMed",
    ("EC", "Beta High"): "EC\nBeta\nHigh",
}


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def _find_input(primary: Path, recursive_root: Path, fallbacks: list[Path]) -> Path:
    if primary.exists():
        return primary

    if recursive_root.exists():
        matches = sorted(recursive_root.rglob(primary.name))
        if matches:
            LOGGER.warning(
                "Primary input not found: %s. Using recursively discovered file: %s",
                primary,
                matches[0],
            )
            return matches[0]

    for fallback in fallbacks:
        if fallback.exists():
            LOGGER.warning(
                "Primary input not found under requested results root: %s. "
                "Using fallback file: %s",
                primary,
                fallback,
            )
            return fallback

    raise FileNotFoundError(
        f"Could not find {primary.name} at {primary}, under {recursive_root}, "
        f"or in fallback paths."
    )


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{source} is missing required columns: {missing_text}")


def _normalise_band(value: object) -> str:
    text = str(value).strip()
    lookup = {band.lower(): band for _, band in STATE_BAND_ORDER}
    return lookup.get(text.lower(), text)


def _prepare_matrix(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {"state", "band", "network_group", "mean_abs_attribution"}
    _require_columns(frame, required, source)

    data = frame.copy()
    data["state"] = data["state"].astype(str).str.strip().str.upper()
    data["band"] = data["band"].map(_normalise_band)
    data["network_group"] = data["network_group"].astype(str).str.strip()
    data["mean_abs_attribution"] = pd.to_numeric(
        data["mean_abs_attribution"], errors="coerce"
    )
    data = data.dropna(subset=["mean_abs_attribution"])

    if data.empty:
        raise ValueError(f"{source} has no valid mean_abs_attribution values.")

    top_groups = (
        data.groupby("network_group", as_index=True)["mean_abs_attribution"]
        .max()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )

    if len(top_groups) < 12:
        LOGGER.warning("Only %d network groups are available.", len(top_groups))

    grouped = (
        data[data["network_group"].isin(top_groups)]
        .groupby(["network_group", "state", "band"], as_index=False)[
            "mean_abs_attribution"
        ]
        .mean()
    )

    grouped["state_band"] = list(zip(grouped["state"], grouped["band"]))
    matrix = grouped.pivot(
        index="network_group", columns="state_band", values="mean_abs_attribution"
    )
    matrix = matrix.reindex(index=top_groups, columns=STATE_BAND_ORDER)
    matrix.index = [label.replace("|", " | ") for label in matrix.index]
    return matrix


def _plot(matrix: pd.DataFrame, output_base: Path) -> None:
    _configure_matplotlib()

    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    vmax = float(values.max()) if values.count() else 1.0

    cmap = LinearSegmentedColormap.from_list(
        "wpli_attr_blue",
        ["#F7FBFF", "#D9EAF7", "#8DBBDB", "#3C7EAD", "#163D5C"],
    )
    cmap.set_bad("#F0F0F0")

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    image = ax.imshow(
        values,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title("WPLI network-group attribution", pad=10, fontweight="bold")
    ax.set_xticks(np.arange(len(STATE_BAND_ORDER)))
    ax.set_xticklabels(
        [STATE_BAND_LABELS[item] for item in STATE_BAND_ORDER],
        ha="center",
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_xticks(np.arange(-0.5, len(STATE_BAND_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=4)

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Mean absolute attribution")
    colorbar.outline.set_linewidth(0.6)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    source = _find_input(INPUT_PATH, RESULTS_ROOT, FALLBACK_INPUT_PATHS)
    frame = pd.read_csv(source)
    matrix = _prepare_matrix(frame, source)
    _plot(matrix, OUTPUT_BASE)

    print(f"Source: {source}")
    print(f"Wrote {OUTPUT_BASE.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
