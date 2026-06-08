from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.config import load_path_config
from eeg_recovery.visualization.topomap import normalize_channel_name


BANDS = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High")
STATES = ("EO", "EC")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render MNE-style scalp WPLI connectivity plots from explainability edges."
    )
    parser.add_argument("--config", default="configs/paths.example.yaml")
    parser.add_argument("--coordinate-file", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--states", nargs="+", default=STATES, choices=STATES)
    parser.add_argument("--bands", nargs="+", default=BANDS, choices=BANDS)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "svg"),
        choices=("png", "svg", "pdf", "tiff"),
    )
    args = parser.parse_args()

    config = load_path_config(args.config)
    output_root = config.output_root
    coordinate_path = Path(args.coordinate_file) if args.coordinate_file else config.standard_1005_ced
    edges = pd.read_csv(output_root / "results" / "explainability" / "wpli_top_edges.csv")

    names, pos = _load_layout_from_topomap_script(coordinate_path, CANONICAL_CHANNELS_62)
    out_dir = output_root / "results" / "figures" / "explainability" / "mne_wpli_connectivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, object]] = []
    for state in args.states:
        for band in args.bands:
            subset = _top_edges_for_plot(edges, state=state, band=band, top_n=args.top_n)
            if subset.empty:
                continue
            basename = out_dir / f"mne_wpli_{state.lower()}_{_slug(band)}_connectivity_top{args.top_n}"
            paths = _plot_wpli_connectivity(
                edges=subset,
                names=names,
                pos=pos,
                basename=basename,
                formats=args.formats,
                title=f"{state} {band} WPLI connectivity",
            )
            generated.append(
                {
                    "state": state,
                    "band": band,
                    "top_n": int(len(subset)),
                    "value_column": "mean_abs_attribution",
                    "signed_column": "mean_signed_attribution",
                    "paths": ";".join(str(path) for path in paths),
                }
            )

    manifest = pd.DataFrame(generated)
    manifest.to_csv(out_dir / "mne_wpli_connectivity_manifest.csv", index=False)
    _write_contact_sheet(manifest, out_dir / "mne_wpli_connectivity_contact_sheet.png")
    print(f"Wrote {len(generated)} MNE WPLI connectivity sets to {out_dir}")


def _top_edges_for_plot(
    frame: pd.DataFrame,
    *,
    state: str,
    band: str,
    top_n: int,
) -> pd.DataFrame:
    subset = frame[(frame["state"] == state) & (frame["band"] == band)].copy()
    if subset.empty:
        return subset
    return subset.sort_values("mean_abs_attribution", ascending=False).head(top_n).reset_index(drop=True)


def _plot_wpli_connectivity(
    *,
    edges: pd.DataFrame,
    names: Sequence[str],
    pos: np.ndarray,
    basename: Path,
    formats: Iterable[str],
    title: str,
) -> list[Path]:
    name_to_index = {normalize_channel_name(name): index for index, name in enumerate(names)}
    max_abs = max(float(edges["mean_abs_attribution"].max()), 1e-12)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )

    fig, ax = plt.subplots(figsize=(4.0, 4.2), constrained_layout=True)
    image, _ = mne.viz.plot_topomap(
        np.zeros(len(names), dtype=float),
        pos,
        axes=ax,
        show=False,
        sensors=False,
        contours=0,
        outlines="head",
        sphere=(0.0, 0.0, 0.0, 1.0),
        extrapolate="head",
        border="mean",
        cmap="Greys",
        vlim=(-1.0, 1.0),
    )
    image.set_alpha(0.0)

    for _, edge in edges.iterrows():
        source = normalize_channel_name(edge["channel_i"])
        target = normalize_channel_name(edge["channel_j"])
        if source not in name_to_index or target not in name_to_index:
            continue
        start = pos[name_to_index[source]]
        end = pos[name_to_index[target]]
        signed_value = float(edge.get("mean_signed_attribution", 0.0))
        magnitude = float(edge["mean_abs_attribution"])
        color = "#c83b3b" if signed_value >= 0 else "#3777c8"
        linewidth = 0.6 + 2.6 * magnitude / max_abs
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=linewidth,
            alpha=0.68,
            solid_capstyle="round",
            zorder=2,
        )

    ax.scatter(pos[:, 0], pos[:, 1], s=10, c="#202020", linewidths=0, zorder=3)
    for name, (x, y) in zip(names, pos):
        ax.text(
            x * 1.035,
            y * 1.035,
            name,
            fontsize=5.5,
            ha="center",
            va="center",
            color="#202020",
            zorder=4,
        )

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")

    written: list[Path] = []
    for fmt in formats:
        output = basename.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tiff"}:
            save_kwargs["dpi"] = 600
        fig.savefig(output, **save_kwargs)
        written.append(output)
    plt.close(fig)
    return written


def _load_layout_from_topomap_script(path: Path, channel_order: Sequence[str]) -> tuple[list[str], np.ndarray]:
    script_path = PROJECT_ROOT / "scripts" / "45_make_mne_explainability_topomaps.py"
    spec = importlib.util.spec_from_file_location("make_mne_explainability_topomaps", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_ced_eeglab_topomap_layout(path, channel_order)


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _write_contact_sheet(manifest: pd.DataFrame, output: Path) -> None:
    png_rows = []
    for _, row in manifest.iterrows():
        png_path = next((Path(path) for path in str(row["paths"]).split(";") if path.endswith(".png")), None)
        if png_path is None or not png_path.exists():
            continue
        png_rows.append((f'{row["state"]} {row["band"]}', png_path))
    if not png_rows:
        return

    # Four columns keep the 12-panel summary landscape-oriented, so it fits
    # the supplementary DOCX page without separating the figure title from the image.
    n_cols = 4
    n_rows = int(np.ceil(len(png_rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 3.05), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, (label, png_path) in zip(axes_array, png_rows):
        ax.imshow(plt.imread(png_path))
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    for ax in axes_array[len(png_rows) :]:
        ax.axis("off")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
