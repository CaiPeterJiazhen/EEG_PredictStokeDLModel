from __future__ import annotations

import argparse
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


PSD_BANDS = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High", "Gamma")
WPLI_TARGETS = (
    ("EC", "Beta Medium"),
    ("EC", "Beta High"),
    ("EC", "Alpha"),
    ("EC", "Theta"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render MNE topomaps from locked explainability outputs."
    )
    parser.add_argument("--config", default="configs/paths.example.yaml")
    parser.add_argument("--coordinate-file", default=None)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "svg"),
        choices=("png", "svg", "pdf", "tiff"),
        help="Output formats to write for each topomap.",
    )
    parser.add_argument(
        "--show-names",
        action="store_true",
        help="Draw channel names on each topomap. Disabled by default to avoid label overlap.",
    )
    args = parser.parse_args()

    config = load_path_config(args.config)
    output_root = config.output_root
    coordinate_path = Path(args.coordinate_file) if args.coordinate_file else config.standard_1005_ced

    names, pos = _load_ced_eeglab_topomap_layout(coordinate_path, CANONICAL_CHANNELS_62)
    sphere = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=float)

    out_dir = output_root / "results" / "figures" / "explainability" / "mne_topomaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_topomap_coordinate_validation_table(names, pos, sphere=sphere)).to_csv(
        out_dir / "electrode_coordinate_validation.csv",
        index=False,
    )

    psd_path = output_root / "results" / "explainability" / "psd_channel_band_importance.csv"
    wpli_path = output_root / "results" / "explainability" / "wpli_node_importance.csv"
    psd = pd.read_csv(psd_path)
    wpli = pd.read_csv(wpli_path)

    generated: list[dict[str, str]] = []
    for state in ("EO", "EC"):
        for band in PSD_BANDS:
            values = _channel_values(
                psd[(psd["state"] == state) & (psd["band"] == band)],
                value_column="mean_signed_attribution",
            )
            if not values:
                continue
            data = _values_for_order(values)
            basename = out_dir / f"mne_psd_{state.lower()}_{_slug(band)}_signed_attribution_topomap"
            paths = _plot_mne_topomap(
                data=data,
                pos=pos,
                sphere=sphere,
                names=names if args.show_names else None,
                basename=basename,
                formats=args.formats,
                title=f"PSD {state} {band} signed attribution",
                cmap="RdBu_r",
                colorbar_label="signed attribution",
                symmetric=True,
            )
            generated.append(
                {
                    "kind": "PSD",
                    "state": state,
                    "band": band,
                    "value_column": "mean_signed_attribution",
                    "paths": ";".join(str(path) for path in paths),
                }
            )

    for state, band in WPLI_TARGETS:
        values = _channel_values(
            wpli[(wpli["state"] == state) & (wpli["band"] == band)],
            value_column="node_importance",
        )
        if not values:
            continue
        data = _values_for_order(values)
        basename = out_dir / f"mne_wpli_{state.lower()}_{_slug(band)}_node_importance_topomap"
        paths = _plot_mne_topomap(
            data=data,
            pos=pos,
            sphere=sphere,
            names=names if args.show_names else None,
            basename=basename,
            formats=args.formats,
            title=f"WPLI {state} {band} node importance",
            cmap="viridis",
            colorbar_label="node importance",
            symmetric=False,
        )
        generated.append(
            {
                "kind": "WPLI",
                "state": state,
                "band": band,
                "value_column": "node_importance",
                "paths": ";".join(str(path) for path in paths),
            }
        )

    pd.DataFrame(generated).to_csv(out_dir / "mne_topomap_manifest.csv", index=False)
    _write_contact_sheet(out_dir, generated)
    print(f"Wrote {len(generated)} MNE topomap sets to {out_dir}")


def _load_ced_eeglab_topomap_layout(
    path: str | Path,
    channel_order: Sequence[str],
    *,
    target_radius: float = 0.96,
) -> tuple[list[str], np.ndarray]:
    pos_by_key = _load_ced_eeglab_2d_pos(path)
    missing = [channel for channel in channel_order if normalize_channel_name(channel) not in pos_by_key]
    if missing:
        raise ValueError(f"Missing electrode coordinates for: {', '.join(missing)}")

    ch_names = [_mne_channel_name(channel) for channel in channel_order]
    pos = np.asarray(
        [pos_by_key[normalize_channel_name(channel)] for channel in channel_order],
        dtype=float,
    )
    max_radius = float(np.nanmax(np.linalg.norm(pos, axis=1)))
    if max_radius <= 0:
        raise ValueError("Electrode coordinates have zero spatial extent.")
    return ch_names, pos * (target_radius / max_radius)


def _load_ced_eeglab_2d_pos(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    pos: dict[str, np.ndarray] = {}
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 4 or parts[-1].upper() != "EEG":
                continue
            try:
                theta_degrees = float(parts[2])
                radius = float(parts[3])
            except ValueError:
                continue
            key = normalize_channel_name(parts[1])
            theta = np.deg2rad(theta_degrees)
            pos[key] = np.asarray(
                [radius * np.sin(theta), radius * np.cos(theta)],
                dtype=float,
            )

    if not pos:
        raise ValueError(f"No EEG electrode coordinates could be parsed from {source}")
    return pos


def _mne_channel_name(channel: str) -> str:
    key = normalize_channel_name(channel)
    if key.startswith("FP"):
        return f"Fp{key[2:].lower()}"
    if key.endswith("Z"):
        return f"{key[:-1]}z"
    return key


def _topomap_coordinate_validation_table(
    names: Sequence[str],
    pos: np.ndarray,
    *,
    sphere: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    center = sphere[:2]
    radius = float(sphere[3])
    for index, (channel, xy) in enumerate(zip(names, pos)):
        distance = float(np.linalg.norm(np.asarray(xy) - center))
        rows.append(
            {
                "channel_index": index,
                "channel": channel,
                "mne_topomap_x": float(xy[0]),
                "mne_topomap_y": float(xy[1]),
                "distance_from_center": distance,
                "inside_head_sphere": bool(distance <= radius),
            }
        )
    return rows


def _plot_mne_topomap(
    *,
    data: np.ndarray,
    pos: np.ndarray,
    sphere: np.ndarray,
    names: Sequence[str] | None,
    basename: Path,
    formats: Iterable[str],
    title: str,
    cmap: str,
    colorbar_label: str,
    symmetric: bool,
) -> list[Path]:
    data = np.asarray(data, dtype=float)
    if symmetric:
        limit = max(float(np.nanmax(np.abs(data))), 1e-12)
        vlim = (-limit, limit)
    else:
        upper = max(float(np.nanmax(data)), 1e-12)
        vlim = (0.0, upper)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )
    fig, ax = plt.subplots(figsize=(3.2, 3.4), constrained_layout=True)
    image, _ = mne.viz.plot_topomap(
        data,
        pos,
        axes=ax,
        show=False,
        sensors=True,
        names=names,
        contours=6,
        outlines="head",
        sphere=sphere,
        extrapolate="head",
        border="mean",
        image_interp="cubic",
        cmap=cmap,
        vlim=vlim,
    )
    ax.set_title(title, fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)

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


def _write_contact_sheet(out_dir: Path, generated: Sequence[dict[str, str]]) -> Path:
    png_paths: list[tuple[str, Path]] = []
    for row in generated:
        paths = [Path(value) for value in str(row.get("paths", "")).split(";") if value]
        png = next((path for path in paths if path.suffix.lower() == ".png" and path.exists()), None)
        if png is None:
            continue
        label = f"{row.get('kind', '')} {row.get('state', '')} {row.get('band', '')}".strip()
        png_paths.append((label, png))

    if not png_paths:
        raise ValueError("No PNG topomap exports were available for the contact sheet.")

    columns = 4
    rows = int(np.ceil(len(png_paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.35), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(rows, columns)
    for ax in axes_array.ravel():
        ax.axis("off")

    for ax, (label, path) in zip(axes_array.ravel(), png_paths):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    output = out_dir / "mne_topomap_contact_sheet.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def _channel_values(frame: pd.DataFrame, *, value_column: str) -> dict[str, float]:
    if frame.empty:
        return {}
    tmp = frame.copy()
    tmp["channel_norm"] = tmp["channel"].map(normalize_channel_name)
    grouped = tmp.groupby("channel_norm", as_index=True)[value_column].mean()
    return {str(index): float(value) for index, value in grouped.items()}


def _values_for_order(values: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [values.get(normalize_channel_name(channel), 0.0) for channel in CANONICAL_CHANNELS_62],
        dtype=float,
    )


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    main()
