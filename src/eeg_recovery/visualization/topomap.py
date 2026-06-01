from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


@dataclass(frozen=True)
class ElectrodeCoordinate:
    channel: str
    x: float
    y: float


def normalize_channel_name(channel: str) -> str:
    return str(channel).strip().upper()


def load_electrode_coordinates(path: str | Path) -> dict[str, ElectrodeCoordinate]:
    """Load 2D EEG coordinates from a .ced or simple .node electrode file."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    coords: dict[str, ElectrodeCoordinate] = {}
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.lower().startswith("62channels"):
                continue
            parts = line.split()
            parsed = _parse_coordinate_line(parts)
            if parsed is not None:
                coords[parsed.channel] = parsed
    if not coords:
        raise ValueError(f"No electrode coordinates could be parsed from {source}")
    return coords


def coordinates_for_channels(
    channel_order: Sequence[str],
    coordinates: Mapping[str, ElectrodeCoordinate],
) -> list[ElectrodeCoordinate]:
    ordered: list[ElectrodeCoordinate] = []
    missing: list[str] = []
    normalized = {normalize_channel_name(key): value for key, value in coordinates.items()}
    for channel in channel_order:
        key = normalize_channel_name(channel)
        if key not in normalized:
            missing.append(channel)
            continue
        coord = normalized[key]
        ordered.append(ElectrodeCoordinate(channel=key, x=float(coord.x), y=float(coord.y)))
    if missing:
        raise ValueError(f"Missing electrode coordinates for: {', '.join(missing)}")
    return ordered


def plot_topomap(
    coordinates: Sequence[ElectrodeCoordinate],
    values: Sequence[float],
    output_path: str | Path,
    *,
    title: str,
    cmap: str = "coolwarm",
) -> Path:
    coords = list(coordinates)
    values_array = np.asarray(values, dtype=float)
    if len(coords) != values_array.shape[0]:
        raise ValueError("coordinates and values must have the same length.")
    xy = np.asarray([(coord.x, coord.y) for coord in coords], dtype=float)
    xy = _normalize_xy(xy)
    grid_x, grid_y = np.mgrid[-1.05:1.05:180j, -1.05:1.05:180j]
    grid_z = griddata(xy, values_array, (grid_x, grid_y), method="cubic")
    if np.isnan(grid_z).all():
        grid_z = griddata(xy, values_array, (grid_x, grid_y), method="nearest")
    else:
        nearest = griddata(xy, values_array, (grid_x, grid_y), method="nearest")
        grid_z = np.where(np.isnan(grid_z), nearest, grid_z)
    scalp_mask = grid_x**2 + grid_y**2 <= 1.0
    grid_z = np.where(scalp_mask, grid_z, np.nan)
    max_abs = float(np.nanmax(np.abs(grid_z))) if np.isfinite(grid_z).any() else 1.0
    max_abs = max(max_abs, 1e-9)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.6, 5.6))
    ax = plt.gca()
    image = ax.imshow(
        grid_z.T,
        extent=(-1.05, 1.05, -1.05, 1.05),
        origin="lower",
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
    )
    circle = plt.Circle((0, 0), 1.0, color="black", fill=False, linewidth=1.2)
    ax.add_patch(circle)
    ax.scatter(xy[:, 0], xy[:, 1], s=10, c="black", alpha=0.75)
    for coord, (x, y) in zip(coords, xy):
        ax.text(x * 1.06, y * 1.06, coord.channel, fontsize=5, ha="center", va="center")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="importance")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def coordinate_validation_table(
    channel_order: Sequence[str],
    coordinates: Mapping[str, ElectrodeCoordinate],
) -> list[dict[str, object]]:
    normalized = {normalize_channel_name(key): value for key, value in coordinates.items()}
    rows = []
    for index, channel in enumerate(channel_order):
        key = normalize_channel_name(channel)
        coord = normalized.get(key)
        rows.append(
            {
                "channel_index": index,
                "channel": key,
                "has_coordinate": coord is not None,
                "x": np.nan if coord is None else float(coord.x),
                "y": np.nan if coord is None else float(coord.y),
            }
        )
    return rows


def _parse_coordinate_line(parts: list[str]) -> ElectrodeCoordinate | None:
    if len(parts) >= 10 and parts[-1].upper() in {"EEG", "FID"}:
        channel = normalize_channel_name(parts[1])
        if parts[-1].upper() != "EEG":
            return None
        try:
            return ElectrodeCoordinate(channel=channel, x=float(parts[5]), y=float(parts[6]))
        except ValueError:
            return None
    if len(parts) >= 5 and parts[0].upper() == "EEG":
        channel = normalize_channel_name(parts[1])
        try:
            return ElectrodeCoordinate(channel=channel, x=float(parts[2]), y=float(parts[3]))
        except ValueError:
            return None
    return None


def _normalize_xy(xy: np.ndarray) -> np.ndarray:
    centered = xy - np.nanmean(xy, axis=0, keepdims=True)
    scale = float(np.nanmax(np.linalg.norm(centered, axis=1)))
    if scale <= 0:
        raise ValueError("Electrode coordinates have zero spatial extent.")
    return centered / scale

