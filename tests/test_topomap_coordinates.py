from __future__ import annotations

from pathlib import Path

import numpy as np

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.visualization.topomap import (
    coordinates_for_channels,
    load_electrode_coordinates,
    plot_topomap,
)


def test_topomap_coordinates_cover_canonical_62_without_mastoids(tmp_path: Path) -> None:
    source = tmp_path / "standard_1005_subset.ced"
    _write_synthetic_ced(source)

    coords = load_electrode_coordinates(source)
    ordered = coordinates_for_channels(CANONICAL_CHANNELS_62, coords)

    assert len(ordered) == 62
    assert [item.channel for item in ordered] == list(CANONICAL_CHANNELS_62)
    assert "M1" not in {item.channel for item in ordered}
    assert "M2" not in {item.channel for item in ordered}


def test_topomap_generation_writes_png(tmp_path: Path) -> None:
    source = tmp_path / "standard_1005_subset.ced"
    _write_synthetic_ced(source)
    ordered = coordinates_for_channels(CANONICAL_CHANNELS_62, load_electrode_coordinates(source))
    values = np.linspace(-1.0, 1.0, len(ordered))
    output = tmp_path / "topomap.png"

    plot_topomap(ordered, values, output, title="test topomap")

    assert output.exists()
    assert output.stat().st_size > 0


def _write_synthetic_ced(path: Path) -> None:
    rows = [
        "1\tM1\t0\t0\t0\t-1\t0\t0\t0\tEEG",
        "2\tM2\t0\t0\t0\t1\t0\t0\t0\tEEG",
    ]
    for index, channel in enumerate(CANONICAL_CHANNELS_62, start=3):
        angle = 2 * np.pi * (index - 3) / len(CANONICAL_CHANNELS_62)
        x = float(np.cos(angle))
        y = float(np.sin(angle))
        rows.append(f"{index}\t{channel.title()}\t0\t0\t0\t{x:.6f}\t{y:.6f}\t0\t0\tEEG")
    path.write_text("\n".join(rows), encoding="utf-8")

