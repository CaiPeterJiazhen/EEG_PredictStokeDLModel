from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def test_eeglab_topomap_layout_preserves_head_orientation_and_fills_circle(tmp_path: Path) -> None:
    script = _load_mne_topomap_script()
    source = _write_synthetic_ced(tmp_path)

    names, pos = script._load_ced_eeglab_topomap_layout(
        source,
        ("FPZ", "OZ", "T7", "T8", "CB1", "CB2"),
    )
    coords = dict(zip(names, pos))
    radii = np.linalg.norm(pos, axis=1)

    assert coords["Fpz"][1] > coords["T7"][1]
    assert coords["Oz"][1] < coords["T7"][1]
    assert coords["T7"][0] < coords["Fpz"][0]
    assert coords["T8"][0] > coords["Fpz"][0]
    assert coords["CB1"][0] < 0
    assert coords["CB2"][0] > 0
    assert coords["CB1"][1] < coords["Oz"][1]
    assert coords["CB2"][1] < coords["Oz"][1]
    assert radii.max() <= 0.98
    assert radii.max() >= 0.94


def _write_synthetic_ced(tmp_path: Path) -> Path:
    source = tmp_path / "synthetic_standard_1005.ced"
    source.write_text(
        "\n".join(
            [
                "1 Fpz 0 0.5 1.0 0.0 0.0 0 0 1 EEG",
                "2 Oz 180 0.5 -1.0 0.0 0.0 0 0 1 EEG",
                "3 T7 -90 0.5 0.0 1.0 0.0 0 0 1 EEG",
                "4 T8 90 0.5 0.0 -1.0 0.0 0 0 1 EEG",
                "5 CB1 -150 0.6 -1.2 0.3 -0.1 0 0 1 EEG",
                "6 CB2 150 0.6 -1.2 -0.3 -0.1 0 0 1 EEG",
            ]
        ),
        encoding="utf-8",
    )
    return source


def _load_mne_topomap_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "45_make_mne_explainability_topomaps.py"
    spec = importlib.util.spec_from_file_location("make_mne_explainability_topomaps", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
