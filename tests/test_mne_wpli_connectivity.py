from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def test_top_edges_for_plot_filters_band_state_and_sorts_by_abs_attribution() -> None:
    script = _load_connectivity_script()
    frame = pd.DataFrame(
        [
            {"state": "EC", "band": "Delta", "channel_i": "F3", "channel_j": "C3", "mean_abs_attribution": 0.2},
            {"state": "EC", "band": "Delta", "channel_i": "F4", "channel_j": "C4", "mean_abs_attribution": 0.5},
            {"state": "EC", "band": "Theta", "channel_i": "F7", "channel_j": "P7", "mean_abs_attribution": 0.9},
            {"state": "EO", "band": "Delta", "channel_i": "F8", "channel_j": "P8", "mean_abs_attribution": 0.8},
        ]
    )

    result = script._top_edges_for_plot(frame, state="EC", band="Delta", top_n=2)

    assert list(result["channel_i"]) == ["F4", "F3"]
    assert list(result["channel_j"]) == ["C4", "C3"]


def _load_connectivity_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "46_make_mne_wpli_connectivity.py"
    spec = importlib.util.spec_from_file_location("make_mne_wpli_connectivity", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
