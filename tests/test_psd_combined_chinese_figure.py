from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_psd_eo_combined_chinese_figure.py"
_SPEC = spec_from_file_location("make_psd_eo_combined_chinese_figure", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
compute_z_difference = _MODULE.compute_z_difference
extract_psd_eo_overall_distances = _MODULE.extract_psd_eo_overall_distances
format_frequency_tick = _MODULE.format_frequency_tick


def test_extract_psd_eo_overall_distances_uses_chinese_group_labels() -> None:
    distances = pd.DataFrame(
        {
            "subject_id": ["h1", "p1", "p2"],
            "group": ["healthy", "patient", "patient"],
            "timepoint": ["baseline", "baseline", "post14"],
            "modality": ["psd", "psd", "psd"],
            "state": ["EO", "EO", "EO"],
            "level": ["overall", "overall", "overall"],
            "distance_to_health": [0.9, 3.1, 2.0],
        }
    )

    frame = extract_psd_eo_overall_distances(distances)

    assert frame["group_label"].tolist() == ["Healthy", "Stroke baseline"]


def test_compute_z_difference_handles_zero_healthy_std() -> None:
    stroke = np.array([[3.0, 6.0]], dtype=float)
    healthy = np.array([[1.0, 2.0]], dtype=float)
    healthy_std = np.array([[2.0, 0.0]], dtype=float)

    z = compute_z_difference(stroke, healthy, healthy_std)

    np.testing.assert_allclose(z, np.array([[1.0, 4.0]], dtype=float))


def test_format_frequency_tick_preserves_tens() -> None:
    assert format_frequency_tick(0.5) == "0.5"
    assert format_frequency_tick(10.0) == "10"
