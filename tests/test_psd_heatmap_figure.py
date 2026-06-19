from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_psd_eo_channel_frequency_heatmap.py"
_SPEC = spec_from_file_location("make_psd_eo_channel_frequency_heatmap", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
compute_z_difference = _MODULE.compute_z_difference
format_frequency_tick = _MODULE.format_frequency_tick
robust_symmetric_limit = _MODULE.robust_symmetric_limit


def test_compute_z_difference_uses_healthy_standard_deviation() -> None:
    stroke = np.array([[3.0, 6.0], [9.0, 12.0]], dtype=float)
    healthy = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    healthy_std = np.array([[2.0, 2.0], [3.0, 4.0]], dtype=float)

    z = compute_z_difference(stroke, healthy, healthy_std)

    np.testing.assert_allclose(z, np.array([[1.0, 2.0], [2.0, 2.0]], dtype=float))


def test_robust_symmetric_limit_returns_positive_percentile_bound() -> None:
    matrix = np.array(
        [
            [-1.0, 0.0, 1.0],
            [2.0, -2.0, 100.0],
        ],
        dtype=float,
    )

    limit = robust_symmetric_limit(matrix, percentile=90.0)

    assert limit > 0.0
    assert limit < 100.0


def test_format_frequency_tick_preserves_tens() -> None:
    assert format_frequency_tick(0.5) == "0.5"
    assert format_frequency_tick(10.0) == "10"
    assert format_frequency_tick(40.0) == "40"
