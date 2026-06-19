from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_wpli_ec_betahigh_clustermap.py"
_SPEC = spec_from_file_location("make_wpli_ec_betahigh_clustermap", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
edge_values_to_symmetric_matrix = _MODULE.edge_values_to_symmetric_matrix
robust_centered_limits = _MODULE.robust_centered_limits


def test_edge_values_to_symmetric_matrix_restores_upper_triangle_order() -> None:
    channels = ("A", "B", "C")
    edge_list = (("A", "B"), ("A", "C"), ("B", "C"))
    values = np.array([0.1, -0.2, 0.3], dtype=float)

    matrix = edge_values_to_symmetric_matrix(values, edge_list, channels)

    np.testing.assert_allclose(
        matrix,
        np.array(
            [
                [0.0, 0.1, -0.2],
                [0.1, 0.0, 0.3],
                [-0.2, 0.3, 0.0],
            ],
            dtype=float,
        ),
    )


def test_robust_centered_limits_keep_zero_between_color_bounds() -> None:
    matrix = np.array(
        [
            [0.0, 0.02, 0.08, 0.12],
            [0.02, 0.0, 0.05, 0.10],
            [0.08, 0.05, 0.0, -0.03],
            [0.12, 0.10, -0.03, 0.0],
        ],
        dtype=float,
    )

    vmin, vmax = robust_centered_limits(matrix)

    assert vmin < 0.0 < vmax
