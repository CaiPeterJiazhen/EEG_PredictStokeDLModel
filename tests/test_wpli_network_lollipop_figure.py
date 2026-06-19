from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_wpli_ec_network_lollipop.py"
_SPEC = spec_from_file_location("make_wpli_ec_network_lollipop", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
format_network_label = _MODULE.format_network_label
point_sizes = _MODULE.point_sizes
select_wpli_ec_network_rows = _MODULE.select_wpli_ec_network_rows


def test_select_wpli_ec_network_rows_prefers_requested_five_networks() -> None:
    rows = []
    for network, effect in [
        ("central", 0.93),
        ("central|frontal", 0.92),
        ("central|parietal", 0.91),
        ("central|temporal", 0.90),
        ("frontal|occipital", 0.89),
        ("frontal|parietal", 0.99),
    ]:
        rows.append(
            {
                "modality": "wpli",
                "state": "EC",
                "level": "network",
                "network_group": network,
                "q_value": 0.004,
                "effect_size": effect,
                "effect_size_name": "rank_biserial",
            }
        )

    selected = select_wpli_ec_network_rows(pd.DataFrame(rows))

    assert set(selected["network_group"]) == {
        "central",
        "central|frontal",
        "central|parietal",
        "central|temporal",
        "frontal|occipital",
    }
    assert "frontal|parietal" not in set(selected["network_group"])


def test_format_network_label_replaces_pipe_with_en_dash() -> None:
    assert format_network_label("central|frontal") == "central–frontal"


def test_point_sizes_are_constant_when_q_values_are_equal() -> None:
    sizes = point_sizes(np.array([2.0, 2.0, 2.0], dtype=float))

    assert sizes.tolist() == [115.0, 115.0, 115.0]
