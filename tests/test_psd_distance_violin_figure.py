from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_psd_eo_distance_violin_boxplot.py"
_SPEC = spec_from_file_location("make_psd_eo_distance_violin_boxplot", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
extract_psd_eo_overall_distances = _MODULE.extract_psd_eo_overall_distances
extract_psd_eo_overall_stats = _MODULE.extract_psd_eo_overall_stats
format_p_value = _MODULE.format_p_value


def test_extract_psd_eo_overall_distances_filters_only_requested_groups() -> None:
    distances = pd.DataFrame(
        {
            "subject_id": ["h1", "p1", "p2", "p3", "p4"],
            "group": ["healthy", "patient", "patient", "patient", "patient"],
            "timepoint": ["baseline", "baseline", "post14", "baseline", "baseline"],
            "modality": ["psd", "psd", "psd", "wpli", "psd"],
            "state": ["EO", "EO", "EO", "EO", "EC"],
            "level": ["overall", "overall", "overall", "overall", "overall"],
            "distance_to_health": [0.8, 3.2, 2.1, 4.0, 1.5],
        }
    )

    frame = extract_psd_eo_overall_distances(distances)

    assert frame["group_label"].tolist() == ["Healthy", "Stroke baseline"]
    assert frame["distance_to_health"].tolist() == [0.8, 3.2]


def test_extract_psd_eo_overall_stats_reads_target_feature_id() -> None:
    stats = pd.DataFrame(
        {
            "feature_id": ["distance|psd|EO|overall"],
            "p_value": [1e-6],
            "q_value": [0.005793988878876868],
            "effect_size": [0.9761904761904763],
            "effect_size_name": ["rank_biserial"],
            "test": ["mannwhitney"],
        }
    )

    result = extract_psd_eo_overall_stats(stats)

    assert result.q_value == 0.005793988878876868
    assert result.effect_size == 0.9761904761904763


def test_format_p_value_uses_compact_scientific_notation_for_tiny_values() -> None:
    assert format_p_value(0.0057939) == "0.0058"
    assert format_p_value(0.00042) == "4.2e-04"
