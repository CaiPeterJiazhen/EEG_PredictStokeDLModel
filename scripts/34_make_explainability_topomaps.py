from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.config import load_path_config
from eeg_recovery.visualization.topomap import (
    coordinate_validation_table,
    coordinates_for_channels,
    load_electrode_coordinates,
    normalize_channel_name,
    plot_topomap,
)


PSD_BANDS = ("Delta", "Theta", "Alpha", "Beta Low", "Beta Medium", "Beta High")
WPLI_TARGETS = (
    ("EC", "Beta Medium"),
    ("EC", "Beta High"),
    ("EC", "Alpha"),
    ("EC", "Theta"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make scalp topomaps for locked explainability outputs.")
    parser.add_argument("--config", default="configs/paths.example.yaml")
    parser.add_argument("--coordinate-file", default=None)
    parser.add_argument("--node-file", default=None)
    args = parser.parse_args()

    config = load_path_config(args.config)
    output_root = config.output_root
    coordinate_path = Path(args.coordinate_file) if args.coordinate_file else config.standard_1005_ced
    node_path = Path(args.node_file) if args.node_file else None

    coordinates = load_electrode_coordinates(coordinate_path)
    if node_path is not None and node_path.exists():
        fallback = load_electrode_coordinates(node_path)
        coordinates = {**fallback, **coordinates}

    ordered = coordinates_for_channels(CANONICAL_CHANNELS_62, coordinates)
    out_dir = output_root / "results" / "figures" / "explainability" / "topomaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = coordinate_validation_table(CANONICAL_CHANNELS_62, coordinates)
    pd.DataFrame(validation).to_csv(out_dir / "electrode_coordinate_validation.csv", index=False)

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
            output = out_dir / f"psd_{state.lower()}_{_slug(band)}_signed_attribution_topomap.png"
            plot_topomap(
                ordered,
                _values_for_order(values),
                output,
                title=f"PSD {state} {band} attribution",
                cmap="coolwarm",
            )
            generated.append({"kind": "PSD", "state": state, "band": band, "path": str(output)})

    for state, band in WPLI_TARGETS:
        values = _channel_values(
            wpli[(wpli["state"] == state) & (wpli["band"] == band)],
            value_column="node_importance",
        )
        if not values:
            continue
        output = out_dir / f"wpli_{state.lower()}_{_slug(band)}_node_importance_topomap.png"
        plot_topomap(
            ordered,
            _values_for_order(values),
            output,
            title=f"WPLI {state} {band} node importance",
            cmap="viridis",
        )
        generated.append({"kind": "WPLI", "state": state, "band": band, "path": str(output)})

    manifest_path = out_dir / "topomap_manifest.csv"
    pd.DataFrame(generated).to_csv(manifest_path, index=False)
    _write_doc(output_root / "docs" / "explainability_topomap_notes.md", coordinate_path, node_path, generated)
    print(f"Wrote topomaps to {out_dir}")


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


def _write_doc(path: Path, coordinate_path: Path, node_path: Path | None, generated: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fallback = f" with fallback `{node_path}`" if node_path else ""
    text = f"""# Explainability Topomap Notes

Topomaps were generated from the locked explainability summaries without
retraining or changing the final model. Electrode coordinates came from
`{coordinate_path}`{fallback}. The canonical 62-channel order was validated and
M1/M2 mastoids were excluded from the plotting channel list.

The PSD maps use signed SmoothGrad-smoothed Integrated Gradients attribution.
The WPLI maps use node-level mean absolute edge attribution aggregated to each
channel.

All model inputs had already been affected-side aligned before feature
extraction. The displayed coordinate frame therefore follows the canonical
post-alignment channel order used by the CNN features; interpretation should be
made in that aligned frame rather than as unaligned native-lesion laterality.

Generated maps: {len(generated)}.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
