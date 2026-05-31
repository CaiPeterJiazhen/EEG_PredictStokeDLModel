from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class ImaginaryCoherenceSummaryVector:
    values: np.ndarray
    feature_names: tuple[str, ...]


def compute_subject_imaginary_coherence_summary_features(
    output_root: str | Path,
    subject_id: str,
) -> ImaginaryCoherenceSummaryVector:
    root = Path(output_root) / "data" / "features" / "fc"
    eo = _load_npz(root / f"{subject_id}_EO_fc.npz")
    ec = _load_npz(root / f"{subject_id}_EC_fc.npz")
    return compute_imaginary_coherence_summary_from_payloads(
        imaginary_coherence_eo=np.asarray(eo["imaginary_coherence"], dtype=np.float32),
        imaginary_coherence_ec=np.asarray(ec["imaginary_coherence"], dtype=np.float32),
        edge_list=np.asarray(eo["edge_list"]).astype(str),
        band_names=np.asarray(eo["band_names"]).astype(str),
    )


def compute_imaginary_coherence_summary_from_payloads(
    *,
    imaginary_coherence_eo: np.ndarray,
    imaginary_coherence_ec: np.ndarray,
    edge_list: np.ndarray,
    band_names: np.ndarray,
) -> ImaginaryCoherenceSummaryVector:
    eo = np.asarray(imaginary_coherence_eo, dtype=np.float32)
    ec = np.asarray(imaginary_coherence_ec, dtype=np.float32)
    edges = np.asarray(edge_list).astype(str)
    bands = tuple(_safe_feature_token(name) for name in np.asarray(band_names).astype(str))
    _validate_payloads(eo, ec, edges, bands)

    values: list[float] = []
    names: list[str] = []
    by_state = {
        "eo": np.abs(eo),
        "ec": np.abs(ec),
        "abs_delta": np.abs(ec - eo),
    }
    for state, matrix in by_state.items():
        for band_index, band_name in enumerate(bands):
            band_values = np.asarray(matrix[:, band_index], dtype=np.float32)
            for summary_name, summary_value in _edge_distribution_summary(band_values, edges):
                names.append(f"imagcoh_{state}_{band_name}_{summary_name}")
                values.append(summary_value)
            for summary_name, summary_value in _node_strength_summary(band_values, edges):
                names.append(f"imagcoh_{state}_{band_name}_{summary_name}")
                values.append(summary_value)

    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Imaginary coherence summary features contain non-finite values.")
    return ImaginaryCoherenceSummaryVector(values=vector, feature_names=tuple(names))


def _edge_distribution_summary(
    band_values: np.ndarray,
    edge_list: np.ndarray,
) -> tuple[tuple[str, float], ...]:
    left_intra: list[float] = []
    right_intra: list[float] = []
    interhemispheric: list[float] = []
    for edge_index, (raw_a, raw_b) in enumerate(np.asarray(edge_list).astype(str)):
        if edge_index >= len(band_values):
            break
        value = float(band_values[edge_index])
        hemisphere_a = _channel_hemisphere(raw_a)
        hemisphere_b = _channel_hemisphere(raw_b)
        if hemisphere_a == "left" and hemisphere_b == "left":
            left_intra.append(value)
        elif hemisphere_a == "right" and hemisphere_b == "right":
            right_intra.append(value)
        elif hemisphere_a is not None and hemisphere_b is not None and hemisphere_a != hemisphere_b:
            interhemispheric.append(value)
    left_mean = _finite_mean(left_intra)
    right_mean = _finite_mean(right_intra)
    inter_mean = _finite_mean(interhemispheric)
    return (
        ("global_abs_mean", _finite_mean(band_values.tolist())),
        ("global_abs_std", _finite_std(band_values.tolist())),
        ("ipsilesional_intra_abs_mean", left_mean),
        ("contralesional_intra_abs_mean", right_mean),
        ("interhemispheric_abs_mean", inter_mean),
        ("intra_signed_asymmetry", _signed_asymmetry(left_mean, right_mean)),
    )


def _node_strength_summary(
    band_values: np.ndarray,
    edge_list: np.ndarray,
) -> tuple[tuple[str, float], ...]:
    adjacency, node_names = _weighted_adjacency_from_edges(band_values, edge_list)
    if adjacency.shape[0] < 2:
        return (
            ("global_strength_mean", 0.0),
            ("ipsilesional_strength_mean", 0.0),
            ("contralesional_strength_mean", 0.0),
            ("strength_signed_asymmetry", 0.0),
            ("ipsilesional_motor_strength_mean", 0.0),
            ("contralesional_motor_strength_mean", 0.0),
            ("motor_strength_signed_asymmetry", 0.0),
        )
    strength = adjacency.sum(axis=1)
    ipsi_indices = _indices_by_hemisphere(node_names, "left")
    contra_indices = _indices_by_hemisphere(node_names, "right")
    ipsi_strength = _finite_mean(strength[ipsi_indices].tolist()) if ipsi_indices.size else 0.0
    contra_strength = _finite_mean(strength[contra_indices].tolist()) if contra_indices.size else 0.0
    ipsi_motor_indices = _motor_indices(node_names, "left")
    contra_motor_indices = _motor_indices(node_names, "right")
    ipsi_motor = _finite_mean(strength[ipsi_motor_indices].tolist()) if ipsi_motor_indices.size else 0.0
    contra_motor = _finite_mean(strength[contra_motor_indices].tolist()) if contra_motor_indices.size else 0.0
    return (
        ("global_strength_mean", _finite_mean(strength.tolist())),
        ("ipsilesional_strength_mean", ipsi_strength),
        ("contralesional_strength_mean", contra_strength),
        ("strength_signed_asymmetry", _signed_asymmetry(ipsi_strength, contra_strength)),
        ("ipsilesional_motor_strength_mean", ipsi_motor),
        ("contralesional_motor_strength_mean", contra_motor),
        ("motor_strength_signed_asymmetry", _signed_asymmetry(ipsi_motor, contra_motor)),
    )


def _weighted_adjacency_from_edges(
    band_values: np.ndarray,
    edge_list: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    edges = np.asarray(edge_list).astype(str)
    node_names = tuple(sorted(set(edges.ravel().tolist())))
    node_index = {name: index for index, name in enumerate(node_names)}
    adjacency = np.zeros((len(node_names), len(node_names)), dtype=np.float64)
    for edge_index, (raw_a, raw_b) in enumerate(edges):
        if edge_index >= len(band_values):
            break
        value = float(band_values[edge_index])
        if not np.isfinite(value):
            value = 0.0
        value = abs(value)
        a = node_index[raw_a]
        b = node_index[raw_b]
        adjacency[a, b] = value
        adjacency[b, a] = value
    return adjacency, node_names


def _indices_by_hemisphere(node_names: tuple[str, ...], hemisphere: str) -> np.ndarray:
    return np.asarray(
        [index for index, name in enumerate(node_names) if _channel_hemisphere(name) == hemisphere],
        dtype=np.int64,
    )


def _motor_indices(node_names: tuple[str, ...], hemisphere: str) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, name in enumerate(node_names)
            if _channel_hemisphere(name) == hemisphere and _is_motor_area_channel(name)
        ],
        dtype=np.int64,
    )


def _is_motor_area_channel(label: str) -> bool:
    normalized = str(label).upper()
    return normalized.startswith(("C", "FC", "CP"))


def _channel_hemisphere(label: str) -> str | None:
    match = re.search(r"(\d+)$", str(label).upper())
    if not match:
        return None
    return "left" if int(match.group(1)) % 2 == 1 else "right"


def _signed_asymmetry(ipsilesional_value: float, contralesional_value: float, eps: float = 1e-8) -> float:
    return float(
        (float(ipsilesional_value) - float(contralesional_value))
        / (abs(float(ipsilesional_value)) + abs(float(contralesional_value)) + eps)
    )


def _finite_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.nanmean(np.asarray(values, dtype=np.float32)))
    return mean if np.isfinite(mean) else 0.0


def _finite_std(values: list[float]) -> float:
    if not values:
        return 0.0
    std = float(np.nanstd(np.asarray(values, dtype=np.float32)))
    return std if np.isfinite(std) else 0.0


def _validate_payloads(eo: np.ndarray, ec: np.ndarray, edge_list: np.ndarray, band_names: tuple[str, ...]) -> None:
    if eo.shape != ec.shape:
        raise ValueError("EO and EC imaginary coherence arrays must have the same shape.")
    if eo.ndim != 2:
        raise ValueError("imaginary coherence arrays must be two-dimensional.")
    if edge_list.shape != (eo.shape[0], 2):
        raise ValueError("edge_list must have shape (n_edges, 2).")
    if len(band_names) != eo.shape[1]:
        raise ValueError("band_names length must match imaginary coherence band dimension.")
    if not np.isfinite(eo).all() or not np.isfinite(ec).all():
        raise ValueError("imaginary coherence arrays contain NaN or infinite values.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Required FC feature file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "imaginary_coherence" not in payload.files:
            raise KeyError(f"FC feature file is missing imaginary_coherence: {path}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def _safe_feature_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "band"
