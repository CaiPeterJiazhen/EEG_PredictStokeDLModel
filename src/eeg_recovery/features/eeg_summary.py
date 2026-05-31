from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


PSD_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)

ROI_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("frontal", ("FP", "AF", "F", "FC", "FT")),
    ("central", ("C", "CP")),
    ("temporal", ("T", "TP")),
    ("parietal", ("P", "PO")),
    ("occipital", ("O", "CB")),
)


@dataclass(frozen=True)
class EEGSummaryFeatureVector:
    values: np.ndarray
    feature_names: tuple[str, ...]


def compute_subject_eeg_summary_features(output_root: str | Path, subject_id: str) -> EEGSummaryFeatureVector:
    root = Path(output_root)
    psd_payloads = {
        state: _load_npz(root / "data" / "features" / "psd" / f"{subject_id}_{state}_psd.npz")
        for state in ("EO", "EC")
    }
    fc_payloads = {
        state: _load_npz(root / "data" / "features" / "fc" / f"{subject_id}_{state}_fc.npz")
        for state in ("EO", "EC")
    }
    return compute_eeg_summary_from_feature_payloads(
        psd_eo=np.asarray(psd_payloads["EO"]["psd"], dtype=np.float32),
        psd_ec=np.asarray(psd_payloads["EC"]["psd"], dtype=np.float32),
        frequency_bins=np.asarray(psd_payloads["EO"]["frequency_bins"], dtype=np.float32),
        channel_names=np.asarray(psd_payloads["EO"]["channel_names_after_alignment"]).astype(str),
        wpli_eo=np.asarray(fc_payloads["EO"]["wpli"], dtype=np.float32),
        wpli_ec=np.asarray(fc_payloads["EC"]["wpli"], dtype=np.float32),
        edge_list=np.asarray(fc_payloads["EO"]["edge_list"]).astype(str),
        band_names=np.asarray(fc_payloads["EO"]["band_names"]).astype(str),
    )


def compute_eeg_summary_from_feature_payloads(
    *,
    psd_eo: np.ndarray,
    psd_ec: np.ndarray,
    frequency_bins: np.ndarray,
    channel_names: np.ndarray,
    wpli_eo: np.ndarray,
    wpli_ec: np.ndarray,
    edge_list: np.ndarray,
    band_names: np.ndarray,
) -> EEGSummaryFeatureVector:
    values: list[float] = []
    names: list[str] = []
    psd_by_state = {"eo": psd_eo, "ec": psd_ec}
    for state, psd in psd_by_state.items():
        global_band_means: dict[str, float] = {}
        roi_band_means: dict[str, dict[str, float]] = {}
        total_power = np.asarray(psd[:, _frequency_mask(frequency_bins, 1.0, 45.0)], dtype=np.float32)
        total_power_mean = float(np.nanmean(total_power))
        roi_indices_by_name = _roi_channel_indices(channel_names)
        for summary_name, summary_value in _psd_spectral_shape_summary(psd, frequency_bins):
            names.append(f"psd_{state}_{summary_name}")
            values.append(summary_value)
        for roi_name, roi_indices in roi_indices_by_name:
            for summary_name, summary_value in _psd_spectral_shape_summary(psd[roi_indices], frequency_bins):
                names.append(f"psd_{state}_{roi_name}_{summary_name}")
                values.append(summary_value)
        for band_name, low, high in PSD_BANDS:
            mask = _frequency_mask(frequency_bins, low, high)
            band_values = psd[:, mask]
            band_mean = float(np.nanmean(band_values))
            global_band_means[band_name] = band_mean
            names.append(f"psd_{state}_{band_name}_mean")
            values.append(band_mean)
            names.append(f"psd_{state}_{band_name}_bsi")
            values.append(float(_brain_symmetry_index(band_values, channel_names)))
            names.append(f"psd_{state}_{band_name}_relative_mean")
            values.append(_safe_ratio(band_mean, total_power_mean))
            for summary_name, summary_value in _lesion_aligned_psd_summary(band_values, channel_names):
                names.append(f"psd_{state}_{band_name}_{summary_name}")
                values.append(summary_value)
            for roi_name, roi_indices in roi_indices_by_name:
                roi_values = band_values[roi_indices]
                roi_mean = float(np.nanmean(roi_values))
                roi_band_means.setdefault(roi_name, {})[band_name] = roi_mean
                names.append(f"psd_{state}_{roi_name}_{band_name}_mean")
                values.append(roi_mean)
                names.append(f"psd_{state}_{roi_name}_{band_name}_bsi")
                values.append(float(_brain_symmetry_index(roi_values, channel_names[roi_indices])))
                roi_total_mean = float(np.nanmean(total_power[roi_indices]))
                names.append(f"psd_{state}_{roi_name}_{band_name}_relative_mean")
                values.append(_safe_ratio(roi_mean, roi_total_mean))
                for summary_name, summary_value in _lesion_aligned_psd_summary(roi_values, channel_names[roi_indices]):
                    names.append(f"psd_{state}_{roi_name}_{band_name}_{summary_name}")
                    values.append(summary_value)
        for ratio_name, numerator, denominator in _stroke_band_ratios():
            names.append(f"psd_{state}_{ratio_name}")
            values.append(_safe_ratio(global_band_means[numerator], global_band_means[denominator]))
            for roi_name in roi_band_means:
                names.append(f"psd_{state}_{roi_name}_{ratio_name}")
                values.append(_safe_ratio(roi_band_means[roi_name][numerator], roi_band_means[roi_name][denominator]))
        names.append(f"psd_{state}_delta_theta_alpha_beta_ratio")
        values.append(
            _safe_ratio(
                global_band_means["delta"] + global_band_means["theta"],
                global_band_means["alpha"] + global_band_means["beta"],
            )
        )
        for roi_name in roi_band_means:
            names.append(f"psd_{state}_{roi_name}_delta_theta_alpha_beta_ratio")
            values.append(
                _safe_ratio(
                    roi_band_means[roi_name]["delta"] + roi_band_means[roi_name]["theta"],
                    roi_band_means[roi_name]["alpha"] + roi_band_means[roi_name]["beta"],
                )
            )

    wpli_by_state = {"eo": wpli_eo, "ec": wpli_ec}
    for band_index, raw_band_name in enumerate(band_names):
        band_name = _safe_feature_token(raw_band_name)
        eo_values = wpli_eo[:, band_index]
        ec_values = wpli_ec[:, band_index]
        names.append(f"wpli_eo_{band_name}_mean")
        values.append(float(np.nanmean(eo_values)))
        names.append(f"wpli_ec_{band_name}_mean")
        values.append(float(np.nanmean(ec_values)))
        names.append(f"wpli_abs_delta_{band_name}_mean")
        values.append(float(np.nanmean(np.abs(ec_values - eo_values))))
        for state, wpli in wpli_by_state.items():
            band_values = wpli[:, band_index]
            graph_values = _wpli_graph_summary(band_values, edge_list)
            for graph_name, graph_value in graph_values:
                names.append(f"wpli_{state}_{band_name}_{graph_name}")
                values.append(graph_value)
            theory_values = _wpli_graph_theory_summary(band_values, edge_list)
            for graph_name, graph_value in theory_values:
                names.append(f"wpli_{state}_{band_name}_{graph_name}")
                values.append(graph_value)

    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("EEG summary features contain non-finite values.")
    return EEGSummaryFeatureVector(values=vector, feature_names=tuple(names))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Required EEG summary source file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _frequency_mask(frequency_bins: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequency_bins >= low) & (frequency_bins < high)
    if not mask.any():
        raise ValueError(f"No PSD frequency bins are available for band [{low}, {high}).")
    return mask


def _stroke_band_ratios() -> tuple[tuple[str, str, str], ...]:
    return (
        ("delta_alpha_ratio", "delta", "alpha"),
        ("delta_theta_ratio", "delta", "theta"),
        ("theta_beta_ratio", "theta", "beta"),
    )


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    value = float(numerator) / (float(denominator) + eps)
    if not np.isfinite(value):
        return 0.0
    return value


def _psd_spectral_shape_summary(
    psd_values: np.ndarray,
    frequency_bins: np.ndarray,
    *,
    low: float = 1.0,
    high: float = 45.0,
    eps: float = 1e-12,
) -> tuple[tuple[str, float], ...]:
    mask = _frequency_mask(frequency_bins, low, high)
    frequencies = np.asarray(frequency_bins[mask], dtype=np.float64)
    spectrum = np.nanmean(np.asarray(psd_values[:, mask], dtype=np.float64), axis=0)
    spectrum = np.clip(spectrum, 0.0, None)
    total = float(np.sum(spectrum))
    if total <= eps or not np.isfinite(total):
        return (
            ("spectral_entropy", 0.0),
            ("spectral_centroid_hz", 0.0),
            ("spectral_spread_hz", 0.0),
            ("spectral_edge_95_hz", 0.0),
        )
    probability = spectrum / total
    entropy = -float(np.sum(probability * np.log(probability + eps))) / float(np.log(len(probability) + eps))
    centroid = float(np.sum(probability * frequencies))
    spread = float(np.sqrt(np.sum(probability * (frequencies - centroid) ** 2)))
    cumulative = np.cumsum(probability)
    edge_index = int(np.searchsorted(cumulative, 0.95, side="left"))
    edge_index = min(edge_index, len(frequencies) - 1)
    edge = float(frequencies[edge_index])
    return (
        ("spectral_entropy", float(np.clip(entropy, 0.0, 1.0))),
        ("spectral_centroid_hz", centroid),
        ("spectral_spread_hz", spread),
        ("spectral_edge_95_hz", edge),
    )


def _brain_symmetry_index(band_values: np.ndarray, channel_names: np.ndarray, eps: float = 1e-8) -> float:
    channel_lookup = {str(name).upper(): index for index, name in enumerate(channel_names)}
    numerators = []
    denominators = []
    for left_label, right_label in _left_right_channel_pairs(channel_lookup):
        left = band_values[channel_lookup[left_label]]
        right = band_values[channel_lookup[right_label]]
        numerators.append(np.abs(left - right).mean())
        denominators.append((np.abs(left) + np.abs(right)).mean())
    if not numerators:
        return 0.0
    return float(np.sum(numerators) / (np.sum(denominators) + eps))


def _lesion_aligned_psd_summary(
    band_values: np.ndarray,
    channel_names: np.ndarray,
) -> tuple[tuple[str, float], ...]:
    ipsi_indices, contra_indices = _hemisphere_channel_indices(channel_names)
    ipsi_mean = _finite_mean_or_zero(np.asarray(band_values)[ipsi_indices].ravel().tolist()) if ipsi_indices.size else 0.0
    contra_mean = (
        _finite_mean_or_zero(np.asarray(band_values)[contra_indices].ravel().tolist())
        if contra_indices.size
        else 0.0
    )
    return (
        ("ipsilesional_mean", ipsi_mean),
        ("contralesional_mean", contra_mean),
        ("ipsi_contra_signed_asymmetry", _signed_asymmetry(ipsi_mean, contra_mean)),
    )


def _hemisphere_channel_indices(channel_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ipsi_indices = []
    contra_indices = []
    for index, label in enumerate(np.asarray(channel_names).astype(str)):
        hemisphere = _channel_hemisphere(label)
        if hemisphere == "left":
            ipsi_indices.append(index)
        elif hemisphere == "right":
            contra_indices.append(index)
    return np.asarray(ipsi_indices, dtype=np.int64), np.asarray(contra_indices, dtype=np.int64)


def _roi_channel_indices(channel_names: np.ndarray) -> list[tuple[str, np.ndarray]]:
    labels = np.asarray(channel_names).astype(str)
    result = []
    for roi_name, prefixes in ROI_PREFIXES:
        indices = [
            index
            for index, label in enumerate(labels)
            if any(str(label).upper().startswith(prefix) for prefix in prefixes)
        ]
        if indices:
            result.append((roi_name, np.asarray(indices, dtype=np.int64)))
    return result


def _wpli_graph_summary(
    band_values: np.ndarray,
    edge_list: np.ndarray,
    eps: float = 1e-8,
) -> tuple[tuple[str, float], ...]:
    left_intra: list[float] = []
    right_intra: list[float] = []
    interhemispheric: list[float] = []
    for edge_index, (raw_a, raw_b) in enumerate(np.asarray(edge_list).astype(str)):
        hemisphere_a = _channel_hemisphere(raw_a)
        hemisphere_b = _channel_hemisphere(raw_b)
        if hemisphere_a is None or hemisphere_b is None:
            continue
        value = float(band_values[edge_index])
        if hemisphere_a == "left" and hemisphere_b == "left":
            left_intra.append(value)
        elif hemisphere_a == "right" and hemisphere_b == "right":
            right_intra.append(value)
        elif hemisphere_a != hemisphere_b:
            interhemispheric.append(value)
    left_mean = _finite_mean_or_zero(left_intra)
    right_mean = _finite_mean_or_zero(right_intra)
    inter_mean = _finite_mean_or_zero(interhemispheric)
    return (
        ("left_intra_mean", left_mean),
        ("right_intra_mean", right_mean),
        ("interhemispheric_mean", inter_mean),
        ("intra_asymmetry", abs(left_mean - right_mean) / (abs(left_mean) + abs(right_mean) + eps)),
        ("ipsilesional_intra_mean", left_mean),
        ("contralesional_intra_mean", right_mean),
        ("intra_signed_asymmetry", _signed_asymmetry(left_mean, right_mean, eps=eps)),
    )


def _wpli_graph_theory_summary(
    band_values: np.ndarray,
    edge_list: np.ndarray,
) -> tuple[tuple[str, float], ...]:
    adjacency, node_names = _weighted_adjacency_from_edges(band_values, edge_list)
    if adjacency.shape[0] < 2:
        return (
            ("global_strength_mean", 0.0),
            ("global_efficiency", 0.0),
            ("weighted_clustering", 0.0),
            ("ipsilesional_strength_mean", 0.0),
            ("contralesional_strength_mean", 0.0),
            ("strength_signed_asymmetry", 0.0),
        )
    strength = adjacency.sum(axis=1)
    ipsi_indices = np.asarray(
        [index for index, name in enumerate(node_names) if _channel_hemisphere(name) == "left"],
        dtype=np.int64,
    )
    contra_indices = np.asarray(
        [index for index, name in enumerate(node_names) if _channel_hemisphere(name) == "right"],
        dtype=np.int64,
    )
    ipsi_strength = _finite_mean_or_zero(strength[ipsi_indices].tolist()) if ipsi_indices.size else 0.0
    contra_strength = _finite_mean_or_zero(strength[contra_indices].tolist()) if contra_indices.size else 0.0
    return (
        ("global_strength_mean", _finite_mean_or_zero(strength.tolist())),
        ("global_efficiency", _weighted_global_efficiency(adjacency)),
        ("weighted_clustering", _weighted_clustering_coefficient(adjacency)),
        ("ipsilesional_strength_mean", ipsi_strength),
        ("contralesional_strength_mean", contra_strength),
        ("strength_signed_asymmetry", _signed_asymmetry(ipsi_strength, contra_strength)),
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
        a = node_index[raw_a]
        b = node_index[raw_b]
        value = float(band_values[edge_index])
        if not np.isfinite(value):
            value = 0.0
        value = max(value, 0.0)
        adjacency[a, b] = value
        adjacency[b, a] = value
    return adjacency, node_names


def _weighted_global_efficiency(adjacency: np.ndarray, eps: float = 1e-8) -> float:
    n_nodes = adjacency.shape[0]
    if n_nodes < 2:
        return 0.0
    distances = np.full_like(adjacency, np.inf, dtype=np.float64)
    connected = adjacency > 0
    distances[connected] = 1.0 / (adjacency[connected] + eps)
    np.fill_diagonal(distances, 0.0)
    for via in range(n_nodes):
        distances = np.minimum(distances, distances[:, [via]] + distances[[via], :])
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = 1.0 / distances
    efficiency[~np.isfinite(efficiency)] = 0.0
    np.fill_diagonal(efficiency, 0.0)
    return float(efficiency.sum() / (n_nodes * (n_nodes - 1)))


def _weighted_clustering_coefficient(adjacency: np.ndarray) -> float:
    n_nodes = adjacency.shape[0]
    if n_nodes < 3:
        return 0.0
    max_weight = float(np.nanmax(adjacency))
    if max_weight <= 0 or not np.isfinite(max_weight):
        return 0.0
    normalized = np.clip(adjacency / max_weight, 0.0, 1.0)
    transformed = np.cbrt(normalized)
    triangles = np.diag(transformed @ transformed @ transformed)
    degrees = (adjacency > 0).sum(axis=1)
    valid = degrees >= 2
    if not valid.any():
        return 0.0
    clustering = np.zeros(n_nodes, dtype=np.float64)
    clustering[valid] = triangles[valid] / (degrees[valid] * (degrees[valid] - 1))
    return _finite_mean_or_zero(clustering[valid].tolist())


def _channel_hemisphere(label: str) -> str | None:
    match = re.search(r"(\d+)$", str(label).upper())
    if not match:
        return None
    number = int(match.group(1))
    if number % 2 == 1:
        return "left"
    return "right"


def _finite_mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.nanmean(np.asarray(values, dtype=np.float32)))
    if not np.isfinite(mean):
        return 0.0
    return mean


def _signed_asymmetry(ipsilesional_value: float, contralesional_value: float, eps: float = 1e-8) -> float:
    return float(
        (float(ipsilesional_value) - float(contralesional_value))
        / (abs(float(ipsilesional_value)) + abs(float(contralesional_value)) + eps)
    )


def _left_right_channel_pairs(channel_lookup: dict[str, int]) -> list[tuple[str, str]]:
    pairs = []
    for label in channel_lookup:
        match = re.match(r"^([A-Z]+)(\d+)$", label)
        if not match:
            continue
        prefix, number_text = match.groups()
        number = int(number_text)
        if number % 2 != 1:
            continue
        right_label = f"{prefix}{number + 1}"
        if right_label in channel_lookup:
            pairs.append((label, right_label))
    return sorted(set(pairs))


def _safe_feature_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "band"
