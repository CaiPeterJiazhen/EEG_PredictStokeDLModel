from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


BAND_DEFINITIONS: tuple[tuple[str, float, float], ...] = (
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta Low", 13.0, 18.0),
    ("Beta Medium", 18.0, 21.0),
    ("Beta High", 21.0, 30.0),
)

FeatureModality = Literal["psd", "wpli"]
FeatureTimepoint = Literal["baseline", "post14"]


@dataclass(frozen=True)
class HealthyReference:
    """Per-feature healthy reference center and scale."""

    center: np.ndarray
    scale: np.ndarray
    robust: bool = False


@dataclass(frozen=True)
class LoadedFeature:
    """Feature tensor and metadata loaded from an existing project npz output."""

    values: np.ndarray
    feature_ids: tuple[str, ...]
    subject_id: str
    group: str
    timepoint: str
    state: str
    modality: FeatureModality
    source_path: Path
    channel_names: tuple[str, ...] = ()
    frequency_bins: tuple[float, ...] = ()
    edge_list: tuple[tuple[str, str], ...] = ()
    band_names: tuple[str, ...] = ()


def build_healthy_reference(features_healthy: np.ndarray, robust: bool = False) -> HealthyReference:
    """Build a healthy template using per-feature mean/std or median/MAD.

    The first axis is interpreted as subject. Remaining axes are preserved, so
    PSD channel-frequency and wPLI edge-band tensors are supported without
    hard-coding their shapes.
    """

    values = np.asarray(features_healthy, dtype=float)
    if values.ndim < 2:
        raise ValueError("features_healthy must have subject x feature dimensions.")
    if values.shape[0] == 0:
        raise ValueError("At least one healthy subject is required.")
    if not np.isfinite(values).all():
        raise ValueError("features_healthy contains NaN or infinite values.")

    if robust:
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        scale = 1.4826 * mad
    else:
        center = values.mean(axis=0)
        ddof = 1 if values.shape[0] > 1 else 0
        scale = values.std(axis=0, ddof=ddof)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return HealthyReference(center=center.astype(float), scale=scale.astype(float), robust=robust)


def transform_to_healthy_zscore(x: np.ndarray, reference: HealthyReference) -> np.ndarray:
    """Transform a feature tensor to z-scores against the healthy reference."""

    values = np.asarray(x, dtype=float)
    if values.shape != reference.center.shape:
        raise ValueError(
            f"x shape {values.shape} does not match healthy reference shape {reference.center.shape}."
        )
    return (values - reference.center) / reference.scale


def compute_health_distance(
    z: np.ndarray,
    feature_axis: int | Sequence[int] | None = None,
) -> float | np.ndarray:
    """Return RMS z-distance, optionally over selected feature axes."""

    values = np.asarray(z, dtype=float)
    if values.size == 0:
        return float("nan")
    squared = np.square(values)
    if feature_axis is None:
        return float(np.sqrt(np.nanmean(squared)))
    return np.sqrt(np.nanmean(squared, axis=feature_axis))


def compute_normalization_index(D_pre: float | np.ndarray, D_post: float | np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
    """Return NI = (D_pre - D_post) / (D_pre + eps). Positive values indicate recovery toward healthy."""

    pre = np.asarray(D_pre, dtype=float)
    post = np.asarray(D_post, dtype=float)
    ni = (pre - post) / (pre + float(eps))
    if np.isscalar(D_pre) and np.isscalar(D_post):
        return float(ni)
    return ni


def resolve_feature_root(features_dir: str | Path) -> Path:
    """Resolve features-dir whether it is output_root, data/features, or features itself."""

    root = Path(features_dir)
    candidates = [
        root,
        root / "data" / "features",
        root / "features",
    ]
    for candidate in candidates:
        if (candidate / "psd").exists() or (candidate / "fc").exists():
            return candidate
    return root


def load_feature(
    subject_id: str,
    group: str,
    timepoint: str,
    state: str,
    modality: FeatureModality,
    features_dir: str | Path,
) -> LoadedFeature:
    """Load an existing PSD or wPLI feature tensor without assuming a fixed shape."""

    root = resolve_feature_root(features_dir)
    modality_dir = root / ("psd" if modality == "psd" else "fc")
    key = "psd" if modality == "psd" else "wpli"
    suffix = "psd" if modality == "psd" else "fc"
    path = _find_feature_file(modality_dir, subject_id, state, suffix, timepoint, group)
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"Feature file {path} is missing required key {key!r}.")
        values = np.asarray(payload[key], dtype=float)
        if values.ndim != 2:
            raise ValueError(f"{modality} feature must be 2D, got shape {values.shape} in {path}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{modality} feature contains non-finite values in {path}.")
        channel_names = _payload_strings(payload, "channel_names_after_alignment", values.shape[0])
        frequency_bins = tuple(float(value) for value in _payload_frequencies(payload, values.shape[1])) if modality == "psd" else ()
        edge_list = _payload_edges(payload, values.shape[0]) if modality == "wpli" else ()
        band_names = _payload_strings(payload, "band_names", values.shape[1]) if modality == "wpli" else ()

    feature_ids = _feature_ids(
        values=values,
        modality=modality,
        state=state,
        channel_names=channel_names,
        frequency_bins=frequency_bins,
        edge_list=edge_list,
        band_names=band_names,
    )
    return LoadedFeature(
        values=values,
        feature_ids=feature_ids,
        subject_id=subject_id,
        group=group,
        timepoint=timepoint,
        state=state,
        modality=modality,
        source_path=path,
        channel_names=channel_names,
        frequency_bins=frequency_bins,
        edge_list=edge_list,
        band_names=band_names,
    )


def band_name_for_frequency(frequency_hz: float) -> str:
    """Return the named EEG band for a frequency bin."""

    for name, low_hz, high_hz in BAND_DEFINITIONS:
        if low_hz <= float(frequency_hz) < high_hz:
            return name
    return "Other"


def flatten_feature_values(feature: LoadedFeature) -> np.ndarray:
    """Return feature values flattened in the same order as feature_ids."""

    return np.asarray(feature.values, dtype=float).reshape(-1)


def aggregate_feature_groups(feature: LoadedFeature) -> dict[str, np.ndarray]:
    """Map interpretable feature IDs to the raw tensor indices they summarize."""

    if feature.modality == "psd":
        groups: dict[str, list[int]] = {}
        n_freq = feature.values.shape[1]
        channels = feature.channel_names or tuple(f"ch{index:03d}" for index in range(feature.values.shape[0]))
        frequencies = feature.frequency_bins or tuple(float(index) for index in range(n_freq))
        for channel_index, channel in enumerate(channels):
            for freq_index, frequency in enumerate(frequencies):
                feature_id = f"psd|{feature.state}|{channel}|{band_name_for_frequency(frequency)}"
                groups.setdefault(feature_id, []).append(channel_index * n_freq + freq_index)
        return {key: np.asarray(indices, dtype=int) for key, indices in groups.items()}

    bands = feature.band_names or tuple(f"band{index:02d}" for index in range(feature.values.shape[1]))
    edges = feature.edge_list or tuple((f"edge{index:04d}", "") for index in range(feature.values.shape[0]))
    groups = {}
    n_band = feature.values.shape[1]
    for edge_index, (first, second) in enumerate(edges):
        edge_name = f"{first}-{second}" if second else first
        for band_index, band in enumerate(bands):
            groups[f"wpli|{feature.state}|{edge_name}|{band}"] = np.asarray(
                [edge_index * n_band + band_index],
                dtype=int,
            )
    return groups


def feature_band_distances(feature: LoadedFeature, z: np.ndarray) -> list[dict[str, object]]:
    """Return band and network-level RMS z-distance rows for one feature tensor."""

    rows: list[dict[str, object]] = []
    if feature.modality == "psd":
        frequencies = np.asarray(feature.frequency_bins or tuple(range(feature.values.shape[1])), dtype=float)
        for band_name, low_hz, high_hz in BAND_DEFINITIONS:
            mask = (frequencies >= low_hz) & (frequencies < high_hz)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "level": "band",
                    "band": band_name,
                    "network_group": "",
                    "distance_to_health": compute_health_distance(z[:, mask]),
                }
            )
        return rows

    bands = feature.band_names or tuple(f"band{index:02d}" for index in range(feature.values.shape[1]))
    for band_index, band in enumerate(bands):
        rows.append(
            {
                "level": "band",
                "band": str(band),
                "network_group": "",
                "distance_to_health": compute_health_distance(z[:, band_index]),
            }
        )
    if feature.edge_list:
        edge_groups = np.asarray([edge_network_group(first, second) for first, second in feature.edge_list])
        for group in sorted(set(edge_groups)):
            mask = edge_groups == group
            rows.append(
                {
                    "level": "network",
                    "band": "",
                    "network_group": group,
                    "distance_to_health": compute_health_distance(z[mask, :]),
                }
            )
    return rows


def edge_network_group(first: str, second: str) -> str:
    """Map a channel pair into broad network groups."""

    group_first = channel_network_group(first)
    group_second = channel_network_group(second)
    if group_first == group_second:
        return group_first
    return "|".join(sorted((group_first, group_second)))


def channel_network_group(channel: str) -> str:
    """Map a channel name to frontal/central/parietal/temporal/occipital/other."""

    token = "".join(ch for ch in str(channel).upper() if ch.isalpha())
    if token.startswith(("FP", "AF", "F", "FT", "FC")):
        return "frontal"
    if token.startswith("C"):
        return "central"
    if token.startswith(("CP", "P", "PO")):
        return "parietal"
    if token.startswith(("O", "CB")):
        return "occipital"
    if token.startswith(("T", "TP")):
        return "temporal"
    return "other"


def _find_feature_file(
    modality_dir: Path,
    subject_id: str,
    state: str,
    suffix: str,
    timepoint: str,
    group: str,
) -> Path:
    modality = "psd" if suffix == "psd" else "wpli"
    manifest_path = modality_dir.parent / "feature_manifest.csv"
    manifest_match = _find_feature_file_from_manifest(
        manifest_path=manifest_path,
        modality=modality,
        subject_id=subject_id,
        state=state,
        timepoint=timepoint,
        group=group,
    )
    if manifest_match is not None:
        return manifest_match

    baseline_tokens = ["", "baseline", "pre", "基线"]
    post_tokens = ["最终", "post14", "post", "day14", "d14", "治疗后", "阶段"]
    tokens = baseline_tokens if timepoint == "baseline" or group == "healthy" else post_tokens
    candidates = []
    for token in tokens:
        if token:
            candidates.append(modality_dir / f"{subject_id}_{token}_{state}_{suffix}.npz")
            candidates.append(modality_dir / f"{subject_id}_{state}_{token}_{suffix}.npz")
        else:
            candidates.append(modality_dir / f"{subject_id}_{state}_{suffix}.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    recursive = _find_feature_file_recursive(
        modality_dir=modality_dir,
        subject_id=subject_id,
        state=state,
        suffix=suffix,
        group=group,
        stage_tokens=tokens,
    )
    if recursive is not None:
        return recursive
    joined = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"No {timepoint} {modality_dir.name} feature found for {subject_id} {state}. Tried: {joined}")


def _find_feature_file_from_manifest(
    *,
    manifest_path: Path,
    modality: str,
    subject_id: str,
    state: str,
    timepoint: str,
    group: str,
) -> Path | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception:
        return None
    required = {"group", "stage", "subject_id", "state", "modality", "feature_path"}
    if not required.issubset(manifest.columns):
        return None
    frame = manifest.copy()
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).str.strip().eq("written")]
    group_value = "health" if group == "healthy" else group
    stages = _stage_aliases(timepoint, group)
    subject_matches = _subject_match_mask(frame["subject_id"], subject_id, group=group_value)
    matches = frame[
        frame["group"].eq(group_value)
        & frame["state"].eq(state)
        & frame["modality"].eq(modality)
        & frame["stage"].isin(stages)
        & subject_matches
    ]
    if matches.empty:
        return None
    stage_rank = {stage: index for index, stage in enumerate(stages)}
    matches = matches.assign(_stage_rank=matches["stage"].map(stage_rank).fillna(len(stages)))
    for raw_path in matches.sort_values("_stage_rank")["feature_path"]:
        path = Path(str(raw_path).strip())
        if path.exists():
            return path
        relative = manifest_path.parent / str(raw_path).strip()
        if relative.exists():
            return relative
    return None


def _find_feature_file_recursive(
    *,
    modality_dir: Path,
    subject_id: str,
    state: str,
    suffix: str,
    group: str,
    stage_tokens: Sequence[str],
) -> Path | None:
    pattern = f"*_{state}_{suffix}.npz"
    group_token = "health" if group == "healthy" else group
    for path in sorted(modality_dir.rglob(pattern), key=lambda item: str(item)):
        parts = {part.strip() for part in path.parts}
        if group_token not in parts:
            continue
        if not any(token in parts or token == "" for token in stage_tokens):
            continue
        stem_subject = path.name[: -len(f"_{state}_{suffix}.npz")]
        if _subject_ids_match(stem_subject, subject_id, group=group_token):
            return path
    return None


def _stage_aliases(timepoint: str, group: str) -> list[str]:
    if group == "healthy":
        return ["health"]
    if timepoint == "baseline":
        return ["基线", "baseline", "pre"]
    if timepoint == "post14":
        return ["最终", "post14", "治疗后", "阶段", "post", "day14", "d14"]
    return [timepoint]


def _subject_match_mask(values: pd.Series, subject_id: str, *, group: str) -> pd.Series:
    return values.map(lambda value: _subject_ids_match(str(value), subject_id, group=group))


def _subject_ids_match(candidate: str, requested: str, *, group: str) -> bool:
    candidate_clean = str(candidate).strip()
    requested_clean = str(requested).strip()
    if candidate_clean == requested_clean:
        return True
    if group in {"patient", "health", "healthy"}:
        return _normalize_subject_token(candidate_clean) == _normalize_subject_token(requested_clean)
    return False


def _normalize_subject_token(value: str) -> str:
    match = re.search(r"sub\s*0*(\d+)", str(value), flags=re.IGNORECASE)
    if match is None:
        return str(value).strip()
    return f"sub{int(match.group(1)):02d}"


def _payload_strings(payload: np.lib.npyio.NpzFile, key: str, length: int) -> tuple[str, ...]:
    if key not in payload.files:
        return tuple(f"item{index:03d}" for index in range(length))
    values = np.asarray(payload[key]).astype(str).reshape(-1)
    if len(values) != length:
        return tuple(f"item{index:03d}" for index in range(length))
    return tuple(str(value) for value in values)


def _payload_frequencies(payload: np.lib.npyio.NpzFile, length: int) -> tuple[float, ...]:
    if "frequency_bins" not in payload.files:
        return tuple(float(index + 1) for index in range(length))
    values = np.asarray(payload["frequency_bins"], dtype=float).reshape(-1)
    if len(values) != length:
        raise ValueError("frequency_bins length does not match PSD feature width.")
    return tuple(float(value) for value in values)


def _payload_edges(payload: np.lib.npyio.NpzFile, length: int) -> tuple[tuple[str, str], ...]:
    if "edge_list" not in payload.files:
        return tuple((f"edge{index:04d}", "") for index in range(length))
    values = np.asarray(payload["edge_list"]).astype(str)
    if values.shape != (length, 2):
        return tuple((f"edge{index:04d}", "") for index in range(length))
    return tuple((str(first), str(second)) for first, second in values.tolist())


def _feature_ids(
    *,
    values: np.ndarray,
    modality: FeatureModality,
    state: str,
    channel_names: Sequence[str],
    frequency_bins: Sequence[float],
    edge_list: Sequence[tuple[str, str]],
    band_names: Sequence[str],
) -> tuple[str, ...]:
    if modality == "psd":
        channels = tuple(channel_names) or tuple(f"ch{index:03d}" for index in range(values.shape[0]))
        frequencies = tuple(frequency_bins) or tuple(float(index) for index in range(values.shape[1]))
        return tuple(
            f"psd|{state}|{channel}|{float(frequency):.6g}Hz"
            for channel in channels
            for frequency in frequencies
        )

    edges = tuple(edge_list) or tuple((f"edge{index:04d}", "") for index in range(values.shape[0]))
    bands = tuple(band_names) or tuple(f"band{index:02d}" for index in range(values.shape[1]))
    ids = []
    for first, second in edges:
        edge_name = f"{first}-{second}" if second else first
        for band in bands:
            ids.append(f"wpli|{state}|{edge_name}|{band}")
    return tuple(ids)


def stack_loaded_features(features: Iterable[LoadedFeature]) -> np.ndarray:
    """Stack loaded feature tensors and validate that shapes match."""

    items = list(features)
    if not items:
        raise ValueError("No loaded features were provided.")
    shape = items[0].values.shape
    mismatched = [item.subject_id for item in items if item.values.shape != shape]
    if mismatched:
        raise ValueError(f"Feature shape mismatch for subject(s): {', '.join(mismatched)}")
    return np.stack([item.values for item in items], axis=0)
