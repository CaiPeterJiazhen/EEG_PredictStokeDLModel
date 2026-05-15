from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


CANONICAL_CHANNELS_62 = (
    "FP1", "FPZ", "FP2",
    "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
)

MIDLINE_CHANNELS = ("FPZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ")

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "channel_mapping.yaml"


class ChannelMappingError(ValueError):
    """Raised when channel mapping metadata or input channels are invalid."""


@dataclass(frozen=True)
class ChannelMapping:
    channel_names: tuple[str, ...]
    midline_channels: tuple[str, ...]
    left_right_pairs: tuple[tuple[str, str], ...]
    mirror_map: Mapping[str, str]


def load_channel_mapping(config_path: str | Path | None = None) -> ChannelMapping:
    """Load and validate the fixed 62-channel mirror mapping."""

    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    if not path.exists():
        raise ChannelMappingError(f"Channel mapping config does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ChannelMappingError(f"Channel mapping config must be a YAML mapping: {path}")

    channel_names = _read_string_tuple(payload, "channel_names")
    midline_channels = _read_string_tuple(payload, "midline_channels")
    left_right_pairs = _read_pairs(payload)
    mirror_map = _build_mirror_map(channel_names, midline_channels, left_right_pairs)
    _validate_mapping(channel_names, midline_channels, left_right_pairs, mirror_map)

    return ChannelMapping(
        channel_names=channel_names,
        midline_channels=midline_channels,
        left_right_pairs=left_right_pairs,
        mirror_map=MappingProxyType(dict(mirror_map)),
    )


def _read_string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ChannelMappingError(f"Channel mapping key {key!r} must be a list of strings")
    return tuple(value)


def _read_pairs(payload: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    value = payload.get("left_right_pairs")
    if not isinstance(value, list):
        raise ChannelMappingError("Channel mapping key 'left_right_pairs' must be a list")

    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not all(isinstance(channel, str) for channel in item)
        ):
            raise ChannelMappingError("Each left/right pair must contain exactly two strings")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _build_mirror_map(
    channel_names: tuple[str, ...],
    midline_channels: tuple[str, ...],
    left_right_pairs: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    mirror_map = {channel: channel for channel in midline_channels}
    for left, right in left_right_pairs:
        mirror_map[left] = right
        mirror_map[right] = left
    return mirror_map


def _validate_mapping(
    channel_names: tuple[str, ...],
    midline_channels: tuple[str, ...],
    left_right_pairs: tuple[tuple[str, str], ...],
    mirror_map: Mapping[str, str],
) -> None:
    if channel_names != CANONICAL_CHANNELS_62:
        raise ChannelMappingError("Channel names must match the fixed actual 62-channel order")
    if len(channel_names) != 62:
        raise ChannelMappingError(f"Expected 62 channels, found {len(channel_names)}")
    if len(set(channel_names)) != len(channel_names):
        raise ChannelMappingError("Channel mapping contains duplicate channel names")
    if {"M1", "M2"} & set(channel_names):
        raise ChannelMappingError("Actual 62-channel mapping must not include M1 or M2")
    if tuple(midline_channels) != MIDLINE_CHANNELS:
        raise ChannelMappingError("Midline channels do not match the fixed project convention")

    channel_set = set(channel_names)
    pair_channels = {channel for pair in left_right_pairs for channel in pair}
    if pair_channels & set(midline_channels):
        raise ChannelMappingError("Midline channels must not appear in left/right pairs")
    if not pair_channels <= channel_set:
        unknown = sorted(pair_channels - channel_set)
        raise ChannelMappingError(f"Left/right pairs contain unknown channel(s): {unknown}")
    if set(midline_channels) | pair_channels != channel_set:
        missing = sorted(channel_set - (set(midline_channels) | pair_channels))
        raise ChannelMappingError(f"Mirror map is missing channel(s): {missing}")

    if set(mirror_map) != channel_set:
        missing = sorted(channel_set - set(mirror_map))
        extra = sorted(set(mirror_map) - channel_set)
        raise ChannelMappingError(f"Mirror map keys mismatch; missing={missing}, extra={extra}")
    for channel, mirror in mirror_map.items():
        if mirror not in channel_set:
            raise ChannelMappingError(f"Mirror target for {channel} is unknown: {mirror}")
        if mirror_map[mirror] != channel:
            raise ChannelMappingError(f"Mirror map is not symmetric for {channel} and {mirror}")
    for channel in midline_channels:
        if mirror_map[channel] != channel:
            raise ChannelMappingError(f"Midline channel must map to itself: {channel}")
