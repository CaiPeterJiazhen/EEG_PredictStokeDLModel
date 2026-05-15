from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eeg_recovery.channels.mapping import (
    ChannelMapping,
    ChannelMappingError,
    load_channel_mapping,
)


def flip_channels_for_affected_hand(
    data: np.ndarray,
    channel_names: Sequence[str],
    affected_hand: str,
    *,
    channel_axis: int = 0,
    mapping: ChannelMapping | None = None,
) -> np.ndarray:
    """Align EEG channels to right affected hand / C3 stimulation convention."""

    channel_mapping = mapping if mapping is not None else load_channel_mapping()
    normalized_hand = affected_hand.strip()
    if normalized_hand not in {"右", "左"}:
        raise ChannelMappingError(
            f"affected_hand must be '右' or '左', got {affected_hand!r}"
        )

    names = list(channel_names)
    _validate_input_channels(data, names, channel_axis, channel_mapping)

    if normalized_hand == "右":
        return np.array(data, copy=True)

    input_index = {channel: index for index, channel in enumerate(names)}
    reorder_indices = [
        input_index[channel_mapping.mirror_map[channel]]
        for channel in names
    ]
    return np.take(data, reorder_indices, axis=channel_axis)


def _validate_input_channels(
    data: np.ndarray,
    channel_names: list[str],
    channel_axis: int,
    mapping: ChannelMapping,
) -> None:
    if len(set(channel_names)) != len(channel_names):
        raise ChannelMappingError("Input channel_names contain duplicate channel(s)")

    expected = set(mapping.channel_names)
    observed = set(channel_names)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ChannelMappingError(
            f"Input channel_names are missing or invalid; missing={missing}, extra={extra}"
        )
    if tuple(channel_names) != mapping.channel_names:
        raise ChannelMappingError(
            "Input channel_names must match the fixed 62-channel order"
        )

    try:
        axis_size = data.shape[channel_axis]
    except IndexError as exc:
        raise ChannelMappingError(
            f"channel_axis {channel_axis} is out of bounds for data with shape {data.shape}"
        ) from exc

    if axis_size != len(channel_names):
        raise ChannelMappingError(
            f"Data channel axis length {axis_size} does not match "
            f"{len(channel_names)} channel_names"
        )
