from __future__ import annotations

import numpy as np
import pytest
import yaml

from eeg_recovery.channels import (
    CANONICAL_CHANNELS_62,
    MIDLINE_CHANNELS,
    ChannelMappingError,
    flip_channels_for_affected_hand,
    load_channel_mapping,
)


EXPECTED_CHANNELS = [
    "FP1", "FPZ", "FP2",
    "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]

EXPECTED_PAIRS = [
    ("FP1", "FP2"),
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F5", "F6"),
    ("F3", "F4"),
    ("F1", "F2"),
    ("FT7", "FT8"),
    ("FC5", "FC6"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("T7", "T8"),
    ("C5", "C6"),
    ("C3", "C4"),
    ("C1", "C2"),
    ("TP7", "TP8"),
    ("CP5", "CP6"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
    ("P7", "P8"),
    ("P5", "P6"),
    ("P3", "P4"),
    ("P1", "P2"),
    ("PO7", "PO8"),
    ("PO5", "PO6"),
    ("PO3", "PO4"),
    ("CB1", "CB2"),
    ("O1", "O2"),
]


def test_channel_mapping_yaml_defines_actual_62_channel_order_without_mastoids() -> None:
    mapping = load_channel_mapping()

    assert mapping.channel_names == tuple(EXPECTED_CHANNELS)
    assert CANONICAL_CHANNELS_62 == tuple(EXPECTED_CHANNELS)
    assert len(mapping.channel_names) == 62
    assert len(set(mapping.channel_names)) == 62
    assert "M1" not in mapping.channel_names
    assert "M2" not in mapping.channel_names


def test_mirror_map_is_complete_symmetric_and_midline_self_maps() -> None:
    mapping = load_channel_mapping()

    assert set(mapping.mirror_map) == set(EXPECTED_CHANNELS)
    for left, right in EXPECTED_PAIRS:
        assert mapping.mirror_map[left] == right
        assert mapping.mirror_map[right] == left

    assert MIDLINE_CHANNELS == ("FPZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ")
    for channel in MIDLINE_CHANNELS:
        assert mapping.mirror_map[channel] == channel


def test_loaded_channel_mapping_cannot_be_mutated() -> None:
    mapping = load_channel_mapping()

    with pytest.raises(TypeError):
        mapping.channel_names[0] = "BROKEN"
    with pytest.raises(TypeError):
        mapping.midline_channels[0] = "BROKEN"
    with pytest.raises(TypeError):
        mapping.left_right_pairs[0] = ("BROKEN", "BROKEN")
    with pytest.raises(TypeError):
        mapping.mirror_map["C3"] = "BROKEN"


def test_right_affected_hand_preserves_values_and_channel_order() -> None:
    data = np.arange(62 * 4).reshape(62, 4)

    flipped = flip_channels_for_affected_hand(data, EXPECTED_CHANNELS, affected_hand="右")

    np.testing.assert_array_equal(flipped, data)
    assert flipped is not data


def test_left_affected_hand_reorders_channels_from_their_mirrors() -> None:
    data = np.arange(62 * 3).reshape(62, 3)
    flipped = flip_channels_for_affected_hand(data, EXPECTED_CHANNELS, affected_hand="左")

    c3_out = EXPECTED_CHANNELS.index("C3")
    c4_in = EXPECTED_CHANNELS.index("C4")
    c4_out = EXPECTED_CHANNELS.index("C4")
    c3_in = EXPECTED_CHANNELS.index("C3")
    cz = EXPECTED_CHANNELS.index("CZ")

    np.testing.assert_array_equal(flipped[c3_out], data[c4_in])
    np.testing.assert_array_equal(flipped[c4_out], data[c3_in])
    np.testing.assert_array_equal(flipped[cz], data[cz])
    assert flipped.shape == data.shape


def test_left_affected_hand_reorders_along_configurable_channel_axis() -> None:
    data = np.arange(2 * 62 * 5).reshape(2, 62, 5)
    flipped = flip_channels_for_affected_hand(
        data,
        EXPECTED_CHANNELS,
        affected_hand="左",
        channel_axis=1,
    )

    c3 = EXPECTED_CHANNELS.index("C3")
    c4 = EXPECTED_CHANNELS.index("C4")
    np.testing.assert_array_equal(flipped[:, c3, :], data[:, c4, :])
    np.testing.assert_array_equal(flipped[:, c4, :], data[:, c3, :])


def test_flip_rejects_complete_but_noncanonical_channel_order() -> None:
    data = np.zeros((62, 4))
    shuffled = list(EXPECTED_CHANNELS)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]

    with pytest.raises(ChannelMappingError, match="order|fixed 62-channel order"):
        flip_channels_for_affected_hand(data, shuffled, affected_hand="左")


@pytest.mark.parametrize("affected_hand", ["", "unknown", "双"])
def test_flip_rejects_unknown_affected_hand(affected_hand: str) -> None:
    data = np.zeros((62, 4))

    with pytest.raises(ChannelMappingError, match="affected_hand"):
        flip_channels_for_affected_hand(data, EXPECTED_CHANNELS, affected_hand=affected_hand)


def test_flip_rejects_duplicate_or_missing_channels() -> None:
    data = np.zeros((62, 4))
    duplicated = list(EXPECTED_CHANNELS)
    duplicated[-1] = duplicated[0]

    with pytest.raises(ChannelMappingError, match="duplicate"):
        flip_channels_for_affected_hand(data, duplicated, affected_hand="左")

    with pytest.raises(ChannelMappingError, match="missing"):
        flip_channels_for_affected_hand(data[:-1], EXPECTED_CHANNELS[:-1], affected_hand="左")


@pytest.mark.parametrize(
    "override",
    [
        {"channel_names": EXPECTED_CHANNELS + ["M1", "M2"]},
        {"left_right_pairs": EXPECTED_PAIRS[:-1]},
        {"left_right_pairs": EXPECTED_PAIRS + [("ZZ", "FP1")]},
        {"left_right_pairs": EXPECTED_PAIRS + [("FPZ", "FP2")]},
    ],
)
def test_load_channel_mapping_rejects_malformed_configs(tmp_path, override) -> None:
    config = {
        "channel_names": EXPECTED_CHANNELS,
        "midline_channels": list(MIDLINE_CHANNELS),
        "left_right_pairs": EXPECTED_PAIRS,
    }
    config.update(override)
    config_path = tmp_path / "channel_mapping.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ChannelMappingError):
        load_channel_mapping(config_path)
