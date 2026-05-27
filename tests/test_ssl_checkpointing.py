from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eeg_recovery.models.multimodal_model import MultimodalEEGModel
from eeg_recovery.training.ssl_checkpointing import (
    extract_wpli_encoder_from_dual_segment_barlow_checkpoint,
    merge_branch_pretrained_states,
    prefix_branch_encoder_state_dict,
    save_ssl_encoder_checkpoint,
    validate_ssl_checkpoint_metadata,
)


def _metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "checkpoint_type": "segment_ssl_encoder",
        "branch": "psd",
        "ssl_objective": "barlow",
        "segment_ssl_method": "segment_barlow",
        "base_seed": 0,
        "fold_index": 1,
        "test_subject_id": "sub01",
        "excluded_subject_id": "sub01",
        "ssl_data_scope": "all-patient",
        "feature_kind": "psd",
        "encoder_kind": "cnn",
        "embedding_dim": 32,
        "dropout": 0.0,
        "pretrain_epochs": 20,
        "pretrain_lr": 1e-3,
        "source_feature_manifest_hash": "manifest-a",
    }
    metadata.update(overrides)
    return metadata


def _checkpoint(branch: str) -> dict[str, object]:
    return {
        "checkpoint_type": "segment_ssl_encoder",
        "metadata": _metadata(branch=branch, feature_kind="psd" if branch == "psd" else "fc-wpli"),
        "encoder_state_dict": {
            "encoder.network.0.weight": torch.full((1,), 1.0 if branch == "psd" else 2.0),
            "encoder.network.0.bias": torch.full((1,), 3.0 if branch == "psd" else 4.0),
        },
        "prefixed_state_dict": None,
    }


def test_prefix_psd_encoder_state_dict_keys() -> None:
    state = {"encoder.network.0.weight": torch.ones(1)}

    prefixed = prefix_branch_encoder_state_dict("psd", state)

    assert set(prefixed) == {"branch_models.psd.encoder.encoder.network.0.weight"}
    torch.testing.assert_close(prefixed["branch_models.psd.encoder.encoder.network.0.weight"], torch.ones(1))


def test_prefix_wpli_encoder_state_dict_keys() -> None:
    state = {"encoder.network.0.weight": torch.ones(1)}

    prefixed = prefix_branch_encoder_state_dict("wpli", state)

    assert set(prefixed) == {"branch_models.wpli.encoder.encoder.network.0.weight"}
    torch.testing.assert_close(prefixed["branch_models.wpli.encoder.encoder.network.0.weight"], torch.ones(1))


@pytest.mark.parametrize(
    "field",
    ["branch", "test_subject_id", "embedding_dim", "source_feature_manifest_hash"],
)
def test_checkpoint_metadata_mismatch_raises(field: str) -> None:
    observed = _metadata()
    expected = _metadata()
    expected[field] = "different" if field != "embedding_dim" else 64

    with pytest.raises(ValueError, match=field):
        validate_ssl_checkpoint_metadata(observed, expected)


def test_merge_psd_wpli_checkpoints() -> None:
    psd_checkpoint = _checkpoint("psd")
    wpli_checkpoint = _checkpoint("wpli")

    merged_state, merged_metadata = merge_branch_pretrained_states(psd_checkpoint, wpli_checkpoint)

    assert "branch_models.psd.encoder.encoder.network.0.weight" in merged_state
    assert "branch_models.wpli.encoder.encoder.network.0.weight" in merged_state
    assert merged_metadata["psd"]["branch"] == "psd"
    assert merged_metadata["wpli"]["branch"] == "wpli"


def test_merge_can_load_checkpoint_paths() -> None:
    scratch = Path("results") / "checkpoints" / "ssl_encoders" / "test_io"
    scratch.mkdir(parents=True, exist_ok=True)
    psd_path = scratch / "psd.pt"
    wpli_path = scratch / "wpli.pt"

    torch.save(_checkpoint("psd"), psd_path)
    torch.save(_checkpoint("wpli"), wpli_path)

    merged_state, _ = merge_branch_pretrained_states(psd_path, wpli_path)

    assert "branch_models.psd.encoder.encoder.network.0.bias" in merged_state
    assert "branch_models.wpli.encoder.encoder.network.0.bias" in merged_state


def test_merged_state_loads_into_multimodal_model() -> None:
    psd_model = MultimodalEEGModel(
        feature_kind="psd-fc-wpli",
        fusion="gated",
        embedding_dim=32,
        dropout=0.0,
        encoder_kind="cnn",
    )
    state = psd_model.state_dict()
    psd_encoder = {
        key.removeprefix("branch_models.psd.encoder."): value
        for key, value in state.items()
        if key.startswith("branch_models.psd.encoder.")
    }
    wpli_encoder = {
        key.removeprefix("branch_models.wpli.encoder."): value
        for key, value in state.items()
        if key.startswith("branch_models.wpli.encoder.")
    }
    psd_checkpoint = {
        "checkpoint_type": "segment_ssl_encoder",
        "metadata": _metadata(branch="psd", feature_kind="psd"),
        "encoder_state_dict": psd_encoder,
        "prefixed_state_dict": None,
    }
    wpli_checkpoint = {
        "checkpoint_type": "segment_ssl_encoder",
        "metadata": _metadata(branch="wpli", feature_kind="fc-wpli"),
        "encoder_state_dict": wpli_encoder,
        "prefixed_state_dict": None,
    }

    merged_state, _ = merge_branch_pretrained_states(psd_checkpoint, wpli_checkpoint)
    target = MultimodalEEGModel(
        feature_kind="psd-fc-wpli",
        fusion="gated",
        embedding_dim=32,
        dropout=0.0,
        encoder_kind="cnn",
    )
    incompatible = target.load_state_dict(merged_state, strict=False)

    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys


def test_extract_wpli_encoder_from_dual_checkpoint_saves_branch_checkpoint(tmp_path: Path) -> None:
    dual_path = tmp_path / "dual.pt"
    output_path = tmp_path / "segssl_wpli_barlow_foldstrict_v1.pt"
    expected_dual_metadata = _metadata(
        branch="dual",
        feature_kind="psd-fc-wpli",
        segment_ssl_method="dual_segment_barlow",
        source_feature_manifest_hash="dual-manifest",
    )
    branch_metadata = _metadata(
        branch="wpli",
        feature_kind="fc-wpli",
        source_feature_manifest_hash="wpli-manifest",
    )
    torch.save(
        {
            "checkpoint_type": "dual_segment_ssl_encoder",
            "metadata": expected_dual_metadata,
            "state_dict": {
                "branch_models.psd.encoder.encoder.network.0.weight": torch.full((1,), 1.0),
                "branch_models.wpli.encoder.encoder.network.0.weight": torch.full((1,), 2.0),
                "branch_models.wpli.encoder.encoder.network.0.bias": torch.full((1,), 3.0),
            },
        },
        dual_path,
    )

    extracted = extract_wpli_encoder_from_dual_segment_barlow_checkpoint(
        dual_checkpoint_path=dual_path,
        output_path=output_path,
        expected_dual_metadata=expected_dual_metadata,
        branch_metadata=branch_metadata,
    )

    assert set(extracted) == {"encoder.network.0.weight", "encoder.network.0.bias"}
    torch.testing.assert_close(extracted["encoder.network.0.weight"], torch.full((1,), 2.0))
    saved = torch.load(output_path, map_location="cpu")
    assert saved["metadata"]["branch"] == "wpli"
    assert saved["metadata"]["source_feature_manifest_hash"] == "wpli-manifest"
    assert saved["metadata"]["source_dual_feature_manifest_hash"] == "dual-manifest"
    assert saved["metadata"]["derived_from_dual_checkpoint"] is True
    assert saved["metadata"]["source_dual_checkpoint_metadata"]["branch"] == "dual"


def test_extract_wpli_encoder_from_dual_checkpoint_rejects_supervised_metadata(tmp_path: Path) -> None:
    dual_path = tmp_path / "supervised.pt"
    output_path = tmp_path / "wpli.pt"
    expected_dual_metadata = _metadata(
        branch="dual",
        feature_kind="psd-fc-wpli",
        segment_ssl_method="dual_segment_barlow",
        source_feature_manifest_hash="dual-manifest",
    )
    supervised_metadata = dict(expected_dual_metadata)
    supervised_metadata["supervised_training"] = True
    torch.save(
        {
            "checkpoint_type": "dual_segment_ssl_encoder",
            "metadata": supervised_metadata,
            "state_dict": {
                "branch_models.wpli.encoder.encoder.network.0.weight": torch.ones(1),
            },
        },
        dual_path,
    )

    with pytest.raises(ValueError, match="supervised"):
        extract_wpli_encoder_from_dual_segment_barlow_checkpoint(
            dual_checkpoint_path=dual_path,
            output_path=output_path,
            expected_dual_metadata=expected_dual_metadata,
            branch_metadata=_metadata(branch="wpli", feature_kind="fc-wpli"),
        )


def test_extract_wpli_encoder_from_dual_checkpoint_rejects_metadata_mismatch(tmp_path: Path) -> None:
    dual_path = tmp_path / "dual.pt"
    output_path = tmp_path / "wpli.pt"
    observed_metadata = _metadata(
        branch="dual",
        feature_kind="psd-fc-wpli",
        segment_ssl_method="dual_segment_barlow",
        source_feature_manifest_hash="dual-manifest",
    )
    expected_dual_metadata = dict(observed_metadata)
    expected_dual_metadata["test_subject_id"] = "sub02"
    torch.save(
        {
            "checkpoint_type": "dual_segment_ssl_encoder",
            "metadata": observed_metadata,
            "state_dict": {
                "branch_models.wpli.encoder.encoder.network.0.weight": torch.ones(1),
            },
        },
        dual_path,
    )

    with pytest.raises(ValueError, match="test_subject_id"):
        extract_wpli_encoder_from_dual_segment_barlow_checkpoint(
            dual_checkpoint_path=dual_path,
            output_path=output_path,
            expected_dual_metadata=expected_dual_metadata,
            branch_metadata=_metadata(branch="wpli", feature_kind="fc-wpli"),
        )
