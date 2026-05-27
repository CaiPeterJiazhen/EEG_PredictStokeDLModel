from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import torch

from eeg_recovery.training.ssl_checkpointing import (
    load_reusable_ssl_encoder_checkpoint,
    load_ssl_encoder_checkpoint,
    save_ssl_encoder_checkpoint,
)


def _metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "checkpoint_type": "segment_ssl_encoder",
        "branch": "psd",
        "ssl_objective": "barlow",
        "segment_ssl_method": "segment_barlow",
        "base_seed": 0,
        "effective_seed": 3,
        "fold_index": 3,
        "test_subject_id": "sub08",
        "excluded_subject_id": "sub08",
        "ssl_data_scope": "all-patient",
        "historical_unlabeled_pretraining": True,
        "segment_feature_kind": "psd",
        "supervised_feature_kind": "psd-fc-wpli",
        "encoder_kind": "cnn",
        "embedding_dim": 32,
        "dropout": 0.0,
        "projection_dim": 32,
        "pretrain_epochs": 20,
        "pretrain_lr": 1e-3,
        "batch_size": 16,
        "feature_mask_prob": 0.03,
        "noise_std": 0.02,
        "lambda_latent": 1.0,
        "lambda_local": 0.1,
        "n_ssl_segments": 100,
        "source_feature_manifest_hash": "manifest-a",
        "created_at": "2026-05-27T00:00:00+00:00",
    }
    metadata.update(overrides)
    return metadata


def test_checkpoint_metadata_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "encoder.pt"
    save_ssl_encoder_checkpoint(path, _metadata(), {"weight": torch.ones(1)})
    expected = _metadata(test_subject_id="sub09")

    with pytest.raises(ValueError, match="test_subject_id"):
        load_reusable_ssl_encoder_checkpoint(path, expected_metadata=expected)


def test_reuse_only_missing_checkpoint_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Missing reusable Segment SSL checkpoint"):
        load_reusable_ssl_encoder_checkpoint(
            missing_path,
            expected_metadata=_metadata(),
            reuse_only=True,
        )


def test_encoder_checkpoint_save_and_reload_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "encoder.pt"
    state = {"encoder.network.0.weight": torch.arange(3, dtype=torch.float32)}

    save_ssl_encoder_checkpoint(path, _metadata(), state)
    loaded = load_ssl_encoder_checkpoint(path, expected_metadata=_metadata())

    assert loaded["checkpoint_type"] == "segment_ssl_encoder"
    assert loaded["metadata"]["branch"] == "psd"
    torch.testing.assert_close(
        loaded["encoder_state_dict"]["encoder.network.0.weight"],
        state["encoder.network.0.weight"],
    )


def test_checkpoint_extensions_are_gitignored() -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "results/checkpoints/ssl_encoders/example.pt",
            "results/checkpoints/ssl_encoders/example.pth",
            "results/checkpoints/ssl_encoders/example.ckpt",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "example.pt" in result.stdout
    assert "example.pth" in result.stdout
    assert "example.ckpt" in result.stdout
