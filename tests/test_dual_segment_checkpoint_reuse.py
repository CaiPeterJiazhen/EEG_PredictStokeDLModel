from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from eeg_recovery.training.ssl_checkpointing import REQUIRED_SSL_METADATA_FIELDS


def _load_dual_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "18_train_dual_segment_barlow_transfer.py"
    spec = spec_from_file_location("dual_segment_barlow_transfer", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        ssl_data_scope="all-patient",
        embedding_dim=32,
        dropout=0.0,
        projection_dim=32,
        pretrain_epochs=20,
        pretrain_lr=1e-3,
        pretrain_batch_size=16,
        feature_mask_prob=0.03,
        noise_std=0.02,
        lambda_latent=1.0,
        lambda_local=0.1,
        masked_latent_loss="cosine",
        barlow_offdiag_weight=0.005,
    )


def test_dual_checkpoint_metadata_matches_branch_checkpoint_schema() -> None:
    dual = _load_dual_script_module()
    args = _args()
    ssl_config = SimpleNamespace(seed=3)
    fold = SimpleNamespace(fold_index=3, test_subject_id="sub08")

    metadata = dual._checkpoint_metadata(
        args=args,
        branch="wpli",
        feature_kind="fc-wpli",
        fold_segments=[object(), object()],
        fold=fold,
        seed=0,
        ssl_config=ssl_config,
        source_feature_manifest_hash="branch-specific-hash",
    )

    for field in REQUIRED_SSL_METADATA_FIELDS:
        assert field in metadata
    assert metadata["branch"] == "wpli"
    assert metadata["segment_feature_kind"] == "fc-wpli"
    assert metadata["supervised_feature_kind"] == "psd-fc-wpli"
    assert metadata["historical_unlabeled_pretraining"] is True
    assert metadata["n_ssl_segments"] == 2
    assert metadata["source_feature_manifest_hash"] == "branch-specific-hash"


def test_dual_reuse_hash_accepts_manifest_legacy_project_root(tmp_path: Path) -> None:
    dual = _load_dual_script_module()
    checkpoint_dir = tmp_path / "results" / "checkpoints" / "ssl_encoders"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "encoder.pt"
    legacy_root = Path("E:/AAAProjectList/EEG_PredictStokeDLModel")
    legacy_checkpoint_path = legacy_root / "results" / "checkpoints" / "ssl_encoders" / checkpoint_path.name
    records = [
        SimpleNamespace(
            group="patient",
            subject_id="sub01",
            subject_key="patient:sub01",
            stage="baseline",
            state="EC",
            segment_index=0,
            features={"psd": object()},
            source_path=tmp_path / "data" / "features" / "segment_level" / "psd" / "a.npz",
        )
    ]
    legacy_hash = dual._segment_records_manifest_hash_with_source_root(
        records,
        output_root=tmp_path,
        source_root=legacy_root,
    )
    torch.save(
        {
            "checkpoint_type": "segment_ssl_encoder",
            "metadata": {"source_feature_manifest_hash": legacy_hash},
            "encoder_state_dict": {},
        },
        checkpoint_path,
    )
    pd.DataFrame({"checkpoint_path": [str(legacy_checkpoint_path)]}).to_csv(
        checkpoint_dir / "segment_ssl_checkpoint_manifest.csv",
        index=False,
    )

    selected_hash = dual._source_manifest_hash_for_checkpoint_reuse(
        records,
        output_root=tmp_path,
        checkpoint_path=checkpoint_path,
        current_hash="current-root-hash",
    )

    assert selected_hash == legacy_hash
