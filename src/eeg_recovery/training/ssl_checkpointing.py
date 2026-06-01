from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Mapping

import torch


REQUIRED_SSL_METADATA_FIELDS = (
    "checkpoint_type",
    "branch",
    "ssl_objective",
    "segment_ssl_method",
    "base_seed",
    "effective_seed",
    "fold_index",
    "test_subject_id",
    "excluded_subject_id",
    "ssl_data_scope",
    "historical_unlabeled_pretraining",
    "segment_feature_kind",
    "supervised_feature_kind",
    "encoder_kind",
    "embedding_dim",
    "dropout",
    "projection_dim",
    "pretrain_epochs",
    "pretrain_lr",
    "batch_size",
    "feature_mask_prob",
    "noise_std",
    "lambda_latent",
    "lambda_local",
    "source_feature_manifest_hash",
)

MERGED_METADATA_FIELDS = (
    "fold_index",
    "test_subject_id",
    "excluded_subject_id",
    "base_seed",
    "encoder_kind",
    "embedding_dim",
    "dropout",
    "ssl_data_scope",
)

DUAL_SEGMENT_BARLOW_REQUIRED_METADATA_FIELDS = (
    "ssl_objective",
    "segment_ssl_method",
    "base_seed",
    "fold_index",
    "test_subject_id",
    "excluded_subject_id",
    "ssl_data_scope",
    "encoder_kind",
    "embedding_dim",
    "dropout",
    "pretrain_epochs",
    "pretrain_lr",
    "source_feature_manifest_hash",
)

WPLI_BRANCH_ENCODER_PREFIX = "branch_models.wpli.encoder."

SSL_CHECKPOINT_MANIFEST_NAME = "segment_ssl_checkpoint_manifest.csv"
SSL_CHECKPOINT_MANIFEST_COLUMNS = (
    "checkpoint_path",
    "status",
    "updated_at",
    "checkpoint_type",
    "branch",
    "ssl_objective",
    "segment_ssl_method",
    "base_seed",
    "effective_seed",
    "fold_index",
    "test_subject_id",
    "excluded_subject_id",
    "ssl_data_scope",
    "historical_unlabeled_pretraining",
    "segment_feature_kind",
    "supervised_feature_kind",
    "encoder_kind",
    "embedding_dim",
    "dropout",
    "projection_dim",
    "pretrain_epochs",
    "pretrain_lr",
    "batch_size",
    "feature_mask_prob",
    "noise_std",
    "lambda_latent",
    "lambda_local",
    "n_ssl_segments",
    "source_feature_manifest_hash",
    "graph_smoothness_weight",
    "qeeg_auxiliary_enabled",
    "lambda_qeeg",
    "qeeg_feature_name",
    "qeeg_scaler_hash",
    "qeeg_fit_subject_ids",
    "data_scope",
    "seed",
)


def make_ssl_checkpoint_name(
    *,
    method: str,
    branch: str,
    ssl_objective: str,
    seed: int,
    fold_index: int,
    test_subject_id: str,
    embedding_dim: int,
    pretrain_epochs: int,
    ssl_data_scope: str,
    tag: str | None = None,
) -> str:
    parts = [
        method,
        ssl_objective,
        branch,
        f"seed{seed}",
        f"fold{fold_index:02d}",
        f"test_{test_subject_id}",
        f"emb{embedding_dim}",
        f"pre{pretrain_epochs}",
        ssl_data_scope,
    ]
    if tag:
        parts.append(tag)
    return f"{_safe_token('_'.join(str(part) for part in parts))}.pt"


def save_ssl_encoder_checkpoint(
    path: str | Path,
    metadata: Mapping[str, object],
    encoder_state_dict: Mapping[str, torch.Tensor],
    prefixed_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": "segment_ssl_encoder",
        "metadata": dict(metadata),
        "encoder_state_dict": _cpu_state_dict(encoder_state_dict),
        "prefixed_state_dict": _cpu_state_dict(prefixed_state_dict) if prefixed_state_dict is not None else None,
    }
    payload["metadata"].setdefault("checkpoint_type", payload["checkpoint_type"])
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def update_ssl_checkpoint_manifest(
    checkpoint_dir: str | Path,
    checkpoint_path: str | Path,
    metadata: Mapping[str, object],
    *,
    status: str,
) -> Path:
    manifest_path = Path(checkpoint_dir) / SSL_CHECKPOINT_MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_text = str(Path(checkpoint_path))
    row = {
        "checkpoint_path": checkpoint_text,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    for column in SSL_CHECKPOINT_MANIFEST_COLUMNS:
        if column in row:
            continue
        value = metadata.get(column, "")
        row[column] = str(value) if value is not None else ""

    rows: list[dict[str, str]] = []
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for existing in reader:
                if existing.get("checkpoint_path") != checkpoint_text:
                    rows.append({column: existing.get(column, "") for column in SSL_CHECKPOINT_MANIFEST_COLUMNS})
    rows.append({column: str(row.get(column, "")) for column in SSL_CHECKPOINT_MANIFEST_COLUMNS})
    rows.sort(key=lambda item: (item.get("branch", ""), item.get("base_seed", ""), item.get("fold_index", ""), item.get("checkpoint_path", "")))
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SSL_CHECKPOINT_MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def load_ssl_encoder_checkpoint(
    path: str | Path,
    expected_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("SSL checkpoint must be a dictionary.")
    if checkpoint.get("checkpoint_type") != "segment_ssl_encoder":
        raise ValueError("SSL checkpoint has invalid checkpoint_type.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("SSL checkpoint is missing metadata.")
    metadata.setdefault("checkpoint_type", checkpoint["checkpoint_type"])
    if expected_metadata is not None:
        validate_ssl_checkpoint_metadata(metadata, expected_metadata)
    return checkpoint


def load_reusable_ssl_encoder_checkpoint(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, object],
    reuse_only: bool = True,
    force_retrain_ssl: bool = False,
) -> dict[str, object] | None:
    checkpoint_path = Path(path)
    if force_retrain_ssl:
        if reuse_only:
            raise ValueError("--force-retrain-ssl cannot be combined with --reuse-only.")
        return None
    if not checkpoint_path.exists():
        if reuse_only:
            raise FileNotFoundError(f"Missing reusable Segment SSL checkpoint: {checkpoint_path}")
        return None
    return load_ssl_encoder_checkpoint(checkpoint_path, expected_metadata)


def validate_ssl_checkpoint_metadata(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    allow_mismatched_ssl_seeds: bool = False,
) -> None:
    for field in REQUIRED_SSL_METADATA_FIELDS:
        if field not in observed:
            raise ValueError(f"SSL checkpoint metadata missing required field: {field}")
        if field not in expected:
            raise ValueError(f"Expected SSL checkpoint metadata missing required field: {field}")
        if allow_mismatched_ssl_seeds and field == "base_seed":
            continue
        if observed[field] != expected[field]:
            raise ValueError(
                "SSL checkpoint metadata mismatch for "
                f"{field}: observed={observed[field]!r}, expected={expected[field]!r}"
            )


def prefix_branch_encoder_state_dict(
    branch: str,
    encoder_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if branch not in {"psd", "wpli"}:
        raise ValueError("branch must be 'psd' or 'wpli'.")
    return {
        f"branch_models.{branch}.encoder.{key}": value.detach().cpu().clone()
        for key, value in encoder_state_dict.items()
    }


def merge_branch_pretrained_states(
    psd_checkpoint: str | Path | Mapping[str, object],
    wpli_checkpoint: str | Path | Mapping[str, object],
    *,
    allow_mismatched_ssl_seeds: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    psd = _coerce_checkpoint(psd_checkpoint)
    wpli = _coerce_checkpoint(wpli_checkpoint)
    psd_metadata = _checkpoint_metadata(psd)
    wpli_metadata = _checkpoint_metadata(wpli)

    _validate_branch_checkpoint(psd, psd_metadata, branch="psd")
    _validate_branch_checkpoint(wpli, wpli_metadata, branch="wpli")
    for field in MERGED_METADATA_FIELDS:
        if allow_mismatched_ssl_seeds and field == "base_seed":
            continue
        if psd_metadata.get(field) != wpli_metadata.get(field):
            raise ValueError(
                "Cannot merge SSL checkpoints with metadata mismatch for "
                f"{field}: psd={psd_metadata.get(field)!r}, wpli={wpli_metadata.get(field)!r}"
            )

    merged_state: dict[str, torch.Tensor] = {}
    for branch, checkpoint in (("psd", psd), ("wpli", wpli)):
        encoder_state = checkpoint.get("encoder_state_dict")
        if not isinstance(encoder_state, dict):
            raise ValueError(f"{branch} SSL checkpoint is missing encoder_state_dict.")
        prefixed = prefix_branch_encoder_state_dict(branch, encoder_state)
        collisions = sorted(set(merged_state) & set(prefixed))
        if collisions:
            joined = ", ".join(collisions)
            raise ValueError(f"Merged SSL state would overwrite key(s): {joined}")
        merged_state.update(prefixed)

    shared_metadata = {
        field: psd_metadata[field]
        for field in MERGED_METADATA_FIELDS
        if field in psd_metadata and (field != "base_seed" or not allow_mismatched_ssl_seeds)
    }
    return merged_state, {"psd": dict(psd_metadata), "wpli": dict(wpli_metadata), "shared": shared_metadata}


def extract_wpli_encoder_from_dual_segment_barlow_checkpoint(
    *,
    dual_checkpoint_path: str | Path,
    output_path: str | Path,
    expected_dual_metadata: Mapping[str, object],
    branch_metadata: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Extract WPLI encoder weights from a strict LOSO dual Segment Barlow checkpoint.

    The saved branch checkpoint is a new SSL encoder checkpoint. It carries the
    WPLI branch metadata used by downstream consumers and embeds the original
    dual metadata for provenance.
    """

    checkpoint_path = Path(dual_checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Dual Segment Barlow checkpoint must be a dictionary.")
    checkpoint_type = str(checkpoint.get("checkpoint_type", ""))
    metadata = _dual_checkpoint_metadata(checkpoint)
    _reject_supervised_checkpoint(checkpoint_type=checkpoint_type, metadata=metadata)
    validate_dual_segment_barlow_checkpoint_metadata(metadata, expected_dual_metadata)

    state_dict = _prefixed_state_dict_from_checkpoint(checkpoint)
    extracted = _extract_prefixed_branch_encoder_state(state_dict, WPLI_BRANCH_ENCODER_PREFIX)
    saved_metadata = dict(branch_metadata)
    saved_metadata.update(
        {
            "checkpoint_type": "segment_ssl_encoder",
            "branch": "wpli",
            "ssl_objective": "barlow",
            "segment_ssl_method": "segment_barlow",
            "derived_from_dual_checkpoint": True,
            "derived_from_dual_checkpoint_path": str(checkpoint_path),
            "source_dual_checkpoint_type": checkpoint_type,
            "source_dual_feature_manifest_hash": metadata["source_feature_manifest_hash"],
            "source_dual_checkpoint_metadata": dict(metadata),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_ssl_encoder_checkpoint(
        output_path,
        saved_metadata,
        extracted,
        prefixed_state_dict=prefix_branch_encoder_state_dict("wpli", extracted),
    )
    load_ssl_encoder_checkpoint(output_path, saved_metadata)
    return extracted


def validate_dual_segment_barlow_checkpoint_metadata(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    branch = observed.get("branch")
    branches = observed.get("branches")
    has_dual_branch = branch in {None, "dual", "psd+wpli", "psd_wpli", "psd-wpli", "merged"}
    has_dual_branches = isinstance(branches, (list, tuple, set)) and {"psd", "wpli"}.issubset(set(branches))
    if not has_dual_branch and not has_dual_branches:
        raise ValueError(f"Dual Segment Barlow checkpoint has non-dual branch metadata: {branch!r}")
    for field in DUAL_SEGMENT_BARLOW_REQUIRED_METADATA_FIELDS:
        if field not in observed:
            raise ValueError(f"Dual Segment Barlow metadata missing required field: {field}")
        if field not in expected:
            raise ValueError(f"Expected dual Segment Barlow metadata missing required field: {field}")
        if observed[field] != expected[field]:
            raise ValueError(
                "Dual Segment Barlow metadata mismatch for "
                f"{field}: observed={observed[field]!r}, expected={expected[field]!r}"
            )
    if observed["ssl_objective"] != "barlow":
        raise ValueError("Dual Segment Barlow checkpoint ssl_objective must be 'barlow'.")
    if observed["segment_ssl_method"] != "dual_segment_barlow":
        raise ValueError("Dual Segment Barlow checkpoint segment_ssl_method must be 'dual_segment_barlow'.")
    if observed["excluded_subject_id"] != observed["test_subject_id"]:
        raise ValueError("Dual Segment Barlow checkpoint excluded_subject_id must equal test_subject_id.")


def _coerce_checkpoint(checkpoint: str | Path | Mapping[str, object]) -> dict[str, object]:
    if isinstance(checkpoint, (str, Path)):
        return load_ssl_encoder_checkpoint(checkpoint)
    return dict(checkpoint)


def _dual_checkpoint_metadata(checkpoint: Mapping[str, object]) -> dict[str, object]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Dual Segment Barlow checkpoint is missing metadata.")
    if isinstance(metadata.get("shared"), dict):
        resolved = dict(metadata["shared"])
        resolved.setdefault("branch", "dual")
        for branch in ("wpli", "psd"):
            branch_metadata = metadata.get(branch)
            if not isinstance(branch_metadata, dict):
                continue
            for field in DUAL_SEGMENT_BARLOW_REQUIRED_METADATA_FIELDS:
                resolved.setdefault(field, branch_metadata.get(field))
        resolved.setdefault("source_dual_checkpoint_metadata", dict(metadata))
        return resolved
    return dict(metadata)


def _prefixed_state_dict_from_checkpoint(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    for field in ("state_dict", "pretrained_state_dict", "prefixed_state_dict", "encoder_state_dict"):
        state_dict = checkpoint.get(field)
        if isinstance(state_dict, dict) and any(str(key).startswith(WPLI_BRANCH_ENCODER_PREFIX) for key in state_dict):
            return state_dict
    if any(str(key).startswith(WPLI_BRANCH_ENCODER_PREFIX) for key in checkpoint):
        return checkpoint  # type: ignore[return-value]
    raise ValueError("Dual Segment Barlow checkpoint does not contain WPLI encoder weights.")


def _extract_prefixed_branch_encoder_state(
    state_dict: Mapping[str, object],
    prefix: str,
) -> dict[str, torch.Tensor]:
    extracted = {
        str(key).removeprefix(prefix): value.detach().cpu().clone()
        for key, value in state_dict.items()
        if str(key).startswith(prefix) and isinstance(value, torch.Tensor)
    }
    if not extracted:
        raise ValueError("Dual Segment Barlow checkpoint does not contain WPLI encoder weights.")
    return extracted


def _reject_supervised_checkpoint(*, checkpoint_type: str, metadata: Mapping[str, object]) -> None:
    supervised_markers = {
        "supervised_training",
        "supervised_finetune",
        "fine_tuned_from_ssl",
        "finetuned_from_ssl",
        "supervised_epochs",
        "transfer_mode",
    }
    for marker in supervised_markers:
        if marker in metadata and metadata[marker]:
            raise ValueError("Refusing to reuse supervised-trained checkpoint as an SSL encoder.")
    text_fields = [
        checkpoint_type,
        str(metadata.get("checkpoint_type", "")),
        str(metadata.get("training_stage", "")),
        str(metadata.get("checkpoint_role", "")),
        str(metadata.get("experiment_name", "")),
    ]
    lowered = " ".join(text_fields).lower()
    if any(token in lowered for token in ("supervised", "fine_tune", "finetune", "transfer")):
        raise ValueError("Refusing to reuse supervised-trained checkpoint as an SSL encoder.")


def _checkpoint_metadata(checkpoint: Mapping[str, object]) -> dict[str, object]:
    if checkpoint.get("checkpoint_type") != "segment_ssl_encoder":
        raise ValueError("SSL checkpoint has invalid checkpoint_type.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("SSL checkpoint is missing metadata.")
    resolved = dict(metadata)
    resolved.setdefault("checkpoint_type", checkpoint["checkpoint_type"])
    return resolved


def _validate_branch_checkpoint(
    checkpoint: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    branch: str,
) -> None:
    expected = dict(metadata)
    expected["branch"] = branch
    expected["ssl_objective"] = "barlow"
    validate_ssl_checkpoint_metadata(metadata, expected)
    if checkpoint.get("checkpoint_type") != "segment_ssl_encoder":
        raise ValueError("SSL checkpoint has invalid checkpoint_type.")


def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _safe_token(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_-")
    if not safe:
        raise ValueError("SSL checkpoint name cannot be empty.")
    return safe
