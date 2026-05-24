from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.training.train_feature_ssl import FeatureSSLPairRecord
from eeg_recovery.training.train_structured_ssl import (
    StructuredSSLConfig,
    build_fc_graph_views,
    build_time_frequency_images,
    run_structured_ssl_pretraining,
)


def test_build_time_frequency_images_returns_channel_frequency_slices():
    sampling_rate = 250.0
    time = np.arange(0, 4.0, 1.0 / sampling_rate, dtype=np.float32)
    data = np.stack(
        [
            np.sin(2 * np.pi * (index + 1) * time)
            for index in range(62)
        ]
    ).astype(np.float32)

    images = build_time_frequency_images(
        data,
        sampling_rate=sampling_rate,
        max_windows=3,
        nperseg=128,
        noverlap=64,
    )

    assert images.shape == (3, 62, 90)
    assert np.isfinite(images).all()


def test_time_frequency_images_follow_affected_hand_alignment():
    sampling_rate = 250.0
    time = np.arange(0, 4.0, 1.0 / sampling_rate, dtype=np.float32)
    data = np.zeros((62, time.size), dtype=np.float32)
    c3_index = CANONICAL_CHANNELS_62.index("C3")
    c4_index = CANONICAL_CHANNELS_62.index("C4")
    data[c3_index] = np.sin(2 * np.pi * 10.0 * time)
    data[c4_index] = np.sin(2 * np.pi * 20.0 * time)

    images = build_time_frequency_images(
        data,
        sampling_rate=sampling_rate,
        max_windows=1,
        nperseg=256,
        noverlap=128,
        channel_names=CANONICAL_CHANNELS_62,
        affected_hand="左",
    )

    target_frequencies = np.linspace(0.5, 45.0, 90, dtype=np.float32)
    c4_peak = float(target_frequencies[int(np.argmax(images[0, c4_index]))])
    assert abs(c4_peak - 10.0) <= 0.75


def test_fc_graph_views_drop_incident_edges_for_selected_nodes():
    fc = np.ones((6, 2), dtype=np.float32)
    edge_index = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=np.int64,
    )

    view_a, view_b = build_fc_graph_views(
        fc,
        edge_index=edge_index,
        seed=1,
        node_dropout_prob=1.0,
        edge_dropout_prob=0.0,
        noise_std=0.0,
    )

    assert view_a.shape == fc.shape
    assert view_b.shape == fc.shape
    assert np.count_nonzero(view_a) == 0
    assert np.count_nonzero(view_b) == 0


def test_structured_ssl_pretraining_returns_psd_and_wpli_encoder_states():
    rng = np.random.default_rng(0)
    tfr_images = [
        rng.normal(size=(62, 90)).astype(np.float32),
        rng.normal(size=(62, 90)).astype(np.float32),
        rng.normal(size=(62, 90)).astype(np.float32),
    ]
    pairs = [
        FeatureSSLPairRecord(
            group="patient",
            subject_id=f"sub0{index + 1}",
            subject_key=f"patient:sub0{index + 1}",
            stage="基线",
            is_supervised_subject=True,
            modalities={
                "wpli": (
                    rng.normal(size=(1891, 6)).astype(np.float32),
                    rng.normal(size=(1891, 6)).astype(np.float32),
                )
            },
        )
        for index in range(3)
    ]
    config = StructuredSSLConfig(
        psd_epochs=1,
        fc_epochs=1,
        batch_size=2,
        embedding_dim=4,
        projection_dim=4,
        device="cpu",
        seed=3,
    )

    pretrained_state, history = run_structured_ssl_pretraining(tfr_images, pairs, config)

    assert any(key.startswith("branch_models.psd.encoder.encoder.") for key in pretrained_state)
    assert any(key.startswith("branch_models.wpli.encoder.encoder.") for key in pretrained_state)
    assert set(history["branch"]) == {"psd_tfr", "wpli_graph"}
    assert torch.isfinite(torch.stack([value.float().mean() for value in pretrained_state.values()])).all()


def test_structured_ssl_transfer_script_help_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-B", "scripts/08_train_structured_ssl_transfer.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--psd-ssl-epochs" in result.stdout
    assert "--fc-ssl-epochs" in result.stdout
    assert "--max-tfr-windows-per-record" in result.stdout
