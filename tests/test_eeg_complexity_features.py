from __future__ import annotations

import numpy as np

from eeg_recovery.channels.mapping import CANONICAL_CHANNELS_62
from eeg_recovery.features.complexity import (
    ComplexityConfig,
    compute_complexity_summary_from_aligned_data,
    higuchi_fractal_dimension,
)


def test_higuchi_fractal_dimension_is_finite_for_nonconstant_signal():
    signal = np.sin(np.linspace(0.0, 20.0 * np.pi, 2048)) + 0.1 * np.sin(np.linspace(0.0, 80.0 * np.pi, 2048))

    value = higuchi_fractal_dimension(signal, kmax=8)

    assert np.isfinite(value)
    assert 1.0 <= value <= 2.5


def test_higuchi_fractal_dimension_handles_constant_signal():
    value = higuchi_fractal_dimension(np.ones(512), kmax=8)

    assert np.isfinite(value)
    assert value == 1.0


def test_complexity_summary_outputs_interpretable_global_roi_and_asymmetry_features():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(CANONICAL_CHANNELS_62), 4096)).astype(np.float32)

    vector = compute_complexity_summary_from_aligned_data(
        data,
        channel_names=CANONICAL_CHANNELS_62,
        state="EO",
        config=ComplexityConfig(kmax=4, max_samples=2048),
    )

    assert vector.values.dtype == np.float32
    assert np.isfinite(vector.values).all()
    assert vector.values.shape == (len(vector.feature_names),)
    assert "complexity_eo_higuchi_fd_mean" in vector.feature_names
    assert "complexity_eo_higuchi_fd_ipsilesional_mean" in vector.feature_names
    assert "complexity_eo_higuchi_fd_contralesional_mean" in vector.feature_names
    assert "complexity_eo_higuchi_fd_ipsi_contra_signed_asymmetry" in vector.feature_names
    assert any(name.startswith("complexity_eo_frontal_higuchi_fd") for name in vector.feature_names)
