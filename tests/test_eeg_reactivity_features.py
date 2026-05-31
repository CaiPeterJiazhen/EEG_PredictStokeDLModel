from __future__ import annotations

import numpy as np

from eeg_recovery.features.eeg_reactivity import reactivity_features_from_named_values


def test_reactivity_features_pair_eo_ec_named_values():
    values = np.asarray([2.0, 6.0, 0.4, 0.1, 9.0], dtype=np.float32)
    names = (
        "psd_eo_alpha_mean",
        "psd_ec_alpha_mean",
        "wpli_eo_beta_interhemispheric_mean",
        "wpli_ec_beta_interhemispheric_mean",
        "psd_eo_unpaired",
    )

    vector = reactivity_features_from_named_values(values, names)

    assert "reactivity_psd_alpha_mean_ec_minus_eo" in vector.feature_names
    assert "reactivity_psd_alpha_mean_ec_minus_eo_fractional_change" in vector.feature_names
    assert "reactivity_wpli_beta_interhemispheric_mean_ec_minus_eo" in vector.feature_names
    assert "unpaired" not in " ".join(vector.feature_names)
    assert np.isfinite(vector.values).all()
    lookup = dict(zip(vector.feature_names, vector.values, strict=True))
    assert lookup["reactivity_psd_alpha_mean_ec_minus_eo"] == 4.0
    np.testing.assert_allclose(
        lookup["reactivity_psd_alpha_mean_ec_minus_eo_fractional_change"],
        0.5,
        rtol=1e-6,
    )


def test_reactivity_features_reject_misaligned_lengths():
    try:
        reactivity_features_from_named_values(np.asarray([1.0, 2.0]), ("psd_eo_alpha_mean",))
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("Expected feature/name length mismatch to fail.")
