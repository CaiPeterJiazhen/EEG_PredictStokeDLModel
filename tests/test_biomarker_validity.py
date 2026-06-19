from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from eeg_recovery.validation.feature_overlap import compute_feature_overlap
from eeg_recovery.validation.healthy_reference import (
    build_healthy_reference,
    compute_health_distance,
    compute_normalization_index,
    transform_to_healthy_zscore,
)
from eeg_recovery.validation.statistics import (
    adjust_pvalues_bh,
    partial_spearman_association,
    spearman_association,
)


def test_healthy_reference_zscore_distance_and_normalization_direction() -> None:
    healthy = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[3.0, 6.0], [9.0, 12.0]],
        ],
        dtype=float,
    )

    reference = build_healthy_reference(healthy)
    z = transform_to_healthy_zscore(np.array([[3.0, 6.0], [9.0, 12.0]]), reference)

    np.testing.assert_allclose(reference.center, np.array([[2.0, 4.0], [6.0, 8.0]]))
    np.testing.assert_allclose(reference.scale, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(z, np.ones((2, 2)))
    assert compute_health_distance(z) == 1.0
    assert compute_normalization_index(4.0, 2.0) > 0.0


def test_statistics_helpers_emit_q_values_and_partial_spearman_fields() -> None:
    q_values = adjust_pvalues_bh([0.01, 0.02, 0.20])
    np.testing.assert_allclose(q_values, np.array([0.03, 0.03, 0.20]))

    association = spearman_association(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 3.0, 2.0, 5.0]),
    )
    partial = partial_spearman_association(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.0, 2.5, 2.8, 4.2, 5.1]),
        pd.DataFrame({"age": [61, 62, 63, 64, 65], "FMA_pre": [10, 12, 13, 15, 16]}),
    )

    assert {"rho", "p_value", "n"} <= set(association)
    assert association["rho"] > 0.0
    assert {"partial_rho", "p_value", "n", "covariates"} <= set(partial)
    assert partial["covariates"] == "age;FMA_pre"


def test_feature_overlap_permutation_runs_for_model_and_evidence_sets() -> None:
    result = compute_feature_overlap(
        model_attribution_features={"psd|EO|C3|Alpha", "wpli|EC|C3-C4|Beta High"},
        evidence_sets={
            "abnormal_features": {"psd|EO|C3|Alpha", "psd|EC|C4|Theta"},
            "normalization_features": {"wpli|EC|C3-C4|Beta High"},
            "outcome_associated_features": {"psd|EO|F3|Alpha"},
        },
        universe_features={
            "psd|EO|C3|Alpha",
            "psd|EC|C4|Theta",
            "wpli|EC|C3-C4|Beta High",
            "psd|EO|F3|Alpha",
            "wpli|EO|F3-F4|Alpha",
        },
        n_permutation=25,
        random_state=0,
    )

    assert set(result["evidence_set"]) == {
        "abnormal_features",
        "normalization_features",
        "outcome_associated_features",
    }
    assert {"overlap_count", "fisher_p_value", "permutation_p_value", "q_value"} <= set(result.columns)


def test_biomarker_validity_cli_outputs_group_level_report_without_prediction_probabilities(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.xlsx"
    features_dir = tmp_path / "features"
    explainability_dir = tmp_path / "explainability"
    out_dir = tmp_path / "biomarker_validity"
    _write_synthetic_metadata(metadata_path)
    _write_synthetic_features(features_dir)
    _write_synthetic_explainability(explainability_dir)

    command = [
        sys.executable,
        str(Path("scripts") / "11_validate_psd_wpli_biomarker_validity.py"),
        "--metadata",
        str(metadata_path),
        "--features-dir",
        str(features_dir),
        "--explainability-dir",
        str(explainability_dir),
        "--out-dir",
        str(out_dir),
        "--states",
        "EO",
        "EC",
        "--modalities",
        "psd",
        "wpli",
        "--n-bootstrap",
        "20",
        "--n-permutation",
        "20",
    ]
    subprocess.run(command, check=True)

    forbidden = {
        "baseline_prediction_loso_metrics.csv",
        "baseline_prediction_subject_probs.csv",
        "healthy_distance_classifier.pkl",
    }
    assert forbidden.isdisjoint({path.name for path in out_dir.rglob("*")})
    assert not any("prob" in path.name.lower() for path in out_dir.rglob("*"))

    report = (out_dir / "biomarker_validity_report.md").read_text(encoding="utf-8")
    assert "group-level association, not individual prediction" in report
    assert "post14 PSD/wPLI are used only for longitudinal validation" in report

    normalization = pd.read_csv(out_dir / "normalization_indices.csv")
    assert "baseline_prediction_input" in normalization.columns
    assert not normalization["baseline_prediction_input"].any()
    assert (out_dir / "skipped_subjects.csv").exists()


def test_cli_reads_nested_all_stage_manifest_and_redacts_healthy_names(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.xlsx"
    features_dir = tmp_path / "features_all_stages"
    out_dir = tmp_path / "biomarker_validity"
    _write_nested_metadata_with_named_healthy(metadata_path)
    _write_nested_manifest_features(features_dir)

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "11_validate_psd_wpli_biomarker_validity.py"),
            "--metadata",
            str(metadata_path),
            "--features-dir",
            str(features_dir),
            "--out-dir",
            str(out_dir),
            "--states",
            "EO",
            "EC",
            "--modalities",
            "psd",
            "wpli",
            "--n-bootstrap",
            "10",
            "--n-permutation",
            "10",
        ],
        check=True,
    )

    distances_text = (out_dir / "healthy_reference_distances.csv").read_text(encoding="utf-8")
    report_text = (out_dir / "biomarker_validity_report.md").read_text(encoding="utf-8")
    assert "张三" not in distances_text
    assert "张三" not in report_text
    assert "healthy_001" in distances_text
    assert (out_dir / "normalization_indices.csv").exists()


def _write_synthetic_metadata(path: Path) -> None:
    patients = pd.DataFrame(
        {
            "编号": ["sub01", "sub02", "sub03", "sub04"],
            "姓名": ["A", "B", "C", "D"],
            "年龄": [60, 61, 62, 63],
            "病程": [10, 11, 12, 13],
            "性别": ["男", "女", "男", "女"],
            "患病侧（手）": ["左", "右", "左", "右"],
            "治疗前FMA": [30, 30, 30, 30],
            "治疗后FMA": [54, 50, 36, 34],
            "治疗前MBI": [40, 41, 42, 43],
            "治疗后MBI": [70, 68, 50, 49],
        }
    )
    health = pd.DataFrame(
        {
            "编号": ["hc01", "hc02", "hc03"],
            "姓名": ["E", "F", "G"],
            "年龄": [55, 56, 57],
            "病程": [np.nan, np.nan, np.nan],
            "性别": ["女", "男", "女"],
            "患病侧（手）": [np.nan, np.nan, np.nan],
            "治疗前FMA": [np.nan, np.nan, np.nan],
            "治疗后FMA": [np.nan, np.nan, np.nan],
            "治疗前MBI": [np.nan, np.nan, np.nan],
            "治疗后MBI": [np.nan, np.nan, np.nan],
        }
    )
    pd.concat([patients, health], ignore_index=True).to_excel(path, index=False)


def _write_synthetic_features(features_dir: Path) -> None:
    psd_dir = features_dir / "psd"
    fc_dir = features_dir / "fc"
    psd_dir.mkdir(parents=True)
    fc_dir.mkdir(parents=True)
    health_ids = ["hc01", "hc02", "hc03"]
    patient_ids = ["sub01", "sub02", "sub03", "sub04"]
    frequency_bins = np.array([2.0, 10.0, 24.0], dtype=float)
    channels = np.array(["C3", "C4"])
    band_names = np.array(["Alpha", "Beta High"])
    edge_list = np.array([["F3", "F4"], ["C3", "C4"]])

    for offset, subject_id in enumerate(health_ids):
        psd = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=float) + offset * 0.1
        wpli = np.array([[0.20, 0.30], [0.40, 0.50]], dtype=float) + offset * 0.02
        for state in ("EO", "EC"):
            np.savez_compressed(
                psd_dir / f"{subject_id}_{state}_psd.npz",
                psd=psd,
                frequency_bins=frequency_bins,
                channel_names_after_alignment=channels,
                subject_id=np.array(subject_id),
                state=np.array(state),
            )
            np.savez_compressed(
                fc_dir / f"{subject_id}_{state}_fc.npz",
                wpli=wpli,
                edge_list=edge_list,
                band_names=band_names,
                subject_id=np.array(subject_id),
                state=np.array(state),
            )

    for index, subject_id in enumerate(patient_ids):
        baseline_shift = 0.9 if index < 2 else 1.8
        post_shift = 0.25 if index < 2 else 1.4
        for state in ("EO", "EC"):
            np.savez_compressed(
                psd_dir / f"{subject_id}_{state}_psd.npz",
                psd=np.array([[1.1, 2.1, 3.1], [1.6, 2.6, 3.6]], dtype=float) + baseline_shift,
                frequency_bins=frequency_bins,
                channel_names_after_alignment=channels,
                subject_id=np.array(subject_id),
                state=np.array(state),
            )
            np.savez_compressed(
                psd_dir / f"{subject_id}_post14_{state}_psd.npz",
                psd=np.array([[1.1, 2.1, 3.1], [1.6, 2.6, 3.6]], dtype=float) + post_shift,
                frequency_bins=frequency_bins,
                channel_names_after_alignment=channels,
                subject_id=np.array(subject_id),
                state=np.array(state),
            )
            np.savez_compressed(
                fc_dir / f"{subject_id}_{state}_fc.npz",
                wpli=np.array([[0.21, 0.31], [0.41, 0.51]], dtype=float) + baseline_shift * 0.1,
                edge_list=edge_list,
                band_names=band_names,
                subject_id=np.array(subject_id),
                state=np.array(state),
            )
            np.savez_compressed(
                fc_dir / f"{subject_id}_post14_{state}_fc.npz",
                wpli=np.array([[0.21, 0.31], [0.41, 0.51]], dtype=float) + post_shift * 0.1,
                edge_list=edge_list,
                band_names=band_names,
                subject_id=np.array(subject_id),
                state=np.array(state),
            )


def _write_synthetic_explainability(explainability_dir: Path) -> None:
    explainability_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "state": ["EO", "EC"],
            "channel": ["C3", "C4"],
            "frequency_hz": [10.0, 24.0],
            "band": ["Alpha", "Beta High"],
            "mean_abs_attribution": [0.9, 0.4],
        }
    ).to_csv(explainability_dir / "psd_channel_frequency_top_features.csv", index=False)
    pd.DataFrame(
        {
            "state": ["EC", "EO"],
            "channel_i": ["C3", "F3"],
            "channel_j": ["C4", "F4"],
            "band": ["Beta High", "Alpha"],
            "mean_abs_attribution": [0.8, 0.3],
        }
    ).to_csv(explainability_dir / "wpli_top_edges.csv", index=False)


def _write_nested_metadata_with_named_healthy(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "编号": ["sub01", "sub02", "sub001张三", "sub002李四", "sub003王五"],
            "姓名": ["A", "B", "张三", "李四", "王五"],
            "年龄": [60, 61, 55, 56, 57],
            "病程": [10, 11, np.nan, np.nan, np.nan],
            "性别": ["男", "女", "女", "男", "女"],
            "患病侧（手）": ["左", "右", np.nan, np.nan, np.nan],
            "治疗前FMA": [30, 31, np.nan, np.nan, np.nan],
            "治疗后FMA": [50, 36, np.nan, np.nan, np.nan],
            "治疗前MBI": [40, 41, np.nan, np.nan, np.nan],
            "治疗后MBI": [70, 50, np.nan, np.nan, np.nan],
        }
    )
    frame.to_excel(path, index=False)


def _write_nested_manifest_features(features_dir: Path) -> None:
    manifest_rows = []
    specs = [
        ("health", "health", "sub001张三"),
        ("health", "health", "sub002李四"),
        ("health", "health", "sub003王五"),
        ("patient", "基线", "sub01"),
        ("patient", "最终", "sub01"),
        ("patient", "基线", "sub02"),
        ("patient", "最终", "sub02"),
    ]
    for group, stage, subject_id in specs:
        for state in ("EO", "EC"):
            for modality in ("psd", "wpli"):
                subdir = features_dir / ("psd" if modality == "psd" else "fc") / group / stage
                subdir.mkdir(parents=True, exist_ok=True)
                suffix = "psd" if modality == "psd" else "fc"
                path = subdir / f"{subject_id}_{state}_{suffix}.npz"
                if modality == "psd":
                    _write_psd_npz(path, subject_id, state, shift=_stage_shift(group, stage, subject_id))
                else:
                    _write_fc_npz(path, subject_id, state, shift=_stage_shift(group, stage, subject_id) * 0.1)
                manifest_rows.append(
                    {
                        "group": group,
                        "stage": stage,
                        "subject_id": subject_id,
                        "state": state,
                        "modality": modality,
                        "feature_path": path.as_posix(),
                        "status": "written",
                    }
                )
    pd.DataFrame(manifest_rows).to_csv(features_dir / "feature_manifest.csv", index=False)


def _stage_shift(group: str, stage: str, subject_id: str) -> float:
    if group == "health":
        return {"sub001张三": 0.0, "sub002李四": 0.1, "sub003王五": 0.2}[subject_id]
    if stage == "基线":
        return 0.8 if subject_id == "sub01" else 1.6
    return 0.2 if subject_id == "sub01" else 1.3


def _write_psd_npz(path: Path, subject_id: str, state: str, *, shift: float) -> None:
    np.savez_compressed(
        path,
        psd=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=float) + shift,
        frequency_bins=np.array([2.0, 10.0, 24.0], dtype=float),
        channel_names_after_alignment=np.array(["C3", "C4"]),
        subject_id=np.array(subject_id),
        state=np.array(state),
    )


def _write_fc_npz(path: Path, subject_id: str, state: str, *, shift: float) -> None:
    np.savez_compressed(
        path,
        wpli=np.array([[0.20, 0.30], [0.40, 0.50]], dtype=float) + shift,
        edge_list=np.array([["F3", "F4"], ["C3", "C4"]]),
        band_names=np.array(["Alpha", "Beta High"]),
        subject_id=np.array(subject_id),
        state=np.array(state),
    )
