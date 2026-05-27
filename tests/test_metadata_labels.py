from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eeg_recovery.config import ConfigError, PathConfig, load_path_config
from eeg_recovery.metadata.labels import (
    MODEL_INPUT_COLUMNS,
    MetadataError,
    load_supervised_label_table,
    model_input_metadata,
)
from eeg_recovery.metadata.subjects import normalize_subject_id


EXPECTED_SUPERVISED_ROWS = [
    ("sub01", 71, "男", 17, "左", 63, 65, 80, 95),
    ("sub05", 66, "男", 20, "左", 9, 12, 20, 35),
    ("sub07", 58, "女", 21, "右", 22, 24, 35, 50),
    ("sub08", 70, "女", 18, "右", 10, 12, 30, 45),
    ("sub09", 64, "男", 16, "左", 6, 11, 15, 30),
    ("sub10", 61, "男", 15, "右", 59, 64, 70, 90),
    ("sub011", 59, "男", 14, "左", 7, 13, 25, 45),
    ("sub013", 62, "女", 13, "右", 25, 41, 40, 65),
    ("sub14", 60, "女", 12, "左", 60, 62, 75, 85),
    ("sub15", 56, "男", 11, "左", 51, 63, 60, 80),
    ("sub16", 67, "男", 10, "右", 23, 30, 45, 60),
    ("sub17", 65, "女", 9, "右", 51, 62, 55, 75),
    ("sub18", 57, "男", 8, "左", 64, 65, 85, 95),
    ("sub20", 63, "女", 7, "左", 62, 64, 80, 90),
    ("sub22", 69, "男", 6, "右", 61, 63, 70, 85),
    ("sub24", 55, "女", 5, "左", 11, 18, 25, 45),
    ("sub27", 68, "男", 4, "左", 61, 64, 75, 88),
    ("sub28", 52, "女", 3, "右", 63, 66, 82, 96),
    ("sub29", 72, "男", 2, "左", 61, 65, 78, 92),
]


def write_synthetic_workbooks(
    tmp_path: Path,
    rows: list[tuple[object, ...]] | None = None,
    *,
    integrity_id_header: str = "患者ID",
    whitespace_headers: bool = False,
    extra_clinical_rows: list[tuple[object, ...]] | None = None,
) -> PathConfig:
    integrity_path = tmp_path / "integrity.xlsx"
    clinical_path = tmp_path / "clinical.xlsx"
    patient_eeg_root = tmp_path / "patient_eeg"
    health_eeg_root = tmp_path / "health_eeg"
    ced_path = tmp_path / "standard_1005.ced"
    patient_eeg_root.mkdir()
    health_eeg_root.mkdir()
    ced_path.write_text("placeholder", encoding="utf-8")

    rows = rows or EXPECTED_SUPERVISED_ROWS
    supervised_ids = [row[0] for row in rows]
    pd.DataFrame({integrity_id_header: supervised_ids}).to_excel(integrity_path, index=False)

    columns = [
        "编号",
        "姓名",
        "年龄",
        "病程",
        "性别",
        "患病侧（手）",
        "治疗前FMA",
        "治疗后FMA",
        "治疗前MBI",
        "治疗后MBI",
    ]
    if whitespace_headers:
        columns = [
            " 编号 ",
            "姓名",
            " 年龄\n",
            "病程 ",
            " 性别",
            "患病侧（手）\n",
            "治疗前\nFMA",
            " 治疗后FMA ",
            "治疗前 MBI",
            "治疗后\nMBI",
        ]

    clinical_rows = [["编号", None, None, None, None, None, None, None, None, None]]
    clinical_rows.append(columns)
    for row in rows:
        clinical_rows.append([row[0], "匿名", *row[1:]])
    for row in extra_clinical_rows or []:
        clinical_rows.append([row[0], "匿名", *row[1:]])
    clinical_rows.append(["健康受试者组", None, None, None, None, None, None, None, None, None])
    pd.DataFrame(clinical_rows).to_excel(clinical_path, header=False, index=False)

    return PathConfig(
        patient_info_integrity_xlsx=integrity_path,
        patient_info_clinical_xlsx=clinical_path,
        patient_eeg_root=patient_eeg_root,
        health_eeg_root=health_eeg_root,
        standard_1005_ced=ced_path,
        output_root=tmp_path / "outputs",
    )


def test_normalize_subject_id_uses_two_minimum_digits_without_extra_padding() -> None:
    examples = {
        "sub1": "sub01",
        "sub01": "sub01",
        "sub011": "sub11",
        "sub013": "sub13",
        "sub021": "sub21",
    }

    assert {
        raw: normalize_subject_id(raw)
        for raw in examples
    } == examples


def test_load_supervised_label_table_from_synthetic_workbooks(tmp_path: Path) -> None:
    config = write_synthetic_workbooks(tmp_path)

    labels = load_supervised_label_table(config)

    assert len(labels) == 19
    assert labels["subject_id"].is_unique
    assert labels["Residual"].median() == 1.5
    assert labels["label"].value_counts().sort_index().to_dict() == {0: 9, 1: 10}

    sub22 = labels.set_index("subject_id").loc["sub22"]
    assert sub22["Residual"] == 1.5
    assert sub22["label"] == 1


def test_load_supervised_label_table_raises_for_missing_supervised_fma_post(tmp_path: Path) -> None:
    rows = list(EXPECTED_SUPERVISED_ROWS)
    sub22 = list(rows[14])
    sub22[6] = None
    rows[14] = tuple(sub22)
    config = write_synthetic_workbooks(tmp_path, rows)

    with pytest.raises(MetadataError, match="FMA_post"):
        load_supervised_label_table(config)


def test_load_supervised_label_table_accepts_whitespace_normalized_headers(tmp_path: Path) -> None:
    config = write_synthetic_workbooks(tmp_path, whitespace_headers=True)

    labels = load_supervised_label_table(config)

    assert len(labels) == 19
    assert labels["Residual"].median() == 1.5


def test_load_supervised_label_table_accepts_whitespace_normalized_integrity_header(
    tmp_path: Path,
) -> None:
    config = write_synthetic_workbooks(tmp_path, integrity_id_header=" 患者\nID ")

    labels = load_supervised_label_table(config)

    assert len(labels) == 19
    assert labels["subject_id"].tolist()[0] == "sub01"


def test_load_supervised_label_table_derives_ids_from_clinical_layout_workbook(
    tmp_path: Path,
) -> None:
    clinical_path = tmp_path / "clinical_layout.xlsx"
    patient_eeg_root = tmp_path / "patient_eeg"
    health_eeg_root = tmp_path / "health_eeg"
    ced_path = tmp_path / "standard_1005.ced"
    patient_eeg_root.mkdir()
    health_eeg_root.mkdir()
    ced_path.write_text("placeholder", encoding="utf-8")

    columns = [
        "编号",
        "姓名",
        "年龄",
        "病程",
        "性别",
        "患病侧",
        "治疗前FMA",
        "治疗后FMA",
        "治疗前MBI",
        "治疗后MBI",
    ]
    rows = [
        [row[0], "匿名", f"{row[1]}岁", f"{row[3]}天", row[2], f"{row[4]}手", *row[5:]]
        for row in EXPECTED_SUPERVISED_ROWS
    ]
    rows.append(["sub021", "匿名", "47岁", "1年", "女", "右手", "非常好", "非常好", None, None])
    pd.DataFrame(rows, columns=columns).to_excel(clinical_path, index=False)

    config = PathConfig(
        patient_info_integrity_xlsx=clinical_path,
        patient_info_clinical_xlsx=clinical_path,
        patient_eeg_root=patient_eeg_root,
        health_eeg_root=health_eeg_root,
        standard_1005_ced=ced_path,
        output_root=tmp_path / "outputs",
    )

    labels = load_supervised_label_table(config)

    assert len(labels) == 19
    assert "sub21" not in set(labels["subject_id"])
    assert labels["Residual"].median() == 1.5
    assert labels["label"].value_counts().sort_index().to_dict() == {0: 9, 1: 10}
    assert labels.set_index("subject_id").loc["sub01", "affected_hand"] == "左"


def test_load_supervised_label_table_raises_for_duplicate_supervised_clinical_subject(
    tmp_path: Path,
) -> None:
    duplicate_sub22 = ("sub022", 70, "男", 6, "右", 61, 63, 70, 85)
    config = write_synthetic_workbooks(tmp_path, extra_clinical_rows=[duplicate_sub22])

    with pytest.raises(MetadataError, match="duplicate.*sub22"):
        load_supervised_label_table(config)


def test_model_input_metadata_excludes_post_treatment_labels_and_derived_columns(tmp_path: Path) -> None:
    config = write_synthetic_workbooks(tmp_path)
    labels = load_supervised_label_table(config)

    inputs = model_input_metadata(labels)

    assert list(inputs.columns) == ["subject_id", *MODEL_INPUT_COLUMNS]
    assert "FMA_post" not in inputs.columns
    assert "MBI_post" not in inputs.columns
    assert "Delta_FMA_pred" not in inputs.columns
    assert "Delta_FMA_obs" not in inputs.columns
    assert "Residual" not in inputs.columns
    assert "label" not in inputs.columns


def test_load_supervised_label_table_from_real_workbooks() -> None:
    try:
        config = load_path_config(Path("configs/paths.example.yaml"))
    except ConfigError as error:
        pytest.skip(f"External workbook paths are unavailable: {error}")

    labels = load_supervised_label_table(config)

    assert len(labels) == 19
    assert labels["Residual"].median() == 1.5
    assert labels["label"].value_counts().sort_index().to_dict() == {0: 9, 1: 10}


def test_post_treatment_scores_are_not_model_input_helpers() -> None:
    assert "FMA_post" not in MODEL_INPUT_COLUMNS
    assert "MBI_post" not in MODEL_INPUT_COLUMNS
