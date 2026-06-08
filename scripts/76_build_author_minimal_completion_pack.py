from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "63_validate_author_submission_metadata.py"
CURRENT_METADATA = ROOT / "outputs" / "manuscript_package" / "author_submission_metadata_from_intake.json"
TEMPLATE_METADATA = ROOT / "docs" / "author_submission_metadata_template.json"
PROJECT_PREFILL = ROOT / "outputs" / "manuscript_package" / "author_submission_metadata_project_prefill.json"
REPLACEMENT_MAP = ROOT / "results" / "tables" / "author_field_replacement_map.csv"

OUTPUT_MD = ROOT / "docs" / "author_minimal_completion_pack.md"
OUTPUT_JSON = ROOT / "outputs" / "manuscript_package" / "author_minimal_completion_answers.json"
OUTPUT_CSV = ROOT / "results" / "tables" / "author_minimal_completion_pack.csv"


FIELD_PROMPTS = {
    "target_journal_reference_style": "确认最终投稿期刊、文章类型和参考文献格式。",
    "author_list_affiliations": "提供最终作者顺序、单位、通讯作者和邮箱。",
    "author_contributions": "按 CRediT 或目标期刊格式填写每位作者贡献，并确认所有作者批准终稿。",
    "ethics_approval": "提供伦理委员会全称、批件号、批准日期、适用地点和可投稿英文伦理声明。",
    "informed_consent": "说明知情同意路径、签署对象、书面/豁免状态、覆盖 EEG/tACS/临床评估和数据共享的范围。",
    "trial_or_study_registration": "提供注册平台和注册号；如果未注册，提供作者认可的未注册说明。",
    "study_site_dates_design": "提供医院/科室、招募起止日期、末次评估窗口、前瞻/回顾和单/多中心设计。",
    "eligibility_stroke_timing": "提供纳入排除标准、卒中亚型规则和发病至 EEG/tACS 的时间定义。",
    "tacs_device_electrodes": "补充 tACS 设备型号、电极尺寸/材料，并确认当前刺激方案。",
    "concurrent_rehabilitation": "确认是否并行常规康复，及其频率、单次时长、训练内容和一致性。",
    "tacs_safety_adverse_events": "提供不良事件、耐受性、退出/中止和安全监测方法的汇总。",
    "fma_assessors_timing": "补充 FMA-UE 评估者资质、盲法状态和治疗前后评估时间点。",
    "eeg_hardware_reference_impedance": "提供 EEG 放大器、采集软件、电极帽、在线参考、地线和阻抗阈值。",
    "resting_state_instructions": "确认睁眼/闭眼顺序、目标时长、注视说明和困倦监测。",
    "raw_eeg_preprocessing": "提供导出 .set/.fdt 前的滤波、陷波、重参考、坏道、ICA/伪迹和导出规则。",
    "data_repository_doi_scope": "提供数据仓储平台、DOI/访问号、版本、许可、公开范围和受控访问流程。",
    "code_repository_license": "提供代码仓储 URL/DOI、版本或 commit hash、代码许可证和可投稿英文声明。",
    "funding_competing_acknowledgements": "提供基金号、资助方角色、利益冲突和致谢内容。",
}


KEY_HELP = {
    "evidence_source": "支撑该字段的文件、批件、方案、设备记录、仓储记录或作者批准来源。",
    "ready_to_paste_statement_en": "可直接放入 manuscript/declarations/cover letter 的最终英文句子。",
    "ready_to_paste_methods_sentence_en": "可直接放入 Methods 的最终英文句子。",
    "ready_to_paste_data_availability_en": "含 DOI、访问路径和限制原因的最终英文 Data Availability 句子。",
    "ready_to_paste_code_availability_en": "含仓储、版本和许可的最终英文 Code Availability 句子。",
}


STATEMENT_PATTERNS = {
    "ethics_approval": (
        "Example pattern only: The study was approved by [ethics committee] "
        "([approval number], [approval date]) and was conducted in accordance "
        "with the Declaration of Helsinki."
    ),
    "informed_consent": (
        "Example pattern only: Written informed consent was obtained from all "
        "participants or their legally authorised representatives before EEG, "
        "tACS and clinical assessment procedures."
    ),
    "trial_or_study_registration": (
        "Example pattern only: This study was registered at [registry] under "
        "[registration number]. If not registered, provide an explicit "
        "author-approved non-registration statement."
    ),
    "data_repository_doi_scope": (
        "Controlled-access pattern only: Human-participant raw or linkable data "
        "are not publicly available because [privacy/consent/ethics reason]. "
        "Qualified researchers may request access from [institution/committee] "
        "subject to [ethics approval/data-use agreement]. Derived source data "
        "are available at [repository DOI/accession]."
    ),
    "code_repository_license": (
        "Example pattern only: The analysis code is available at [repository URL "
        "or DOI], version [tag/commit], under [software licence]."
    ),
}


def main() -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    validator = load_validator_module()
    metadata_path = CURRENT_METADATA if CURRENT_METADATA.exists() else TEMPLATE_METADATA
    metadata = load_fields(metadata_path)
    project_prefill = load_fields(PROJECT_PREFILL) if PROJECT_PREFILL.exists() else {}
    replacement_targets = load_replacement_targets()

    field_rows = build_field_rows(validator, metadata, project_prefill, replacement_targets)
    write_csv(field_rows)
    write_answer_skeleton(field_rows, metadata_path, project_prefill)
    OUTPUT_MD.write_text(build_markdown(field_rows, metadata_path), encoding="utf-8")

    print(OUTPUT_MD)
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print(summary_line(field_rows))


def load_validator_module():
    spec = importlib.util.spec_from_file_location("author_metadata_validator", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load validator from {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fields(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = data.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError(f"{path} does not contain a top-level fields object.")
    return fields


def load_replacement_targets() -> dict[str, dict[str, str]]:
    if not REPLACEMENT_MAP.exists():
        return {}
    with REPLACEMENT_MAP.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["field_id"]: row for row in csv.DictReader(handle)}


def build_field_rows(
    validator,
    metadata: dict[str, dict[str, Any]],
    project_prefill: dict[str, dict[str, Any]],
    replacement_targets: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rule in validator.FIELD_RULES:
        target = replacement_targets.get(rule.field_id, {})
        priority = target.get("priority") or rule.priority
        if priority == "optional":
            continue

        values = metadata.get(rule.field_id, {})
        if not isinstance(values, dict):
            values = {}
        required_keys = list(rule.required_keys)
        if "evidence_source" not in required_keys:
            required_keys.append("evidence_source")
        alternative_keys = sorted({key for group in rule.alternative_key_groups for key in group})
        prefill_values = project_prefill.get(rule.field_id, {})
        if not isinstance(prefill_values, dict):
            prefill_values = {}
        prefill_keys = [key for key, value in prefill_values.items() if not validator.is_blank(value)]
        keys_to_show = list(dict.fromkeys(required_keys + alternative_keys + prefill_keys))

        missing_required = [key for key in required_keys if validator.is_blank(values.get(key))]
        alternative_status = alternative_requirement_status(rule, values, validator)
        complete = not missing_required and not alternative_status
        if complete:
            continue
        for key in keys_to_show:
            prefill_value = prefill_values.get(key, "")
            rows.append(
                {
                    "field_id": rule.field_id,
                    "priority": priority,
                    "metadata_key": key,
                    "strict_required": key_requirement_label(key, required_keys, alternative_keys),
                    "missing_now": "yes" if validator.is_blank(values.get(key)) else "no",
                    "current_author_response": stringify(values.get(key, "")),
                    "project_prefill_suggestion": stringify(prefill_value),
                    "author_action": FIELD_PROMPTS.get(rule.field_id, "Provide author-approved metadata."),
                    "key_guidance": KEY_HELP.get(key, "Author-approved response required for final insertion."),
                    "alternative_requirement": alternative_status or "none",
                    "statement_pattern": STATEMENT_PATTERNS.get(rule.field_id, ""),
                    "target_files": target.get("target_files", ""),
                    "replacement_action": target.get("replacement_action", ""),
                }
            )
    return rows


def alternative_requirement_status(rule, values: dict[str, Any], validator) -> str:
    if not rule.alternative_key_groups:
        return ""
    for group in rule.alternative_key_groups:
        if all(not validator.is_blank(values.get(key)) for key in group):
            return ""
    return "provide one of: " + " OR ".join(" + ".join(group) for group in rule.alternative_key_groups)


def key_requirement_label(key: str, required_keys: list[str], alternative_keys: list[str]) -> str:
    if key in required_keys:
        return "yes"
    if key in alternative_keys:
        return "alternative"
    return "project_prefill_only"


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "field_id",
        "priority",
        "metadata_key",
        "strict_required",
        "missing_now",
        "current_author_response",
        "project_prefill_suggestion",
        "author_action",
        "key_guidance",
        "alternative_requirement",
        "statement_pattern",
        "target_files",
        "replacement_action",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_answer_skeleton(
    rows: list[dict[str, str]],
    metadata_path: Path,
    project_prefill: dict[str, dict[str, Any]],
) -> None:
    fields: dict[str, dict[str, str]] = {}
    suggestions: dict[str, dict[str, str]] = {}
    for row in rows:
        field = fields.setdefault(row["field_id"], {})
        field.setdefault(row["metadata_key"], row["current_author_response"])
        if row["project_prefill_suggestion"]:
            suggestions.setdefault(row["field_id"], {})[row["metadata_key"]] = row["project_prefill_suggestion"]

    payload = {
        "_instructions": {
            "purpose": "Minimal author-answer skeleton for fields that currently block final manuscript replacement.",
            "source_metadata_file": metadata_path.as_posix(),
            "do_not_guess": "Fill fields only with author-approved evidence. Project prefill suggestions are not automatically submission-ready.",
            "how_to_validate": (
                "After copying approved values into docs/author_submission_metadata_template.json "
                "or the XLSX intake workbook, run scripts/63_validate_author_submission_metadata.py --strict."
            ),
        },
        "fields": fields,
        "_project_prefill_suggestions": suggestions,
        "_project_prefill_source": PROJECT_PREFILL.as_posix() if project_prefill else "",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown(rows: list[dict[str, str]], metadata_path: Path) -> str:
    fields = sorted({row["field_id"] for row in rows})
    blocking = sorted({row["field_id"] for row in rows if row["priority"].startswith("blocking")})
    high = sorted({row["field_id"] for row in rows if row["priority"] == "high"})

    lines = [
        "# Author Minimal Completion Pack",
        "",
        "This pack compresses the final author-metadata blocker into the smallest actionable answer set. It is generated from the same validation rules used by `scripts/63_validate_author_submission_metadata.py`, so it should stay aligned with the strict submission gate.",
        "",
        "## Summary",
        "",
        f"- Current metadata source: `{metadata_path}`",
        f"- Fields needing author action: {len(fields)}",
        f"- Blocking or blocking-if-applicable fields: {len(blocking)}",
        f"- High-priority fields: {len(high)}",
        f"- Minimal answer JSON skeleton: `{OUTPUT_JSON.relative_to(ROOT).as_posix()}`",
        f"- Machine-readable CSV: `{OUTPUT_CSV.relative_to(ROOT).as_posix()}`",
        "",
        "## How To Use",
        "",
        "1. Use the tables below to answer each field with verified author, ethics, protocol, device, repository, or institutional evidence.",
        "2. Review project-prefill suggestions, but copy them only after the corresponding author confirms they are accurate.",
        "3. For the fastest XLSX route, fill `Minimal_Completion.author_response` and `Minimal_Completion.evidence_source` in `outputs/manuscript_package/Author_Submission_Metadata_Intake.xlsx`; columns N:P provide formula-based completion status, should not be edited manually, and are ignored by the importer.",
        "4. Alternatively, transfer approved answers into `docs/author_submission_metadata_template.json` or the full `Author_Input` sheet.",
        "5. Run `python scripts/66_import_author_metadata_intake_workbook.py`, then `python scripts/63_validate_author_submission_metadata.py --strict` and `python scripts/64_build_author_metadata_insertion_protocol.py --strict`.",
        "6. Only after both strict checks pass should clean manuscripts, declarations, Data Availability, Code Availability, cover letter, and repository README be finalized.",
        "",
        "## Blocking Fields",
        "",
    ]
    lines.extend(field_table(rows, priorities=("blocking", "blocking_if_applicable")))
    lines.extend(["", "## High-Priority Fields", ""])
    lines.extend(field_table(rows, priorities=("high",)))
    lines.extend(["", "## Field Details", ""])
    for field_id in fields:
        lines.extend(field_detail(field_id, [row for row in rows if row["field_id"] == field_id]))
    return "\n".join(lines) + "\n"


def field_table(rows: list[dict[str, str]], priorities: tuple[str, ...]) -> list[str]:
    by_field: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["priority"] in priorities:
            by_field.setdefault(row["field_id"], []).append(row)
    if not by_field:
        return ["No fields in this category."]
    lines = [
        "| Field | Author action | Required keys still to fill | Project-prefill suggestions | Target files |",
        "|---|---|---|---|---|",
    ]
    for field_id, field_rows in sorted(by_field.items()):
        missing_keys = [row["metadata_key"] for row in field_rows if row["missing_now"] == "yes"]
        suggestions = [
            f"{row['metadata_key']}: {shorten(row['project_prefill_suggestion'], 130)}"
            for row in field_rows
            if row["project_prefill_suggestion"]
        ]
        lines.append(
            "| "
            + " | ".join(
                md_escape(value)
                for value in [
                    field_id,
                    field_rows[0]["author_action"],
                    ", ".join(missing_keys) if missing_keys else "none",
                    "; ".join(suggestions) if suggestions else "none",
                    field_rows[0]["target_files"],
                ]
            )
            + " |"
        )
    return lines


def field_detail(field_id: str, rows: list[dict[str, str]]) -> list[str]:
    first = rows[0]
    lines = [
        f"### {field_id}",
        "",
        f"- Priority: `{first['priority']}`",
        f"- Author action: {first['author_action']}",
        f"- Replacement target: {first['replacement_action'] or 'not mapped'}",
    ]
    if first["statement_pattern"]:
        lines.append(f"- Statement pattern: {first['statement_pattern']}")
    lines.extend(
        [
            "",
            "| Metadata key | Required | Current value | Project prefill suggestion | Guidance |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_escape(value)
                for value in [
                    row["metadata_key"],
                    row["strict_required"],
                    shorten(row["current_author_response"], 120) or "blank",
                    shorten(row["project_prefill_suggestion"], 160) or "none",
                    row["key_guidance"],
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def shorten(value: str, limit: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def summary_line(rows: list[dict[str, str]]) -> str:
    fields = {row["field_id"] for row in rows}
    blocking = {row["field_id"] for row in rows if row["priority"].startswith("blocking")}
    high = {row["field_id"] for row in rows if row["priority"] == "high"}
    return f"fields={len(fields)} blocking={len(blocking)} high={len(high)} rows={len(rows)}"


if __name__ == "__main__":
    main()
