# 作者必填信息表 / Author Required Information Form

请在每一行的最后一列填写真实、可投稿的信息。未核实的信息不要直接写入正式稿；伦理、知情同意、原始 EEG 预处理和数据共享范围尤其不能推断。

Please complete the final column with submission-ready information. Unverified ethics, consent, raw EEG preprocessing, and data-sharing details should not be inferred.

Machine-readable option: the same required fields are available in `docs/author_submission_metadata_template.json`. After filling that JSON file, run `python scripts/63_validate_author_submission_metadata.py --strict` to check whether the metadata are ready for final insertion.

Fastest path: use `docs/author_quick_response_request_zh.md`, `docs/author_minimal_completion_pack.md`, `outputs/manuscript_package/author_minimal_completion_answers.json`, or the `Minimal_Completion` sheet in `outputs/manuscript_package/Author_Submission_Metadata_Intake.xlsx` first. These files list only the 18 fields that currently prevent final replacement, with project-prefill suggestions kept separate from author-approved answers. In the Excel sheet, fill the author_response and evidence_source columns; columns N:P are formula-based status helpers, should not be edited manually, and are ignored by the importer.

## 最小补全清单 / Minimum Completion Checklist

先补齐下表字段，才能把 clean manuscript、Declarations、Data Availability、Code Availability 和投稿系统字段推进到最终可上传状态。

Complete these items first. They are the fields that currently block final replacement in the clean manuscript, declarations, Data Availability, Code Availability, and submission portal.

| 优先项 / Priority item | 当前边界 / Current boundary | 最少需要作者提供 / Minimum author response | 填写位置 / Destination |
|---|---|---|---|
| 伦理审批 / Ethics approval | 源工作簿未发现伦理委员会、批件号或批准日期。 | 伦理委员会全称、批件号、批准日期、适用地点，以及可直接粘贴的英文伦理声明。 | Author_Input: ethics_approval; manuscript Methods/Declarations. |
| 知情同意和数据共享授权 / Consent and data-sharing permission | 项目文件未发现同意书原文，也不能判断是否覆盖 EEG、tACS、临床量表和数据共享。 | 同意方式、签署对象、书面或豁免路径、是否允许公开或受控共享数据，以及英文声明。 | Author_Input: informed_consent; Data Availability; Declarations. |
| 注册或未注册说明 / Registration or non-registration statement | 项目文件未发现注册号。 | 注册平台、注册号和注册日期；如未注册，请提供作者认可的未注册说明。 | Author_Input: trial_or_study_registration; cover letter; declarations. |
| 研究地点、日期和设计 / Site, dates, and design | 源工作簿只有有限入组相关线索，不能形成投稿级研究设计句子。 | 医院/科室、招募起止日期、随访或末次评估窗口、前瞻/回顾、单中心/多中心。 | Author_Input: study_site_dates_design; Methods cohort paragraph. |
| 纳入排除和卒中时程 / Eligibility and stroke timing | 源工作簿有病程和 MMSE 类字段，但没有完整方案级纳排标准或病程定义。 | 纳入标准、排除标准、卒中亚型标准、病灶/侧别规则、发病至 EEG/tACS 的时间定义。 | Author_Input: eligibility_stroke_timing; Methods; TRIPOD+AI checklist. |
| tACS 设备和电极 / tACS device and electrodes | 当前只支持靶点、20 Hz、1000 microampere、20 min、14 sessions 的核心方案。 | 设备型号、电极尺寸/材料、盐水或导电介质细节，并确认现有刺激方案是否准确。 | Author_Input: tacs_device_electrodes; Methods intervention paragraph. |
| 并行常规康复 / Concurrent conventional rehabilitation | 源工作簿不能确认是否接受常规康复、剂量或训练内容。 | 是否并行常规康复、频率、单次时长、主要训练内容、不同患者或组间是否一致。 | Author_Input: concurrent_rehabilitation; Methods; Discussion claim boundary. |
| 安全性和退出 / Safety and withdrawals | 源工作簿有缺失/脱落/不适笔记，但不是完整不良事件监测数据。 | 不良事件汇总、耐受性、退出/中止人数和原因；如无正式安全性数据，需要明确说明。 | Author_Input: tacs_safety_adverse_events; Methods or Declarations. |
| FMA-UE 评估流程 / FMA-UE assessment procedure | 项目数据支持治疗前后 FMA-UE，但不支持评估者资质或盲法状态。 | 评估者资质、是否盲法、基线和治疗后评估时间点、量表版本。 | Author_Input: fma_assessors_timing; Methods outcome definition. |
| EEG 采集硬件 / EEG acquisition hardware | 项目文件支持 250 Hz、64-channel design 和 M1/M2 去除后 62 channels，但不能确认硬件细节。 | 放大器型号、采集软件、电极帽/导联系统、在线参考、地线、阻抗阈值。 | Author_Input: eeg_hardware_reference_impedance; EEG Methods. |
| 原始 EEG 预处理 / Raw EEG preprocessing | 当前只确认特征管线加载 `.set/.fdt` 后没有额外滤波、ICA 或坏道处理。 | 导出 `.set/.fdt` 前的滤波、陷波、重参考、坏道处理、ICA/眼动肌电处理、连续/分段规则。 | Author_Input: raw_eeg_preprocessing; EEG Methods. |
| 数据/代码仓储和许可 / Data, code, repository, and licences | 派生数据和代码包已准备，但 DOI、许可、公开范围和受控访问流程未确认。 | 仓储平台、DOI/访问号、版本、数据许可、代码许可、公开数据范围、受限数据申请流程。 | Author_Input: data_repository_doi_scope and code_repository_license; Data/Code Availability. |

## 1. 投稿目标和作者信息 / Submission Target and Authorship

| 字段 / Field | 为什么需要 / Why needed | 当前证据 / Current evidence | 作者填写 / Author response |
|---|---|---|---|
| 目标期刊、文章类型、参考文献格式 / Target journal, article type, and reference style | 决定字数、摘要结构、图表数量、补充材料格式和最终参考文献样式。 | 已有 Nature-style 和 Journal of Neural Engineering structured-abstract 两个版本；最终期刊未确认。 | 请填写期刊全名、文章类型、目标投稿日期、参考文献格式。 |
| 作者名单和单位 / Author list and affiliations | 正式 DOCX、封面信、贡献声明和投稿系统均需要完全一致。 | 当前稿件未包含最终作者姓名、单位和通讯作者信息。 | 请按投稿顺序填写姓名、单位编号、ORCID、通讯作者邮箱。 |
| 作者贡献 / Author contributions | 多数一区期刊要求 CRediT 或等效贡献声明。 | 项目文件无法可靠推断每位作者贡献。 | 请填写 Conceptualization, Data curation, Formal analysis, Funding, Investigation, Methodology, Software, Supervision, Writing 等贡献。 |

## 2. 伦理、知情同意和注册 / Ethics, Consent, and Registration

| 字段 / Field | 为什么需要 / Why needed | 当前证据 / Current evidence | 作者填写 / Author response |
|---|---|---|---|
| 伦理审批机构、批件号和批准日期 / Ethics committee, approval number, and approval date | 人体受试者研究的投稿硬性字段；不能根据项目文件猜测。 | 当前审计未发现可核验的伦理批件号或机构名称。 | 请填写伦理委员会全称、批件号、批准日期、适用研究地点。 |
| 知情同意 / Informed consent | 需说明所有参与者或法定代理人是否签署书面知情同意。 | 当前审计未发现可核验的同意书文本。 | 请填写同意方式、签署对象、是否覆盖 EEG、tACS、临床量表和数据共享。 |
| 临床试验注册 / Trial or study registration | 若为前瞻性干预研究，许多期刊要求注册号或说明未注册原因。 | 当前项目文件未发现注册号。 | 请填写注册平台、注册号、注册日期；如未注册，请提供可投稿的说明。 |
| 数据共享授权 / Consent for data sharing | 决定原始 EEG、临床数据、处理后特征能否公开或受控共享。 | 当前审计无法确认受试者同意范围。 | 请说明同意书是否允许公开共享、受控访问共享、或仅共享去标识化派生数据。 |

## 3. 研究设计和受试者 / Study Design and Participants

| 字段 / Field | 为什么需要 / Why needed | 当前证据 / Current evidence | 作者填写 / Author response |
|---|---|---|---|
| 研究地点和招募日期 / Study site and recruitment dates | Methods 和伦理声明需要清楚界定研究窗口和地点。 | 项目中发现可能的工作簿字段，但最终日期和地点未确认。 | 请填写医院/科室、招募开始日期、招募结束日期、随访或末次评估日期。 |
| 研究设计 / Study design | 决定表述为前瞻性、回顾性、探索性、单中心或多中心。 | 当前稿件按小样本探索性预测建模研究处理。 | 请确认研究是前瞻性还是回顾性、单中心还是多中心、是否随机或盲法。 |
| 纳入标准 / Inclusion criteria | TRIPOD+AI 和临床可解释性要求明确样本来源。 | 当前审计未发现完整纳入标准。 | 请填写年龄范围、卒中诊断标准、病程、上肢运动障碍标准、能否配合 EEG/tACS 等。 |
| 排除标准 / Exclusion criteria | 影响可重复性和人群外推边界。 | 当前审计未发现完整排除标准。 | 请填写癫痫、金属植入、严重认知障碍、其他神经疾病、药物或 tACS 禁忌等。 |
| 卒中亚型、病灶侧和病程 / Stroke subtype, lesion side, and time since stroke | 解释神经生理特征和比例恢复模型适用范围。 | 当前模型按受累手进行左右镜像对齐；更详细病灶信息未核验。 | 请填写缺血/出血、皮质/皮质下、病灶侧、发病到 EEG、发病到 tACS 的时间统计。 |

## 4. tACS 干预和临床量表 / tACS Intervention and Clinical Assessments

| 字段 / Field | 为什么需要 / Why needed | 当前证据 / Current evidence | 作者填写 / Author response |
|---|---|---|---|
| tACS 方案确认 / tACS protocol confirmation | Methods 中的刺激靶点、强度和频次必须与实际方案一致。 | 项目设计文档提示：对侧 M1，右手受累 C3、左手受累 C4，20 Hz，1000 microampere，20 min/session，每日一次，共 14 次，2 周。 | 请确认或修正靶点、频率、强度、时长、疗程、设备和电极尺寸。 |
| 并行常规康复 / Concurrent conventional rehabilitation | 若 tACS 与常规康复同时进行，需要明确剂量，避免误读为单独 tACS 效应。 | 当前审计发现可能相关术语，但无法确认实际剂量。 | 请填写是否同时接受常规康复、频率、每次时长、主要训练内容、两组是否一致。 |
| 安全性和不良事件 / Safety and adverse events | 神经调控研究通常需要报告耐受性和不良事件。 | 当前项目文件未发现完整安全性汇总。 | 请填写是否发生头痛、刺痛、疲劳、皮肤反应、癫痫等，及退出情况。 |
| FMA-UE 评估者和时间点 / FMA-UE assessors and timing | 结果标签来自 FMA-UE 变化；评估流程需可复现。 | 当前分析使用治疗前和治疗后 FMA-UE，并按比例恢复残差中位数分组。 | 请填写评估者资质、是否盲法、治疗前后具体时间点、量表版本。 |

## 5. EEG 采集和预处理 / EEG Acquisition and Preprocessing

| 字段 / Field | 为什么需要 / Why needed | 当前证据 / Current evidence | 作者填写 / Author response |
|---|---|---|---|
| EEG 设备和电极帽 / EEG amplifier, cap, and montage | EEG 研究的可重复性核心信息。 | 项目 `.set` 文件显示 250 Hz、去除 M1/M2 后 62 channels；硬件和电极帽未核验。 | 请填写放大器型号、采集软件、电极帽系统、原始通道数、10-20/10-10 montage。 |
| 在线参考、地线和阻抗 / Online reference, ground, and impedance | 影响功率谱和连接特征，必须在 Methods 报告。 | 当前项目文件未发现在线参考、地线或阻抗标准。 | 请填写在线参考、地线位置、阻抗阈值、采集前检查流程。 |
| 静息态任务说明 / Resting-state instructions | EO/EC 的记录条件影响频谱解释。 | 文件名约定提示 baseline `*1.set` 为 EO，`*2.set` 为 EC；持续时间由元数据估计。 | 请填写睁眼/闭眼顺序、每段目标时长、是否注视固定点、是否监测困倦。 |
| 导出为 `.set/.fdt` 前的原始预处理 / Raw preprocessing before `.set/.fdt` export | 当前特征脚本从预处理后的 EEGLAB 文件开始；原始预处理步骤必须由作者确认。 | 已核验：特征管线加载后未再进行额外滤波、ICA、坏道剔除或插值。 | 请填写原始滤波范围、陷波、重参考、坏道处理、ICA/眼动肌电伪迹剔除、分段或连续导出规则。 |
| EEG 数据排除规则 / EEG data exclusion criteria | 需要说明哪些记录或片段被排除，避免选择偏倚。 | 当前可核验的是最终纳入 `.set/.fdt` 文件；原始排除规则未确认。 | 请填写记录级和片段级排除标准、丢弃比例、是否由盲法人员处理。 |

## 6. 数据、代码和仓储 / Data, Code, and Repository

| 字段 / Field | 为什么需要 / Why needed | 当前证据 / Current evidence | 作者填写 / Author response |
|---|---|---|---|
| 仓储平台和 DOI / Repository platform and DOI | Data Availability 和数据集引用需要稳定标识符。 | 已生成 repository README 草案和 source-data workbook；正式 DOI 未生成。 | 请填写 Zenodo、OSF、Figshare、机构库或受控访问库名称、DOI/访问号、版本号。 |
| 可公开数据范围 / Publicly shareable data scope | 决定哪些文件进入公开仓储，哪些只在补充材料或受控访问。 | 已整理 figure source data、统计表、数据字典和复现脚本；原始 EEG/临床表可能含敏感信息。 | 请勾选可公开项：派生特征、统计表、图表源数据、代码、匿名临床变量、预处理 EEG、原始 EEG。 |
| 受限数据访问流程 / Controlled or restricted access route | 若不能公开人类受试者数据，需说明限制原因和申请流程。 | 当前无法确认伦理和同意范围。 | 请填写访问审批机构/联系人、申请材料、评审标准、数据使用协议、预期回复时间。 |
| 代码许可和数据许可 / Code and data licences | 影响数据复用和期刊合规性。 | 当前 repository README 使用待填写许可字段。 | 请填写代码许可证、数据许可证、是否禁止商业使用、是否有专利或单位限制。 |

## 7. 声明、基金和投稿材料 / Declarations, Funding, and Submission Materials

| 字段 / Field | 为什么需要 / Why needed | 当前证据 / Current evidence | 作者填写 / Author response |
|---|---|---|---|
| 基金资助 / Funding | 需要进入 Acknowledgements 和投稿系统。 | 当前项目文件未核验基金号。 | 请填写基金名称、项目编号、资助对象、资助方是否参与研究设计或写作。 |
| 利益冲突 / Competing interests | 所有作者需统一确认。 | 当前未核验。 | 请填写是否存在设备、软件、资金、专利、顾问费或其他利益关系；无则写 None declared。 |
| 致谢 / Acknowledgements | 技术协助、数据采集、统计咨询等贡献需准确署名。 | 当前未核验。 | 请填写需要致谢的个人、团队、平台和语言编辑支持。 |
| 推荐或回避审稿人 / Suggested or opposed reviewers | 不少期刊投稿系统要求或允许填写。 | 当前未准备。 | 请填写姓名、单位、邮箱、推荐理由；回避审稿人请说明利益冲突原因。 |

## 8. 可直接粘贴的最终声明 / Ready-to-Paste Final Statements

| 字段 / Field | 为什么需要 / Why needed | 当前证据 / Current evidence | 作者填写 / Author response |
|---|---|---|---|
| Ethics approval statement | 放入 Methods 或 Declarations。 | 需要作者提供真实审批信息后才能定稿。 | 请提供英文最终句子，或填写中文信息由我转换为英文。 |
| Informed consent statement | 放入 Declarations。 | 需要作者确认书面同意和数据共享范围。 | 请提供英文最终句子，或填写中文信息由我转换为英文。 |
| Data Availability statement | 放入 Data Availability。 | 已有草案，但 DOI、许可、受限访问流程和可公开范围未确认。 | 请填写最终仓储链接、DOI、哪些数据公开、哪些数据受限、访问流程。 |
| Code Availability statement | 放入 Code Availability 或 Data Availability。 | 已有复现脚本，但公开仓储和许可证未确认。 | 请填写代码仓储、版本、许可证、运行环境限制。 |

## 使用建议 / How to Use

- 先补齐伦理、知情同意、EEG 采集和预处理字段，再替换正式稿中的相应占位说明。
- 若希望一次性核对所有字段，请填写 `docs/author_submission_metadata_template.json`，再运行 `scripts/63_validate_author_submission_metadata.py` 生成缺口报告。
- 若原始 EEG 或临床变量不能公开，请给出限制原因、审批流程和可公开的替代数据层级。
- 若目标期刊不是 Nature 系列，请先确认摘要结构、参考文献格式、数据共享政策和 AI/ML 报告清单要求。
