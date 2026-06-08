# 作者快速回复清单：项目记录版回答

生成日期：2026-06-02

本文件仅根据当前项目仓库和既有项目记录填写。凡伦理、知情同意、注册、设备型号、康复方案、安全性、作者、基金、数据许可等不能由项目文件核实的项目，均写为“暂不能确认”。这些内容不能替代作者、伦理文件、设备记录、方案文件或医院/科室记录确认。

主要证据文件：

- `docs/author_metadata_project_prefill_report.md`
- `docs/author_required_evidence_trace.md`
- `docs/source_workbook_author_metadata_audit.md`
- `docs/methods_detail_provenance.md`
- `docs/methods_gap_resolution_from_project_files.md`
- `docs/project_context.md`
- `docs/eeg_metadata_audit.md`
- `docs/cohort_characteristics.md`
- `docs/data_availability_and_fair_audit.md`
- `docs/repository_readme_for_deposit.md`
- `docs/participant_flow_and_safety_source_notes.md`
- `docs/target_journal_strategy.md`

## 1. ethics_approval

- 伦理委员会全称：暂不能确认。
- 批件号：暂不能确认。
- 批准日期：暂不能确认。
- 适用医院/科室/研究地点：暂不能确认。
- 是否符合 Declaration of Helsinki：暂不能确认。
- 可直接放入论文的英文伦理声明：暂不能提供投稿可用声明。当前项目记录没有可核实的伦理委员会名称、批件号或批准日期。
- 证据来源：`docs/author_required_evidence_trace.md` 和 `docs/source_workbook_author_metadata_audit.md` 均记录该字段为 `author_required`；`docs/patient_record_pdf_text_audit.md` 显示本地病历 PDF 多为扫描件，不能自动提取伦理/同意信息。

## 2. informed_consent

- 知情同意方式：暂不能确认。
- 签署对象：暂不能确认。
- 是否覆盖 EEG：暂不能确认。
- 是否覆盖 tACS：暂不能确认。
- 是否覆盖 FMA-UE 等临床评估：暂不能确认。
- 是否覆盖数据共享：暂不能确认。
- 若允许共享，请说明公开共享、受控访问或仅派生数据共享：暂不能确认。项目记录建议原始 EEG、预处理 EEG 和个体级临床源记录应受控访问或暂不公开；去标识派生表、模型预测、统计表和图源数据可作为公开派生数据候选，但需伦理和同意范围确认。
- 可直接放入论文的英文知情同意声明：暂不能提供投稿可用声明。
- 证据来源：`docs/author_required_evidence_trace.md`、`docs/source_workbook_author_metadata_audit.md`、`docs/data_availability_and_fair_audit.md`。

## 3. trial_or_study_registration

- 是否注册：项目记录未发现注册号；是否“未注册”仍需作者确认。
- 注册平台：暂不能确认。
- 注册号：暂不能确认。
- 注册日期：暂不能确认。
- 如果未注册，请提供可投稿的英文未注册说明：暂不能直接提供最终投稿句子。若作者确认未注册，可基于 `docs/jne_final_declaration_templates.md` 使用类似模板：`The study was not prospectively registered because [author-approved reason]. This limitation is reported explicitly because the present manuscript is an exploratory prediction-model analysis rather than a confirmatory clinical efficacy trial.`
- 证据来源：`docs/author_required_evidence_trace.md` 记录未发现 NCT、ChiCTR、ClinicalTrials.gov 或其他注册编号。

## 4. study_site_dates_design

- 医院/科室：暂不能确认。
- 招募开始日期：暂不能确认。
- 招募结束日期：暂不能确认。
- 随访或末次评估窗口：项目设计记录为完成最后一次 tACS 后进行治疗后评估；没有更长期随访窗口记录。
- 前瞻性或回顾性：暂不能确认。当前代码和论文分析属于基于既有数据的预测建模分析，但原始临床研究设计属性需作者确认。
- 单中心或多中心：暂不能确认。
- 是否随机：暂不能确认。
- 是否盲法：暂不能确认。
- 可直接放入 Methods 的英文研究设计句子：暂不能提供投稿可用完整句子。
- 补充项目证据：原始患者 CNT 文件记录窗口为 2024-01-12 至 2025-08-14，但这是 EEG 文件记录时间窗口，不等同于正式招募窗口。
- 证据来源：`docs/author_metadata_project_prefill_report.md`、`docs/methods_detail_provenance.md`、`docs/eeg_metadata_audit.md`。

## 5. eligibility_stroke_timing

- 纳入标准：暂不能确认。
- 排除标准：暂不能确认。项目记录只支持最终监督队列为 19 例，并记录非监督/脱落/缺失数据相关情况；不能替代方案级纳排标准。
- 卒中亚型标准：暂不能确认。
- 病灶侧/受累手规则：项目临床表包含“患病侧（手）/ affected hand”字段；模型管线按受累手做半球对齐。右手受累不翻转；左手受累进行左右通道镜像，使数据统一到右患手/C3 刺激约定。
- 发病至 EEG 的时间定义：暂不能确认。
- 发病至 tACS 的时间定义：暂不能确认。
- 当前项目中的“病程”字段具体指什么时间间隔：暂不能确认。项目只确认临床源表中存在“病程”字段；19 例监督队列病程均值 36.3 天，median 33 天，范围 17-83 天，但该字段究竟是发病至 EEG、发病至 tACS、发病至入组还是其他时间间隔，需作者确认。
- 证据来源：`docs/cohort_characteristics.md`、`docs/project_context.md`、`src/eeg_recovery/channels/hemisphere_flip.py`、`configs/channel_mapping.yaml`、`docs/source_workbook_author_metadata_audit.md`。

## 6. tacs_device_electrodes

- tACS 设备型号：暂不能确认。
- 电极尺寸：暂不能确认。
- 电极材料或导电介质：暂不能确认。
- 靶点是否确认为对侧 M1：项目设计记录为对侧 M1。
- 右手受累是否为 C3、左手受累是否为 C4：项目设计记录为右手受累刺激 C3，左手受累刺激 C4。
- 频率是否为 20 Hz：是，项目设计支持。
- 强度是否为 1000 microampere：是，项目设计支持。
- 每次是否为 20 min：是，项目设计支持。
- 疗程是否为每日 1 次、14 次、2 周：是，项目设计支持。
- 可直接放入 Methods 的英文 tACS 设备/电极句子：设备和电极细节不足，不能写完整设备/电极句。可作为已支持方案骨架使用：`The project protocol specified tACS over the contralateral M1, targeting C3 for right-hand impairment and C4 for left-hand impairment, at 20 Hz and 1000 microampere for 20 min per session, once daily for 14 sessions over 2 weeks.` 设备型号、电极尺寸和材料需补齐后再作为最终 Methods 句子。
- 证据来源：`tacs_eeg_proportional_recovery_project_design.md`、`docs/methods_detail_provenance.md`、`docs/author_metadata_project_prefill_report.md`。

## 7. concurrent_rehabilitation

- 是否并行常规康复：暂不能确认。
- 康复频率：暂不能确认。
- 每次时长：暂不能确认。
- 主要训练内容：暂不能确认。
- 所有患者是否一致：暂不能确认。
- 若不一致，请说明差异：暂不能确认。
- 可直接放入 Methods 的英文句子：暂不能提供投稿可用声明。
- 证据来源：`docs/author_required_evidence_trace.md` 和 `docs/source_workbook_author_metadata_audit.md` 均记录当前项目文件不能验证常规康复剂量和内容。

## 8. tacs_safety_adverse_events

- 是否有正式不良事件监测记录：当前项目文件中未找到正式不良事件监测表；是否实际存在需作者确认。
- 不良事件汇总：暂不能确认。项目只记录源表中有缺失/脱落/不适相关备注分类，不能作为正式 AE 汇总。
- 耐受性汇总：暂不能确认。
- 退出/中止人数和原因：项目源表支持非正式 participant-flow 级别分类：M1 患者记录 29 例，当前 EEG 索引患者 28 例，最终监督队列 19 例，非监督 EEG 索引患者 9 例。源表备注中包括未治疗 2、出院 1、依从性差或认知/沟通困难 2、EEG 帽热/不适 1、MRI 相关不适/不愿入组 1、治疗后不适 1、治疗后不适并次日高血压 1、完整 FMA 但基线接近天花板 1。上述不是正式 AE 表。
- 是否发生头痛、刺痛、皮肤反应、疲劳、癫痫、血压异常或其他事件：血压异常相关备注有 1 条“治疗后不适并次日高血压”源表分类；其他具体 AE 项目暂不能确认。
- 如无正式安全性数据，请提供可投稿的英文说明：不能直接写“无正式安全性数据”，除非作者确认。项目记录可支持的边界句为：`The current project files did not contain an investigator-adjudicated adverse-event monitoring table; source-workbook discontinuation and discomfort notes were therefore used only for participant-flow review and not as a formal safety analysis.`
- 证据来源：`docs/participant_flow_and_safety_source_notes.md`、`docs/author_metadata_project_prefill_report.md`。

## 9. eeg_hardware_reference_impedance

- EEG 放大器型号：暂不能确认。
- 采集软件：暂不能确认。
- 电极帽/导联系统：暂不能确认。项目只支持 64 通道源设计和 62 通道分析保留结果。
- 原始通道数：项目设计为 64 通道；实际分析使用 62 通道，原因是预处理后删除 M1/M2。
- 在线参考：暂不能确认。
- 地线：暂不能确认。
- 阻抗阈值：暂不能确认。
- 可直接放入 EEG Methods 的英文采集硬件句子：暂不能提供完整采集硬件句。可写入已支持的分析边界：`The retained analysis montage contained 62 channels after removal of M1 and M2, and the indexed EEGLAB files were sampled at 250 Hz.` 放大器、采集软件、在线参考、地线和阻抗阈值需作者补充。
- 证据来源：`docs/project_context.md`、`docs/eeg_metadata_audit.md`、`docs/methods_detail_provenance.md`、`configs/channel_mapping.yaml`。

## 10. raw_eeg_preprocessing

- 导出 `.set/.fdt` 前原始滤波：项目代码不能确认具体滤波参数。用户早期说明预处理已完成滤波，但未提供高通/低通/带通/阶数/算法等投稿级参数。
- 陷波滤波：暂不能确认。
- 重参考：用户早期说明预处理已完成重参考，但具体参考方式暂不能确认。
- 坏道处理：用户早期说明预处理已完成坏道处理，但坏道判定、插值或删除规则暂不能确认。
- ICA 或眼动/肌电伪迹处理：用户早期说明预处理已完成 ICA 和伪迹剔除，但 ICA 组件剔除标准、眼动/肌电处理规则暂不能确认。
- 片段剔除或连续数据保留规则：项目索引的 EEGLAB 文件加载后均为连续 one-trial 数据；当前分析不再做片段剔除。上游预处理阶段是否分段后再导出连续文件、或具体剔除规则暂不能确认。
- EEGLAB `.set/.fdt` 导出规则：当前分析从项目提供的预处理 EEGLAB `.set/.fdt` 文件开始；`.set` 元数据由 `scipy.io.loadmat` 读取，`.fdt` 以 float32 按 channels x samples 加载。个别 `.set` 内部声明旧 `.fdt` 文件名时，代码优先使用实际存在的同名 companion `.fdt`。
- 是否所有分析文件均为连续 one-trial 数据：是，当前索引文件元数据显示 `trials=1`。
- 可直接放入 EEG Methods 的英文预处理句子：可作为当前分析边界句使用：`Analyses started from project-provided preprocessed EEGLAB .set/.fdt files. After loading these files, the feature pipeline performed no additional temporal filtering, artifact rejection, channel interpolation, bad-channel removal, or ICA.` 上游原始滤波、重参考、ICA、坏道、伪迹剔除参数仍需作者补充。
- 证据来源：`docs/methods_gap_resolution_from_project_files.md`、`docs/methods_detail_provenance.md`、`docs/eeg_metadata_audit.md`、`src/eeg_recovery/io/eeglab.py`、`src/eeg_recovery/features/psd.py`、`src/eeg_recovery/features/connectivity.py`。

## 11. data_repository_doi_scope

- 数据仓储平台：暂不能确认。
- DOI 或 accession：暂不能确认。
- 版本号：暂不能确认。
- 数据许可：暂不能确认。
- 可公开数据范围：项目准备的候选公开派生材料包括去标识 subject-level derived analysis tables、locked LOSO predictions、bootstrap/permutation/paired-comparison outputs、figure source summaries、validation/audit tables、author-field replacement maps、reproducibility scripts、source-data workbook。
- 受限数据范围：原始 EEG、预处理/最小处理 EEG、可识别临床源记录、可直接链接到个体的参与者级源文件。
- 受限数据申请联系人或委员会：暂不能确认。
- 申请材料和审核要求：暂不能确认；项目草案建议需 ethics approval、project proposal、data-use agreement 和机构审核。
- 可直接放入 Data Availability 的英文声明：不能最终投稿，因为 DOI、平台、许可和受控访问联系人缺失。可用草案见 `docs/repository_readme_for_deposit.md` 和 `docs/jne_final_declaration_templates.md`。
- 证据来源：`docs/data_availability_and_fair_audit.md`、`docs/repository_readme_for_deposit.md`、`results/tables/source_data_dictionary.csv`。

## 12. code_repository_license

- 代码仓储 URL 或 DOI：当前 git 远端为 `https://github.com/CaiPeterJiazhen/EEG_PredictStokeDLModel.git`。
- 版本号或 commit hash：当前 HEAD 为 `811b3b2e915b9048f05f2ab2a0281e4534939d20`，当前分支为 `main`。注意：当前工作区存在未提交修改和未跟踪文件，因此该 hash 不能视为最终冻结投稿版本。
- 代码许可证：暂不能确认；项目根目录未发现 `LICENSE` 文件。
- 运行环境说明：`pyproject.toml` 记录项目名 `eeg-recovery`，版本 `0.1.0`，Python `>=3.10`，依赖包括 `numpy`、`openpyxl`、`pandas`、`pyyaml`、`scipy`、`scikit-learn`、`torch`、`matplotlib`、`mne`、`mne-connectivity`。
- 可直接放入 Code Availability 的英文声明：暂不能提供最终投稿句子，因为许可证、最终 commit/tag 和归档 DOI 尚未冻结。可用草案为：`Analysis code is available at https://github.com/CaiPeterJiazhen/EEG_PredictStokeDLModel.git, version [final tag or commit], under [software licence].`
- 证据来源：`git remote -v`、`git rev-parse HEAD`、`pyproject.toml`、`docs/repository_readme_for_deposit.md`。

## 13. target_journal_reference_style

- 最终目标期刊：项目策略建议首选 Journal of Neural Engineering (JNE)，但最终目标期刊需作者确认。
- 文章类型：暂不能确认；项目当前准备的是 JNE structured manuscript / original research 风格稿件。
- 参考文献格式：暂不能确认；若投 JNE，应按 IOP/JNE 要求处理。
- 目标投稿日期：暂不能确认。
- 证据来源或作者确认方式：`docs/target_journal_strategy.md`、`docs/jne_submission_checklist.md`。

## 14. author_list_affiliations

- 最终作者顺序：暂不能确认。
- 每位作者单位：暂不能确认。
- ORCID：暂不能确认。
- 通讯作者姓名：暂不能确认。
- 通讯作者邮箱：暂不能确认。
- 通讯地址：暂不能确认。
- 证据来源或作者确认方式：当前项目没有作者最终清单；需作者邮件、投稿系统信息或机构记录确认。

## 15. author_contributions

- 每位作者 CRediT 贡献：暂不能确认。
- 是否所有作者已批准最终稿：暂不能确认。
- 可直接放入声明的英文作者贡献：暂不能提供投稿可用声明。模板见 `docs/jne_final_declaration_templates.md`。
- 证据来源或作者确认方式：需作者确认。

## 16. fma_assessors_timing

- FMA-UE 评估者资质：暂不能确认。
- 是否盲法：暂不能确认。
- 基线评估时间点：项目记录支持治疗前/基线 FMA-UE 作为输入标签计算基础；模型输入只使用治疗前基线 EEG 和可选治疗前临床变量。
- 治疗后评估时间点：项目设计支持最后一次 tACS 后进行治疗后 FMA-UE 评估。
- FMA-UE 版本或评分依据：项目标签代码和上下文使用 FMA-UE 上肢满分 66。
- 可直接放入 Methods 的英文评估流程句子：可作为部分句子使用：`FMA-UE was scored on a 66-point upper-extremity scale at baseline and after the final tACS session for proportional-recovery labelling.` 评估者资质和盲法需作者补充后才能最终投稿。
- 证据来源：`docs/project_context.md`、`docs/methods_detail_provenance.md`、`src/eeg_recovery/metadata/labels.py`、`docs/cohort_characteristics.md`。

## 17. resting_state_instructions

- 睁眼/闭眼采集顺序：项目文件名规则支持 `*1.set` 为 eyes-open、`*2.set` 为 eyes-closed；实际采集顺序是否即 EO 后 EC 需作者确认。
- 每个状态目标时长：目标时长暂不能确认。当前 38 个监督基线 EO/EC 文件实际平均时长 188.4 s，范围 101.0-247.8 s。
- 睁眼时是否注视固定点：暂不能确认。
- 是否监测困倦或闭眼入睡：暂不能确认。
- 可直接放入 EEG Methods 的英文静息态说明：可作为部分句子使用：`Baseline resting-state EEG files ending in *1.set and *2.set were mapped to eyes-open and eyes-closed conditions, respectively; across the supervised baseline files, recording duration averaged 188.4 s and ranged from 101.0 to 247.8 s.` 采集指令、注视点和困倦监测需作者补充。
- 证据来源：`docs/project_context.md`、`docs/eeg_metadata_audit.md`、`results/tables/eeg_recording_metadata_audit.csv`。

## 18. funding_competing_acknowledgements

- 基金名称和项目编号：暂不能确认。
- 资助方是否参与研究设计、数据分析或写作：暂不能确认。
- 利益冲突声明：暂不能确认。
- 致谢对象和授权：暂不能确认。
- 可直接放入 Declarations/Acknowledgements 的英文声明：暂不能提供投稿可用声明。模板见 `docs/jne_final_declaration_templates.md`。
- 证据来源或作者确认方式：当前项目文件未发现基金、利益冲突或致谢授权证据；需作者确认。

## 已可较稳妥保留的项目事实

- 项目目标：用卒中患者 tACS 治疗前基线静息态 EEG 预测 tACS 后 FMA-UE proportional recovery。
- 监督队列：19 例患者，LOSO-CV；标签分布 label 0 为 9 例，label 1 为 10 例。
- 全患者 EEG 队列：28 例，其中 9 例非监督训练患者为 `sub02, sub03, sub06, sub12, sub19, sub21, sub23, sub25, sub26`。
- EEG 状态规则：`*1.set` 为睁眼，`*2.set` 为闭眼。
- EEG 分析通道：实际使用 62 通道，M1/M2 已删除。
- EEG 元数据：采样率 250 Hz；监督基线 EO/EC 文件为连续 one-trial；38 个监督基线文件平均时长 188.4 s，范围 101.0-247.8 s。
- 特征管线边界：加载预处理 `.set/.fdt` 后，项目特征脚本不再额外进行滤波、ICA、坏道处理、插值、伪迹剔除。
- tACS 方案骨架：对侧 M1；右手受累 C3、左手受累 C4；20 Hz；1000 microampere；20 min/session；每日 1 次；14 次；2 周。
- 数据共享草案：派生表、锁定预测、统计表、图源数据、审计表、代码可作为公开/归档候选；原始 EEG、预处理 EEG、个体级临床源记录需受控访问或暂不公开，等待伦理和同意范围确认。

## 仍然最阻断投稿的缺口

1. 伦理委员会全称、批件号、批准日期、Declaration of Helsinki 句子。
2. 知情同意方式、签署对象、是否覆盖 EEG/tACS/临床评估/数据共享。
3. 注册状态或作者认可的未注册说明。
4. 医院/科室、招募日期、研究设计属性、纳排标准、卒中亚型和时间定义。
5. tACS 设备型号、电极尺寸/材料、常规康复方案、安全性/不良事件正式记录。
6. EEG 放大器、采集软件、在线参考、地线、阻抗阈值，以及上游预处理详细参数。
7. 最终数据仓储 DOI、数据许可、代码许可、最终 commit/tag。
8. 作者列表、单位、贡献、基金、利益冲突和致谢授权。
