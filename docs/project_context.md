# tACS EEG Proportional Recovery Project Context

本文档是后续 thread 或 subagent 的固定上下文入口。任何开发任务开始前，都应先读取本文档和 `docs/implementation_plan.md`，再读取原始设计文档 `tacs_eeg_proportional_recovery_project_design.md`。

## 1. Project Goal

基于卒中患者接受 tACS 治疗前的基线静息态 EEG，预测患者完成 tACS 后是否达到 FMA-UE proportional recovery。主任务是二分类：

- `1`: Proportional Recovery
- `0`: Poor Recovery

主要输入为治疗前基线 EEG 的两个状态：

- EO: 睁眼静息态
- EC: 闭眼静息态

扩展模型可以加入治疗前临床变量。

## 2. Fixed Data Paths

项目代码目录：

```text
F:\CJZProjectFile\EEG_PredictStokeDLModel
```

患者信息表：

```text
F:\CJZFile\EEG_M1\19例患者脑电数据完整性检查.xlsx
F:\CJZFile\EEG_M1\脑卒中患者信息记录表.xlsx
```

预处理后 EEG：

```text
F:\CJZFile\EEG_M1\Patient_tACS_M1_RestingStateEEG_afterProcess
F:\CJZFile\EEG_M1\Health_tACS_M1_RestingStateEEG_afterProcess
```

通道坐标文件：

```text
F:\CJZFile\EEG_M1\standard_1005.ced
```

## 3. Data Facts Confirmed In This Project

- 监督训练名单以 `19例患者脑电数据完整性检查.xlsx` 中 19 例为准。
- 设计文档中写 64 通道，但实际 `.set` 文件均为 62 通道。
- 62 通道原因：预处理后删除了 `M1` 和 `M2`。
- 后续全部模型、特征和解释分析均按 62 通道实现。
- 所有已扫到的 `.set` 文件采样率均为 `250 Hz`。
- 所有已扫到的 `.set` 文件均为连续数据，`trials=1`。
- 所有已扫到的 `.set` 文件通道顺序一致。
- `.set` 数据字段指向同名 `.fdt` 文件。
- 已确认 4 个 `.set` 文件内部声明的 `.fdt` 名称与实际同名 companion 不一致，这是预处理后重命名文件但未同步更新 `.set` 内部 `data/datfile` 字段导致，不代表数据缺失：
  - `即时\sub024赵全增\zqz1.set`: 内部声明 `zqc1.fdt`，实际对应 `zqz1.fdt`
  - `基线\sub011单庆明\sqm1.set`: 内部声明 `dqm1.fdt`，实际对应 `sqm1.fdt`
  - `基线\sub02翟玉琴\zyq1.set`: 内部声明 `cyq1.fdt`，实际对应 `zyq1.fdt`
  - `阶段\sub014迟承萱\ccx1.set`: 内部声明 `ccx.fdt`，实际对应 `ccx1.fdt`
  - 代码应优先使用 `.set` 内声明且实际存在的 `.fdt`；若声明文件不存在但同目录同名 companion `.fdt` 存在，则使用同名 companion。
- 文件名状态规则：
  - `*1.set`: EO, 睁眼
  - `*2.set`: EC, 闭眼
- 患者 EEG 目录包含 `基线`、`即时`、`阶段`、`最终`。
- 主预测模型只使用患者 `基线` 目录下的 EO/EC EEG。
- 健康人 EEG 可用于无标签自监督预训练。

## 4. Subject ID Rules

不同文件中患者 ID 可能出现 `sub01`、`sub011`、`sub013`、`sub021` 等写法。开发时必须统一解析为规范 ID：

```text
sub01, sub02, ..., sub29
```

建议规则：

1. 从目录名或表格 ID 中提取 `sub` 后面的数字。
2. 转为整数。
3. 再格式化为两位数字：`sub{number:02d}`。

示例：

```text
sub013 -> sub13
sub011 -> sub11
sub021 -> sub21
sub1   -> sub01
```

监督训练 19 例规范 ID：

```text
sub01, sub05, sub07, sub08, sub09,
sub10, sub11, sub13, sub14, sub15,
sub16, sub17, sub18, sub20, sub22,
sub24, sub27, sub28, sub29
```

## 5. Label Definition

FMA-UE 满分为 66。

```text
Delta_FMA_pred = 0.7 * (66 - FMA_pre)
Delta_FMA_obs  = FMA_post - FMA_pre
Residual       = Delta_FMA_pred - Delta_FMA_obs
```

使用 19 例监督训练患者 residual 的中位数作为阈值。已确认：

```text
median_residual = 1.5
```

标签：

```text
label = 1 if residual <= 1.5 else 0
```

当前标签分布：

```text
Proportional Recovery: 10
Poor Recovery: 9
```

监督训练标签表：

| subject_id | affected_hand | FMA_pre | FMA_post | Delta_obs | Delta_pred | residual | label |
|---|---:|---:|---:|---:|---:|---:|---:|
| sub01 | 左 | 63 | 65 | 2 | 2.1 | 0.1 | 1 |
| sub05 | 左 | 9 | 12 | 3 | 39.9 | 36.9 | 0 |
| sub07 | 右 | 22 | 24 | 2 | 30.8 | 28.8 | 0 |
| sub08 | 右 | 10 | 12 | 2 | 39.2 | 37.2 | 0 |
| sub09 | 左 | 6 | 11 | 5 | 42.0 | 37.0 | 0 |
| sub10 | 右 | 59 | 64 | 5 | 4.9 | -0.1 | 1 |
| sub11 | 左 | 7 | 13 | 6 | 41.3 | 35.3 | 0 |
| sub13 | 右 | 25 | 41 | 16 | 28.7 | 12.7 | 0 |
| sub14 | 左 | 60 | 62 | 2 | 4.2 | 2.2 | 0 |
| sub15 | 左 | 51 | 63 | 12 | 10.5 | -1.5 | 1 |
| sub16 | 右 | 23 | 30 | 7 | 30.1 | 23.1 | 0 |
| sub17 | 右 | 51 | 62 | 11 | 10.5 | -0.5 | 1 |
| sub18 | 左 | 64 | 65 | 1 | 1.4 | 0.4 | 1 |
| sub20 | 左 | 62 | 64 | 2 | 2.8 | 0.8 | 1 |
| sub22 | 右 | 61 | 63 | 2 | 3.5 | 1.5 | 1 |
| sub24 | 左 | 11 | 18 | 7 | 38.5 | 31.5 | 0 |
| sub27 | 左 | 61 | 64 | 3 | 3.5 | 0.5 | 1 |
| sub28 | 右 | 63 | 66 | 3 | 2.1 | -0.9 | 1 |
| sub29 | 左 | 61 | 65 | 4 | 3.5 | -0.5 | 1 |

## 6. EEG Channel Facts

实际 62 通道顺序：

```text
FP1, FPZ, FP2,
AF3, AF4,
F7, F5, F3, F1, FZ, F2, F4, F6, F8,
FT7, FC5, FC3, FC1, FCZ, FC2, FC4, FC6, FT8,
T7, C5, C3, C1, CZ, C2, C4, C6, T8,
TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, TP8,
P7, P5, P3, P1, PZ, P2, P4, P6, P8,
PO7, PO5, PO3, POZ, PO4, PO6, PO8,
CB1, O1, OZ, O2, CB2
```

中线通道保持不变：

```text
FPZ, FZ, FCZ, CZ, CPZ, PZ, POZ, OZ
```

左右翻转必须在 PSD 和 FC 计算之前完成。项目统一方向：

```text
统一到“患手右侧 / 刺激 C3 / 左半球为刺激侧”的空间表示
```

因此：

- 患手右侧患者：不翻转。
- 患手左侧患者：执行左右通道翻转，使 C4 对应到统一 C3 位置。

注意：原始设计文档中写 64 通道；后续代码必须覆盖实际 62 通道，并在测试中断言没有 `M1`、`M2`。

## 7. Feature Dimensions

PSD：

```text
PSD_EO: 62 x 90
PSD_EC: 62 x 90
```

频率范围和分辨率：

```text
0.5-45 Hz, 0.5 Hz resolution
```

FC 边数：

```text
62 * 61 / 2 = 1891
```

单状态 FC：

```text
FC_EO: 1891 x 6
FC_EC: 1891 x 6
```

频段：

| band | range |
|---|---|
| Delta | 1-3 Hz |
| Theta | 4-7 Hz |
| Alpha | 8-13 Hz |
| Beta Low | 13-18 Hz |
| Beta Medium | 18-21 Hz |
| Beta High | 21-30 Hz |

论文解释重点为 tACS 20 Hz 与 Beta Medium `18-21 Hz`。

## 8. Leakage Control Rules

必须遵守：

- 监督训练和评估以患者为单位。
- 主验证方法为 LOSO-CV。
- 禁止片段级随机划分导致同一患者片段进入 train 和 test。
- 标准化、PCA、特征筛选、特征选择必须只在训练折内 fit，再应用到测试折。
- 标签由 `FMA_pre` 和 `FMA_post` 计算，但模型输入不能包含 `FMA_post` 或任何治疗后变量。
- 主模型输入只使用治疗前基线 EEG。
- 严格 LOSO 的 SSL 设置中，当前测试折患者的 EEG 不参与该折 SSL 预训练。
- 如果使用所有无标签 EEG 预训练，包括测试患者 EEG，必须作为 transductive/semi-supervised 设置单独报告。

## 9. MATLAB And Python Notes

MATLAB 可用路径：

```text
F:\Matlab2020a\bin\matlab.exe
```

在 Codex 中调用 MATLAB 时，需要用沙箱外方式运行：

```powershell
F:\Matlab2020a\bin\matlab.exe -wait -batch "..."
```

优先使用 Python 实现可复现流水线。已确认本机 `D:\anaconda\python.exe` 有 `scipy`、`h5py`、`mne`，但 MNE 读取某个 `.set` 头信息曾超过 60 秒。后续可优先使用 `scipy.io.loadmat` 读取 `.set` 元数据，并直接读取 `.fdt` 数据。

## 10. Project Cleanliness Rules

- 不在项目根目录保留临时文件。
- 临时脚本、调试输出、预览文件只允许在系统临时目录中短暂使用，并在结束前删除。
- 正式输出只放入约定目录，例如 `data/processed`、`data/features`、`results`、`docs`。
- 不要把外部原始数据复制进项目目录；代码通过配置读取外部路径。
- 后续创建代码时应补充 `.gitignore`，排除缓存、模型权重、临时结果和大体积中间文件。
