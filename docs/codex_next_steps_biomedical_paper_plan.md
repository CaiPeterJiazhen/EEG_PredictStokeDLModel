# 后续 Codex 执行计划：面向生物医学期刊的 tACS-EEG 恢复预测论文

更新时间：2026-05-24  
适用项目：`CaiPeterJiazhen/EEG_PredictStokeDLModel`  
目标论文方向：基于 tACS 治疗前基线静息态 EEG，预测脑卒中患者治疗后上肢运动功能是否属于 proportional recovery，并从 PSD 与功能连接中提取可解释神经标志物。

---

## 0. 写给 Codex 的总原则

本项目后续不应继续盲目追求某个 seed 的最高 accuracy。下一阶段目标应从“模型探索”转向“可以写入生物医学论文的确认性分析、统计可靠性和神经机制解释”。

所有任务必须遵守以下原则：

1. **严格 patient-level LOSO**：任何训练、标准化、特征选择、PCA、阈值校准、SSL 预训练和模型选择都不能使用当前外层测试患者。
2. **区分探索性与确认性结果**：已经做过的多模型、多 seed、多超参数扫描只能作为探索性结果；论文主结果必须来自冻结后的分析协议。
3. **禁止治疗后变量泄漏**：`FMA_post`、`MBI_post`、`Delta_FMA_obs`、`Residual`、`label` 只能用于标签和评估，不能作为模型输入。
4. **优先完成统计与解释**：先补统计显著性、置信区间、错误患者分析和可解释性，再考虑新的复杂模型。
5. **结果必须可复现**：每个正式输出都必须有命令、配置、seed、输入文件、输出文件路径和测试记录。
6. **项目目录保持整洁**：使用 `python -B` 和 `pytest -p no:cacheprovider`；不在根目录保留临时文件；正式输出只进入 `results/`、`data/features/`、`docs/` 等约定目录。
7. **面向生物医学审稿**：文章不仅要报告模型性能，还要解释哪些 PSD 频段、通道、FC 边、脑区网络与 proportional recovery 相关，并和已有脑卒中 EEG/tACS/康复文献对齐。

---

## 1. 当前项目事实与论文主线

### 1.1 项目任务

监督任务：

```text
Input: tACS 治疗前基线 EO/EC 静息态 EEG
Features: PSD, wPLI FC, imaginary coherence FC
Primary target: proportional recovery vs poor recovery
Evaluation: 19-patient LOSO
```

当前标签定义：

```text
residual = 0.7 * (66 - FMA_pre) - (FMA_post - FMA_pre)
label = 1 if residual <= 1.5 else 0
```

当前监督样本：

```text
n = 19
positive = 10
negative = 9
```

当前主特征：

```text
PSD:
  EO: 62 x 90
  EC: 62 x 90
  frequency range: 0.5-45 Hz
  resolution: 0.5 Hz

FC:
  metric: wPLI / imaginary coherence
  EO: 1891 x 6
  EC: 1891 x 6
  bands: delta, theta, alpha, beta low, beta medium, beta high
```

tACS 目标频率为 20 Hz，因此 **beta medium 18-21 Hz** 是后续解释分析中的重点频段。

### 1.2 推荐论文主线

建议论文不要写成“某个深度学习模型达到最高准确率”，而应写成：

> 本研究建立了一个严格 LOSO 的小样本 EEG 预测框架，用 tACS 前基线 EO/EC EEG 的 PSD 与 wPLI 功能连接预测卒中患者 tACS 后上肢 proportional recovery。我们系统比较传统机器学习、无 SSL 的 CNN 和 SSL-CNN。结果重点不是单 seed 的最高 accuracy，而是证明结构化 EEG-CNN 与无负样本 SSL 能在小样本场景中提高预测稳定性，并通过 PSD/FC attribution 揭示与 tACS 20 Hz、运动网络、额-中央-顶叶连接相关的可解释神经标志物。

### 1.3 需要模仿的生物医学论文结构

参考论文：

```text
Lin et al., A Transferable Deep Learning Prognosis Model for Predicting Stroke Patients' Recovery in Different Rehabilitation Trainings
```

后续应借鉴其 `C. Key Factors of Prognosis Model` 的写法：

1. 先说明解释方法如何从每个 LOOCV/LOSO 模型得到 key factors。
2. 再把 key factors 转换成可统计分析的 PSD/FC biomarkers。
3. 对重要 biomarker 做相关性分析、组间差异分析和 FDR 校正。
4. 输出 PSD topomap、FC connectome、重要通道/频段/边列表。
5. 在 Discussion 中把模型解释结果与既往脑卒中恢复、tACS、运动网络和 EEG PSD/FC 文献对齐。
6. 明确指出哪些结果只是模型解释，哪些结果在原始 EEG 统计中也能被重复观察到。

---

## 2. 冻结后的主比较框架

后续所有正式统计和图表应围绕以下三大模型组进行。不要再随意改变主模型，除非明确标记为 Supplementary exploratory analysis。

### 2.1 Traditional ML baseline

Primary baseline:

```text
Logistic Regression L1
Logistic Regression L2
```

Supplementary baselines:

```text
SVM linear
SVM RBF
Random Forest
Gaussian NB
KNN
XGBoost/LightGBM if installed
```

输入：

```text
PSD band-power features
wPLI FC flattened edge-band features
optional iCOH features as supplementary
```

要求：

- scaler、selector、PCA 必须 fold-local。
- 报告所有 baseline，但论文主比较可用 Logistic L1/L2 作为传统 ML 代表。
- 不要只报告弱 ML baseline，否则会被审稿人质疑深度模型优势是人为造成的。

### 2.2 Supervised CNN without SSL

Primary no-SSL CNN:

```text
architecture: multimodal
feature_kind: psd-fc-wpli
fusion: gated
encoder: cnn
embedding_dim: 32
dropout: 0
lr: 0.002
weight_decay: 1e-5
epochs: 100
patience: 100
seeds: 0,1,2,3,7,13
```

理由：

- 与当前项目中最合理的 `PSD + wPLI gated CNN` 结构一致。
- 使用稳定性方案，而不是只用 seed 2 的 best run。
- 能直接解释 EO/EC gate、PSD branch、wPLI branch。

### 2.3 SSL-CNN

Primary SSL candidate:

```text
multi-task VICReg SSL
local masked reconstruction + VICReg alignment
contrastive_weight: 0.001
vicreg_invariance_weight: 25
vicreg_variance_weight: 25
vicreg_covariance_weight: 1
data_scope: all-patient
strict_loso_exclusion: true
seeds: 0,1,2,3,7,13
```

Secondary SSL candidate for supplement:

```text
tuned Barlow Twins feature-space SSL
projection_dim: 32
feature_mask_prob: 0.01
seeds: 0,1,2,3,7,13,21,42,99,123
```

理由：

- NT-Xent 需要 batch 内负样本，但本项目没有语义上明确的负样本；VICReg 和 Barlow Twins 更符合当前数据场景。
- 当前证据显示 SSL 的主要价值是提高 seed 稳定性和平均表现，而不是稳定提高单 seed 最高 accuracy。
- 如果主文篇幅有限，主文只放 VICReg，Barlow Twins 放 Supplementary。

---

## 3. Phase 1：代码审计与分析协议冻结

### 3.1 目标

建立一个冻结版 analysis protocol，明确后续论文主结果只使用固定模型、固定 seeds、固定指标、固定统计方法。避免继续通过外层 LOSO 结果反复调参。

### 3.2 Codex 任务

#### Task 1.1 创建冻结分析协议文档

创建：

```text
docs/paper_analysis_protocol.md
```

内容必须包括：

- 研究问题。
- supervised cohort 和标签定义。
- 输入特征。
- 主模型组：ML、no-SSL CNN、SSL-CNN。
- 主 seeds。
- 主评价指标。
- 统计方法。
- 可解释性方法。
- 哪些结果属于 exploratory，哪些属于 confirmatory。
- 禁止事项：不能使用外层测试结果继续调参。

完成标准：

- 文档能被论文 Methods 直接转化。
- 每个模型配置都有完整命令或配置项。
- 明确说明当前项目没有外部验证集，因此结论为 internal validation / pilot study。

#### Task 1.2 建立结果 manifest

创建：

```text
src/eeg_recovery/utils/result_manifest.py
scripts/12_build_result_manifest.py
tests/test_result_manifest.py
```

输出：

```text
results/metrics/result_manifest.csv
```

manifest 至少包含：

```text
run_name
model_family
model_type
feature_kind
ssl_method
seed
data_scope
prediction_path
metric_path
loss_history_path
is_exploratory
is_confirmatory
notes
```

完成标准：

- 能扫描 `results/predictions` 和 `results/metrics`。
- 能标记哪些已有结果是探索性结果。
- 后续统计脚本只从 manifest 读取需要比较的预测文件。

#### Task 1.3 泄漏审计报告

创建：

```text
docs/leakage_audit_report.md
```

重点审计：

- `loso.py`
- `train_baselines.py`
- `train_supervised.py`
- `train_feature_ssl.py`
- `train_masked_ssl.py`
- `frozen_ssl_heads.py`
- `ensemble_calibration.py`
- `feature_tables.py`
- `labels.py`

报告格式：

```text
模块
潜在泄漏点
当前代码如何避免
仍需修复/确认的问题
结论：pass / warning / fail
```

特别注意：

- SSL 是否排除当前测试患者。
- 阈值校准是否 OOF。
- pair-diff embedding 是否把标签或测试集信息带入。
- RBF SVM 1.0 结果必须单独审查，不能作为主结果。

---

## 4. Phase 2：统计可靠性与模型比较

### 4.1 目标

完成论文中必须有的统计验证：置信区间、permutation test、binomial test、模型间比较、校准分析。

### 4.2 Codex 任务

#### Task 2.1 扩展 metrics 模块

创建或修改：

```text
src/eeg_recovery/training/statistics.py
src/eeg_recovery/training/metrics.py
tests/test_statistics.py
tests/test_metrics.py
```

必须实现：

```text
accuracy
balanced_accuracy
sensitivity
specificity
precision
NPV
PPV
F1
ROC-AUC
PR-AUC
Brier score
confusion matrix
exact binomial test
bootstrap 95% CI
permutation test
paired model comparison
calibration curve table
```

bootstrap 要求：

- patient-level bootstrap。
- 默认 `n_bootstrap=10000`，可命令行调整。
- 固定随机种子。
- 对每个模型输出 metric、mean、lower_95、upper_95。

permutation test 分两级：

1. 快速版：固定预测概率，打乱 `y_true`。
2. 严格版：打乱标签后重新训练整个 LOSO pipeline；可以只对主模型做，耗时较高。

完成标准：

- 现有 predictions CSV 可直接输入。
- 单一类别 bootstrap sample 需要安全处理 ROC-AUC/PR-AUC 为 NaN 或跳过。
- 测试覆盖小样本边界情况。

#### Task 2.2 创建统计验证脚本

创建：

```text
scripts/12_run_statistical_validation.py
```

输入：

```text
--manifest results/metrics/result_manifest.csv
--confirmatory-only true
--n-bootstrap 10000
--n-permutations 10000
--seed 20260524
```

输出：

```text
results/metrics/confirmatory_model_metrics.csv
results/metrics/confirmatory_bootstrap_ci.csv
results/metrics/confirmatory_permutation_tests.csv
results/metrics/confirmatory_binomial_tests.csv
results/metrics/confirmatory_model_comparisons.csv
results/metrics/confirmatory_calibration_metrics.csv
results/figures/calibration_curve_<model>.png
results/figures/roc_curve_confirmatory_models.png
results/figures/pr_curve_confirmatory_models.png
```

模型间比较至少包括：

```text
Logistic L1 vs no-SSL CNN
Logistic L2 vs no-SSL CNN
Logistic L1 vs SSL-CNN
no-SSL CNN vs SSL-CNN
```

完成标准：

- 所有输出都是 patient-level。
- 模型间比较使用相同 subject_id 顺序。
- 如果某模型缺少某个 subject，脚本必须报错而不是自动 inner join 后沉默继续。

#### Task 2.3 生成可写入论文的统计摘要

创建：

```text
docs/statistical_validation_summary.md
```

内容：

- 主模型结果表。
- 每个模型的 95% CI。
- permutation p-value。
- binomial p-value。
- paired comparison 结果。
- 对结果可信度的文字解释。
- 哪些结论能写进主文，哪些只能写 supplement。

---

## 5. Phase 3：确认性重跑主实验

### 5.1 目标

在冻结配置后重跑主模型，获得一组不再调参的 confirmatory results。

### 5.2 Codex 任务

#### Task 3.1 重跑 ML baseline

命令模板：

```powershell
python -B scripts\04_train_ml_baselines.py --config configs\paths.example.yaml --feature-kind psd-fc-wpli --seed 20260524
```

输出重命名为：

```text
results/predictions/confirmatory_ml_baseline_predictions.csv
results/metrics/confirmatory_ml_baseline_metrics.csv
```

完成标准：

- 至少包含 Logistic L1/L2。
- 保存每个 subject 的 y_true、y_score、y_pred。
- 更新 result manifest。

#### Task 3.2 重跑 no-SSL CNN

命令模板：

```powershell
python -B scripts\05_train_supervised_loso.py --config configs\paths.example.yaml --architecture multimodal --feature-kind psd-fc-wpli --fusion gated --encoder cnn --device cuda --epochs 100 --patience 100 --embedding-dim 32 --dropout 0 --lr 0.002 --weight-decay 0.00001 --seed <SEED>
```

seeds：

```text
0, 1, 2, 3, 7, 13
```

输出命名：

```text
results/predictions/confirmatory_no_ssl_psdfcwpli_gated_cnn_seed<seed>.csv
results/metrics/confirmatory_no_ssl_psdfcwpli_gated_cnn_seed<seed>.csv
results/training_logs/confirmatory_no_ssl_psdfcwpli_gated_cnn_seed<seed>.csv
results/figures/confirmatory_no_ssl_psdfcwpli_gated_cnn_seed<seed>_loss.png
```

另建 ensemble：

```text
results/predictions/confirmatory_no_ssl_psdfcwpli_gated_cnn_seedensemble.csv
results/metrics/confirmatory_no_ssl_psdfcwpli_gated_cnn_seedensemble.csv
```

完成标准：

- 每个 seed 都包含 19 个 subject。
- 更新 result manifest。
- 统计每个 subject 在不同 seed 中预测错误的次数。

#### Task 3.3 重跑 SSL-CNN：multi-task VICReg

命令模板：

```powershell
python -B scripts\09_train_masked_ssl_transfer.py --config configs\paths.example.yaml --data-scope all-patient --device cuda --seed <SEED> --psd-ssl-epochs 20 --fc-ssl-epochs 20 --ssl-batch-size 8 --ssl-lr 0.001 --embedding-dim 32 --decoder-hidden-dim 128 --psd-channel-mask-prob 0.15 --psd-frequency-mask-prob 0.15 --psd-element-mask-prob 0.02 --fc-node-mask-prob 0.02 --fc-edge-mask-prob 0.05 --fc-band-mask-prob 0.02 --eo-ec-consistency-weight 0 --contrastive-weight 0.001 --alignment-method vicreg --projection-dim 16 --contrastive-noise-std 0.02 --contrastive-feature-mask-prob 0.05 --vicreg-invariance-weight 25 --vicreg-variance-weight 25 --vicreg-covariance-weight 1 --vicreg-variance-target 1 --supervised-epochs 100 --patience 100 --supervised-lr 0.001 --dropout 0 --transfer-mode finetune --summary-name confirmatory_mtvicreg_seed<SEED>_summary.csv
```

seeds：

```text
0, 1, 2, 3, 7, 13
```

完成标准：

- 每折 SSL 预训练必须排除当前测试患者。
- 记录每折 SSL pair 数。
- 保存 SSL history、supervised predictions、metrics、loss history。
- 建立 seed ensemble。
- 更新 result manifest。

#### Task 3.4 错误患者稳定性分析

创建：

```text
scripts/14_analyze_error_subjects.py
src/eeg_recovery/analysis/error_subjects.py
tests/test_error_subjects.py
```

输出：

```text
results/metrics/error_subject_frequency_by_model.csv
results/metrics/error_subject_clinical_table.csv
results/figures/error_subject_heatmap.png
docs/error_subject_analysis.md
```

重点患者：

```text
sub09
sub14
sub05
sub28
```

分析内容：

- 每个模型/seed 是否预测错误。
- 概率是否接近 0.5。
- FMA_pre、FMA_post、residual 是否接近阈值。
- 患侧、病程、年龄、基线严重程度是否异常。
- PSD/FC 是否离群。
- 是否可能属于标签边界样本或数据质量问题。

---

## 6. Phase 4：可解释性分析

### 6.1 目标

建立可写入生物医学论文的解释体系，类似 Lin et al. 的 `C. Key Factors of Prognosis Model`，但更适合本项目的 tACS 上肢恢复预测。

解释分析必须回答：

1. 模型预测 proportional recovery 时，哪些 PSD 频率-通道最重要？
2. 20 Hz tACS 对应的 beta medium 18-21 Hz 是否出现突出贡献？
3. 哪些 wPLI 边和脑区网络最重要？
4. EO 和 EC 哪个状态更重要？
5. 模型解释结果是否跨 LOSO folds、seeds 稳定？
6. 解释结果是否能被 raw EEG group difference / clinical correlation 支持？
7. 解释结果是否与已有脑卒中恢复、tACS、运动网络 EEG 文献一致？

### 6.2 Codex 任务

#### Task 4.1 建立 explainability 模块

创建：

```text
src/eeg_recovery/explainability/__init__.py
src/eeg_recovery/explainability/attribution_base.py
src/eeg_recovery/explainability/psd_attribution.py
src/eeg_recovery/explainability/fc_attribution.py
src/eeg_recovery/explainability/gate_analysis.py
src/eeg_recovery/explainability/stability.py
src/eeg_recovery/explainability/clinical_correlation.py
src/eeg_recovery/explainability/visualization.py
tests/test_explainability_shapes.py
tests/test_explainability_statistics.py
```

必须支持：

```text
Integrated Gradients
Gradient x Input
Occlusion / permutation importance
Optional GradientSHAP if dependency available
Model randomization sanity check
```

不要求一开始支持所有方法，但 Integrated Gradients 和 occlusion/permutation 必须有。

完成标准：

- PSD attribution shape：`62 x 90`。
- FC attribution shape：`1891 x 6`。
- EO/EC 分开输出。
- PSD/wPLI branch 分开输出。
- 支持多个 folds/seeds 聚合。
- 单元测试使用 mock tensors，不依赖真实大数据。

#### Task 4.2 训练时保存可解释模型 checkpoint

当前许多训练脚本只保存预测和 metrics，不一定保存每折模型。可解释性需要每折模型参数。

修改：

```text
src/eeg_recovery/training/train_supervised.py
scripts/05_train_supervised_loso.py
scripts/09_train_masked_ssl_transfer.py
```

新增选项：

```text
--save-fold-checkpoints
--checkpoint-prefix <name>
```

输出：

```text
results/checkpoints/<run_name>/fold_<fold_index>_<subject_id>.pt
results/checkpoints/<run_name>/fold_manifest.csv
```

checkpoint manifest 包含：

```text
run_name
fold_index
test_subject_id
seed
model_config
checkpoint_path
scaler_path
feature_kind
architecture
fusion
encoder
```

完成标准：

- checkpoint 不进入 git。
- `.gitignore` 确认忽略 `results/checkpoints/` 或 `checkpoints/`。
- 能从 checkpoint 完整重建模型并复现该 fold 的 test probability。

#### Task 4.3 PSD attribution 分析

创建脚本：

```text
scripts/13_run_psd_explainability.py
```

输入：

```text
--checkpoint-manifest results/checkpoints/<run_name>/fold_manifest.csv
--predictions results/predictions/<run>.csv
--method integrated_gradients
--baseline zero
--n-steps 64
```

输出：

```text
results/explainability/psd_attribution_subject_level.csv
results/explainability/psd_attribution_channel_frequency.npz
results/explainability/psd_attribution_band_channel_summary.csv
results/explainability/psd_attribution_roi_band_summary.csv
results/explainability/psd_attribution_seed_stability.csv
results/figures/psd_topomap_delta.png
results/figures/psd_topomap_theta.png
results/figures/psd_topomap_alpha.png
results/figures/psd_topomap_beta_low.png
results/figures/psd_topomap_beta_medium.png
results/figures/psd_topomap_beta_high.png
results/figures/psd_channel_frequency_heatmap.png
```

PSD summary 必须包含：

```text
state: EO / EC / averaged
channel
frequency_hz
band
attribution_mean
attribution_abs_mean
attribution_std
seed_stability
fold_stability
rank
```

重点统计：

- beta medium 18-21 Hz 的 top channels。
- C3/C4、FC3/FC4、CP3/CP4、CZ、CPZ 等运动相关通道。
- 刺激侧 vs 非刺激侧。
- frontal / central / parietal / occipital ROI summary。
- proportional vs poor group 原始 PSD 是否有对应差异。
- attribution 与 `Delta_FMA_obs`、`Residual`、`FMA_pre` 的相关。

完成标准：

- 每个图都有 CSV 源数据。
- topomap 使用统一 channel coordinates。
- 如果缺少部分通道坐标，必须报告，不得静默插值。
- 多重比较使用 FDR 校正。

#### Task 4.4 FC attribution 分析

创建脚本：

```text
scripts/13_run_fc_explainability.py
```

输出：

```text
results/explainability/fc_attribution_subject_level.csv
results/explainability/fc_attribution_edge_band.npz
results/explainability/fc_attribution_edge_band_summary.csv
results/explainability/fc_attribution_node_summary.csv
results/explainability/fc_attribution_roi_pair_summary.csv
results/explainability/fc_attribution_seed_stability.csv
results/figures/fc_connectome_delta.png
results/figures/fc_connectome_theta.png
results/figures/fc_connectome_alpha.png
results/figures/fc_connectome_beta_low.png
results/figures/fc_connectome_beta_medium.png
results/figures/fc_connectome_beta_high.png
results/figures/fc_node_importance_beta_medium.png
```

FC summary 必须包含：

```text
state
edge_index
channel_i
channel_j
roi_i
roi_j
hemisphere_i
hemisphere_j
band
attribution_mean
attribution_abs_mean
attribution_std
seed_stability
fold_stability
rank
```

重点统计：

- beta medium 18-21 Hz 的 top edges。
- 刺激侧 M1 区域相关边，例如 C3/FC3/CP3/CZ 周边。
- interhemispheric vs intrahemispheric。
- frontal-central、central-parietal、fronto-parietal、occipital-parietal 等 ROI pair。
- node-level degree importance。
- top FC edges 的 raw wPLI 是否与 recovery label / residual 相关。
- top FC edges 是否在 correct predictions 中更稳定，在 error subjects 中异常。

完成标准：

- 输出 edge-level、node-level、ROI-pair-level 三层解释。
- 图中边数不超过 top 20 或 top 30，避免 connectome 过密。
- 正负 attribution 分开处理，不要只取绝对值。

#### Task 4.5 EO/EC gate 与 branch importance

创建：

```text
scripts/13_run_gate_analysis.py
```

输出：

```text
results/explainability/gate_state_importance.csv
results/explainability/branch_ablation_importance.csv
results/figures/gate_weights_by_state_and_group.png
results/figures/branch_ablation_importance.png
```

分析内容：

- PSD branch 中 EO vs EC gate weight。
- wPLI branch 中 EO vs EC gate weight。
- proportional vs poor group 的 gate weight 差异。
- 正确预测 vs 错误预测的 gate weight 差异。
- branch ablation：只遮蔽 PSD、只遮蔽 wPLI、只遮蔽 EO、只遮蔽 EC 后概率变化。

完成标准：

- gate weight 不能被解释为绝对因果，只能作为模型内部状态权重。
- branch ablation 与 attribution 结果是否一致需要写入 summary。

#### Task 4.6 解释稳定性与 sanity checks

创建：

```text
scripts/13_run_explainability_stability.py
```

输出：

```text
results/explainability/attribution_seed_rank_correlation.csv
results/explainability/attribution_fold_stability.csv
results/explainability/attribution_model_randomization_sanity.csv
results/explainability/attribution_label_permutation_sanity.csv
docs/explainability_sanity_check_report.md
```

必须实现：

- seed 间 top-k overlap。
- seed 间 Spearman rank correlation。
- fold 间 top-k overlap。
- model randomization：随机化分类头或 encoder 后 attribution 应显著改变。
- label permutation model 的 attribution 不应稳定聚焦于同一神经通路。

完成标准：

- 没有稳定性的 attribution 不进入主文，只能进 supplement 或不报告。
- 若 sanity check fail，必须停止把 attribution 当作神经机制解释。

#### Task 4.7 临床相关与原始 EEG 验证

创建：

```text
scripts/13_run_biomarker_validation.py
src/eeg_recovery/explainability/biomarker_validation.py
```

输出：

```text
results/explainability/clinical_correlation.csv
results/explainability/raw_psd_group_difference.csv
results/explainability/raw_fc_group_difference.csv
results/explainability/top_biomarker_validation_summary.csv
results/figures/top_biomarker_scatter_residual.png
results/figures/top_biomarker_group_boxplot.png
```

验证对象：

- top PSD channel-band biomarkers。
- top FC edge-band biomarkers。
- beta medium 18-21 Hz motor-network biomarkers。
- gate/branch importance summaries。

统计：

```text
normality test
t-test or Wilcoxon / Mann-Whitney
Pearson or Spearman correlation
FDR correction
effect size
95% CI
```

关联变量：

```text
label
Residual
Delta_FMA_obs
FMA_pre
FMA_post
MBI_pre
duration
age
affected_hand
```

注意：

- `FMA_post` 只能用于 post-hoc validation，不能作为模型输入。
- 对 19 例小样本，所有 p-value 都必须谨慎解释。
- 重点看 effect size、方向一致性、稳定性，而不只看 p < 0.05。

---

## 7. Phase 5：文献对齐与生物医学解释

### 7.1 目标

把模型解释结果转化为生物医学论文里的“Key Factors of Prognosis Model”部分，而不是只展示 AI saliency 图。

### 7.2 Codex 任务

#### Task 5.1 建立文献特征表

创建：

```text
docs/biomedical_interpretation_literature_map.md
data/manual/literature_eeg_recovery_features.csv
```

CSV 字段：

```text
paper
year
population
treatment
task
EEG_state
feature_type
frequency_band
channel_or_roi
connectivity_pair
direction_good_recovery
statistical_method
main_finding
notes
```

至少纳入：

- Lin et al. 2022 JBHI 这篇 transferable prognosis model。
- tACS 后恢复好/差患者 PSD 特点相关论文。
- tACS 后恢复好/差患者 FC 特点相关论文。
- 卒中上肢恢复 EEG biomarker 论文。
- beta-band / motor network / interhemispheric connectivity 相关论文。

完成标准：

- 每个文献结论都要能追溯到原文。
- 不要把下肢康复结果直接等同于上肢 tACS；只能作为相似证据或方法学参考。
- 对不一致文献也要记录，不能只选支持本研究的文献。

#### Task 5.2 写解释结果与文献对齐报告

创建：

```text
docs/biomedical_interpretation_summary.md
```

建议结构：

```text
1. Model-derived key factors
2. PSD biomarkers
3. FC biomarkers
4. EO/EC state dependence
5. Relationship to 20 Hz tACS and beta medium band
6. Relationship to motor network and hemispheric rebalancing
7. Comparison with Lin et al. style analysis
8. Findings supported by raw EEG statistics
9. Findings only supported by model attribution
10. Limitations
```

必须明确分层：

- Tier 1：attribution 稳定 + raw feature 相关/组差异 + 文献支持。
- Tier 2：attribution 稳定 + 文献支持，但 raw feature 统计未显著。
- Tier 3：仅模型 attribution，暂不作为神经机制结论。

完成标准：

- 形成论文 Results/Discussion 可直接使用的段落。
- 不夸大因果性。
- 对 tACS 20 Hz beta medium 的解释必须有数据支撑。

---

## 8. Phase 6：临床特征与基线表

### 8.1 目标

面向生物医学期刊，需要有患者基线特征表、组间差异、临床-only baseline、EEG+clinical 融合对比。

### 8.2 Codex 任务

#### Task 6.1 患者基线特征表

创建：

```text
scripts/15_generate_patient_characteristics_table.py
```

输出：

```text
results/tables/table1_patient_characteristics.csv
results/tables/table1_patient_characteristics.md
```

字段：

```text
age
sex
duration
affected_hand
FMA_pre
FMA_post
Delta_FMA_obs
Residual
MBI_pre
MBI_post
label
```

统计：

- proportional recovery vs poor recovery。
- continuous variables：median/IQR 或 mean/SD，按正态性选择。
- categorical variables：count/percentage。
- p-value：Fisher exact / t-test / Mann-Whitney。
- effect size。

完成标准：

- 明确哪些变量治疗前可用，哪些只用于描述和标签。
- 主文 Table 1 不应误导为模型输入包含治疗后变量。

#### Task 6.2 临床-only 与 EEG+clinical baseline

创建或扩展：

```text
scripts/16_train_clinical_baselines.py
src/eeg_recovery/training/train_clinical_baselines.py
tests/test_clinical_baselines.py
```

模型：

```text
clinical-only logistic
FMA_pre-only logistic
EEG-only logistic
EEG+clinical logistic
no-SSL CNN EEG-only
no-SSL CNN EEG+clinical
SSL-CNN EEG-only
SSL-CNN EEG+clinical
```

允许输入：

```text
age
sex
duration
affected_hand
FMA_pre
MBI_pre
```

禁止输入：

```text
FMA_post
MBI_post
Delta_FMA_obs
Residual
label
```

输出：

```text
results/metrics/clinical_baseline_comparison.csv
results/predictions/clinical_baseline_predictions.csv
```

完成标准：

- 证明 EEG 特征提供的信息是否超过 baseline clinical severity。
- 如果 clinical-only 与 EEG 模型差不多，论文必须诚实报告。

---

## 9. Phase 7：论文图表计划

### 9.1 主文图

建议主文 5-6 张图：

#### Figure 1：Study design and pipeline

内容：

- tACS 前 baseline EEG。
- EO/EC。
- 患侧翻转。
- PSD/wPLI extraction。
- ML vs CNN vs SSL-CNN。
- LOSO evaluation。
- Explainability。

输出：

```text
results/figures/paper_figure1_pipeline.png
```

#### Figure 2：Model architecture

内容：

- PSD branch。
- wPLI branch。
- EO/EC gated fusion。
- SSL pretraining objective。
- supervised classifier。

输出：

```text
results/figures/paper_figure2_model_architecture.png
```

#### Figure 3：Predictive performance

内容：

- 三类模型性能比较。
- accuracy / balanced accuracy / ROC-AUC / PR-AUC。
- 95% CI。
- ROC/PR curves。

输出：

```text
results/figures/paper_figure3_performance.png
```

#### Figure 4：Seed stability and SSL effect

内容：

- no-SSL vs SSL-CNN multi-seed distribution。
- seed-level accuracy boxplot/pointplot。
- variance reduction。
- per-subject prediction consistency heatmap。

输出：

```text
results/figures/paper_figure4_seed_stability.png
```

#### Figure 5：PSD key factors

内容：

- PSD attribution topomap by band。
- beta medium channel-frequency heatmap。
- top biomarker scatter vs residual / Delta FMA。
- proportional vs poor group boxplot。

输出：

```text
results/figures/paper_figure5_psd_key_factors.png
```

#### Figure 6：FC key factors

内容：

- beta medium top FC edges connectome。
- node importance。
- ROI-pair importance。
- top FC biomarker correlation.

输出：

```text
results/figures/paper_figure6_fc_key_factors.png
```

### 9.2 主文表

#### Table 1：Patient characteristics

输出：

```text
results/tables/table1_patient_characteristics.csv
```

#### Table 2：Model performance

输出：

```text
results/tables/table2_model_performance.csv
```

字段：

```text
model
input
n_subjects
accuracy
balanced_accuracy
sensitivity
specificity
ROC-AUC
PR-AUC
95% CI
permutation p
binomial p
```

#### Table 3：Key PSD/FC biomarkers

输出：

```text
results/tables/table3_key_biomarkers.csv
```

字段：

```text
feature_type
state
band
channel_or_edge
roi_or_roi_pair
attribution_rank
raw_feature_effect_direction
correlation_with_residual
FDR_q
literature_consistency
```

### 9.3 Supplementary

建议：

- 所有 ML baseline。
- 所有 feature combinations。
- SSL objective ablation。
- seed-wise results。
- error patient detailed table。
- attribution stability checks。
- raw PSD/FC group-difference maps。
- leakage audit report。

---

## 10. Phase 8：README、运行脚本与复现性

### 10.1 Codex 任务

#### Task 8.1 README

创建：

```text
README.md
```

必须包括：

- 项目目标。
- 数据路径配置。
- 62 通道事实。
- EO/EC 文件命名。
- 标签定义。
- 患侧翻转。
- PSD/FC 特征。
- LOSO 防泄漏原则。
- 如何运行每一步。
- 输出路径。
- 如何复现主结果。
- 如何运行测试。
- 当前限制。

#### Task 8.2 End-to-end runner

创建：

```text
scripts/run_confirmatory_pipeline.py
tests/test_pipeline_smoke.py
```

支持 stages：

```text
prepare-metadata
compute-psd
compute-fc
train-ml
train-no-ssl-cnn
train-ssl-cnn
build-manifest
statistics
explainability
figures
tables
```

要求：

- 默认 dry-run 不访问外部 EEG 大文件。
- 真正运行需要显式传 `--run-real`.
- 每个 stage 写入 log。
- 可以从中断处继续。

#### Task 8.3 环境记录

创建：

```text
environment.yml
requirements-lock.txt
scripts/99_record_environment.py
```

输出：

```text
results/reproducibility/environment_report.txt
```

包括：

```text
python version
package versions
torch version
CUDA availability
GPU name
OS
git commit
random seeds
```

---

## 11. Phase 9：论文写作框架

### 11.1 Codex 任务

创建：

```text
docs/manuscript_outline_biomedical.md
```

建议标题：

```text
Self-supervised multimodal EEG representation learning for predicting upper-limb proportional recovery after tACS in stroke: a pilot leave-one-subject-out study
```

建议摘要结构：

```text
Background
Objective
Methods
Results
Interpretation
Limitations
Conclusion
```

### 11.2 主文结构

```text
1. Introduction
   - Stroke upper-limb recovery heterogeneity
   - tACS treatment response prediction
   - EEG PSD/FC as biomarkers
   - Small-sample challenge
   - Why SSL and explainability

2. Materials and Methods
   - Participants
   - tACS intervention and outcome
   - EEG acquisition/preprocessing
   - Label definition
   - PSD and FC features
   - ML baselines
   - CNN model
   - SSL pretraining
   - Evaluation protocol
   - Statistical analysis
   - Explainability and biomarker validation

3. Results
   - Patient characteristics
   - Predictive performance
   - Seed stability and SSL effect
   - Error subject analysis
   - PSD key factors
   - FC key factors
   - Clinical/biomarker correlation

4. Discussion
   - Prediction feasibility
   - SSL improves stability rather than only best accuracy
   - Beta medium and tACS-related interpretation
   - Motor network / hemispheric connectivity interpretation
   - Comparison with previous EEG prognosis models
   - Limitations
   - Future work

5. Conclusion
```

### 11.3 论文表述边界

可以写：

- “internal LOSO validation”
- “pilot study”
- “model-derived biomarkers”
- “SSL improved seed stability”
- “findings suggest involvement of beta-band PSD/FC and motor-related networks”

不建议写：

- “clinically deployable”
- “causal biomarkers”
- “SSL significantly outperforms all baselines” unless statistics support it
- “tACS mechanism confirmed”
- “external generalization proven”

---

## 12. 推荐 Codex 执行顺序

按以下顺序一步步执行，不要跳步：

1. `docs/paper_analysis_protocol.md`
2. `docs/leakage_audit_report.md`
3. `result_manifest.py` 和 `scripts/12_build_result_manifest.py`
4. `statistics.py` 和 `scripts/12_run_statistical_validation.py`
5. confirmatory ML / no-SSL CNN / SSL-CNN 重跑
6. `docs/statistical_validation_summary.md`
7. checkpoint 保存与复现
8. explainability 基础模块和测试
9. PSD attribution
10. FC attribution
11. gate and branch ablation
12. explainability stability/sanity checks
13. raw EEG biomarker validation
14. patient characteristics table
15. clinical-only / EEG+clinical baselines
16. literature map
17. biomedical interpretation summary
18. paper figures and tables
19. README
20. run_confirmatory_pipeline
21. manuscript outline

每完成一步都必须：

```powershell
python -B -m pytest tests -v -p no:cacheprovider
```

如果全量测试太慢，至少运行该任务相关测试，然后在阶段结束时运行全量测试。

每步结束必须更新：

```text
docs/task_status.md
```

---

## 13. Definition of Done

项目达到可以开始正式写论文初稿的标准：

1. 冻结分析协议完成。
2. 主模型 confirmatory results 完成。
3. 统计验证完成，包括 CI、permutation、binomial、paired comparisons。
4. 可解释性完成，包括 PSD、FC、gate、branch ablation。
5. attribution 通过 seed/fold stability 和 sanity checks。
6. top PSD/FC biomarkers 做了 raw feature validation。
7. 完成 patient characteristics table。
8. 完成 clinical-only baseline，确认 EEG 是否提供额外信息。
9. 完成文献对齐和 biomedical interpretation。
10. 完成主文图表和 supplement 图表。
11. README 和复现脚本完成。
12. 全量测试通过。
13. `docs/statistical_validation_summary.md` 和 `docs/biomedical_interpretation_summary.md` 能直接转化为论文 Results/Discussion。

---

## 14. 近期最优先的三个任务

如果只能先做三件事，按此顺序：

### Priority 1：统计验证

先完成：

```text
src/eeg_recovery/training/statistics.py
scripts/12_run_statistical_validation.py
docs/statistical_validation_summary.md
```

原因：

- 没有 CI 和 permutation p-value，任何 0.8421 accuracy 都很难说服生物医学审稿人。

### Priority 2：冻结确认性模型重跑

先固定：

```text
Logistic L1/L2
no-SSL PSD+wPLI gated CNN
multi-task VICReg SSL-CNN
```

原因：

- 必须避免继续用同一 19 个 LOSO 测试结果反复选模型。

### Priority 3：可解释性主线

先实现：

```text
PSD attribution
FC attribution
gate analysis
raw biomarker validation
```

原因：

- 生物医学期刊更看重可解释神经机制。
- 本项目的主要创新不应只是 accuracy，而应是 tACS 前 EEG 中可解释的 PSD/FC 恢复预测特征。
