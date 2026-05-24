# GPT Pro 项目审查说明：tACS EEG Proportional Recovery

更新时间：2026-05-24

本文档用于交给网页版 GPT Pro 对当前项目进行代码检查、项目评估和下一步研究方向建议。请重点审查代码是否存在数据泄漏、交叉验证实现是否严格、实验结论是否被小样本和反复调参高估，以及后续研究应如何更稳健地推进。

## 我理解的项目主要目的

这个项目的核心目的，是利用脑卒中患者 tACS 治疗前的静息态 EEG 数据，预测患者治疗后上肢运动功能是否符合 proportional recovery。项目希望回答的问题不是“治疗后 FMA 会是多少”，而是“基线 EEG 中是否存在能够预测 tACS 后恢复潜力的神经特征”。

当前监督训练的主任务是一个二分类任务：

- 输入：tACS 前静息态 EEG，包含睁眼 EO 和闭眼 EC 两个状态。
- 主要特征：PSD、wPLI 功能连接、imaginary coherence 功能连接，其中当前重点是 `PSD + wPLI`。
- 输出：是否属于 proportional recovery 组。
- 评估：19 例患者 leave-one-subject-out，简称 LOSO，也就是 19 折验证，每一折有 1 个患者作为测试集。

从研究目的上看，项目目前更接近“小样本神经生物标志物探索 + 严格受试者级预测验证”，而不是一个已经可以临床部署的模型。后续最重要的是保证验证流程可信、结果稳定、可解释性足够强，并且避免在 19 个样本上通过反复调参把测试集信息隐式用掉。

## 给 GPT Pro 的审查任务

建议 GPT Pro 按以下优先级检查：

1. 检查 LOSO 划分、标准化、特征选择、PCA、自监督预训练和模型选择是否存在测试患者泄漏。
2. 检查标签构建是否完全来自治疗前和治疗后 FMA 结果，且没有把治疗后变量作为模型输入。
3. 检查患者 ID、健康人 ID、EO/EC 文件识别、`.set/.fdt` 匹配、左右半球翻转是否一致。
4. 检查自监督学习的数据作用域是否在每折中正确排除了测试患者。
5. 评估目前多个实验结果是否可信，尤其是小样本、多 seed、反复调参、ensemble、阈值校准带来的乐观偏差。
6. 给出下一步研究方向，重点是如何让 SSL 真正提高稳定性或可解释性，而不是只追求某个 seed 的最高准确率。

## 数据来源

外部原始数据不存放在项目仓库中，当前项目通过本地路径读取。

患者信息表：

- `F:\CJZFile\EEG_M1\19例患者脑电数据完整性检查.xlsx`
- `F:\CJZFile\EEG_M1\脑卒中患者信息记录表.xlsx`

EEG 预处理后数据：

- 患者 EEG：`F:\CJZFile\EEG_M1\Patient_tACS_M1_RestingStateEEG_afterProcess`
- 健康人 EEG：`F:\CJZFile\EEG_M1\Health_tACS_M1_RestingStateEEG_afterProcess`
- 通道文件：`F:\CJZFile\EEG_M1\standard_1005.ced`

项目内生成数据：

- PSD 特征：`data/features/psd`
- FC 特征：`data/features/fc`
- 指标汇总：`results/metrics`
- LOSO 逐患者预测：`results/predictions`
- SSL 输出：`results/ssl`
- 训练损失日志：`results/training_logs`
- 图表：`results/figures`

## 监督训练名单与标签

监督训练最终使用 `19例患者脑电数据完整性检查.xlsx` 中确认的 19 例患者：

`sub01, sub05, sub07, sub08, sub09, sub10, sub11, sub13, sub14, sub15, sub16, sub17, sub18, sub20, sub22, sub24, sub27, sub28, sub29`

标签定义：

```text
residual = 0.7 * (66 - FMA_pre) - (FMA_post - FMA_pre)
```

当前使用监督训练 19 例患者的 residual 中位数作为阈值，阈值为 `1.5`。

- `label = 1`：`residual <= 1.5`，定义为 proportional recovery。
- `label = 0`：`residual > 1.5`，定义为 poor recovery。

当前标签分布：

- positive：10 例。
- negative：9 例。

注意：该标签是项目中的主要监督目标。GPT Pro 应检查标签计算代码是否只在标签构建阶段使用 FMA_post，模型输入阶段不能使用任何治疗后变量。

## EEG 文件与预处理事实

EEG 文件是 EEGLAB `.set/.fdt` 文件。预处理在本项目之前已经完成，包括滤波、重参考、ICA、坏道处理、分段长度控制、伪迹剔除等步骤。

文件命名规则：

- `1` 代表睁眼 EO。
- `2` 代表闭眼 EC。
- 需要避免把 `session12.set` 这类数字后缀误判为 EC。虽然原始文件中目前没有 `xxxx12.set`，代码层面已经为该类歧义加了保护。

通道事实：

- 当前实际通道数是 62。
- M1、M2 已删除，后续均按 62 通道处理。
- 中线电极在患侧镜像翻转时不改变。
- 采样率为 250 Hz。
- `.fdt` 为连续数据。

已知 `.set` 内部声明的 `.fdt` 文件名与外部实际文件名不一致的情况：

| `.set` 内部旧名 | 实际 `.fdt` 文件 |
|---|---|
| `zqc1.fdt` | `zqz1.fdt` |
| `dqm1.fdt` | `sqm1.fdt` |
| `cyq1.fdt` | `zyq1.fdt` |
| `ccx.fdt` | `ccx1.fdt` |

这些不是新数据，也不是错误配对，而是预处理后保存文件时重命名造成的内部声明和外部文件名不一致。当前 IO 代码会优先使用存在的 companion `.fdt`，从而兼容这些情况。

## 患侧对齐策略

为了让不同患手方向的患者进入同一模型空间，项目在特征计算前做了患侧对齐。

统一目标表示：

- 所有患者都被映射到“患手在右侧、刺激侧为 C3、左半球为刺激侧”的表示。

处理原则：

- 右侧患手患者：不翻转。
- 左侧患手患者：左右半球镜像通道顺序。
- 中线电极：保持不变。

该部分对结论很重要，因为 PSD 和 FC 都依赖通道空间含义。GPT Pro 应检查 `channels/hemisphere_flip.py`、`channels/mapping.py` 以及 PSD/FC 特征生成代码是否一致使用该映射。

## 当前主要特征

PSD：

- EO/EC 各自计算。
- 形状：`(62, 90)`。
- 频率范围：0.5-45 Hz。
- 频率分辨率：0.5 Hz。

FC：

- 指标 1：wPLI。
- 指标 2：imaginary coherence，简称 iCOH。
- EO/EC 各自计算。
- 62 通道全连接无向边数：`62 * 61 / 2 = 1891`。
- 形状：`(1891, 6)`。

频段：

| 频段 | 范围 |
|---|---|
| delta | 1-3 Hz |
| theta | 4-7 Hz |
| alpha | 8-13 Hz |
| beta low | 13-18 Hz |
| beta medium | 18-21 Hz |
| beta high | 21-30 Hz |

tACS 目标频率为 20 Hz，因此 beta medium 频段具有特别高的解释价值。

## 项目代码结构

主要包路径：`src/eeg_recovery`

核心模块：

- `src/eeg_recovery/config.py`：路径和全局配置。
- `src/eeg_recovery/metadata/labels.py`：标签构建。
- `src/eeg_recovery/metadata/subjects.py`：受试者元数据。
- `src/eeg_recovery/io/eeglab.py`：EEGLAB `.set/.fdt` 读取。
- `src/eeg_recovery/io/index.py`：数据文件索引和 EO/EC 识别。
- `src/eeg_recovery/channels/mapping.py`：通道读取和映射。
- `src/eeg_recovery/channels/hemisphere_flip.py`：患侧镜像翻转。
- `src/eeg_recovery/features/psd.py`：PSD 计算。
- `src/eeg_recovery/features/connectivity.py`：FC 计算。
- `src/eeg_recovery/features/feature_tables.py`：监督特征表构建。
- `src/eeg_recovery/models/encoders.py`：PSD/FC encoder。
- `src/eeg_recovery/models/dual_state_model.py`：EO/EC 双状态模型。
- `src/eeg_recovery/models/fusion_3d_model.py`：融合模型。
- `src/eeg_recovery/models/multimodal_model.py`：多模态 PSD/FC 模型。
- `src/eeg_recovery/models/ssl_model.py`：SSL 模型部件。
- `src/eeg_recovery/training/loso.py`：LOSO 训练和评估。
- `src/eeg_recovery/training/train_baselines.py`：传统 ML baseline。
- `src/eeg_recovery/training/train_supervised.py`：监督 CNN。
- `src/eeg_recovery/training/train_ssl.py`：早期 SSL。
- `src/eeg_recovery/training/train_feature_ssl.py`：feature-space SSL，包括 NT-Xent、VICReg、Barlow Twins、BYOL。
- `src/eeg_recovery/training/train_structured_ssl.py`：时频图和图结构 SSL。
- `src/eeg_recovery/training/train_masked_ssl.py`：masked modeling SSL。
- `src/eeg_recovery/training/ensemble_calibration.py`：ensemble 和阈值校准。
- `src/eeg_recovery/training/frozen_ssl_heads.py`：冻结 encoder 后接下游 head。

主要脚本：

- `scripts/01_prepare_metadata.py`
- `scripts/02_compute_psd.py`
- `scripts/03_compute_fc.py`
- `scripts/04_train_ml_baselines.py`
- `scripts/05_train_supervised_loso.py`
- `scripts/06_train_ssl.py`
- `scripts/07_train_feature_ssl_transfer.py`
- `scripts/08_train_structured_ssl_transfer.py`
- `scripts/09_train_masked_ssl_transfer.py`
- `scripts/10_ensemble_ssl_predictions.py`
- `scripts/11_train_frozen_vicreg_heads.py`

测试：

- 测试目录：`tests`
- 最近完整测试命令：`python -B -m pytest tests -v -p no:cacheprovider`
- 最近结果：173 passed。

## 评估协议

监督训练使用 19 折 LOSO：

- 每折 1 名患者作为测试集。
- 其余 18 名患者用于训练和内部验证。
- 所有 fold-local 操作必须只在训练患者上拟合，包括标准化、特征选择、PCA、阈值校准等。
- 如果 SSL 使用包含患者的数据，则当前折的测试患者必须从 SSL 预训练数据中排除。

虽然 LOSO 固定了测试患者，但深度模型仍然会有不同 seed，因为：

- 神经网络权重随机初始化不同。
- mini-batch 顺序不同。
- dropout 或数据增强随机性不同。
- CUDA 算子可能存在随机性。
- 内部验证划分或早停选择可能受 seed 影响。

因此，多 seed 稳定性是本项目的重要评估项。

## 已完成的主要尝试与结果

### 1. 传统机器学习 baseline

结果文件：`results/metrics/ml_baseline_model_comparison.csv`

| 模型 | Accuracy | Balanced Acc | ROC-AUC | PR-AUC | 备注 |
|---|---:|---:|---:|---:|---|
| Logistic L1 | 0.8421 | 0.8389 | 0.8444 | 0.8771 | 当前最强简单 baseline 之一 |
| Logistic L2 | 0.8421 | 0.8389 | 0.8333 | 0.8527 | 与 L1 同准确率 |
| SVM RBF | 0.7895 | 0.7889 | 0.8444 | 0.9040 | PR-AUC 高 |
| SVM Linear | 0.7368 | 0.7389 | 0.7333 | 0.6785 | 中等 |
| Random Forest | 0.6842 | 0.6778 | 0.6056 | 0.6355 | 较弱 |
| Gaussian NB | 0.6842 | 0.6889 | 0.7889 | 0.7306 | 较弱 |
| KNN | 0.6316 | 0.6167 | 0.7444 | 0.7006 | 较弱 |
| XGBoost | skipped | skipped | skipped | skipped | 未安装 |
| LightGBM | skipped | skipped | skipped | skipped | 未安装 |

结论：在 19 例小样本上，稀疏或正则化线性模型非常强，是深度学习模型必须超越或至少对齐的硬 baseline。XGBoost、LightGBM 不是因为失败，而是当前环境未安装对应包。

### 2. 初始 CNN 与 concat 融合

代表结果文件：

- `results/metrics/dl_model_comparison_all_concat_cnn.csv`
- `results/metrics/dl_model_comparison_multimodal_psd_fc_wpli_concat_cnn.csv`

早期 concat CNN 表现不佳，例如：

- PSD-only CNN：约 0.5263。
- FC-wPLI-only CNN：约 0.5263。
- FC-iCOH-only CNN：约 0.4737。
- PSD + FC-wPLI concat CNN：约 0.4737。

结论：简单 concat 融合不能充分利用 EO/EC 和 PSD/FC 信息，训练损失可以下降，但 LOSO 泛化很差。这推动了后续 gated fusion 结构和多 seed 稳定性评估。

### 3. 当前主要无 SSL 神经网络：PSD + wPLI gated CNN

模型说明文件：`docs/best_psd_fc_wpli_gated_cnn_model.md`

输入：

- PSD EO：`(batch, 62, 90)`
- PSD EC：`(batch, 62, 90)`
- wPLI EO：`(batch, 1891, 6)`
- wPLI EC：`(batch, 1891, 6)`

结构概念：

- PSD branch：Conv2D encoder。
- wPLI branch：Conv1D edge-band encoder。
- EO/EC 融合：每个 branch 内使用 gated fusion，而不是简单 concat。
- 最终融合：PSD embedding 与 wPLI embedding 拼接后进入 MLP classifier。
- 参数量：约 10,293。

结果：

- 单 seed 最好：seed 2，accuracy 0.8421，balanced accuracy 0.8333，ROC-AUC 0.8222，PR-AUC 0.8158。
- 10 seed gated ensemble：accuracy 0.7895。
- 原始 no-SSL 6 seed 稳定性：mean accuracy 0.6930，std 0.1173，min 0.5263，max 0.8421。
- no-SSL 稳定性方案 A：`lr=0.002, weight_decay=1e-5 或 1e-4, dropout=0`，6 seed mean accuracy 0.7193，std 0.1035，min 0.5789，max 0.8421。

结论：该 CNN 可以在某些 seed 达到 0.8421，但平均表现和最差 seed 仍不够稳定。

### 4. Feature-space NT-Xent SSL

说明文件：`docs/feature_ssl_psd_fc_wpli_transfer_results.md`

实现位置：

- `src/eeg_recovery/training/train_feature_ssl.py`
- `scripts/07_train_feature_ssl_transfer.py`

SSL 数据作用域曾比较四类：

| 数据作用域 | 每折可用 pair 数 | 含义 |
|---|---:|---|
| supervised-baseline | 18 | 当前 LOSO 训练折中的 18 名监督患者基线 EO/EC |
| all-patient-baseline | 27 | 所有患者基线，但排除当前测试患者 |
| all-patient | 88 | 所有患者所有可用 EEG pair，排除当前测试患者 |
| all-patient-health | 101 | 所有患者加健康人，排除当前测试患者 |

代表结果：

- all-patient + seed 2：accuracy 0.8421，balanced accuracy 0.8333。
- 多 seed 复查：seeds 0,1,2,3,7,13，mean accuracy 0.7368，std 0.0666，min 0.6842，max 0.8421。
- 3 seed ensemble：accuracy 0.6842。

结论：NT-Xent feature-space SSL 能在某个 seed 达到较好准确率，并且比原始 no-SSL 稍稳定，但没有稳定超过 logistic baseline，也没有稳定超过 no-SSL 最好 seed。

重要问题：NT-Xent 是对比学习，需要 batch 内其他样本作为负样本。用户指出当前数据中其实没有真正意义上的负样本，因此后续又尝试了不需要负样本的方法。

### 5. 时频图 SSL 和图结构 SSL

说明文件：`docs/structured_ssl_transfer_results.md`

方法：

- PSD branch 改成时频图视角。
- wPLI branch 使用图结构 view/dropout。
- 尝试对齐 TFR 与 graph 表征。

结果：

| 方法 | Accuracy | Balanced Acc | ROC-AUC | 备注 |
|---|---:|---:|---:|---|
| TFR + graph 5/5 epochs | 0.6316 | 0.6222 | 0.6444 | 较弱 |
| TFR + graph 10/10 epochs | 0.7368 | 0.7278 | 0.7556 | 有改善 |
| aligned TFR + graph 10/10 epochs | 0.6316 | 0.6278 | 0.7778 | AUC 尚可但 accuracy 低 |

结论：预训练 loss 可以下降，但下游 transfer 不稳定，没有明显优于当前 feature-space 或 no-SSL gated CNN。

### 6. Masked Modeling SSL

说明文件：`docs/masked_ssl_psd_fc_wpli_transfer_results.md`

方法演化：

1. PSD masked reconstruction + wPLI graph/edge masked reconstruction，从全局 pooled embedding 重建完整 PSD/FC。
2. 改成 pre-pooling feature-map 局部重建，即只要求局部 feature map 重建，避免用全局 embedding 重建完整输入造成任务过重。
3. 降低 wPLI mask 强度，并尝试 EO/EC consistency loss。

代表结果：

| 方法 | Accuracy | Balanced Acc | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| global reconstruction | 0.6316 | 0.6167 | 0.6333 | 0.6327 |
| local pre-pooling reconstruction | 0.7368 | 0.7278 | 0.7333 | 0.7235 |
| lower wPLI mask | 0.7895 | 0.7833 | 0.8111 | 0.7957 |
| EO/EC consistency 0.05 | 0.6842 | 0.6778 | 0.7111 | 0.6655 |

结论：local masked SSL 明显优于 global reconstruction，降低 wPLI mask 强度有帮助，但 EO/EC consistency 在当前设置下反而伤害性能。

### 7. Multi-task SSL：local reconstruction + feature alignment

说明文件：`docs/multitask_ssl_psd_fc_wpli_transfer_results.md`

方法：

- 保留 local masked reconstruction。
- 同时加入 feature-space alignment，使 embedding 更贴近下游分类需要。

NT-Xent 结果：

| Contrastive weight | Accuracy | Balanced Acc | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|
| 0.25 | 0.5789 | 0.5667 | 0.5889 | 0.5877 |
| 0.05 | 0.7368 | 0.7278 | 0.7889 | 0.7751 |
| 0.01 | 0.8421 | 0.8333 | 0.7667 | 0.6828 |

多 seed NT-Xent：

- seeds 0,1,2,3,7,13。
- mean accuracy 0.7281。
- std 0.0843。
- min 0.6316。
- max 0.8421。

结论：非常轻的 alignment loss 更合适。较大的 contrastive weight 会压过局部重建任务，导致下游准确率下降。NT-Xent 在 seed 2 可达 0.8421，但平均值不够强。

### 8. Multi-task VICReg SSL

方法：

- 不需要负样本。
- local masked reconstruction + VICReg feature-space objective。
- 代表设置：`contrastive_weight=0.001, inv=25, var=25, cov=1`。

结果：

- seed 2：accuracy 0.7895，balanced accuracy 0.7833，ROC-AUC 0.8333，PR-AUC 0.8802。
- 6 seed 稳定性，seeds 0,1,2,3,7,13：
  - mean accuracy 0.7544。
  - std 0.0430。
  - min 0.6842。
  - max 0.7895。

结论：VICReg 没有达到单 seed 0.8421，但它是当时最稳定的 SSL 方案之一。相对原始 no-SSL 的 mean 0.6930 和 NT-Xent 的 mean 0.7281，VICReg 提高了平均准确率并降低了 seed 方差。

### 9. 不需要负样本的 feature-space SSL：VICReg、BYOL、Barlow Twins

说明文件：`docs/negative_free_feature_ssl_results.md`

实现位置：

- `src/eeg_recovery/training/train_feature_ssl.py`
- `scripts/07_train_feature_ssl_transfer.py`

VICReg feature-space SSL：

- 6 seed mean accuracy 0.7193。
- std 0.0248。
- max 0.7368。
- ensemble accuracy 0.7895。

BYOL feature-space SSL：

- seeds 2,7,13。
- mean accuracy 0.7018。
- max 0.7368。
- ensemble accuracy 0.7895。

Barlow Twins default：

- 10 seed mean accuracy 0.7263。
- std 0.0647。
- min 0.6316。
- max 0.8421。
- ensemble accuracy 0.7895。

Tuned Barlow Twins：

- 设置：`projection_dim=32, feature_mask_prob=0.01`，监督训练设置回到原始配置。
- 10 seed mean accuracy 0.7474。
- std 0.0544。
- min 0.6842。
- max 0.8421。
- ensemble accuracy 0.7895。
- 结果文件：`results/metrics/feature_barlow_proj32_mask001_10seed_combined_summary.csv`

结论：Tuned Barlow Twins 是目前不需要负样本的 feature-space SSL 中较好的选择。它提升了平均值和最低 seed 表现，但没有稳定超过 0.8，也没有超过 logistic baseline 的 0.8421。

错误患者分析：

- tuned Barlow 10 seeds 中，`sub09` 和 `sub14` 每个 seed 都预测错误。
- `sub05` 在 9/10 个 seed 中错误。
- `sub28` 在 8/10 个 seed 中错误。
- 这些患者应作为下一步临床和 EEG 特征解释的重点。

### 10. Frozen VICReg encoder + downstream head

说明文件：

- `docs/frozen_vicreg_cnn_head_results.md`
- `docs/neural_network_classifier_selection.md`

尝试：

- 冻结 VICReg encoder。
- 使用 pair-diff embedding。
- 下游 head 包括 neural head 和 RBF SVM。

结果：

- RBF SVM head 在某些设置中达到 1.0 accuracy，并且多 seed 也很高。
- 但用户后续要求分类器仍应使用神经网络。
- 神经网络 head 的结果没有稳定超过 0.8。

重要审查点：

RBF SVM 1.0 结果必须被当作探索性结果，而不能直接作为结论。它需要被 GPT Pro 重点检查是否存在泄漏、内层模型选择是否严格嵌套、pair-diff 构造是否引入标签信息、阈值或超参数是否由外层测试集间接选择。

### 11. Ensemble 与阈值校准

实现位置：

- `src/eeg_recovery/training/ensemble_calibration.py`
- `scripts/10_ensemble_ssl_predictions.py`

代表结果文件：

- `results/metrics/ensemble_metrics_ntxent_vicreg_seed2_ensemble.csv`

结果：

- NT-Xent seed 2 单模型：accuracy 0.8421。
- NT-Xent + VICReg seed 2 ensemble：accuracy 0.7895。
- OOF threshold calibration 没有超过最佳单模型。

结论：ensemble 和阈值校准在当前样本量下不一定提升 accuracy。它可能提升概率平滑，但不能解决关键患者的系统性错误。

## 当前最值得保留的结论

1. 简单 logistic baseline 已经很强，accuracy 0.8421，是所有深度学习方法必须面对的主要比较对象。
2. 当前最合理的神经网络主结构是 `PSD + wPLI gated CNN`，因为它贴合原设计，也显式处理 EO/EC 与 PSD/FC 融合。
3. no-SSL CNN 某些 seed 能达到 0.8421，但平均值和最差 seed 不稳定。
4. SSL 目前最有意义的提升不是最高 accuracy，而是 seed 稳定性：
   - multi-task VICReg 的 6 seed mean accuracy 0.7544，std 0.0430。
   - tuned Barlow Twins 的 10 seed mean accuracy 0.7474，std 0.0544，min 0.6842。
5. 负样本式 NT-Xent 在数据语义上有争议，因为项目中没有真正明确的负样本。
6. 不需要负样本的 SSL，包括 VICReg、BYOL、Barlow Twins，更符合当前数据场景，但仍需更强的设计才能稳定超过 no-SSL。
7. `sub09`、`sub14`、`sub05`、`sub28` 是目前最需要解释的错误样本。

## 当前主要风险

1. 样本量只有 19 例，任何 1 个患者预测改变都会使 accuracy 改变约 0.0526。
2. 多次调参都在同一 19 折 LOSO 上观察结果，容易产生选择偏差。
3. 深度模型 seed 方差大，单 seed 高准确率不能代表稳健性能。
4. 自监督学习数据量虽然比 19 例多，但仍然很小，且患者与健康人分布可能不同。
5. healthy EEG 没有监督标签，只能用于 SSL，不能直接证明对患者预测有帮助。
6. `.set/.fdt` 文件命名和内部 datfile 不一致需要持续防护。
7. 左右半球翻转如果在 PSD、FC、TFR 中有任一处不一致，会直接破坏生理解释。
8. RBF SVM 1.0 结果过高，必须优先审查是否存在泄漏或过拟合。
9. 当前没有外部验证集，所有结论都应表述为探索性。
10. 目前 EEG-only 是主线，临床特征融合尚未成为主结论。

## 建议 GPT Pro 重点检查的代码问题

1. `src/eeg_recovery/training/loso.py`：每一折 test subject 是否完全隔离。
2. `src/eeg_recovery/training/train_feature_ssl.py`：SSL pretraining 是否在每折排除测试患者。
3. `src/eeg_recovery/training/train_masked_ssl.py`：local reconstruction target 是否仅来自训练数据。
4. `src/eeg_recovery/training/frozen_ssl_heads.py`：pair-diff embedding、SVM head、inner model selection 是否严格嵌套。
5. `src/eeg_recovery/features/feature_tables.py`：特征表是否只包含基线 EEG 特征，没有混入标签或后测变量。
6. `src/eeg_recovery/metadata/labels.py`：label 计算、median threshold、天花板效应处理是否符合设计。
7. `src/eeg_recovery/io/index.py`：EO/EC 文件识别是否避免数字后缀误判。
8. `src/eeg_recovery/io/eeglab.py`：`.set` 内部旧 datfile 名与外部 `.fdt` 的 fallback 是否安全。
9. `src/eeg_recovery/channels/hemisphere_flip.py`：中线电极是否保持不变，左右通道是否一一对应。
10. `scripts/*.py`：脚本默认输出路径是否规范，是否会生成临时文件或污染项目目录。

## 建议下一步研究方向

### 1. 先做统计可靠性，而不是继续追某个 seed 的最高准确率

建议补充：

- permutation test：打乱标签后重复 LOSO，估计当前 accuracy 是否显著高于随机。
- bootstrap confidence interval：给 accuracy、balanced accuracy、ROC-AUC、PR-AUC 加置信区间。
- exact binomial test：对 19 个样本的正确数进行显著性分析。
- nested model selection：所有超参数选择在内层完成，外层 LOSO 只报一次最终结果。

### 2. 固定一个最终候选模型，避免继续使用测试集驱动调参

建议候选：

- 解释性强的 logistic L1/L2 baseline。
- neural 主线：`PSD + wPLI gated CNN`。
- SSL 主线：tuned Barlow Twins 或 multi-task VICReg。

之后应冻结模型配置，只做一次严格复现和统计检验。

### 3. 做错误患者分析

重点患者：

- `sub09`
- `sub14`
- `sub05`
- `sub28`

建议检查：

- FMA_pre、FMA_post、residual 是否接近阈值。
- 是否存在天花板效应。
- 患侧、病灶侧、病程、年龄、基线严重程度是否特殊。
- EEG 质量、缺失通道、PSD/FC 分布是否离群。

### 4. 做可解释性分析

建议输出：

- PSD 重要频段和通道，特别是 beta medium 18-21 Hz。
- wPLI 重要边，尤其是刺激侧 M1 相关网络。
- EO/EC gate 权重，判断模型更依赖 EO 还是 EC。
- 患侧对齐后的刺激侧和非刺激侧差异。

这一步比继续堆模型更重要，因为项目最终需要形成神经机制解释。

### 5. 加入临床特征融合，但要严格控制泄漏

可尝试输入：

- 年龄。
- 性别。
- 病程。
- 患侧。
- FMA_pre。
- 基线严重程度。

禁止输入：

- FMA_post。
- improvement。
- residual。
- 任何治疗后变量。

建议模型：

- EEG embedding + clinical MLP gated fusion。
- logistic baseline 中加入临床变量，与 EEG-only 做对比。

### 6. 对 FC 使用更合理的图结构建模

当前 FC 的 CNN 输入是 edge sequence，虽然可运行，但边序列的局部卷积未必具有明确图意义。

建议尝试：

- ROI-level FC，先把 62 通道聚合成更少脑区。
- graph Laplacian eigen features。
- 简单 GNN 或 graph attention，但参数量必须非常小。
- 只保留理论相关边，例如刺激侧 M1 与运动网络相关边。

### 7. 控制模型复杂度

19 例监督样本下，深度模型容易过拟合。建议优先考虑：

- 少参数 CNN。
- 冻结或半冻结 encoder。
- 强正则化 logistic 或 elastic net。
- 稳定特征选择。

### 8. 外部验证或扩展数据

如果后续能增加数据，优先级最高的是：

- 增加监督患者数量。
- 保留一批完全未参与调参的 hold-out 患者。
- 用独立中心或新批次数据验证当前 pipeline。

## 如何复现主要结果

常用入口脚本：

```powershell
python -B scripts\01_prepare_metadata.py
python -B scripts\02_compute_psd.py
python -B scripts\03_compute_fc.py
python -B scripts\04_train_ml_baselines.py
python -B scripts\05_train_supervised_loso.py
python -B scripts\07_train_feature_ssl_transfer.py
python -B scripts\09_train_masked_ssl_transfer.py
python -B scripts\10_ensemble_ssl_predictions.py
python -B scripts\11_train_frozen_vicreg_heads.py
```

测试命令：

```powershell
python -B -m pytest tests -v -p no:cacheprovider
```

如果需要调用 MATLAB，用户已说明需要使用沙箱外方式：

```powershell
F:\Matlab2020a\bin\matlab.exe -wait -batch ...
```

## 项目文件整洁要求

用户明确要求项目文件夹保持整洁，不保留生成的临时文件。当前建议：

- Python 运行使用 `python -B`，避免生成 `__pycache__`。
- pytest 使用 `-p no:cacheprovider`，避免 `.pytest_cache`。
- 不在项目根目录散落临时 CSV、图片或调试脚本。
- 新文档放在 `docs/`。
- 结果统一放在 `results/metrics`、`results/predictions`、`results/training_logs`、`results/ssl`、`results/figures`。

## 总体判断

当前项目已经形成了完整的 EEG 预测流水线：元数据和标签构建、EEGLAB 文件读取、患侧对齐、PSD/FC 特征计算、传统 ML、监督 CNN、多种 SSL、ensemble、错误患者分析、测试覆盖和文档记录。

目前最强的可报告结果仍应谨慎表述：

- 简单 logistic baseline：accuracy 0.8421。
- PSD + wPLI gated CNN：单 seed 最好 accuracy 0.8421，但 seed 稳定性不足。
- SSL 的主要贡献暂时体现在稳定性改善，而不是稳定提高最高 accuracy。
- tuned Barlow Twins 和 multi-task VICReg 是当前更符合“无真实负样本”场景的 SSL 方向。
- RBF SVM 1.0 结果需要审查，不能作为主要结论。

我建议 GPT Pro 把本项目看作一个已经完成工程闭环、但仍需严格统计验证和泄漏审计的小样本神经预测研究。下一步不宜只继续增加复杂模型，而应优先做严格统计检验、错误患者解释、模型配置冻结和可解释性分析。
