# 当前论文整体结构与已有成果记录

更新时间：2026-06-08  
项目目录：`F:/CJZProjectFile/EEG_PredictStokeDLModel`  
当前主稿参考文件：`docs/manuscript_residual_aware_ssl_cnn_nature_polished.md`

## 一、论文当前定位

当前论文主题为：基于基线静息态 EEG 的残差感知自监督 CNN，用于预测卒中后上肢比例恢复结局。

核心论点是：在小样本卒中 tACS 干预队列中，单纯二分类标签会丢失连续恢复残差信息；残差感知训练可以在保持二分类临床推断的同时，把比例恢复残差、残差距离和恢复排序信息作为辅助监督信号纳入模型训练。自监督学习部分用于利用监督标签不足但已有基线 EEG 的患者数据，扩大表征学习样本池；不过目前证据更支持“残差感知辅助监督”的稳定贡献，自监督预训练的独立增益仍需要更大样本验证。



## 二、当前论文主结构

### 1. Title

当前题目：

`Residual-aware EEG learning for post-stroke proportional recovery`

题目强调三个元素：EEG、残差感知学习、卒中后比例恢复。

### 2. Abstract

摘要当前按照如下逻辑组织：

1. 临床问题：卒中后上肢恢复预测对康复分层有意义。
2. 方法挑战：标注样本小、EEG 分段泄漏风险、比例恢复二分类标签丢失连续残差信息。
3. 方法贡献：基线 EO/EC 静息态 PSD + WPLI，patient-level Barlow SSL-CNN，残差感知辅助训练。
4. 主要结果：最终模型 accuracy 0.842、balanced accuracy 0.833、ROC-AUC 0.844、PR-AUC 0.836、Brier 0.126。
5. 结论边界：结果有探索性，需要更大样本和外部验证。

### 3. Introduction

Introduction 当前逻辑为：

1. 卒中后上肢恢复存在个体差异，比例恢复框架是重要临床基础。
2. EEG 是低成本、可重复采集、反映振荡和连接变化的候选预后生物标志物。
3. 既往 EEG/机器学习/深度学习研究提示潜力，但存在样本小、patient-level 验证不足、分段泄漏、二分类标签信息损失等问题。
4. 本研究提出 residual-aware SSL-CNN，利用未标注 EEG 做表征学习，再用二分类标签和残差相关辅助损失进行监督训练。
5. 本文报告 patient-level LOSO、10 seed、bootstrap、permutation、ablation 和 explainability 结果。

Introduction 不放图片，目标长度应控制在英文约 900 词左右。

### 4. Materials and Methods

当前 Methods 建议结构：

1. Study cohort  
   写 29 例临床记录、28 例 EEG-indexed、19 例监督标签队列；说明另外 9 例可用于自监督/描述性分析。伦理批准号和正式知情同意语句仍用待填。

2. Intervention and clinical assessment  
   已确认信息包括：tACS 设备为 Neuroscan 1x1 经颅电刺激器 DC-STIMULATOR PLUS；刺激靶点为患手对侧 M1；C3/C4 根据患手侧选择；20 Hz、1000 microampere、20 min/session、每日一次、14 次；阻抗低于 30 kOhm。FMA-UE 和 MBI 在治疗前及 14 次治疗后评估。评估者、盲法和正式记录时间仍需最终表述。

3. EEG acquisition and preprocessing  
   已确认信息包括：Compumedics Neuroscan SynAmps2，64 导 EEG，10-20 布局；预处理使用 EEGLAB；平均参考；0.5 Hz 高通、45 Hz 低通、50 Hz 工频去除；ICA 使用 EEGLAB；坏道和伪迹由 EEGLAB 人工手动剔除。当前分析管线从预处理后的 EEGLAB `.set/.fdt` 文件开始。

4. Outcome definition  
   需要写清公式：

   `Predicted Delta FMA = 0.7 x (66 - baseline FMA-UE)`

   `Observed Delta FMA = post-treatment FMA-UE - baseline FMA-UE`

   `Residual = predicted Delta FMA - observed Delta FMA`

   监督队列残差中位数阈值为 1.5。Residual <= 1.5 定义为比例恢复组，Residual > 1.5 定义为恢复不良组。该阈值应写成 cohort-specific modelling endpoint，而不是外部验证过的临床 cut-off。

5. Feature extraction  
   PSD：EO/EC 两状态，每状态 62 channels x 90 frequency bins，0.5-45 Hz。  
   WPLI：EO/EC 两状态，每状态 1891 edges x 6 bands；频段包括 delta、theta、alpha、beta low、beta medium、beta high。PSD 图中当前已加入 Gamma topomap；表格中仍需检查 `Other` 是否全部统一为 Gamma。

6. Traditional ML  
   单独小标题，描述 Logistic L1/L2、SVM、Random forest、Gaussian NB、KNN 等传统模型，强调 LOSO 和 fold-local feature selection。

7. Self-supervised learning  
   单独小标题，描述 patient-level Barlow Twins 预训练、two augmented views、泄漏控制、仅使用训练池患者 EEG。这里需要加入 Barlow Twins 损失公式。

8. CNN and residual-aware training  
   单独小标题，描述 PSD branch、WPLI branch、EO/EC gated fusion、64-d multimodal embedding、binary classification head、residual regression auxiliary head、pairwise ranking loss、soft-label residual-distance BCE。推断阶段只使用 binary classification head。

9. Cross-validation and statistics  
   Patient-level LOSO，CNN 模型重复 10 seeds：`0,1,2,3,4,5,7,13,21,42`。报告 accuracy、balanced accuracy、sensitivity、specificity、ROC-AUC、PR-AUC、Brier。使用 subject-level bootstrap、permutation test、paired bootstrap 和 McNemar test。

10. Explainability  
      Methods 部分只写用了什么方法，不写结果：SmoothGrad integrated gradients、branch/state occlusion、PSD topomap、WPLI connectivity、网络级归因汇总。

### 5. Results

当前 Results 建议结构：

1. Cohort and outcome definition  
   展示 Table 1，说明队列规模、标签比例、FMA/MBI 基线与治疗后变化。

2. Traditional EEG-ML provides limited discrimination  
   展示传统机器学习模型表格和对比结果。当前最强 EEG-only Logistic L1 no selector：accuracy 0.737、balanced accuracy 0.733、ROC-AUC 0.711、PR-AUC 0.775、Brier 0.208。

3. Final residual-aware SSL-CNN supports patient-level prediction  
   展示 ROC 图和混淆矩阵。主稿锁定最终模型结果：accuracy 0.842、balanced accuracy 0.833、sensitivity 1.000、specificity 0.667、ROC-AUC 0.844、PR-AUC 0.836、Brier 0.126。

4. Model comparison and conservative uncertainty  
   展示 Figure 4C。当前最新 Figure 4C 已改为四模型比较：Logistic L1、No-SSL CNN、Residual-aware CNN、Residual-aware SSL-CNN，并使用 10 seed 逐 seed 均值口径。

5. Ablation and robustness  
   展示 Figure 5 和 Table 3。这里必须保留 Barlow CNN / SSL-CNN without residual-aware heads 这一行，用来单独评估“只有 Barlow 自监督预训练、没有残差感知辅助监督”的贡献。重点说明残差感知辅助监督贡献最大，自监督预训练独立贡献尚不稳定。

6. Explainability and EEG biomarker localization  
   展示 PSD topomap 和 WPLI connectivity。解释性结果应写成 hypothesis-generating，不写成因果机制。

### 6. Discussion

Discussion 当前应围绕四点展开：

1. 残差感知训练是当前模型最稳定的方法学贡献。
2. EEG 模型在内部 LOSO 中有较好判别和校准表现，但样本小，paired superiority 不确定。
3. 临床变量在本队列中很强，不能宣称 EEG 已有稳定临床增量价值。
4. 可解释性结果与既往 beta 震荡、静息态连接、运动网络恢复文献有生理一致性，但仍是模型依赖的探索性证据。

## 三、现有表格成果

### Table 1：患者信息

文件：

- `results/tables/table1_patient_information_zh.csv`
- `results/tables/table1_patient_information_zh.md`
- `results/figures/revised_initial/table1_patient_information_zh.png`

当前列设计：

`指标 | 全部临床记录 | 比例恢复组 | 恢复不良组 | P值`

关键结果：

| 指标 | 全部临床记录 | 比例恢复组 | 恢复不良组 | P值 |
|---|---:|---:|---:|---:|
| 受试者数 | 29 | 10 | 9 |  |
| 女性 | 15 (51.7%) | 6 (60.0%) | 6 (66.7%) | 1 |
| 年龄 | 64.03 ± 9.03 | 64.00 ± 7.60 | 66.11 ± 5.30 | 0.49 |
| 病程，月 | 5.86 ± 22.47 (n=28) | 0.97 ± 0.52 | 1.27 ± 0.71 | 0.29 |
| FMA-UE，治疗前 | 38.88 ± 23.72 (n=25) | 59.60 ± 4.74 | 19.22 ± 16.95 | <0.001 |
| FMA-UE，14次治疗后 | 45.84 ± 23.03 (n=19) | 64.10 ± 1.20 | 25.56 ± 17.66 | <0.001 |
| 比例恢复残差 | 12.87 ± 16.23 (n=19) | -0.02 ± 0.87 | 27.19 ± 12.39 | <0.001 |
| MBI，治疗前 | 60.00 ± 18.42 (n=24) | 72.50 ± 13.59 | 43.75 ± 15.29 (n=8) | <0.001 |
| MBI，14次治疗后 | 79.12 ± 18.73 (n=17) | 92.22 ± 8.33 (n=9) | 64.38 ± 15.91 (n=8) | 0.001 |
| 基线静息态 EEG 可用 | 28 (96.6%) | 10 (100.0%) | 9 (100.0%) |  |

注意：Table 1 当前是中文三线表方向，后续如投稿英文期刊，需要翻译为英文表格并统一缩写说明。

### Table 2：主模型性能

文件：

- `results/tables/table2_main_model_performance.csv`
- `results/tables/model_performance_main_table.csv`

主稿锁定口径中，最终模型为 `residual_aware_SSL_CNN_seedmean10`：

| 模型 | Accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic L1 EEG baseline | 0.737 | 0.733 | 0.800 | 0.667 | 0.711 | 0.775 | 0.208 |
| Logistic L2 EEG baseline | 0.684 | 0.689 | 0.600 | 0.778 | 0.778 | 0.840 | 0.221 |
| SVM RBF EEG baseline | 0.684 | 0.694 | 0.500 | 0.889 | 0.778 | 0.771 | 0.219 |
| Residual-aware SSL-CNN | 0.842 | 0.833 | 1.000 | 0.667 | 0.844 | 0.836 | 0.126 |

Permutation tests:

| 指标 | Final model observed | Permutation P |
|---|---:|---:|
| Accuracy | 0.842 | 0.004 |
| Balanced accuracy | 0.833 | 0.003 |
| ROC-AUC | 0.844 | 0.005 |
| PR-AUC | 0.836 | 0.015 |

### Figure 4C 最新四模型比较口径

文件：

- `results/figures/revised_initial/figure4c_a_model_metric_histogram.png`
- `results/figures/revised_initial/figure4c_model_metric_source_data.csv`

当前最新图不是使用 seed ensemble，而是使用 `0,1,2,3,4,5,7,13,21,42` 的逐 seed 指标均值：

| 模型 | Mean accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic L1 | 0.737 | 0.733 | 0.800 | 0.667 | 0.711 | 0.775 | 0.208 |
| No-SSL CNN | 0.763 | 0.757 | 0.880 | 0.633 | 0.773 | 0.753 | 0.198 |
| Residual-aware CNN | 0.816 | 0.812 | 0.880 | 0.744 | 0.899 | 0.908 | 0.130 |
| Residual-aware SSL-CNN | 0.837 | 0.831 | 0.950 | 0.711 | 0.860 | 0.858 | 0.142 |

重要口径说明：主稿 Table 2/ROC/混淆矩阵使用 locked final seedmean/ensemble 推断结果；Figure 4C 最新版使用逐 seed 均值，目的是展示随机种子平均表现。写正文时必须明确区分这两个口径，避免把 0.842 和 0.837 混写。

Barlow CNN / SSL-CNN without residual-aware heads 没有出现在当前 Figure 4C，是因为该图最近按指定改成四组模型：Logistic L1、No-SSL CNN、Residual-aware CNN、Residual-aware SSL-CNN，用来突出从传统 ML 到 CNN、再到残差感知和最终组合模型的递进。Barlow CNN 并不是没有数据，而是应放在 Table 3 和消融结果段落中，作为“单独 SSL 贡献”的对照模型。

Barlow CNN / SSL-CNN without residual-aware heads 的 10 seed 逐 seed 均值为：

| 模型 | Mean accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Barlow CNN / SSL-CNN | 0.789 | 0.782 | 0.920 | 0.644 | 0.777 | 0.753 | 0.198 |

### Table 3：消融分析

文件：

- `results/tables/table3_ablation.csv`
- `results/tables/core_ablation_10seed_summary.md`

核心消融结果：

| 模型/消融 | Row type | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Brier | 解释 |
|---|---|---:|---:|---:|---:|---:|---|
| ML PSD+WPLI Logistic L1 | reported | 0.737 | 0.733 | 0.711 | 0.775 | 0.208 | 传统 EEG-ML 基线 |
| No-SSL CNN same architecture | mean | 0.763 | 0.757 | 0.773 | 0.753 | 0.198 | CNN 架构贡献 |
| Patient-level Barlow CNN, no residual heads | mean | 0.789 | 0.782 | 0.777 | 0.753 | 0.198 | 自监督但无残差辅助 |
| No-SSL residual-aware CNN | mean | 0.816 | 0.812 | 0.899 | 0.908 | 0.130 | 残差辅助贡献 |
| Patient-level Barlow SSL + residual-aware heads | mean | 0.837 | 0.831 | 0.860 | 0.858 | 0.142 | 最终组合模型的 seed 均值 |

解释：当前最强、最稳定的增益来自 residual-aware auxiliary heads。Barlow SSL 的独立增益不稳定，不能写成已证明有效。

### EEG-only ML 三线表

文件：

- `results/tables/eeg_only_ml_three_line_table.md`
- `results/tables/eeg_only_ml_three_line_table.csv`
- `output/doc/eeg_only_ml_three_line_table.docx`

该表只包含 EEG-only 传统机器学习，不包含 Clinical 和 DeepLearning，适合复制到 PPT 或作为补充表。

### Table 4：可解释性 biomarker 表

文件：

- `results/tables/table4_explainability_biomarkers.csv`
- `results/tables/table4_explainability_biomarkers.md`

当前可解释性结果包括：

1. PSD channel-frequency attribution。
2. WPLI edge-band attribution。
3. Spearman residual/distance association。
4. Label-group permutation/FDR 字段。
5. 网络分组字段，如 frontal、central、temporal。

需要注意：表中仍出现 `Other` 频段命名；此前已决定 30-45 Hz 应统一写作 Gamma。后续需要检查 Table 4、正文和图注是否全部完成 `Other -> Gamma` 的统一。

## 四、现有图件成果

### Figure 1：总体技术框架图

文件：

- `results/figures/revised_initial/figure1_overall_framework.png`
- `results/figures/revised_initial/figure1_overall_framework.svg`
- `results/figures/revised_initial/figure1_overall_framework.pdf`
- `results/figures/revised_initial/figure1_overall_framework.tiff`

用途：放在 Materials and Methods 或 Study design 部分，展示从基线 EEG 采集、预处理、特征提取、SSL 预训练、残差感知监督训练到最终评估的整体流程。

### Figure 2：SSL 框架图

文件：

- `results/figures/revised_initial/figure2_ssl_framework.png`
- `results/figures/revised_initial/figure2_ssl_framework.svg`
- `results/figures/revised_initial/figure2_ssl_framework.pdf`
- `results/figures/revised_initial/figure2_ssl_framework.tiff`

用途：放在 Methods 的 self-supervised learning 小标题下，强调两个 augmented views、shared EEG encoder、projection head、Barlow Twins loss 和 leakage control。

### Figure 3：CNN + residual-aware 架构图

文件：

- `results/figures/revised_initial/figure3_cnn_residual_aware.png`
- `results/figures/revised_initial/figure3_cnn_residual_aware.svg`
- `results/figures/revised_initial/figure3_cnn_residual_aware.pdf`
- `results/figures/revised_initial/figure3_cnn_residual_aware.tiff`

用途：放在 Methods 的 CNN/residual-aware 小标题下，展示 PSD/WPLI 双分支、EO/EC gated fusion、classification head、residual regression head、ranking 和 soft-label auxiliary losses。

### Figure 4：最终模型表现

当前拆分图件：

- ROC：`results/figures/revised_initial/figure4a_final_model_roc.png`
- Confusion matrix：`results/figures/revised_initial/figure4b_final_model_confusion_matrix.png`
- Model scorecard：`results/figures/revised_initial/figure4c_a_model_metric_histogram.png`
- Brier calibration：`results/figures/revised_initial/figure4c_b_brier_calibration.png`
- Loss curves：`results/figures/revised_initial/figure4d_final_model_loss_curves.png`

当前 Figure 4C-a 已按用户要求改成四模型比较：Logistic L1、No-SSL CNN、Residual-aware CNN、Residual-aware SSL-CNN。No-SSL CNN、Residual-aware CNN 和 Residual-aware SSL-CNN 均使用 `0,1,2,3,4,5,7,13,21,42` 的逐 seed 均值。

### Figure 5：稳定性和 EEG 特征消融

当前拆分图件：

- Seed stability：`results/figures/revised_initial/figure5a_seed_stability.png`
- Core model ablation bars：`results/figures/revised_initial/figure5b_core_model_ablation_bars.png`
- Feature/state/band ablation ranking：`results/figures/revised_initial/figure5c_feature_state_band_ablation_ranking.png`
- Information efficiency：`results/figures/revised_initial/figure5d_information_efficiency.png`

用途：Results 的 ablation and robustness 部分。当前图形已经处理过 b 图颜色、d 图文字遮挡等问题。

### Figure 6：可解释性结果

当前拆分图件：

- PSD topomap：`results/figures/revised_initial/figure6a_psd_topomap_bands.png`
- WPLI connectivity：`results/figures/revised_initial/figure6b_wpli_connectivity_bands.png`
- Composite：`results/figures/revised_initial/figure6_eeg_explainability.png`

PSD topomap 已用 MNE 重新绘制；右侧 colorbar 只显示 `Max / 0 / Min`，并将色阶改为 95% 分位数对称范围，以增强不同权重大小对比。当前图包含 EO/EC 两行和 Delta、Theta、Alpha、Beta Low、Beta Mid、Beta High、Gamma 七列。

## 五、当前主要结果解释

### 1. 监督队列和标签

监督标签队列为 19 例，比例恢复 10 例，恢复不良 9 例。标签来自比例恢复残差中位数阈值 1.5。

### 2. 传统 EEG-ML

传统 EEG-only 模型能达到中等判别，但整体不稳定。当前主要 EEG-ML baseline 为 Logistic L1 no selector：accuracy 0.737、balanced accuracy 0.733、ROC-AUC 0.711、PR-AUC 0.775、Brier 0.208。

### 3. CNN 与残差感知

No-SSL CNN 10 seed 均值：accuracy 0.763、balanced accuracy 0.757、ROC-AUC 0.773、PR-AUC 0.753、Brier 0.198。

Residual-aware CNN 10 seed 均值：accuracy 0.816、balanced accuracy 0.812、ROC-AUC 0.899、PR-AUC 0.908、Brier 0.130。

这说明在当前数据中，残差感知训练带来的提升比单纯 CNN 架构更明显。

### 4. 自监督学习

Barlow CNN without residual heads 10 seed 均值：accuracy 0.789、balanced accuracy 0.782、ROC-AUC 0.777、PR-AUC 0.753、Brier 0.198。

Residual-aware SSL-CNN 10 seed 均值：accuracy 0.837、balanced accuracy 0.831、ROC-AUC 0.860、PR-AUC 0.858、Brier 0.142。

主稿锁定最终模型 seedmean/ensemble 结果：accuracy 0.842、balanced accuracy 0.833、ROC-AUC 0.844、PR-AUC 0.836、Brier 0.126。

解释边界：可以说 SSL-CNN 提供了完整建模框架，可以利用本来不能进入监督训练的基线 EEG；但不能说 Barlow SSL 单独带来了稳定增益。

写 Results 时建议明确区分：

1. `Barlow CNN / SSL-CNN`：只有 Barlow 自监督预训练 + CNN 分类头，不含 residual-aware auxiliary heads。
2. `Residual-aware CNN`：没有 Barlow 自监督预训练，但含 residual-aware auxiliary heads。
3. `Residual-aware SSL-CNN`：Barlow 自监督预训练 + residual-aware auxiliary heads，是最终组合模型。

### 5. 临床变量

当前 exploratory clinical-only 模型表现很强，clinical-only logistic model 的 ROC-AUC 约 0.911、PR-AUC 约 0.899、Brier 约 0.105。因此正文必须保守写作：本研究不能证明 EEG 对临床变量有稳定增量价值。

### 6. 可解释性

当前解释性结果可写为：

1. PSD 归因显示 EO/EC 状态和频段依赖的头皮分布。
2. EO 低频前中央区和更高频的中央/颞区通道有较明显归因。
3. WPLI 归因强调 EC beta-band connectivity，涉及额区、中央区和颞区节点。
4. 这些结果与卒中恢复中 beta 活动、运动网络连接和静息态 EEG 预测价值的既往文献方向一致。
5. 所有解释性结果均为模型依赖、关联性和假设生成，不应写成已验证因果 biomarker。
