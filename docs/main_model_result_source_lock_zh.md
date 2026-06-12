# 主模型绘图结果来源锁定记录

更新日期：2026-06-12

## 当前主模型结果来源

后续论文中凡涉及最终主模型、`Residual_Barlow_CNN`、`Residual-aware SSL-CNN` 或残差感知 Barlow SSL-CNN 的汇总性能结果，统一以以下文件为准：

`F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_main\results\tables\final_Residual_ssl_cnn.csv`

该文件替代此前误用的：

`F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_main\results\tables\final_ssl_cnn.csv`

旧文件 `final_ssl_cnn.csv` 不再作为主模型汇总指标来源。

## Seed 口径

当前主模型结果为 10 个随机种子：

`0, 1, 2, 3, 4, 5, 7, 13, 21, 42`

## Seed-Level 汇总指标

主模型性能图、模型对比表、正文性能描述和摘要性结果，均使用 `final_Residual_ssl_cnn.csv` 中 10 个 seed 的均值和离散度。

当前读取到的 10 seed 均值为：

| 指标 | 均值 | 标准差 |
|---|---:|---:|
| Accuracy | 0.8474 | 0.0388 |
| Balanced accuracy | 0.8411 | 0.0391 |
| Sensitivity | 0.9600 | 0.0516 |
| Specificity | 0.7223 | 0.0586 |
| Precision | 0.7944 | 0.0358 |
| F1 | 0.8687 | 0.0338 |
| ROC AUC | 0.8867 | 0.0599 |
| PR AUC | 0.8910 | 0.0573 |
| Brier | 0.1324 | 0.0315 |

该新文件与旧 `final_ssl_cnn.csv` 的主要差异在于 ROC AUC、PR AUC 和 Brier；Accuracy、Balanced accuracy、Sensitivity、Specificity、Precision、F1 以及各 seed 的 TN/FP/FN/TP 计数一致。

## 患者级曲线和混淆矩阵

患者级 ROC 曲线、PR 曲线、校准曲线和混淆矩阵，统一使用以下逐患者 10-seed 预测文件：

`F:\CJZProjectFile\EEG_PredictStokeDLModel\results\predictions\final_Residual_ssl_cnn_10seed_patient_predictions.csv`

该文件包含 10 个 seed × 19 名受试者的预测结果。绘制患者级图时，先按 `subject_id` 聚合，`y_true` 取该患者标签，`y_score` 取 10 个 seed 的平均预测概率，再使用固定阈值 0.5 得到最终二分类预测。

当前基于该逐患者文件聚合得到：

- n = 19
- Accuracy = 0.8421
- Balanced accuracy = 0.8333
- ROC AUC = 0.8667
- PR AUC = 0.8613
- Brier = 0.1153
- Confusion matrix: TP = 10, FN = 0, FP = 3, TN = 6

## 后续执行规则

1. 任何模型对比图中最终模型的柱状指标，统一读取 `final_Residual_ssl_cnn.csv`。
2. 任何表格中最终模型的 seed-level 均值±标准差，统一读取 `final_Residual_ssl_cnn.csv`。
3. 任何患者级 ROC、PR、校准和混淆矩阵，统一读取 `final_Residual_ssl_cnn_10seed_patient_predictions.csv` 并按患者聚合 10 个 seed。
4. 旧的 `paper_locked_model_predictions.csv`、`residual_aware_SSL_CNN_seedmean10`、`final_ssl_cnn.csv` 和 `final_ssl_cnn_feature_state_band_ablation_predictions.csv` 不再作为最终主模型结果来源。
5. 除非特别说明为“旧版锁定结果”或“历史对照”，后续 `Residual_Barlow_CNN` 均指本文件锁定的新主模型结果。
