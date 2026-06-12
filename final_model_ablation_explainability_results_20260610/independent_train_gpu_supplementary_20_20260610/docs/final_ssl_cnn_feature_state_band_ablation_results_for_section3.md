# 补充 20 组最终 residual-aware SSL-CNN 消融结果

本目录为补充 20 组消融的真实独立重训结果。每个消融组均使用 final residual-aware patient-level Barlow SSL-CNN、highrank variant、patient-level LOSO、10 seeds、SWA，推断时读取 classification head。每个 LOSO fold 的测试患者在 scaler、SSL 预训练、残差目标标准化、监督微调和阈值前均被排除。

预测文件包含 20 组 x 10 seeds x 19 folds = 3800 行。恢复运行结束后，原始汇总文件只包含最后恢复段；当前文档和表格已从完整预测文件重新聚合，seed_metrics 为 200 行，seedmean_metrics 为 20 行。

补充 20 组的 delta 均以主文 9 组独立重训中的 `full_psd_wpli` 为参考：其 per-seed mean accuracy = 0.811，seedmean balanced accuracy = 0.833，seedmean ROC-AUC = 0.933。

补充组中，seedmean balanced accuracy 最高的是 `full_minus_alpha`，balanced accuracy = 0.889，ROC-AUC = 0.978，PR-AUC = 0.981，Brier = 0.083。seedmean ROC-AUC 最高的是 `full_minus_alpha`，ROC-AUC = 0.978。seedmean balanced accuracy 最低的是 `beta_medium_only`，balanced accuracy = 0.383。

完整表格见：`results/tables/supplementary_final_ssl_cnn_feature_state_band_ablation.csv`。
