# 3.3 特征、状态、频段消融实验结果摘要草稿

## 1. 为什么进行特征、状态、频段消融
本分析用于评估最终模型相关 EEG 信息在模态、状态、频段和运动网络连接层面的预测信息保留情况。所有 feature-subset 结果均为 patient-level LOSO，标准化和 SelectK 特征选择仅在训练折拟合。

## 2. 每类消融组的设置
设置包括 PSD/WPLI 模态消融、EO/EC 状态消融、单频段 only、leave-one-band-out，以及运动相关 WPLI 边保留或删除分析。

## 3. 模态消融结果
完整 PSD+WPLI 的 balanced accuracy 为 0.628，ROC-AUC 为 0.678；PSD only 为 0.733/0.856，WPLI only 为 0.628/0.689（来源：`rerun_ablation_explainability_secondary_20260609/results/tables/table3_feature_state_band_ablation_for_paper.csv`）。

## 4. 状态消融结果
EO only 的 balanced accuracy/ROC-AUC 为 0.317/0.333，EC only 为 0.633/0.700（来源：`rerun_ablation_explainability_secondary_20260609/results/tables/table3_feature_state_band_ablation_for_paper.csv`）。

## 5. 频段消融结果
按 balanced accuracy 排名前 5 的组合为：full_minus_alpha, beta_medium_only, psd_only, psd_gamma_only, full_minus_beta_low（来源：`rerun_ablation_explainability_secondary_20260609/results/tables/table3_feature_state_band_ablation_for_paper.csv`）。

## 6. 运动网络连接结果
motor WPLI edges only 在低维输入下的 balanced accuracy/ROC-AUC 为 0.683/0.756（来源：`rerun_ablation_explainability_secondary_20260609/results/tables/table3_feature_state_band_ablation_for_paper.csv`）。

## 7. final-model occlusion 与 feature-subset ablation 的一致性
最终模型 occlusion 摘要见 `rerun_ablation_explainability_secondary_20260609/results/explainability/final_model_occlusion_modality_state_band_summary.csv`。正的 probability drop 表示被遮蔽部分对比例恢复预测有正向贡献。

## 8. 谨慎结论
这些结果说明模型在当前小样本内部 LOSO 中更依赖哪些 EEG 信息，只能作为模型依赖和关联性证据，不能写成因果机制。