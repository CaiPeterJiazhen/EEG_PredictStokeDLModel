# Tabular Logistic Feature-Subset Support Analysis Note

旧 `modality_state_band_ablation_rerun.csv` 是 tabular Logistic Regression 消融。

它的 `full_psd_wpli` performance 反映的是高维表格 EEG 特征在线性模型和 fold-local SelectK 下的表现，不能与最终 residual-aware SSL-CNN 的 `full_psd_wpli` 直接等价比较。

如果保留在论文中，建议只放入 Supplementary Table，并命名为 tabular feature-subset support analysis。