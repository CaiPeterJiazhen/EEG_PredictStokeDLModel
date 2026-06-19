# 3.5 可解释性分析结果摘要草稿

## 1. 可解释性分析目的
本分析用于描述最终 highrank + SWA Residual-aware patient-level Barlow SSL-CNN 在分类 logit 上依赖的 PSD、WPLI、状态、频段和网络连接信息。

## 2. 全局重要性排序
当前总表中排名第 1 的特征为 WPLI EC Beta High F8-CP1，mean_abs_attribution=0.1353（来源：`rerun_ablation_explainability_secondary_20260609/results/tables/table4_explainability_top_features_for_paper.csv`）。

## 3. PSD topomap
PSD topomap 使用 mean_signed_attribution，正值表示推动比例恢复预测，负值表示推动恢复不良预测。图件来源：`rerun_ablation_explainability_secondary_20260609/results/figures/revised_initial/figure6a_psd_topomap_bands.*`。

## 4. WPLI connectivity
WPLI connectivity 图中线条粗细表示 mean_abs_attribution，颜色表示 mean_signed_attribution 的方向。图件来源：`rerun_ablation_explainability_secondary_20260609/results/figures/revised_initial/figure6b_wpli_connectivity_bands.*`。

## 5. Top 特征统计验证
FDR 后 raw feature group p < 0.05 的条目数为 0（来源：`rerun_ablation_explainability_secondary_20260609/results/tables/table4_explainability_top_features_for_paper.csv` 及 top_*_feature_group_statistics.csv）。

## 6. 谨慎结论
所有解释性结果均为模型依赖、关联性、hypothesis-generating 证据，不应写成因果机制或已验证临床 biomarker。