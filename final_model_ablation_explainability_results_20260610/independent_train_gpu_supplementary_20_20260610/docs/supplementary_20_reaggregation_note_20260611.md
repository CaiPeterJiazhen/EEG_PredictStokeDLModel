# 2026-06-11 补充 20 组汇总重建说明

训练恢复命令使用 `--skip-existing-predictions` 后，预测 CSV 正确累计到 3800 行，但派生的 seed metrics 和表格只覆盖恢复段。当前已从完整预测 CSV 重建以下派生文件：

- `F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_supplementary_20_20260610\results\metrics\final_ssl_cnn_feature_state_band_ablation_seed_metrics.csv`：200 行，20 组 x 10 seeds。
- `F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_supplementary_20_20260610\results\metrics\final_ssl_cnn_feature_state_band_ablation_seedmean_metrics.csv`：20 行，20 组。
- `F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_supplementary_20_20260610\results\tables\supplementary_final_ssl_cnn_feature_state_band_ablation.csv`：20 行，20 组。
- `F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_supplementary_20_20260610\results\metrics\final_ssl_cnn_feature_state_band_ablation_delta_vs_full.csv`：20 行，delta 参考主文 9 组独立重训的 `full_psd_wpli`。

未改动原始预测 CSV。
