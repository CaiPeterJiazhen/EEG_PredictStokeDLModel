# 可写入论文正文的数字结论

所有数字均来自本目录下重新生成的最终 residual-aware SSL-CNN 近似 masked-inference 结果，analysis_status = approximate_fixed_final_checkpoint_masked_inference，均为 exploratory，不能写成独立重训消融。

## full_psd_wpli

- balanced accuracy = 0.833，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.844，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.836，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.126，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = 0.000，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。

## psd_only

- balanced accuracy = 0.633，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.689，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.728，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.218，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = -0.156，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。

## wpli_only

- balanced accuracy = 0.667，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.711，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.642，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.237，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = -0.133，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。

## eo_only

- balanced accuracy = 0.633，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.667，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.717，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.224，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = -0.178，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。

## ec_only

- balanced accuracy = 0.722，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.744，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.659，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.209，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = -0.100，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。

## beta_medium_beta_high

- balanced accuracy = 0.628，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.567，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.668，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.313，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = -0.278，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。

## full_minus_beta_medium_beta_high

- balanced accuracy = 0.517，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.622，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.681，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.298，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = -0.222，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。

## motor_wpli_edges_only

- balanced accuracy = 0.456，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.511，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.591，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.298，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = -0.333，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。

## full_minus_motor_wpli_edges

- balanced accuracy = 0.733，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- ROC-AUC = 0.856，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c/Figure 5d，exploratory: yes。
- PR-AUC = 0.861，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- Brier = 0.153，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3/Figure 5c，exploratory: yes。
- delta ROC-AUC vs full = 0.011，来源 `final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`，口径：seedmean prediction，对应 Table 3，exploratory: yes。
