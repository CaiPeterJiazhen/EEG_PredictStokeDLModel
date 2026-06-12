# 最终 residual-aware SSL-CNN 特征、状态和频段消融结果

## 1. 为什么旧 Logistic 消融不能作为最终模型消融主结果

旧 `modality_state_band_ablation_rerun.csv` 是 tabular Logistic Regression feature-subset support analysis。它使用折内特征选择和线性分类器，不是 10-seed residual-aware SSL-CNN，也没有对每个消融组重新执行 Patient-level Barlow 预训练和残差感知微调。因此旧结果只能作为补充支持分析，不能替代主文 Results 3.3 的最终模型消融。

## 2. 本轮固定口径

本轮结果固定为 final residual-aware patient-level Barlow SSL-CNN，highrank variant，patient-level LOSO，10 seeds，SWA，推断时只读取 classification head。

计算资源说明：当前输出标记为 `approximate_fixed_final_checkpoint_masked_inference`。它使用已保存的 full-input final SWA checkpoint，在各 checkpoint 的 fold-local scaler 后对输入子集置零并重新推断；没有对每个消融组重新进行 Barlow 预训练和监督微调。因此它是最终模型 masked-input retention/occlusion 近似分析，不能写成独立重训主消融。

## 3. 输入构造

每个样本包含 PSD_EO、PSD_EC、WPLI_EO、WPLI_EC。被移除的输入部分在 fold-local standardized space 中置零，因此零表示该训练折标准化后的均值。PSD 30-45 Hz 统一作为 Gamma；WPLI 只包含 Delta、Theta、Alpha、Beta Low、Beta Medium、Beta High，不构造 WPLI Gamma。

## 4. full_psd_wpli 表现

`full_psd_wpli` 的 seed-mean balanced accuracy 为 0.833，ROC-AUC 为 0.844，PR-AUC 为 0.836，Brier 为 0.126。相对 full_psd_wpli 的 delta balanced accuracy 为 0.000，delta ROC-AUC 为 0.000。

## 5. psd_only 与 full_psd_wpli

`psd_only` 的 seed-mean balanced accuracy 为 0.633，ROC-AUC 为 0.689，PR-AUC 为 0.728，Brier 为 0.218。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.200，delta ROC-AUC 为 -0.156。

## 5. wpli_only 与 full_psd_wpli

`wpli_only` 的 seed-mean balanced accuracy 为 0.667，ROC-AUC 为 0.711，PR-AUC 为 0.642，Brier 为 0.237。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.167，delta ROC-AUC 为 -0.133。

## 6. EO only 与 full_psd_wpli

`eo_only` 的 seed-mean balanced accuracy 为 0.633，ROC-AUC 为 0.667，PR-AUC 为 0.717，Brier 为 0.224。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.200，delta ROC-AUC 为 -0.178。

## 6. EC only 与 full_psd_wpli

`ec_only` 的 seed-mean balanced accuracy 为 0.722，ROC-AUC 为 0.744，PR-AUC 为 0.659，Brier 为 0.209。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.111，delta ROC-AUC 为 -0.100。

## 7. beta_medium_beta_high

`beta_medium_beta_high` 的 seed-mean balanced accuracy 为 0.628，ROC-AUC 为 0.567，PR-AUC 为 0.668，Brier 为 0.313。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.206，delta ROC-AUC 为 -0.278。

## 7. full_minus_beta_medium_beta_high

`full_minus_beta_medium_beta_high` 的 seed-mean balanced accuracy 为 0.517，ROC-AUC 为 0.622，PR-AUC 为 0.681，Brier 为 0.298。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.317，delta ROC-AUC 为 -0.222。

## 8. motor_wpli_edges_only

`motor_wpli_edges_only` 的 seed-mean balanced accuracy 为 0.456，ROC-AUC 为 0.511，PR-AUC 为 0.591，Brier 为 0.298。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.378，delta ROC-AUC 为 -0.333。

## 8. full_minus_motor_wpli_edges

`full_minus_motor_wpli_edges` 的 seed-mean balanced accuracy 为 0.733，ROC-AUC 为 0.856，PR-AUC 为 0.861，Brier 为 0.153。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.100，delta ROC-AUC 为 0.011。

## 9. 解释范围

本轮结果回答的是：在已训练最终 residual-aware SSL-CNN checkpoint 下，不同 EEG 输入子集在 masked inference 时保留多少预测信息。结果不直接等同于独立重训消融、因果机制，也不证明某个频段或网络决定恢复结局。

## 10. 谨慎结论

这些结果应表述为 exploratory approximate masked-inference evidence。建议使用“提示”“保留信息”“模型依赖”这类措辞，避免使用“证明”“决定”“因果机制”或“独立重训消融主结果”等表述。

来源表：`final_model_ablation_explainability_results_20260610\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`
种子均值指标：`final_model_ablation_explainability_results_20260610\results\metrics\final_ssl_cnn_feature_state_band_ablation_seedmean_metrics.csv`
种子级指标：`final_model_ablation_explainability_results_20260610\results\metrics\final_ssl_cnn_feature_state_band_ablation_seed_metrics.csv`
Figure 5c：`final_model_ablation_explainability_results_20260610\results\figures\revised_initial\figure5c_final_ssl_cnn_feature_state_band_ablation_ranking.png`
Figure 5d：`final_model_ablation_explainability_results_20260610\results\figures\revised_initial\figure5d_final_ssl_cnn_information_efficiency.png`