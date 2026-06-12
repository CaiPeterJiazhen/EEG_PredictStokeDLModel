# 最终 residual-aware SSL-CNN 特征、状态和频段消融结果

## 1. 为什么旧 Logistic 消融不能作为最终模型消融主结果

旧 `modality_state_band_ablation_rerun.csv` 是 tabular Logistic Regression feature-subset support analysis。它使用折内特征选择和线性分类器，不是 10-seed residual-aware SSL-CNN，也没有对每个消融组重新执行 Patient-level Barlow 预训练和残差感知微调。因此旧结果只能作为补充支持分析，不能替代主文 Results 3.3 的最终模型消融。

## 2. 本轮固定口径

本轮结果固定为 final residual-aware patient-level Barlow SSL-CNN，highrank variant，patient-level LOSO，10 seeds，SWA，推断时只读取 classification head。每个 LOSO fold 的测试患者在 scaler、SSL 预训练、残差目标标准化、监督微调和阈值前均被排除。

本轮为每个消融组独立执行 Barlow 预训练和 residual-aware supervised fine-tuning。

## 3. 输入构造

每个样本包含 PSD_EO、PSD_EC、WPLI_EO、WPLI_EC。被移除的输入部分在 fold-local standardized space 中置零，因此零表示该训练折标准化后的均值。PSD 30-45 Hz 统一作为 Gamma；WPLI 只包含 Delta、Theta、Alpha、Beta Low、Beta Medium、Beta High，不构造 WPLI Gamma。

## 4. full_psd_wpli 表现

`full_psd_wpli` 的 seed-mean balanced accuracy 为 0.833，ROC-AUC 为 0.933，PR-AUC 为 0.942，Brier 为 0.109。相对 full_psd_wpli 的 delta balanced accuracy 为 0.000，delta ROC-AUC 为 0.000。

## 5. psd_only 与 full_psd_wpli

`psd_only` 的 seed-mean balanced accuracy 为 0.578，ROC-AUC 为 0.722，PR-AUC 为 0.763，Brier 为 0.274。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.256，delta ROC-AUC 为 -0.211。

## 5. wpli_only 与 full_psd_wpli

`wpli_only` 的 seed-mean balanced accuracy 为 0.783，ROC-AUC 为 0.733，PR-AUC 为 0.671，Brier 为 0.201。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.050，delta ROC-AUC 为 -0.200。

## 6. EO only 与 full_psd_wpli

`eo_only` 的 seed-mean balanced accuracy 为 0.500，ROC-AUC 为 0.544，PR-AUC 为 0.585，Brier 为 0.525。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.333，delta ROC-AUC 为 -0.389。

## 6. EC only 与 full_psd_wpli

`ec_only` 的 seed-mean balanced accuracy 为 0.583，ROC-AUC 为 0.733，PR-AUC 为 0.789，Brier 为 0.241。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.250，delta ROC-AUC 为 -0.200。

## 7. beta_medium_beta_high

`beta_medium_beta_high` 的 seed-mean balanced accuracy 为 0.522，ROC-AUC 为 0.489，PR-AUC 为 0.555，Brier 为 0.392。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.311，delta ROC-AUC 为 -0.444。

## 7. full_minus_beta_medium_beta_high

`full_minus_beta_medium_beta_high` 的 seed-mean balanced accuracy 为 0.422，ROC-AUC 为 0.456，PR-AUC 为 0.610，Brier 为 0.355。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.411，delta ROC-AUC 为 -0.478。

## 8. motor_wpli_edges_only

`motor_wpli_edges_only` 的 seed-mean balanced accuracy 为 0.789，ROC-AUC 为 0.800，PR-AUC 为 0.804，Brier 为 0.184。相对 full_psd_wpli 的 delta balanced accuracy 为 -0.044，delta ROC-AUC 为 -0.133。

## 8. full_minus_motor_wpli_edges

`full_minus_motor_wpli_edges` 的 seed-mean balanced accuracy 为 0.833，ROC-AUC 为 0.978，PR-AUC 为 0.981，Brier 为 0.102。相对 full_psd_wpli 的 delta balanced accuracy 为 0.000，delta ROC-AUC 为 0.044。

## 9. 解释范围

本轮结果回答的是：在最终 residual-aware SSL-CNN 框架下，不同 EEG 输入子集保留多少预测信息。结果不直接等同于因果机制，也不证明某个频段或网络决定恢复结局。

## 10. 谨慎结论

这些消融结果应表述为 exploratory final-model evidence。建议使用“提示”“保留信息”“模型依赖”这类措辞，避免使用“证明”“决定”“因果机制”等表述。

来源表：`F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_main\results\tables\table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv`
种子均值指标：`F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_main\results\metrics\final_ssl_cnn_feature_state_band_ablation_seedmean_metrics.csv`
种子级指标：`F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_main\results\metrics\final_ssl_cnn_feature_state_band_ablation_seed_metrics.csv`
Figure 5c：`F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_main\results\figures\revised_initial\figure5c_final_ssl_cnn_feature_state_band_ablation_ranking.png`
Figure 5d：`F:\CJZProjectFile\EEG_PredictStokeDLModel\final_model_ablation_explainability_results_20260610\independent_train_gpu_main\results\figures\revised_initial\figure5d_final_ssl_cnn_information_efficiency.png`