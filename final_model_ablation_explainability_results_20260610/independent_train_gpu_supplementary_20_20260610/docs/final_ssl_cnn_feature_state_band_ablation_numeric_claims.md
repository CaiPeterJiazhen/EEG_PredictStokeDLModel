# 可写入论文正文的补充消融数字结论

所有数字均来自本目录下补充 20 组最终 residual-aware SSL-CNN 独立重训结果；delta 参考主文 9 组独立重训中的 `full_psd_wpli`。

- 补充 20 组均已完成 10 seeds x 19 LOSO folds，共 3800 条预测。
- 补充组中 seedmean balanced accuracy 最高的是 `full_minus_alpha`：balanced accuracy = 0.889，ROC-AUC = 0.978，PR-AUC = 0.981，Brier = 0.083。
- 补充组中 seedmean ROC-AUC 最高的是 `full_minus_alpha`：ROC-AUC = 0.978，balanced accuracy = 0.889。
- 补充组中 seedmean balanced accuracy 最低的是 `beta_medium_only`：balanced accuracy = 0.383，ROC-AUC = 0.467。
