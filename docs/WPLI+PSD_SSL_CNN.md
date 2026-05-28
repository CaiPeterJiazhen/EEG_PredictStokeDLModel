你是一个熟悉 PyTorch、EEG、自监督学习、LOSO-CV 防泄漏、小样本医学预测和论文级实验设计的工程研究助手。

当前项目：
EEG_PredictStokeDLModel

当前最强结果：
1. PSD Segment Barlow -> patient-level PSD+FC-wPLI gated CNN：
   - 6-seed mean accuracy 约 0.7982
   - max accuracy 约 0.8947
   - supervised model 仍然输入 PSD + WPLI

2. FC/wPLI Segment Barlow -> patient-level PSD+FC-wPLI gated CNN：
   - 6-seed mean accuracy 约 0.7982
   - accuracy std 约 0.0562
   - ROC AUC mean 约 0.8185
   - PR AUC mean 约 0.8171
   - supervised model 仍然输入 PSD + WPLI

3. WPLI Segment Barlow + attention MIL seed0 明显失败，不作为主线继续扩展。

本任务目标：
实现并运行一个更稳妥的模型选择与融合流程：

A. 确认并保存/复用 PSD Segment Barlow encoder checkpoint
B. 确认并保存/复用 FC/wPLI Segment Barlow encoder checkpoint
C. 汇总 PSD Segment Barlow SSL-CNN 和 FC/wPLI Segment Barlow SSL-CNN 的 6-seed 结果
D. 做 probability-level ensemble
E. 做严格 OOF threshold calibration
F. 输出论文可用的模型选择结果与文档

重要：
不要继续扩展 attention MIL。
不要把 Dual encoder 作为主线。
Dual encoder 只保留为 supplementary ablation。

============================================================
一、必须保证 encoder 权重被保存并可复用
============================================================

现在所有 Segment SSL 训练都必须支持 checkpoint cache，避免以后重复预训练。

请检查并完善：

src/eeg_recovery/training/ssl_checkpointing.py
src/eeg_recovery/training/train_segment_ssl.py
scripts/17_train_segment_ssl_transfer.py

如果 checkpoint 逻辑已经存在，请增强并验证；如果不完整，请补齐。

必须支持以下参数：

--save-ssl-encoders
--reuse-ssl-encoders
--ssl-checkpoint-dir
--force-retrain-ssl
--reuse-only
--checkpoint-tag

默认 checkpoint 目录：

results/checkpoints/ssl_encoders/

该目录必须进入 .gitignore，不要提交 .pt/.pth/.ckpt 权重文件。

每个 fold 的 SSL encoder checkpoint 必须是 fold-specific，因为 SSL data_scope 是 all-patient，包含 patient EEG。外层 LOSO 中，当前 test subject 必须从 SSL pool 中排除。

checkpoint 文件命名建议：

segssl_barlow_psd_seed0_fold01_test_sub01_emb32_pre20_all-patient.pt
segssl_barlow_wpli_seed0_fold01_test_sub01_emb32_pre20_all-patient.pt

注意：
branch 名称统一使用：
- psd
- wpli

不要混用 fc-wpli 作为 checkpoint branch 名称。fc-wpli 是 feature kind，checkpoint branch 应是 wpli。

============================================================
二、checkpoint 内容要求
============================================================

每个 checkpoint 必须是 torch 保存的 dict，至少包含：

{
  "checkpoint_type": "segment_ssl_encoder",
  "metadata": {
      "checkpoint_type": "segment_ssl_encoder",
      "branch": "psd" or "wpli",
      "segment_ssl_method": "segment_barlow",
      "ssl_objective": "barlow",
      "base_seed": seed,
      "effective_seed": seed + fold_index 或当前实现中的有效 seed,
      "fold_index": fold_index,
      "test_subject_id": test_subject_id,
      "excluded_subject_id": test_subject_id,
      "ssl_data_scope": "all-patient",
      "historical_unlabeled_pretraining": true,
      "segment_feature_kind": "psd" or "fc-wpli",
      "supervised_feature_kind": "psd-fc-wpli",
      "encoder_kind": "cnn",
      "embedding_dim": 32,
      "dropout": 0.0,
      "projection_dim": 32,
      "pretrain_epochs": 20,
      "pretrain_lr": ...,
      "batch_size": ...,
      "feature_mask_prob": 0.03,
      "noise_std": 0.02,
      "lambda_latent": 1.0,
      "lambda_local": 0.1,
      "n_ssl_segments": ...,
      "source_feature_manifest_hash": "...",
      "created_at": "..."
  },
  "encoder_state_dict": <branch encoder state_dict>,
  "prefixed_state_dict": <optional state_dict with full model prefix>
}

不要保存 projection head。
不要保存 classifier。
不要保存 supervised fine-tuned model 作为 SSL encoder checkpoint。

如果已有旧 checkpoint 只保存了 merged_state_dict，请尝试提取：
- branch_models.psd.encoder.* -> psd encoder
- branch_models.wpli.encoder.* -> wpli encoder

提取后另存为 branch-specific checkpoint，并保留/重建 metadata。
如果 metadata 无法证明 strict LOSO，不允许复用，必须重新跑该 fold 的 SSL。

============================================================
三、checkpoint 复用规则
============================================================

每次运行 Segment SSL transfer 时：

1. 根据当前 fold 和参数生成 expected_metadata。
2. 如果 --reuse-ssl-encoders 且 checkpoint 存在：
   - 加载 checkpoint；
   - validate metadata；
   - 如果完全匹配，跳过 SSL pretraining；
   - 直接返回 encoder_state_dict。
3. 如果 checkpoint 不存在：
   - 如果 --reuse-only，raise FileNotFoundError；
   - 否则运行 SSL pretraining，并在结束后保存 checkpoint。
4. 如果 checkpoint 存在但 metadata 不匹配：
   - 如果 --force-retrain-ssl，重新训练并覆盖；
   - 否则 raise ValueError。
5. 不允许 silent fallback。
6. 不允许加载包含当前 test subject 的 global all-patient checkpoint 作为 strict LOSO 主结果。

必须验证 metadata 字段：
- branch
- ssl_objective
- segment_ssl_method
- fold_index
- test_subject_id
- excluded_subject_id
- base_seed
- ssl_data_scope
- segment_feature_kind
- supervised_feature_kind
- encoder_kind
- embedding_dim
- dropout
- pretrain_epochs
- projection_dim
- feature_mask_prob
- noise_std
- source_feature_manifest_hash

============================================================
四、保留/重跑单分支 SSL-CNN
============================================================

请确认以下两类模型结果完整：

B. PSD Segment Barlow SSL-CNN
- segment_feature_kind = psd
- supervised_feature_kind = psd-fc-wpli
- objective = barlow
- seeds = 0, 1, 2, 3, 7, 13
- transfer_mode = finetune
- supervised_lr = 0.002
- weight_decay = 1e-5
- dropout = 0
- embedding_dim = 32

C. FC/wPLI Segment Barlow SSL-CNN
- segment_feature_kind = fc-wpli
- supervised_feature_kind = psd-fc-wpli
- objective = barlow
- seeds = 0, 1, 2, 3, 7, 13
- transfer_mode = finetune
- supervised_lr = 0.002
- weight_decay = 1e-5
- dropout = 0
- embedding_dim = 32

如果 predictions / metrics 已经存在并完整，不要重跑 supervised。
如果 encoder checkpoints 缺失，请不要为了生成 checkpoint 而重跑已经完成的 supervised 结果；只在以后需要复用 encoder 时再补 checkpoint。
如果某个 seed/fold 的 checkpoint 缺失但 predictions 已经存在，请在文档里标记：
"metrics available, encoder checkpoint unavailable for reuse"
并建议后续重跑 SSL-only cache。

============================================================
五、实现 ensemble 和 threshold calibration
============================================================

新增脚本：

scripts/22_ensemble_segment_barlow_predictions.py

功能：
读取已有 prediction CSV，生成 ensemble 与 threshold calibration 结果。

输入模型组：

A. no-SSL stable CNN，如结果可用
B. PSD Segment Barlow SSL-CNN
C. FC/wPLI Segment Barlow SSL-CNN
可选 D. Dual Segment Barlow finetune，只作为 ablation，不作为主线

必须实现以下 ensemble：

1. PSD + WPLI equal-weight ensemble
   score = 0.5 * score_psd + 0.5 * score_wpli

2. noSSL + WPLI equal-weight ensemble
   score = 0.5 * score_no_ssl + 0.5 * score_wpli

3. noSSL + PSD + WPLI equal-weight ensemble
   score = mean(score_no_ssl, score_psd, score_wpli)

4. Optional OOF-weighted ensemble
   score = α * score_wpli + (1 - α) * score_psd
   α 只能通过 inner/OOF 方式确定，不能直接用 19 个 LOSO test labels 调。
   如果当前没有严格 OOF 权重机制，请只输出 equal-weight ensemble，并把 weighted ensemble 标记为 exploratory disabled。

必须实现 threshold 策略：

1. fixed threshold = 0.5
2. OOF-calibrated threshold
   - 不能直接在最终 19 个 LOSO test predictions 上搜索最佳阈值。
   - 如果当前无法获得内层 OOF prediction，请实现 leave-one-seed-out 或 training-only threshold calibration。
   - 如果只能在最终 LOSO predictions 上调阈值，则必须标记为 exploratory，不可作为主结果。

输出指标：

- accuracy
- balanced_accuracy
- sensitivity
- specificity
- precision
- f1
- ROC AUC
- PR AUC
- Brier score
- tn/fp/fn/tp
- threshold
- threshold_method
- ensemble_members
- ensemble_weighting
- n_subjects
- n_seeds

输出文件：

results/predictions/ensemble_predictions_psd_wpli_segbarlow_equal_weight.csv
results/metrics/ensemble_metrics_psd_wpli_segbarlow_equal_weight.csv
results/metrics/segment_barlow_ensemble_threshold_summary.csv
results/metrics/segment_barlow_model_selection_summary.csv
results/metrics/segment_barlow_subject_error_frequency.csv

============================================================
六、per-subject error frequency
============================================================

请实现或复用函数：

compute_per_subject_error_frequency(prediction_files)

对以下模型分别输出 error frequency：

A. no-SSL stable CNN
B. PSD Segment Barlow SSL-CNN
C. FC/wPLI Segment Barlow SSL-CNN
E. PSD+WPLI ensemble
F. noSSL+PSD+WPLI ensemble

输出字段：

- subject_id
- y_true
- model_group
- n_runs
- n_errors
- error_rate
- mean_y_score
- std_y_score
- min_y_score
- max_y_score
- mean_y_pred
- repeatedly_wrong_flag
- borderline_label_flag，如果可以根据 residual distance 计算

重点额外输出 sub09 和 sub14：

results/metrics/sub09_sub14_model_scores_summary.csv

============================================================
七、统计比较
============================================================

新增脚本或扩展现有 metrics：

scripts/23_compare_segment_barlow_models.py

至少实现：

1. patient-level bootstrap 95% CI
   - 对 19 个 subject bootstrap
   - 输出 accuracy、balanced accuracy、ROC AUC、PR AUC、Brier score 的 CI

2. paired prediction comparison
   - noSSL vs WPLI Segment Barlow
   - noSSL vs PSD Segment Barlow
   - WPLI Segment Barlow vs PSD Segment Barlow
   - WPLI Segment Barlow vs PSD+WPLI ensemble
   - 使用 paired subject-level correctness 或 paired score difference
   - 小样本下可用 exact/sign test 或 bootstrap paired difference

3. permutation test
   - label permutation 或 paired model permutation
   - 至少用于 final selected model vs random-label baseline

输出：

results/metrics/segment_barlow_statistical_comparison.csv

注意：
不要把不同 seed 当作独立患者。seed 只能评估训练随机性，不是增加样本量。

============================================================
八、文档
============================================================

新增文档：

docs/segment_barlow_single_branch_ensemble_results.md

内容必须包括：

1. 为什么停止 attention MIL：
   - seed0 accuracy 0.5263
   - ROC AUC 0.4778
   - PR AUC 0.5202
   - 不作为主线继续扩展

2. 为什么不把 Dual encoder 作为主线：
   - seed0 Dual 没有提升 Acc/Bal Acc
   - AUC/PR 下降
   - 小样本下可能存在 branch conflict

3. 为什么采用单分支 SSL-CNN：
   - PSD Segment Barlow 和 FC/wPLI Segment Barlow 6-seed 表现都较好
   - 监督模型仍然输入 PSD + WPLI
   - 单分支预训练减少双 encoder 联合微调的不稳定性

4. 为什么 ensemble：
   - 检验 PSD SSL 与 WPLI SSL 是否互补
   - probability-level fusion 比 feature-level dual fusion 更稳
   - 小样本下减少 branch conflict

5. 结果表：
   - no-SSL stable protocol
   - PSD Segment Barlow
   - FC/wPLI Segment Barlow
   - PSD+WPLI ensemble
   - noSSL+PSD+WPLI ensemble，如有

6. threshold calibration：
   - fixed threshold 0.5
   - OOF-calibrated threshold
   - 明确说明没有在 test labels 上调阈值

7. checkpoint 保存：
   - encoder checkpoint 保存路径
   - metadata 校验字段
   - 如何复用
   - 哪些 checkpoint 不提交 git
   - 哪些 metrics/predictions 已经可复现

8. 论文建议：
   - final candidate model
   - supplementary ablations
   - negative pilots
   - 后续 explainability 应重点做 WPLI edge-band attribution + PSD channel-frequency attribution

同时更新：

docs/task_status.md

============================================================
九、测试
============================================================

新增测试：

tests/test_segment_barlow_ensemble.py
tests/test_threshold_calibration.py
tests/test_ssl_checkpoint_reuse.py

至少覆盖：

1. ensemble prediction 合并时 subject_id/y_true 必须一致；
2. prediction CSV 缺少 subject 或 y_score 时会报错；
3. equal-weight ensemble 输出正确；
4. threshold calibration 不允许使用 test labels 直接调主结果；
5. checkpoint metadata 不匹配时报错；
6. reuse-only 且 checkpoint 缺失时报错；
7. encoder checkpoint 保存后可重新加载；
8. .pt/.pth/.ckpt 不进入 git。

运行：

python -B -m pytest tests/test_segment_barlow_ensemble.py tests/test_threshold_calibration.py tests/test_ssl_checkpoint_reuse.py -v -p no:cacheprovider

如可行，再运行：

python -B -m pytest tests -v -p no:cacheprovider

结束前清理：

- no __pycache__
- no .pytest_cache
- no .pyc
- no root temp files
- no checkpoint files committed
- docs/ and metrics CSV 可以提交
- results/checkpoints/ 必须 gitignored

============================================================
十、建议命令
============================================================

先只做 ensemble / calibration，不重跑 SSL：

python -B scripts/22_ensemble_segment_barlow_predictions.py --config configs/paths.example.yaml --psd-results-dir docs/experiment_exports/20260527_local_psd_segssl --wpli-results-dir docs/experiment_exports/20260527_remote_fc_wpli_segssl --seeds 0 1 2 3 7 13 --ensemble psd_wpli no_ssl_wpli no_ssl_psd_wpli --threshold-method fixed oof --output-tag segment_barlow_model_selection

然后做统计比较：

python -B scripts/23_compare_segment_barlow_models.py --config configs/paths.example.yaml --model-selection-summary results/metrics/segment_barlow_model_selection_summary.csv --bootstrap 5000 --permutation 5000

如果需要补 encoder checkpoint：

python -B scripts/17_train_segment_ssl_transfer.py --config configs/paths.example.yaml --data-scope all-patient --device cuda --objective barlow --segment-feature-kind fc-wpli --supervised-feature-kind psd-fc-wpli --pretrain-epochs 20 --pretrain-batch-size 16 --embedding-dim 32 --projection-dim 32 --feature-mask-prob 0.03 --noise-std 0.02 --lambda-latent 1.0 --lambda-local 0.1 --supervised-epochs 0 --save-ssl-encoders --reuse-ssl-encoders --seeds 0 1 2 3 7 13

如果 supervised-epochs=0 当前脚本不支持，请实现 SSL-only checkpoint mode：
--ssl-only-cache

============================================================
十一、验收标准
============================================================

任务完成标准：

1. PSD Segment Barlow 和 FC/wPLI Segment Barlow 6-seed 结果被统一汇总；
2. PSD+WPLI equal-weight ensemble 完成；
3. noSSL+PSD+WPLI ensemble 如 noSSL predictions 可用则完成；
4. fixed threshold 和 OOF threshold 结果分开报告；
5. 没有使用最终 test labels 直接调主结果阈值；
6. per-subject error frequency 完成；
7. sub09/sub14 score summary 完成；
8. bootstrap CI 和 paired comparison 完成；
9. encoder checkpoint 保存/复用逻辑完成；
10. checkpoint metadata 不匹配会报错；
11. checkpoint 文件不提交 git；
12. 文档 `docs/segment_barlow_single_branch_ensemble_results.md` 完成；
13. `docs/task_status.md` 更新；
14. 所有相关测试通过。

重要：
不要为了让 ensemble 看起来更好而在 19 个 LOSO test labels 上直接调权重或阈值。所有主结果必须是 strict LOSO / OOF 合法结果。探索性结果可以保留，但必须明确标记 exploratory。