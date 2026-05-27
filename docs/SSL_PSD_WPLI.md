你是一个熟悉 PyTorch、EEG 自监督学习、小样本医学预测、LOSO-CV 防泄漏、医学论文实验复现的工程研究助手。请在当前仓库 `EEG_PredictStokeDLModel` 中完成一个工程任务：实现并运行 dual Segment Barlow encoder transfer，同时实现 Segment SSL encoder checkpoint 的保存与复用，避免每次监督训练前重复跑耗时的 SSL pretraining。

本任务不是简单跑实验，而是要补齐完整工程闭环：
1. 检查已有 A/B/C 结果是否完整；
2. 如果已有 A/B/C 可直接比较，则不要重跑；
3. 实现 SSL encoder checkpoint 保存/复用；
4. 实现 PSD Segment Barlow encoder 与 FC/wPLI Segment Barlow encoder 的双分支合并加载；
5. 只补跑 D/E/F 三个剩余实验；
6. 输出 metrics、predictions、loss history、summary、文档；
7. 严格保证 LOSO 防泄漏。

============================================================
一、当前项目背景
============================================================

主模型是：

`MultimodalEEGModel(feature_kind="psd-fc-wpli", fusion="gated", encoder_kind="cnn")`

它包含两个 branch：

1. `psd` branch
   - 输入：
     - `psd_eo`: 62 x 90
     - `psd_ec`: 62 x 90
   - encoder:
     - `SharedPSDEncoder`
     - 内部 CNN backend 是 `PSDConv2DEncoder`

2. `wpli` branch
   - 输入：
     - `wpli_eo`: 1891 x 6
     - `wpli_ec`: 1891 x 6
   - encoder:
     - `SharedFCEncoder`
     - 内部 CNN backend 是 `FCConv1DEncoder`

每个 branch 内部 EO/EC 共享同一个 encoder：

EO -> encoder -> EO embedding  
EC -> encoder -> EC embedding  

如果 fusion 是 `gated`，则：

pair = concat(EO embedding, EC embedding)  
state_weights = softmax(gate(pair))  
branch_embedding = weighted_sum(EO embedding, EC embedding)

最后：

combined_embedding = concat(psd_branch_embedding, wpli_branch_embedding)  
classifier(combined_embedding) -> binary probability

因此，当前结构天然支持分别加载两个不同的 SSL encoder：

- PSD branch 加载 PSD Segment Barlow encoder；
- wPLI branch 加载 FC/wPLI Segment Barlow encoder；
- gate 和 classifier 不从 SSL 加载，仍然随机初始化；
- 之后做 patient-level supervised LOSO fine-tuning。

============================================================
二、已有实验状态
============================================================

用户已经完成以下三个实验：

A. no-SSL CNN  
B. PSD Segment Barlow encoder only  
C. FC/wPLI Segment Barlow encoder only  

请你先检查仓库中的结果文件，确认 A/B/C 是否完整且可比较。

检查内容包括：

1. 是否使用同一版重新预处理后的 EEG 特征；
2. 是否使用相同 LOSO folds；
3. 是否使用相同 supervised seed set，例如：
   - 0
   - 1
   - 2
   - 3
   - 7
   - 13
4. 是否使用相同 supervised protocol：
   - architecture = multimodal
   - feature_kind = psd-fc-wpli
   - fusion = gated
   - encoder_kind = cnn
   - embedding_dim = 32
   - dropout = 0.0
   - epochs = 100
   - patience = 100
   - weight_decay = 1e-5
   - supervised_lr 与对应实验设置一致
5. 是否保存了：
   - predictions CSV
   - metrics CSV
   - loss history CSV
   - loss curve PNG
6. B/C 是否严格 fold-specific，且 SSL pretraining pool 排除了当前 LOSO test subject；
7. B/C 是否保存了对应 fold-specific encoder checkpoint。

如果 A/B/C 满足 1–6，则不要重跑 A/B/C，只把它们作为对照纳入最终 summary。

如果 A/B/C 缺少 encoder checkpoint，这不影响它们作为对照结果，但 D/E/F 无法直接复用 encoder 参数。此时请在 D/E/F 第一次运行时重新训练对应 Segment Barlow SSL encoder，并保存 checkpoint，以后复用。

如果 A/B/C 的 supervised protocol、seed set 或预处理版本与 D/E/F 不一致，请不要静默比较；请在文档中明确标记“不完全可比”，并建议后续统一重跑。

============================================================
三、本次只需要补跑的实验
============================================================

请只补跑以下三个实验：

D. dual Segment Barlow encoder + finetune  
E. dual Segment Barlow encoder + freeze-encoder  
F. dual Segment Barlow encoder + lower supervised LR  

定义如下。

------------------------------------------------------------
D. dual Segment Barlow encoder + finetune
------------------------------------------------------------

PSD branch:
  load PSD Segment Barlow encoder

wPLI branch:
  load FC/wPLI Segment Barlow encoder

gate:
  random initialization

classifier:
  random initialization

supervised training:
  all parameters trainable

suggested protocol:
  supervised_lr = 0.002
  weight_decay = 1e-5
  epochs = 100
  patience = 100
  dropout = 0.0
  embedding_dim = 32

------------------------------------------------------------
E. dual Segment Barlow encoder + freeze-encoder
------------------------------------------------------------

PSD branch:
  load PSD Segment Barlow encoder, then freeze encoder

wPLI branch:
  load FC/wPLI Segment Barlow encoder, then freeze encoder

gate:
  trainable

classifier:
  trainable

suggested protocol:
  supervised_lr = 0.002
  weight_decay = 1e-5
  epochs = 100
  patience = 100
  dropout = 0.0
  embedding_dim = 32

注意：
这里的 freeze-encoder 是机制验证，不一定追求最高 accuracy。它用于判断 SSL encoder 本身是否已学到可迁移表征。

------------------------------------------------------------
F. dual Segment Barlow encoder + lower supervised LR
------------------------------------------------------------

PSD branch:
  load PSD Segment Barlow encoder

wPLI branch:
  load FC/wPLI Segment Barlow encoder

gate:
  random initialization

classifier:
  random initialization

supervised training:
  all parameters trainable

suggested protocol:
  supervised_lr = 0.001
  weight_decay = 1e-5
  epochs = 100
  patience = 100
  dropout = 0.0
  embedding_dim = 32

目的：
判断较低 supervised learning rate 是否能更好保留 SSL encoder 表征，减少小样本 fine-tuning 对预训练表征的破坏。

============================================================
四、严格防泄漏规则
============================================================

这是医学预测论文级实验，必须严格防止 leakage。

1. 外层 LOSO 中，当前 test subject 不能出现在：
   - supervised training set
   - validation set
   - scaler fitting
   - SSL pretraining pool
   - checkpoint 训练来源
   - feature normalization statistics
   - augmentation statistics
   - manifest hash 中的训练数据集合

2. 如果 SSL pool 包含 patient EEG：
   每个 fold 必须有 fold-specific checkpoint。

示例：

fold 01, test subject sub01:
  PSD Segment Barlow encoder 必须排除 sub01
  wPLI Segment Barlow encoder 必须排除 sub01

fold 02, test subject sub02:
  PSD Segment Barlow encoder 必须排除 sub02
  wPLI Segment Barlow encoder 必须排除 sub02

3. PSD checkpoint 和 wPLI checkpoint 合并时，必须属于同一个：
   - fold_index
   - test_subject_id
   - excluded_subject_id
   - base_seed
   - embedding_dim
   - encoder_kind
   - ssl_data_scope
   - preprocessing/data manifest

4. 如果 metadata 不匹配，必须 raise error，不能警告后继续运行。

5. 如果 SSL pool 是 health-only 或 external-only，可以允许 global checkpoint；但为了避免混淆，本任务默认仍使用 fold-specific checkpoint 命名。

6. 不要使用包含当前 test patient 的全局 all-patient checkpoint 作为主结果。如果实现了 transductive setting，必须单独命名并在文档中标记，不能混入 main strict LOSO result。

============================================================
五、实现 Phase 1：新增 SSL checkpoint 工具
============================================================

新增文件：

`src/eeg_recovery/training/ssl_checkpointing.py`

需要实现以下函数：

1. `make_ssl_checkpoint_name(...)`

输入至少包括：
- method
- branch
- ssl_objective
- seed
- fold_index
- test_subject_id
- embedding_dim
- pretrain_epochs
- ssl_data_scope
- optional tag

返回 filename-safe 的 `.pt` 文件名。

命名示例：

`segssl_barlow_psd_seed0_fold01_test_sub01_emb32_pre100_all_patient.pt`

`segssl_barlow_wpli_seed0_fold01_test_sub01_emb32_pre100_all_patient.pt`

2. `save_ssl_encoder_checkpoint(path, metadata, encoder_state_dict, prefixed_state_dict=None)`

保存格式必须是 torch checkpoint，内容至少包括：

{
    "checkpoint_type": "segment_ssl_encoder",
    "metadata": metadata,
    "encoder_state_dict": encoder_state_dict,
    "prefixed_state_dict": prefixed_state_dict,
}

其中：

- `encoder_state_dict` 是 branch encoder 自身的 state_dict；
- `prefixed_state_dict` 是已经加好 full model prefix 的 state_dict，可选；
- 不要保存 projection head；
- 不要保存 classifier；
- 不要保存 gate，除非现有 SSL 确实训练了 gate 且另有明确实验目的。本任务默认不加载 gate。

3. `load_ssl_encoder_checkpoint(path, expected_metadata=None)`

功能：
- 用 `torch.load(..., map_location="cpu")` 读取；
- 检查 checkpoint_type；
- 如果提供 expected_metadata，则调用 `validate_ssl_checkpoint_metadata`；
- 返回 checkpoint dict。

4. `validate_ssl_checkpoint_metadata(observed, expected)`

必须严格校验以下字段：

- checkpoint_type
- branch
- ssl_objective
- segment_ssl_method
- base_seed
- fold_index
- test_subject_id
- excluded_subject_id
- ssl_data_scope
- feature_kind
- encoder_kind
- embedding_dim
- dropout
- pretrain_epochs
- pretrain_lr
- source_feature_manifest_hash

如果字段缺失或不匹配，raise ValueError。

允许提供参数：
- `allow_mismatched_ssl_seeds=False`

默认不允许 seed 不匹配。

5. `prefix_branch_encoder_state_dict(branch, encoder_state_dict)`

输入：
- branch = "psd" or "wpli"
- encoder_state_dict keys 例如：
  - `encoder.network.0.weight`
  - `encoder.network.0.bias`

输出 keys：

如果 branch = psd:
  `branch_models.psd.encoder.encoder.network.0.weight`

如果 branch = wpli:
  `branch_models.wpli.encoder.encoder.network.0.weight`

注意：
当前 `MultimodalEEGModel` 的 full state_dict 中，branch encoder key 前缀应是：

`branch_models.<branch>.encoder.`

而 `SharedPSDEncoder` 或 `SharedFCEncoder` 自己的 state_dict 通常从：

`encoder.network...`

开始。

因此 prefix 逻辑是：

`f"branch_models.{branch}.encoder.{key}"`

6. `merge_branch_pretrained_states(psd_checkpoint, wpli_checkpoint, allow_mismatched_ssl_seeds=False)`

输入可以是 path 或 checkpoint dict。

功能：
- 加载 PSD checkpoint；
- 加载 wPLI checkpoint；
- 验证 PSD metadata:
  - branch == "psd"
  - ssl_objective == "barlow"
- 验证 wPLI metadata:
  - branch == "wpli"
  - ssl_objective == "barlow"
- 验证二者共享：
  - fold_index
  - test_subject_id
  - excluded_subject_id
  - base_seed
  - encoder_kind
  - embedding_dim
  - dropout
  - ssl_data_scope
  - source_feature_manifest_hash
- 将 PSD encoder state 加前缀；
- 将 wPLI encoder state 加前缀；
- 合并为一个 dict[str, torch.Tensor]；
- 返回 merged_state_dict 和 merged_metadata。

不允许出现 key collision。

============================================================
六、实现 Phase 2：checkpoint 测试
============================================================

新增测试文件：

`tests/test_ssl_checkpointing.py`

至少包含以下测试：

1. `test_prefix_psd_encoder_state_dict_keys`
   - 构造 dummy encoder state；
   - branch="psd"；
   - 确认 key 变成 `branch_models.psd.encoder.*`。

2. `test_prefix_wpli_encoder_state_dict_keys`
   - branch="wpli"；
   - 确认 key 变成 `branch_models.wpli.encoder.*`。

3. `test_checkpoint_metadata_mismatch_raises`
   - branch 不匹配时报错；
   - test_subject_id 不匹配时报错；
   - embedding_dim 不匹配时报错；
   - source_feature_manifest_hash 不匹配时报错。

4. `test_merge_psd_wpli_checkpoints`
   - 构造 PSD 和 wPLI checkpoint；
   - 合并后同时包含 `branch_models.psd.encoder.*` 和 `branch_models.wpli.encoder.*`。

5. `test_merged_state_loads_into_multimodal_model`
   - 构造 `MultimodalEEGModel(feature_kind="psd-fc-wpli", fusion="gated", embedding_dim=32, dropout=0.0, encoder_kind="cnn")`；
   - 使用 merged_state_dict 调用 `model.load_state_dict(merged_state_dict, strict=False)`；
   - 确认没有 unexpected_keys；
   - missing_keys 可以存在，因为 gate 和 classifier 不从 SSL 加载。

============================================================
七、实现 Phase 3：把 checkpoint 保存/复用接入 Segment SSL
============================================================

请定位当前实现以下实验的代码：

- PSD Segment Barlow
- FC/wPLI Segment Barlow
- PSD Segment VICReg
- FC/wPLI Segment VICReg

如果当前 segment SSL 代码尚未模块化，请先做最小重构，不要大改训练逻辑。

给对应训练入口增加参数：

- `--save-ssl-encoders`
- `--reuse-ssl-encoders`
- `--ssl-checkpoint-dir`
- `--force-retrain-ssl`
- `--reuse-only`
- `--checkpoint-tag`

默认目录：

`results/checkpoints/ssl_encoders/`

逻辑：

1. 每个 fold 开始时，根据当前参数生成 expected_metadata：
   - checkpoint_type = "segment_ssl_encoder"
   - segment_ssl_method = "segment_barlow"
   - ssl_objective = "barlow"
   - branch = "psd" or "wpli"
   - base_seed
   - effective_seed
   - fold_index
   - test_subject_id
   - excluded_subject_id
   - ssl_data_scope
   - feature_kind
   - encoder_kind
   - embedding_dim
   - dropout
   - projection_dim
   - pretrain_epochs
   - pretrain_lr
   - batch_size
   - augmentation config
   - n_ssl_segments
   - source_feature_manifest_hash
   - created_at

2. 如果 `--reuse-ssl-encoders` 且 checkpoint 存在：
   - load checkpoint；
   - validate metadata；
   - 如果通过校验，跳过 SSL pretraining；
   - 返回 encoder state。

3. 如果 checkpoint 不存在：
   - 如果 `--reuse-only`，raise FileNotFoundError；
   - 否则运行 SSL pretraining。

4. 如果 checkpoint 存在但 metadata 不匹配：
   - 如果 `--force-retrain-ssl`，重新训练并覆盖；
   - 否则 raise ValueError，不能静默复用。

5. SSL pretraining 完成后：
   - 提取 branch encoder state_dict；
   - 保存 checkpoint；
   - 不保存 projection head；
   - 不保存 classifier；
   - 不保存 supervised model。

6. 如果 SSL 脚本当前返回的是 full branch model state，请新增一个函数明确提取 encoder：

示例：

`extract_branch_encoder_state(model, branch)`

对于 multimodal model：

`model.branch_models[branch].encoder.state_dict()`

7. 如果 segment SSL 是单 branch 模型，也请统一保存为 branch encoder state，而不是 full model state。

============================================================
八、实现 Phase 4：新增 dual Segment Barlow transfer 脚本
============================================================

新增脚本：

`scripts/18_train_dual_segment_barlow_transfer.py`

这个脚本只负责 D/E/F 三个实验，不要重跑 A/B/C，除非用户明确要求。

脚本参数建议：

- `--config`
- `--device`
- `--seeds`
- `--experiment-set`
  - default: `remaining`
  - allowed:
    - `remaining`
    - `dual-finetune`
    - `dual-freeze`
    - `dual-low-lr`
    - `all-dual`
- `--ssl-data-scope`
- `--psd-checkpoint-dir`
- `--wpli-checkpoint-dir`
- `--ssl-checkpoint-dir`
- `--save-ssl-encoders`
- `--reuse-ssl-encoders`
- `--force-retrain-ssl`
- `--reuse-only`
- `--pretrain-epochs`
- `--pretrain-lr`
- `--projection-dim`
- `--barlow-offdiag-weight`
- `--supervised-epochs`
- `--patience`
- `--supervised-lr-main`
- `--supervised-lr-low`
- `--weight-decay`
- `--embedding-dim`
- `--dropout`
- `--output-tag`
- `--allow-mismatched-ssl-seeds`

默认：

- seeds = 0 1 2 3 7 13
- embedding_dim = 32
- dropout = 0.0
- supervised_epochs = 100
- patience = 100
- supervised_lr_main = 0.002
- supervised_lr_low = 0.001
- weight_decay = 1e-5
- transfer_mode for D = finetune
- transfer_mode for E = freeze-encoder
- transfer_mode for F = finetune

脚本内部流程：

for seed in seeds:
  for fold in LOSO folds:

    1. 确定 test_subject_id；
    
    2. 构造 PSD expected checkpoint metadata；
       branch = "psd"
       ssl_objective = "barlow"
       fold_index = current fold
       test_subject_id = current test subject
       excluded_subject_id = current test subject
    
    3. 构造 wPLI expected checkpoint metadata；
       branch = "wpli"
       ssl_objective = "barlow"
       fold_index = current fold
       test_subject_id = current test subject
       excluded_subject_id = current test subject
    
    4. 对 PSD：
       如果 checkpoint 存在且 metadata 匹配，直接加载；
       否则跑 PSD Segment Barlow SSL，并保存 checkpoint。
    
    5. 对 wPLI：
       如果 checkpoint 存在且 metadata 匹配，直接加载；
       否则跑 FC/wPLI Segment Barlow SSL，并保存 checkpoint。
    
    6. 调用 `merge_branch_pretrained_states(psd_checkpoint, wpli_checkpoint)`；
       得到 merged_state_dict。
    
    7. 将 merged_state_dict 放入：
       `pretrained_state_by_test_subject[test_subject_id] = merged_state_dict`

  fold loop 结束后：

    8. 构造 supervised config；
    9. 调用现有 `run_loso_supervised_with_history(...)`；
    10. 输出 predictions、metrics、loss history、loss curve。

注意：
不要在每个 fold 内单独跑 supervised training。当前已有 supervised helper 是一次接受全部 fold 的 `pretrained_state_by_test_subject`，请尽量复用现有函数，保持行为一致。

============================================================
九、实现 Phase 5：D/E/F 输出命名
============================================================

D 输出：

run name:
`dual_segbarlow_psd_wpli_finetune_lr0_002_seed<seed>`

files:
- `results/predictions/dl_loso_predictions_dual_segbarlow_psd_wpli_finetune_lr0_002_seed<seed>.csv`
- `results/metrics/dl_model_comparison_dual_segbarlow_psd_wpli_finetune_lr0_002_seed<seed>.csv`
- `results/training_logs/dl_loss_history_dual_segbarlow_psd_wpli_finetune_lr0_002_seed<seed>.csv`
- `results/figures/dl_loss_curve_dual_segbarlow_psd_wpli_finetune_lr0_002_seed<seed>.png`

E 输出：

run name:
`dual_segbarlow_psd_wpli_freeze_lr0_002_seed<seed>`

files 同上替换 run name。

F 输出：

run name:
`dual_segbarlow_psd_wpli_finetune_lr0_001_seed<seed>`

files 同上替换 run name。

最终 summary：

`results/metrics/dual_segment_barlow_psd_wpli_transfer_summary.csv`

summary 至少包含：

- experiment_name
- seed
- transfer_mode
- supervised_lr
- weight_decay
- pretrained_psd = True
- pretrained_wpli = True
- psd_ssl_objective = barlow
- wpli_ssl_objective = barlow
- strict_loso_ssl = True
- n_subjects
- accuracy
- balanced_accuracy
- sensitivity
- specificity
- precision
- f1
- roc_auc
- pr_auc
- Brier score，如果当前 metrics 支持或容易加入
- run_name

============================================================
十、实现 Phase 6：与 A/B/C 汇总比较
============================================================

新增或复用 summary 脚本，生成：

`results/metrics/segment_barlow_ablation_comparison_summary.csv`

包含 A/B/C/D/E/F 六组：

A. no-SSL CNN  
B. PSD Segment Barlow encoder only  
C. FC/wPLI Segment Barlow encoder only  
D. dual Segment Barlow + finetune  
E. dual Segment Barlow + freeze-encoder  
F. dual Segment Barlow + lower supervised LR  

每组汇总：

- n_seeds
- accuracy_mean
- accuracy_std
- accuracy_min
- accuracy_max
- balanced_accuracy_mean
- balanced_accuracy_std
- balanced_accuracy_min
- balanced_accuracy_max
- roc_auc_mean
- roc_auc_std
- pr_auc_mean
- pr_auc_std
- sensitivity_mean
- specificity_mean
- ensemble_accuracy
- ensemble_balanced_accuracy
- ensemble_roc_auc
- ensemble_pr_auc
- per-subject error frequency
- delta_accuracy_mean_vs_no_ssl
- delta_accuracy_mean_vs_psd_only
- delta_accuracy_mean_vs_wpli_only
- delta_std_vs_no_ssl
- whether_best_mean
- whether_best_stability

如果找不到 A/B/C 对应结果文件，不要崩溃。请输出 warning，并在文档中写明哪些对照缺失。

============================================================
十一、实现 Phase 7：per-subject error frequency
============================================================

新增或复用函数：

`compute_per_subject_error_frequency(prediction_files)`

输入：
- 多个 seed 的 prediction CSV

输出：
- subject_id
- y_true
- n_runs
- n_errors
- error_rate
- mean_y_score
- std_y_score
- experiments

保存：

`results/metrics/dual_segment_barlow_error_frequency.csv`

还要把 A/B/C/D/E/F 的 error frequency 尽量汇总成：

`results/metrics/segment_barlow_ablation_error_frequency.csv`

目的：
判断 dual encoder 是否减少了之前反复预测错误的患者。

============================================================
十二、实现 Phase 8：文档
============================================================

新增文档：

`docs/dual_segment_barlow_transfer_results.md`

内容必须包括：

1. 背景：
   - 为什么要做 PSD + wPLI dual Segment Barlow；
   - 为什么 Barlow 适合当前小样本 EEG；
   - 为什么不用 negative-sample 方法作为主线。

2. 模型结构：
   - PSD branch 加载 PSD Segment Barlow encoder；
   - wPLI branch 加载 FC/wPLI Segment Barlow encoder；
   - EO/EC gate 和 classifier 随机初始化；
   - supervised LOSO fine-tuning。

3. checkpoint 机制：
   - checkpoint 保存在哪里；
   - 保存了什么；
   - 不保存什么；
   - 如何校验 metadata；
   - 如何防止测试患者泄漏；
   - 如何复用，避免重复跑 SSL。

4. A/B/C 是否已存在：
   - 如果完整，说明没有重跑；
   - 如果不完整，说明缺失项。

5. D/E/F 结果：
   - 每个 seed；
   - mean/std/min/max；
   - ensemble；
   - best row；
   - 与 A/B/C 的差异。

6. 结论：
   - dual encoder 是否提高 mean accuracy；
   - dual encoder 是否降低 seed std；
   - dual encoder 是否提高 min accuracy；
   - dual encoder 是否减少错误患者；
   - dual encoder 是否适合作为论文主 SSL-CNN；
   - 如果没有提升，也要如实说明。

7. 下一步：
   - 如果 D 最好，后续进入 explainability；
   - 如果 F 更稳定，后续采用 lower LR；
   - 如果 E 很差，说明 encoder 需要 supervised fine-tuning；
   - 如果 dual 不如单分支，说明存在 negative transfer，需要保留单分支 Barlow 作为主线。

同时更新：

`docs/task_status.md`

加入本任务完成情况、测试命令、输出文件路径和主要结论。

============================================================
十三、实现 Phase 9：.gitignore 与目录清洁
============================================================

检查 `.gitignore`。

如果没有，请加入：

- `results/checkpoints/`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `__pycache__/`
- `.pytest_cache/`

注意：
checkpoint 文件不能提交到 git。
metrics、predictions、docs 可以提交。
不要在仓库根目录生成临时 CSV、PNG、日志或 debug 文件。

============================================================
十四、测试命令
============================================================

先运行：

`python -B -m pytest tests/test_ssl_checkpointing.py -v -p no:cacheprovider`

再运行相关新增测试：

`python -B -m pytest tests -v -p no:cacheprovider`

如果全量测试太慢，至少先跑：

`python -B -m pytest tests/test_ssl_checkpointing.py tests/test_segment_ssl_dataset.py tests/test_segment_ssl_training.py tests/test_deep_models.py -v -p no:cacheprovider`

最后清洁检查：

- no `__pycache__`
- no `.pytest_cache`
- no `.pyc`
- no `.pyo`
- no root debug files
- checkpoint files only under `results/checkpoints/ssl_encoders/`
- checkpoint files ignored by git

============================================================
十五、建议实际运行命令
============================================================

如果实现完成，建议先 smoke test：

`python -B scripts/18_train_dual_segment_barlow_transfer.py --config configs/paths.example.yaml --device cuda --seeds 0 --experiment-set dual-finetune --save-ssl-encoders --reuse-ssl-encoders --supervised-epochs 2 --patience 2 --pretrain-epochs 1 --output-tag smoke`

smoke test 完成后删除 smoke outputs，确认没有缓存和临时文件。

正式运行：

`python -B scripts/18_train_dual_segment_barlow_transfer.py --config configs/paths.example.yaml --device cuda --seeds 0 1 2 3 7 13 --experiment-set remaining --save-ssl-encoders --reuse-ssl-encoders --supervised-epochs 100 --patience 100 --pretrain-epochs <使用已有SegmentBarlow最佳设置> --supervised-lr-main 0.002 --supervised-lr-low 0.001 --weight-decay 0.00001 --embedding-dim 32 --dropout 0.0`

如果已有 PSD/wPLI Segment Barlow checkpoint：

`python -B scripts/18_train_dual_segment_barlow_transfer.py --config configs/paths.example.yaml --device cuda --seeds 0 1 2 3 7 13 --experiment-set remaining --reuse-only --reuse-ssl-encoders --supervised-epochs 100 --patience 100 --supervised-lr-main 0.002 --supervised-lr-low 0.001 --weight-decay 0.00001 --embedding-dim 32 --dropout 0.0`

============================================================
十六、验收标准
============================================================

本任务完成的标准：

1. `tests/test_ssl_checkpointing.py` 通过；
2. 全量或相关 pytest 通过；
3. 可以保存 PSD Segment Barlow encoder checkpoint；
4. 可以保存 wPLI Segment Barlow encoder checkpoint；
5. 可以复用 checkpoint，跳过重复 SSL pretraining；
6. metadata 不匹配时会报错；
7. 可以合并 PSD + wPLI checkpoint；
8. merged state 能被 `MultimodalEEGModel("psd-fc-wpli")` 加载，且没有 unexpected keys；
9. D/E/F 三组实验跑完至少 seeds 0,1,2,3,7,13；
10. 输出 `dual_segment_barlow_psd_wpli_transfer_summary.csv`；
11. 输出 `segment_barlow_ablation_comparison_summary.csv`；
12. 输出 per-subject error frequency；
13. 更新 `docs/dual_segment_barlow_transfer_results.md`；
14. 更新 `docs/task_status.md`；
15. checkpoint 文件不进入 git；
16. 没有根目录临时文件和缓存文件。

============================================================
十七、重要写作要求
============================================================

不要在文档或 commit message 中声称 dual Segment Barlow 一定提高准确率。

请根据真实结果写：

- improved mean accuracy / did not improve mean accuracy
- improved seed stability / did not improve seed stability
- improved minimum seed performance / did not improve minimum seed performance
- reduced recurrent patient errors / did not reduce recurrent patient errors

对于小样本医学预测，mean accuracy、std、min seed、ensemble 和 error frequency 比单 seed best accuracy 更重要。