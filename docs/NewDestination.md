你是一个熟悉 PyTorch、EEG、自监督学习、脑卒中比例恢复模型、小样本医学预测、LOSO-CV 防泄漏和论文级实验设计的工程研究助手。



当前仓库：

EEG_PredictStokeDLModel

当前分支：

codex-segment-level-fc-wpli-ssl



背景：

我们已经系统尝试了很多方法：

\- Segment Barlow

\- Patient-level Barlow

\- Hard-negative fine-tuning

\- Manifold Mixup + modality dropout

\- Masked Barlow

\- Masked reconstruction + VICReg

\- Graph-smoothed masked-VICReg

\- Dual encoder

\- MIL

\- qEEG-guided CNN branch

\- qEEG auxiliary SSL

\- SAM/SWA stabilization



结论：

1. qEEG-guided SSL-CNN 没有提高 10-seed mean accuracy。
2. Hard-negative 是 mixed/negative。
3. MIL 明显失败。
4. Dual encoder 不作为主线。
5. Patient-level Barlow 曾经有较好的 mean accuracy，比 qEEG-guided branch 更接近当前目标。
6. SWA-only seed0 有提升，但 10-seed mean accuracy 仍低于 no-SSL。
7. 当前真正需要解决的是 seed-level stability，而不是 seedmean ensemble accuracy。



本任务目标：

回到 Patient-level Barlow SSL-CNN，以“连续 recovery residual / signed distance”为核心，做 Residual-aware Multi-task SSL-CNN，目标是提高 10-seed mean accuracy、min accuracy、ROC AUC、PR AUC 和 Brier，并降低 seed-to-seed 方差。



不要继续：

\- qEEG-guided branch 主线

\- MIL

\- Dual encoder 主线

\- hard-negative loss

\- new raw EEG CNN-BLSTM

\- large architecture search

\- test-label optimized threshold

\- 直接针对 sub09/sub14 调参



============================================================

一、主方向

============================================================



实现新模型方向：



Residual-aware Patient-level Barlow SSL-CNN



核心思想：

当前标签是由 continuous residual 二值化得到的。仅用 binary BCE 会丢掉 residual distance 信息，导致小样本下决策边界不稳定。请让模型同时学习：



1. binary proportional recovery label
2. continuous signed residual distance
3. pairwise recovery ranking



定义：



residual = predicted_recovery - observed_recovery

threshold = 1.5



signed_distance = threshold - residual



解释：

signed_distance > 0 表示更接近 proportional recovery；

signed_distance < 0 表示 poor recovery；

距离越大，标签越确定。



============================================================

二、模型基座

============================================================



主基座：



Patient-level Barlow SSL-CNN



模型：

MultimodalEEGModel(

​    feature_kind="psd-fc-wpli",

​    fusion="gated",

​    encoder_kind="cnn",

​    embedding_dim=32,

​    dropout=0.0

)



输入：

\- psd_eo: 62 x 90

\- psd_ec: 62 x 90

\- wpli_eo: 1891 x 6

\- wpli_ec: 1891 x 6



不要加入 qEEG branch。

不要加入 clinical branch。

不要加入 high-dimensional handcrafted features。



Patient-level Barlow checkpoint：

必须复用已有 Patient-level Barlow encoder checkpoint。

如 checkpoint 缺失，只允许 SSL-only cache。

不要重新跑无必要 SSL。

不要覆盖旧 prediction/metric。



============================================================

三、多任务输出头

============================================================



在 CNN embedding 后新增多任务头：



embedding = model.extract_embedding(batch)



Heads:

1. classification_head:

   embedding -> logit_cls



2. residual_head:

   embedding -> residual_score_hat

   预测 signed_distance 的 fold-local z-score



3. optional uncertainty_head:

   embedding -> log_var

   如果实现复杂，第一版不要做 uncertainty。



最终输出：

\- p_cls = sigmoid(logit_cls)

\- d_hat = residual_score_hat

\- p_residual = sigmoid(k * d_hat)

\- p_final = alpha * p_cls + (1 - alpha) * p_residual



默认：

alpha = 0.5

k = 1.0



如果要选择 alpha：

只能在 inner training folds 中选择。

不能用最终 LOSO test labels 选择 alpha。



============================================================

四、loss 设计

============================================================



总 loss：



loss =

  loss_bce

\+ lambda_reg * loss_residual_regression

\+ lambda_rank * loss_pairwise_ranking

\+ lambda_soft * loss_soft_label



1. Binary BCE:

loss_bce = BCEWithLogitsLoss(logit_cls, y_binary)



2. Residual regression:

target = fold-local z-scored signed_distance

loss_residual_regression = HuberLoss(d_hat, target)



3. Pairwise ranking loss:

For training pairs i, j:

if signed_distance_i > signed_distance_j:

​    encourage score_i > score_j



Use pairwise logistic ranking:

loss_rank = log(1 + exp(-(score_i - score_j)))



Only include pairs where distance difference exceeds a small margin:

abs(signed_distance_i - signed_distance_j) > rank_margin



Default:

rank_margin = 0.5 train-fold residual SD



4. Soft label loss:

soft_y = sigmoid(signed_distance / tau)

loss_soft = BCEWithLogitsLoss(logit_cls, soft_y)



Default tau:

tau = train-fold residual SD



Default loss weights for seed0 pilot:

lambda_reg = 0.3

lambda_rank = 0.1

lambda_soft = 0.2



Also test:

A. lambda_reg=0.1, lambda_rank=0.1, lambda_soft=0.2

B. lambda_reg=0.3, lambda_rank=0.0, lambda_soft=0.2

C. lambda_reg=0.3, lambda_rank=0.3, lambda_soft=0.1



Do not run a huge grid.

Seed0 pilot only.



============================================================

五、训练策略

============================================================



Use Patient-level Barlow encoder.



Fine-tuning schedule:

Use the best stable schedule from prior experiments as reference.



Primary:

\- optimizer = AdamW or Adam

\- lr = 0.002

\- weight_decay = 1e-5

\- epochs = 100

\- patience = 100

\- dropout = 0.0

\- embedding_dim = 32



Optional stabilization:

\- SWA-only can be combined if seed0 base multi-task is positive

\- Do not combine SAM unless explicitly needed; SAM was poor in the previous pilot.



Stage schedule:

First implementation can use normal fine-tuning.

If unstable, add:

\- freeze encoder for first 20 epochs

\- train heads/gates

\- unfreeze final encoder layers

\- SWA from epoch 50



But first run should isolate whether residual-aware loss helps.



============================================================

六、实验顺序

============================================================



Phase 1: Implement residual-aware target construction.



Add functions:

\- compute_signed_distance_from_label_table

\- fold_local_standardize_signed_distance

\- make_soft_labels_from_signed_distance

\- pairwise_ranking_loss



Tests must verify:

\- test subject is excluded from residual scaler

\- signed_distance sign is correct

\- soft labels are in [0,1]

\- ranking loss finite

\- batch size 1 does not break



Phase 2: Seed0 pilot.



Compare:



A. prior no-SSL seed0

B. prior Patient-level Barlow seed0

C. Patient-level Barlow + residual regression only seed0

D. Patient-level Barlow + residual regression + soft labels seed0

E. Patient-level Barlow + residual regression + soft labels + pairwise ranking seed0

F. best seed0 + optional SWA



Metrics:

\- accuracy

\- balanced_accuracy

\- sensitivity

\- specificity

\- ROC AUC

\- PR AUC

\- Brier

\- sub09_y_score

\- sub14_y_score

\- sub05_y_score

\- sub13_y_score



Seed0 success criteria:

Continue only if any variant satisfies:

1. accuracy > prior Patient-level Barlow seed0; or
2. balanced accuracy improves; or
3. ROC AUC and PR AUC both improve without accuracy drop; or
4. Brier improves without accuracy drop; or
5. specificity improves without sensitivity dropping more than 0.10.



If seed0 is negative, stop and write negative pilot.



Phase 3: If seed0 positive, run 10 seeds.



Seeds:

0, 1, 2, 3, 4, 5, 7, 13, 21, 42



Primary comparison:

\- no-SSL locked reference

\- Patient-level Barlow baseline

\- Residual-aware Patient-level Barlow

\- Residual-aware Patient-level Barlow + SWA, if seed0 supported it



============================================================

七、主要评价目标

============================================================



Primary target is 10-seed per-run stability, not seedmean ensemble.



Report:

\- mean accuracy

\- std accuracy

\- min accuracy

\- max accuracy

\- mean balanced accuracy

\- mean ROC AUC

\- mean PR AUC

\- mean Brier

\- min ROC AUC

\- min PR AUC

\- per-subject error frequency



Target threshold for success:

The method is successful only if it improves at least two of:



1. mean accuracy > no-SSL locked reference
2. min accuracy >= no-SSL locked reference min or >= 0.7368
3. std accuracy lower than Patient-level Barlow baseline
4. mean ROC AUC higher than no-SSL
5. mean PR AUC higher than no-SSL
6. Brier lower than no-SSL



Do not promote a method based only on seedmean accuracy.



============================================================

八、统计比较

============================================================



Use patient-level statistics.



Do not treat 10 seeds x 19 subjects as 190 independent clinical samples.



Compute:

\- patient-level bootstrap CI

\- paired correctness comparison

\- paired score bootstrap

\- random-label permutation

\- Brier comparison



Comparisons:

1. no-SSL vs Patient-level Barlow baseline
2. no-SSL vs Residual-aware Patient-level Barlow
3. Patient-level Barlow baseline vs Residual-aware Patient-level Barlow



============================================================

九、sub09/sub14 rules

============================================================



sub09, sub14, sub05, sub13 are watched hard cases only.



Do not optimize directly for these subjects.

Do not include their names in loss design.

Do not choose hyperparameters based on fixing them.



Output:

results/metrics/residual_aware_patient_barlow_watched_subject_scores.csv



============================================================

十、outputs

============================================================



Scripts:

scripts/30_train_residual_aware_patient_barlow.py



New code:

src/eeg_recovery/training/residual_targets.py

src/eeg_recovery/training/residual_aware_losses.py



Results:

results/predictions/dl_loso_predictions_patient_barlow_residualaware_seed<seed>.csv

results/metrics/dl_model_comparison_patient_barlow_residualaware_seed<seed>.csv

results/metrics/patient_barlow_residualaware_seed0_comparison.csv

results/metrics/patient_barlow_residualaware_10seed_summary.csv

results/metrics/patient_barlow_residualaware_subject_error_frequency.csv

results/metrics/patient_barlow_residualaware_statistical_comparison.csv



Docs:

docs/patient_barlow_residual_aware_seed0_results.md

docs/patient_barlow_residual_aware_10seed_results.md



Update:

docs/task_status.md



============================================================

十一、tests

============================================================



Add tests:

tests/test_residual_targets.py

tests/test_residual_aware_losses.py

tests/test_patient_barlow_residual_aware_training.py



Tests:

1. signed_distance = threshold - residual
2. positive label corresponds to signed_distance >= 0
3. fold-local residual scaler excludes test subject
4. soft labels are in [0,1]
5. Huber residual loss finite
6. ranking loss finite
7. no ranking pairs returns zero loss
8. seed0 output does not overwrite old Patient-level Barlow results
9. reuse-only missing checkpoint raises error



Run:

python -B -m pytest tests/test_residual_targets.py tests/test_residual_aware_losses.py tests/test_patient_barlow_residual_aware_training.py -v -p no:cacheprovider



If feasible:

python -B -m pytest tests -v -p no:cacheprovider



============================================================

十二、final instruction

============================================================



This task is not qEEG-guided branch work.

This task is not architecture search.

This task uses Patient-level Barlow as the main candidate and tests whether continuous residual-aware supervision improves seed-level stability.



Do not proceed to 10 seeds unless seed0 passes the success criteria.

Do not tune on final test labels.

Do not optimize named subjects.