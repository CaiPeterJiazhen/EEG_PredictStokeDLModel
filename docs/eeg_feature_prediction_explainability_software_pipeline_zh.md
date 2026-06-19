# 预处理 EEG 到预测、解释性分析和 MNE 出图的软件封装流程

本文档说明如何在当前项目中，把一个已经预处理完成的 EEGLAB `.set/.fdt` 脑电文件转成 PSD 和 WPLI 特征，再输入现有传统机器学习模型或当前最终模型 `Residual_Barlow_CNN / Residual-aware SSL-CNN` 做预测，随后完成解释性分析，并用 MNE 绘制 PSD topomap 和 WPLI connectivity 图。

目标是给后续封装软件使用。本文只描述现有项目代码和推荐调用方式，不重新定义模型或特征。

## 0. 当前项目真实保留的模型和结果

当前项目中做过很多历史尝试，但后续软件封装和论文/比赛展示应优先贴近以下“正式保留”模型与结果。模型对比图的数据源是：

```text
docs/model_metrics/patient_level_model_scorecard_with_barlow.csv
docs/model_metrics/core_ablation_10seed_summary.csv
final_model_ablation_explainability_results_20260610/independent_train_gpu_main/results/tables/final_Residual_ssl_cnn.csv
```

### 0.1 五个正式对比模型

| 模型 | 项目中含义 | 主要入口/来源 | Accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC | PR-AUC | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic L1 | 传统机器学习，PSD+WPLI，L1 逻辑回归 | `scripts/04_train_ml_baselines.py` | 0.7368 | 0.7333 | 0.8000 | 0.6667 | 0.7111 | 0.7754 | 0.2080 |
| No-SSL CNN | 不加自监督的 PSD+WPLI gated CNN | `scripts/05_train_supervised_loso.py` | 0.7632 | 0.7567 | 0.8800 | 0.6333 | 0.7733 | 0.7535 | 0.1983 |
| Barlow CNN | Patient-level Barlow SSL + CNN，但无 residual-aware heads | `scripts/29_train_patient_barlow_stabilized.py` | 0.7895 | 0.7822 | 0.9200 | 0.6444 | 0.7767 | 0.7532 | 0.1978 |
| Residual-aware CNN | 无 SSL，但加入残差感知多任务训练 | `scripts/30_train_residual_aware_patient_barlow.py --pretraining-mode no_ssl` | 0.8158 | 0.8122 | 0.8800 | 0.7444 | 0.8989 | 0.9084 | 0.1304 |
| Residual-aware SSL-CNN | 当前最终模型：Patient-level Barlow SSL + residual-aware heads | `scripts/30_train_residual_aware_patient_barlow.py --pretraining-mode patient_barlow` | 0.8474 | 0.8411 | 0.9600 | 0.7223 | 0.8867 | 0.8910 | 0.1324 |

注意：`Residual-aware SSL-CNN` 的主模型结果以后统一以这个文件为准：

```text
final_model_ablation_explainability_results_20260610/independent_train_gpu_main/results/tables/final_Residual_ssl_cnn.csv
```

不要再用旧的 `final_ssl_cnn.csv` 作为主模型结果。

### 0.2 当前最终模型的 10-seed 汇总

最终模型使用 10 个 seed：

```text
0, 1, 2, 3, 4, 5, 7, 13, 21, 42
```

当前锁定的 10-seed 均值和标准差：

| 指标 | mean | sd |
|---|---:|---:|
| Accuracy | 0.8474 | 0.0388 |
| Balanced accuracy | 0.8411 | 0.0391 |
| Sensitivity | 0.9600 | 0.0516 |
| Specificity | 0.7223 | 0.0586 |
| Precision | 0.7944 | 0.0358 |
| F1 | 0.8687 | 0.0338 |
| ROC AUC | 0.8867 | 0.0599 |
| PR AUC | 0.8910 | 0.0573 |
| Brier | 0.1324 | 0.0315 |

### 0.3 当前最终模型 masked-inference 特征/状态/频段消融

当前项目后续画特征消融图时，使用的是 masked-inference 数据，而不是重新训练每个子模型。数据源：

```text
final_model_ablation_explainability_results_20260610/results/tables/table3_final_ssl_cnn_feature_state_band_ablation_for_paper.csv
```

该消融是在固定最终 residual-aware SSL-CNN checkpoint 下，把对应输入分支置零后推理，保持模型权重不变。

| 消融 | 输入说明 | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|---:|---:|
| full_psd_wpli | PSD + WPLI，EO + EC，全频段 | 0.8368 | 0.8306 | 0.8600 | 0.8585 | 0.1416 |
| psd_only | 只保留 PSD，WPLI 置零 | 0.5842 | 0.5867 | 0.6189 | 0.6880 | 0.3337 |
| wpli_only | 只保留 WPLI，PSD 置零 | 0.6474 | 0.6328 | 0.6644 | 0.6708 | 0.2761 |
| eo_only | 只保留 EO，EC 置零 | 0.5632 | 0.5650 | 0.6100 | 0.6743 | 0.3400 |
| ec_only | 只保留 EC，EO 置零 | 0.6895 | 0.6778 | 0.6944 | 0.6845 | 0.2474 |
| beta_medium_beta_high | 只保留 Beta Medium + Beta High | 0.5474 | 0.5433 | 0.5856 | 0.6601 | 0.3564 |
| full_minus_beta_medium_beta_high | 去掉 Beta Medium + Beta High | 0.5105 | 0.5011 | 0.6178 | 0.7033 | 0.3189 |
| motor_wpli_edges_only | 只保留运动相关 WPLI 边，PSD 置零 | 0.5263 | 0.5139 | 0.5122 | 0.5905 | 0.3628 |
| full_minus_motor_wpli_edges | 去掉运动相关 WPLI 边 | 0.7263 | 0.7250 | 0.8044 | 0.8260 | 0.1990 |

因此软件界面如果要做“特征贡献/消融展示”，应该优先复用这套 masked-inference 逻辑，而不是临时训练新的子模型。

## 1. 输入数据约定

当前项目的特征和模型默认输入不是单个 `.set` 文件，而是同一受试者的两个静息态文件：

- `EO`：睁眼，项目中由文件名数字后缀 `1` 识别。
- `EC`：闭眼，项目中由文件名数字后缀 `2` 识别。

输入 EEG 必须满足：

- 文件格式：EEGLAB `.set` + `.fdt`。
- 预处理已完成：滤波、重参考、ICA、坏道处理、分段、伪迹剔除。
- 通道数：62 个实际通道，M1/M2 已删除。
- 通道顺序：必须等于项目的 `CANONICAL_CHANNELS_62`。
- 患者需要提供患侧手：`左` 或 `右`。
- 健康人没有患侧，当前项目按 `右` 处理，即不做左右通道翻转。

关键代码：

| 功能 | 代码 |
|---|---|
| 读取 `.set` 元数据和 `.fdt` 数据 | `src/eeg_recovery/io/eeglab.py` |
| 索引患者/健康人 EEG 文件 | `src/eeg_recovery/io/index.py` |
| 规范化患者 ID | `src/eeg_recovery/metadata/subjects.py` |
| 读取患者信息、患侧、标签 | `src/eeg_recovery/metadata/labels.py` |
| 固定 62 通道顺序 | `src/eeg_recovery/channels/mapping.py` |
| 患侧半球对齐 | `src/eeg_recovery/channels/hemisphere_flip.py` |

## 2. 通道翻转和半球对齐

项目不是把所有 EEG 都反转，而是按患侧做半球对齐：

- `affected_hand == "右"`：不反转。
- `affected_hand == "左"`：左右半球镜像通道互换。
- 中线通道保持不变。

实现代码：

```python
from eeg_recovery.channels.hemisphere_flip import flip_channels_for_affected_hand
```

具体函数：

```python
flip_channels_for_affected_hand(
    data,
    channel_names,
    affected_hand,
    channel_axis=0,
)
```

它会把左患侧患者对齐到项目统一的“右患侧 / C3 刺激”约定。PSD 和 WPLI 都是在这个半球对齐之后计算的。

## 3. PSD 特征计算

PSD 计算代码：

```python
src/eeg_recovery/features/psd.py
```

核心函数：

```python
from eeg_recovery.features.psd import (
    compute_single_state_psd,
    compute_psd_for_eeg_record,
    write_psd_feature,
)
```

项目 PSD 的参数：

- 方法：`scipy.signal.welch`
- 频率分辨率：0.5 Hz
- 频率范围：0.5-45 Hz
- 输出频点数：90
- 输出形状：`(62, 90)`
- 窗函数：Hann
- Welch overlap：50%
- 先按患侧半球对齐，再计算 PSD

单个 state 的概念性调用：

```python
from eeg_recovery.io.eeglab import read_eeglab_set_metadata, read_eeglab_fdt
from eeg_recovery.features.psd import compute_single_state_psd

metadata = read_eeglab_set_metadata(r"F:\path\to\subject_eo.set")
data = read_eeglab_fdt(metadata, subject_id="subXX", state="EO")

feature = compute_single_state_psd(
    data=data,
    sampling_rate=metadata.srate,
    channel_names=metadata.ch_names,
    affected_hand="左",
    subject_id="subXX",
    state="EO",
    source_set_path=metadata.set_path,
)

psd = feature.psd
```

如果已经通过项目索引得到 `EEGFileRecord`，可以直接调用：

```python
feature = compute_psd_for_eeg_record(record, affected_hand="左")
```

## 4. WPLI 特征计算

WPLI 计算代码：

```python
src/eeg_recovery/features/connectivity.py
```

核心函数：

```python
from eeg_recovery.features.connectivity import (
    compute_single_state_connectivity,
    compute_connectivity_for_eeg_record,
    write_connectivity_feature,
)
```

项目 WPLI 的参数：

- 方法：STFT 后按频段计算 wPLI。
- STFT 窗长：`2 * sampling_rate`
- STFT overlap：50%
- 通道数：62
- 连接边数：`62 * 61 / 2 = 1891`
- 频段数：6
- 输出形状：`(1891, 6)`
- 频段：
  - Delta：1-3 Hz
  - Theta：4-7 Hz
  - Alpha：8-13 Hz
  - Beta Low：13-18 Hz
  - Beta Medium：18-21 Hz
  - Beta High：21-30 Hz

单个 state 的概念性调用：

```python
from eeg_recovery.io.eeglab import read_eeglab_set_metadata, read_eeglab_fdt
from eeg_recovery.features.connectivity import compute_single_state_connectivity

metadata = read_eeglab_set_metadata(r"F:\path\to\subject_eo.set")
data = read_eeglab_fdt(metadata, subject_id="subXX", state="EO")

feature = compute_single_state_connectivity(
    data=data,
    sampling_rate=metadata.srate,
    channel_names=metadata.ch_names,
    affected_hand="左",
    subject_id="subXX",
    state="EO",
    source_set_path=metadata.set_path,
)

wpli = feature.wpli
edge_list = feature.edge_list
band_names = feature.band_names
```

如果已经通过项目索引得到 `EEGFileRecord`，可以直接调用：

```python
feature = compute_connectivity_for_eeg_record(record, affected_hand="左")
```

## 5. 当前已有特征生成脚本

### 5.1 生成 19 例监督训练患者的基线 PSD

```powershell
python -B scripts/02_compute_psd.py --config configs/paths.example.yaml
```

输出：

```text
data/features/psd/{subject_id}_{EO|EC}_psd.npz
```

### 5.2 生成 19 例监督训练患者的基线 FC/WPLI

```powershell
python -B scripts/03_compute_fc.py --config configs/paths.example.yaml
```

输出：

```text
data/features/fc/{subject_id}_{EO|EC}_fc.npz
```

其中 `fc` 文件里包含 `wpli`，当前模型主要使用 `wpli`。

### 5.3 生成所有可用患者阶段和健康人的 PSD/WPLI

新增的全阶段/健康人特征脚本：

```text
scripts/113_compute_all_stage_patient_health_features.py
```

调用：

```powershell
python -B scripts/113_compute_all_stage_patient_health_features.py --config configs/paths.example.yaml
```

输出：

```text
data/features_all_stages/psd/{group}/{stage}/{subject_id}_{EO|EC}_psd.npz
data/features_all_stages/fc/{group}/{stage}/{subject_id}_{EO|EC}_fc.npz
data/features_all_stages/feature_manifest.csv
data/features_all_stages/feature_coverage_summary.csv
```

这个脚本不会覆盖旧的 `data/features/psd` 和 `data/features/fc`。

## 6. 输入传统机器学习模型

传统机器学习入口：

```text
scripts/04_train_ml_baselines.py
src/eeg_recovery/training/train_baselines.py
src/eeg_recovery/features/feature_tables.py
```

PSD/WPLI 会先被转换成表格特征：

```python
from eeg_recovery.features.feature_tables import (
    load_psd_band_power_table,
    load_fc_feature_table,
    merge_feature_tables,
)

psd_table = load_psd_band_power_table("data/features/psd", subject_ids=subject_ids)
wpli_table = load_fc_feature_table("data/features/fc", metric="wpli", subject_ids=subject_ids)
feature_table = merge_feature_tables(psd_table, wpli_table)
```

机器学习模型训练和 LOSO 评估：

```powershell
python -B scripts/04_train_ml_baselines.py `
  --config configs/paths.example.yaml `
  --feature-set psd-fc-wpli `
  --selector-k 100 `
  --random-state 42 `
  --output-tag psd_fc_wpli_loso
```

输出：

```text
results/predictions/ml_baseline_loso_predictions_{tag}.csv
results/metrics/ml_baseline_model_comparison_{tag}.csv
```

当前传统 ML 支持：

- Logistic Regression L1
- Logistic Regression L2
- Linear SVM
- RBF SVM
- Random Forest
- Gaussian Naive Bayes
- KNN
- XGBoost，如果本机已安装
- LightGBM，如果本机已安装

注意：当前 `04_train_ml_baselines.py` 是评估脚本，不会保存一个单独的部署模型。如果要在封装软件里预测一个新患者，需要新增一个部署函数：用 19 例训练集特征重新 fit 所选传统模型和 scaler/selector，然后对新患者特征做 transform 和 predict。

## 7. 输入当前最终模型 Residual_Barlow_CNN

当前最终模型对应：

```text
Residual_Barlow_CNN
Residual-aware SSL-CNN
patient_barlow_residualaware_highrank_swa_clsalpha1
```

模型代码：

```text
scripts/30_train_residual_aware_patient_barlow.py
src/eeg_recovery/models/multimodal_model.py
src/eeg_recovery/training/residual_aware_losses.py
src/eeg_recovery/training/residual_targets.py
src/eeg_recovery/training/train_supervised.py
```

最终模型结构：

- 输入特征：PSD + WPLI
- 输入状态：EO + EC
- 每个模态一个 dual-state CNN branch
- 融合方式：`gated`
- 每个 branch embedding_dim：32
- PSD branch embedding：32
- WPLI branch embedding：32
- 融合后 CNN embedding：64
- 分类头：MLP classification head
- 残差头：MLP residual head
- 当前锁定主模型 `clsalpha1` 使用 `selected_alpha = 1.0`，最终预测概率等于 classification head 概率
- 残差感知损失参数：
  - `lambda_reg = 0.3`
  - `lambda_rank = 0.3`
  - `lambda_soft = 0.1`
  - `rank_margin = 0.5`

当前最终模型 checkpoint：

```text
results/checkpoints/supervised/residualaware_highrank_swa_clsalpha1/
```

该目录包含 190 个 fold checkpoint：

```text
10 seeds x 19 LOSO folds = 190 .pt files
```

每个 checkpoint 包含：

- `checkpoint_type`
- `state_dict`
- `state_scaler`
- `metadata`
  - seed
  - fold_index
  - test_subject_id
  - feature_kind
  - fusion
  - embedding_dim
  - selected_alpha
  - loss weights

### 7.1 复现最终模型训练和 checkpoint

如果要重新生成当前这种 fold checkpoint，使用：

```powershell
python -B scripts/30_train_residual_aware_patient_barlow.py `
  --config configs/paths.example.yaml `
  --device cuda `
  --seeds 0 1 2 3 4 5 7 13 21 42 `
  --variants highrank `
  --pretraining-mode patient_barlow `
  --reuse-ssl-encoders `
  --reuse-only `
  --data-scope all-patient `
  --embedding-dim 32 `
  --projection-dim 32 `
  --dropout 0.0 `
  --epochs 100 `
  --patience 100 `
  --lr 0.002 `
  --weight-decay 0.00001 `
  --rank-margin 0.5 `
  --residual-alpha 1.0 `
  --alpha-candidates 1.0 `
  --use-swa `
  --swa-start-epoch 50 `
  --swa-lr 0.0005 `
  --save-supervised-fold-checkpoints `
  --checkpoint-tag residualaware_highrank_swa_clsalpha1 `
  --output-prefix patient_barlow_residualaware
```

输出：

```text
results/checkpoints/supervised/residualaware_highrank_swa_clsalpha1/*.pt
results/predictions/dl_loso_predictions_patient_barlow_residualaware_highrank_swa_clsalpha1_seed{seed}.csv
results/metrics/dl_model_comparison_patient_barlow_residualaware_highrank_swa_clsalpha1_seed{seed}.csv
```

### 7.2 对 19 例内部患者复现 LOSO 预测

如果输入患者是训练名单中的 19 例之一，应使用该患者对应的 LOSO fold checkpoint。已有预测文件在：

```text
results/predictions/dl_loso_predictions_patient_barlow_residualaware_highrank_swa_clsalpha1_seed{seed}.csv
results/predictions/final_Residual_ssl_cnn_10seed_patient_predictions.csv
```

论文图表中的主模型汇总指标锁定在：

```text
final_model_ablation_explainability_results_20260610/independent_train_gpu_main/results/tables/final_Residual_ssl_cnn.csv
```

### 7.3 对新患者做软件预测

当前仓库保存的是 LOSO fold 模型，不是一个单独的 deployment model。因此封装软件有两种实现路线：

1. 推荐后续新增：用 19 例训练集训练一个 all-subject deployment checkpoint。
2. 当前可直接复用：加载 190 个 fold checkpoint，对新患者预测后取平均概率作为 ensemble 结果。

第二种方式的伪代码：

```python
from pathlib import Path
import importlib.util
import numpy as np
import torch

from eeg_recovery.training.train_supervised import SupervisedFeatureRecord, _make_batch

script_path = Path("scripts/31_explain_residual_aware_ssl_cnn.py")
spec = importlib.util.spec_from_file_location("explain_residual_aware_ssl_cnn", script_path)
explain_script = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(explain_script)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

record = SupervisedFeatureRecord(
    subject_id="new_patient",
    label=0,  # 新患者真实标签未知；推理时可填 dummy
    eo=psd_eo.astype(np.float32),
    ec=psd_ec.astype(np.float32),
    modalities={
        "psd": (psd_eo.astype(np.float32), psd_ec.astype(np.float32)),
        "wpli": (wpli_eo.astype(np.float32), wpli_ec.astype(np.float32)),
    },
)

checkpoint_dir = Path("results/checkpoints/supervised/residualaware_highrank_swa_clsalpha1")
scores = []

for checkpoint_path in sorted(checkpoint_dir.glob("*.pt")):
    payload = explain_script._load_checkpoint(checkpoint_path, device)
    metadata = payload["metadata"]
    model = explain_script._build_model_from_metadata(metadata, payload["state_dict"], device)
    scaler = explain_script._scaler_from_payload(payload["state_scaler"])
    batch = _make_batch([record], scaler, device, architecture="multimodal")

    with torch.no_grad():
        outputs = model(batch)
        # 当前主模型 selected_alpha=1.0，因此使用 classification_probability。
        score = float(outputs["classification_probability"].cpu().numpy()[0, 0])
    scores.append(score)

y_score = float(np.mean(scores))
y_pred = int(y_score >= 0.5)
```

封装时更推荐把这些推理函数整理进 `src/eeg_recovery/inference/`，避免软件长期依赖 `scripts/` 下的私有函数。

## 8. 解释性分析

解释性分析入口：

```text
scripts/31_explain_residual_aware_ssl_cnn.py
src/eeg_recovery/explainability/attribution.py
src/eeg_recovery/explainability/occlusion.py
src/eeg_recovery/explainability/tables.py
src/eeg_recovery/explainability/stability.py
```

当前解释性方法：

- Integrated Gradients
- SmoothGrad Integrated Gradients
- Occlusion sensitivity
- classifier randomization sanity check
- IG/SmoothGrad/Occlusion 一致性检查
- PSD channel-band 聚合
- WPLI edge-band 聚合
- WPLI node/network 聚合
- branch/state gate 权重输出

主命令：

```powershell
python -B scripts/31_explain_residual_aware_ssl_cnn.py `
  --config configs/paths.example.yaml `
  --device cuda `
  --model-group residualaware_highrank_swa_clsalpha1 `
  --seeds 0 1 2 3 4 5 7 13 21 42 `
  --method integrated_gradients smoothgrad occlusion `
  --target classification_logit `
  --output-tag residualaware_10seed `
  --ig-steps 64 `
  --smoothgrad-samples 4 `
  --smoothgrad-noise-std 0.02
```

主要输出：

```text
results/explainability/psd_attribution_long.csv
results/explainability/wpli_edge_attribution_long.csv
results/explainability/psd_channel_band_importance.csv
results/explainability/psd_frequency_importance.csv
results/explainability/psd_channel_frequency_top_features.csv
results/explainability/wpli_top_edges.csv
results/explainability/wpli_band_importance.csv
results/explainability/wpli_node_importance.csv
results/explainability/wpli_network_group_importance.csv
results/explainability/occlusion_branch_state.csv
results/explainability/occlusion_psd_band_channel.csv
results/explainability/occlusion_wpli_edge_node_band.csv
results/explainability/ig_smoothgrad_occlusion_consistency.csv
results/explainability/attribution_stability_summary.csv
```

解释性分析使用的输入 key：

```python
INPUT_KEYS = ("psd_eo", "psd_ec", "wpli_eo", "wpli_ec")
```

归因目标：

```python
classification_logit
```

即解释分类头 logit，而不是直接解释最终二分类标签。

## 9. 用 MNE 画 PSD topomap

当前项目真实用于 PSD topomap 的核心数据源是：

```text
results/explainability/psd_channel_band_importance.csv
```

该表由 `scripts/31_explain_residual_aware_ssl_cnn.py` 从 Integrated Gradients / SmoothGrad 归因结果聚合得到，关键列包括：

```text
state
channel
channel_index
band
mean_signed_attribution
mean_abs_attribution
std_abs_attribution
n_samples
```

当前论文图中的 PSD topomap 用 `mean_signed_attribution` 上色，正负号表示对 proportional recovery 分类 logit 的方向性贡献。

单张 MNE topomap 入口：

```text
scripts/45_make_mne_explainability_topomaps.py
```

调用：

```powershell
python -B scripts/45_make_mne_explainability_topomaps.py `
  --config configs/paths.example.yaml `
  --formats png svg pdf
```

输入：

```text
results/explainability/psd_channel_band_importance.csv
results/explainability/wpli_node_importance.csv
```

输出：

```text
results/figures/explainability/mne_topomaps/
```

当前论文使用的 2 x 7 PSD topomap grid 可由以下函数生成：

```text
scripts/84_make_focused_final_and_explainability_figures.py
```

相关函数：

```python
make_psd_topomap_grid(...)
```

这个函数的真实调用链是：

```python
psd = pd.read_csv(config.output_root / "results" / "explainability" / "psd_channel_band_importance.csv")
names, pos = _load_ced_eeglab_topomap_layout(config.standard_1005_ced, CANONICAL_CHANNELS_62)
make_psd_topomap_grid(psd, names, pos, output_dir / "figure6a_psd_topomap_bands")
```

图的布局：

- 2 行：`EO`, `EC`
- 7 列：`Delta`, `Theta`, `Alpha`, `Beta Low`, `Beta Mid`, `Beta High`, `Gamma`
- 颜色：`RdBu_r`
- 色值：`mean_signed_attribution`
- 电极坐标：`F:/CJZFile/EEG_M1/standard_1005.ced`
- 输出目录：`results/figures/revised_initial/`

当前已经生成的真实图片：

```text
results/figures/revised_initial/figure6a_psd_topomap_bands.png
results/figures/revised_initial/figure6a_psd_topomap_bands.pdf
results/figures/revised_initial/figure6a_psd_topomap_bands.svg
results/figures/revised_initial/figure6a_psd_topomap_bands.tiff
```

也可以使用单独脚本：

```text
scripts/112_make_psd_channel_band_topomap_grid.py
```

该脚本读取：

```text
results/explainability/psd_channel_band_importance.csv
```

并输出：

```text
results/figures/paper_panels/figure6b_psd_topomap_grid.png
results/figures/paper_panels/figure6b_psd_topomap_grid.pdf
```

## 10. 用 MNE 画 WPLI connectivity

当前项目真实用于 WPLI connectivity 的核心数据源是：

```text
results/explainability/wpli_top_edges.csv
```

该表同样由 `scripts/31_explain_residual_aware_ssl_cnn.py` 聚合得到，关键列包括：

```text
state
edge_index
channel_i
channel_j
band
band_index
mean_signed_attribution
mean_abs_attribution
network_group
interhemispheric
```

当前 WPLI connectivity 图用：

- `mean_abs_attribution` 控制连线粗细。
- `mean_signed_attribution >= 0` 画红色，表示 positive signed attribution。
- `mean_signed_attribution < 0` 画蓝色，表示 negative signed attribution。

单张 MNE connectivity 入口：

```text
scripts/46_make_mne_wpli_connectivity.py
```

调用：

```powershell
python -B scripts/46_make_mne_wpli_connectivity.py `
  --config configs/paths.example.yaml `
  --top-n 10 `
  --formats png svg pdf
```

输入：

```text
results/explainability/wpli_top_edges.csv
```

输出：

```text
results/figures/explainability/mne_wpli_connectivity/
```

当前论文使用的 2 x 6 WPLI connectivity grid 可由以下脚本生成：

```text
scripts/84_make_focused_final_and_explainability_figures.py
```

相关函数：

```python
make_wpli_connectivity_grid(...)
```

这个函数的真实调用链是：

```python
wpli = pd.read_csv(config.output_root / "results" / "explainability" / "wpli_top_edges.csv")
names, pos = _load_ced_eeglab_topomap_layout(config.standard_1005_ced, CANONICAL_CHANNELS_62)
make_wpli_connectivity_grid(wpli, names, pos, output_dir / "figure6b_wpli_connectivity_bands", top_n=20)
```

图的布局：

- 2 行：`EO`, `EC`
- 6 列：`Delta`, `Theta`, `Alpha`, `Beta Low`, `Beta Mid`, `Beta High`
- 每个 panel 默认选该 state-band 下 `mean_abs_attribution` 最高的 top 20 条边。
- 线宽按 `mean_abs_attribution` 归一化。
- 线颜色按 `mean_signed_attribution` 正负映射。
- 电极坐标：`F:/CJZFile/EEG_M1/standard_1005.ced`
- 输出目录：`results/figures/revised_initial/`

该图读取：

```text
results/explainability/wpli_top_edges.csv
```

并输出：

```text
results/figures/revised_initial/figure6b_wpli_connectivity_bands.png
results/figures/revised_initial/figure6b_wpli_connectivity_bands.pdf
results/figures/revised_initial/figure6b_wpli_connectivity_bands.svg
results/figures/revised_initial/figure6b_wpli_connectivity_bands.tiff
```

如果需要更清爽的 top 10 connectivity 图，可以复用同一函数，把 `top_n` 改为 `10`。项目中已有 top 10 版本输出：

```text
results/figures/revised_initial/figure6b_wpli_connectivity_bands_top10.png
results/figures/revised_initial/figure6b_wpli_connectivity_bands_top10.pdf
results/figures/revised_initial/figure6b_wpli_connectivity_bands_top10.svg
results/figures/revised_initial/figure6b_wpli_connectivity_bands_top10.tiff
```

## 11. 封装软件建议模块

建议后续把当前分散在 `scripts/` 中的流程整理成软件模块：

```text
src/eeg_recovery/inference/
  eeg_input.py
  feature_extractor.py
  final_model_predictor.py
  ml_predictor.py
  explainability_runner.py
  mne_figure_exporter.py
```

推荐职责：

| 模块 | 职责 |
|---|---|
| `eeg_input.py` | 检查 `.set/.fdt`、通道、EO/EC、采样率、患侧 |
| `feature_extractor.py` | 调用 PSD/WPLI 计算函数，返回标准 numpy arrays |
| `final_model_predictor.py` | 加载 residual-aware fold checkpoints，做 190 checkpoint ensemble 或 deployment checkpoint 推理 |
| `ml_predictor.py` | 训练/加载传统 ML 模型，对新患者表格特征预测 |
| `explainability_runner.py` | 对单患者或批量患者运行 IG/SmoothGrad/Occlusion |
| `mne_figure_exporter.py` | 根据归因表导出 PSD topomap 和 WPLI connectivity |

### 11.1 软件默认贴近当前项目的配置

为了让封装软件默认行为和当前项目真实结果一致，建议写死或配置化以下默认值：

| 软件配置项 | 当前项目默认值 |
|---|---|
| 默认主模型名称 | `Residual-aware SSL-CNN / Residual_Barlow_CNN` |
| 默认模型 group | `residualaware_highrank_swa_clsalpha1` |
| 默认模型 checkpoint 目录 | `results/checkpoints/supervised/residualaware_highrank_swa_clsalpha1/` |
| 默认模型结果表 | `final_model_ablation_explainability_results_20260610/independent_train_gpu_main/results/tables/final_Residual_ssl_cnn.csv` |
| 默认模型对比表 | `docs/model_metrics/patient_level_model_scorecard_with_barlow.csv` |
| 默认输入特征 | PSD + WPLI |
| 默认状态 | EO + EC |
| 默认 CNN fusion | `gated` |
| 默认 encoder | `cnn` |
| 默认 embedding_dim | 每个分支 32，PSD+WPLI 融合后 64 |
| 默认解释目标 | `classification_logit` |
| 默认 PSD topomap 数据 | `results/explainability/psd_channel_band_importance.csv` |
| 默认 WPLI connectivity 数据 | `results/explainability/wpli_top_edges.csv` |
| 默认电极坐标 | `F:/CJZFile/EEG_M1/standard_1005.ced` |

软件界面中如果要展示模型历史，不建议把所有探索性 SSL 方法全部列为正式模型。更合适的默认展示顺序是：

```text
1. Logistic L1
2. No-SSL CNN
3. Barlow CNN
4. Residual-aware CNN
5. Residual-aware SSL-CNN
```

其中第 5 个是当前主模型。

## 12. 最小端到端流程

封装软件最小流程如下：

```text
1. 用户输入：
   - EO .set/.fdt
   - EC .set/.fdt
   - subject_id
   - affected_hand

2. 读取 EEG：
   - read_eeglab_set_metadata
   - read_eeglab_fdt

3. 检查通道：
   - CANONICAL_CHANNELS_62

4. 半球对齐：
   - flip_channels_for_affected_hand

5. 特征计算：
   - compute_single_state_psd -> PSD EO/EC
   - compute_single_state_connectivity -> WPLI EO/EC

6. 预测：
   - 传统 ML：转为 feature table 后调用 sklearn 模型
   - 最终模型：构造 SupervisedFeatureRecord，加载 checkpoint + scaler，调用 ResidualAwarePatientBarlowModel

7. 解释性：
   - manual_integrated_gradients
   - smoothgrad_integrated_gradients
   - occlusion
   - make_psd_attribution_long_table
   - make_wpli_edge_attribution_long_table

8. 出图：
   - MNE topomap
   - WPLI connectivity
```

## 13. 需要特别注意的工程问题

1. 当前最终模型是 LOSO fold checkpoint 集合，不是单一部署模型。
2. 新患者软件预测时，建议后续训练一个 all-subject deployment checkpoint。
3. 如果暂时不训练 deployment checkpoint，可以对 190 个 fold checkpoint 做平均集成。
4. 传统 ML 当前没有保存部署模型，需要新增 `joblib.dump/load` 或每次从训练集 fit。
5. 新输入 EEG 必须和训练集保持同样的预处理、通道顺序和 EO/EC 规则。
6. 患者预测必须提供患侧，否则无法做半球对齐。
7. 健康人特征可以计算，但最终康复预测模型是患者模型，不能把健康人直接解释成康复类别。
8. 解释性图必须基于模型实际预测时使用的同一组特征和同一套 scaler。
9. MNE 图使用 `standard_1005.ced` 或项目中已解析的 62 通道二维坐标。
10. 软件封装时应把临时文件写到明确的 `outputs/` 或用户指定目录，运行结束清理缓存和中间文件。
