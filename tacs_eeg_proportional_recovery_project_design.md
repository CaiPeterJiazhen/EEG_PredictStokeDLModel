# 项目开发设计文档：基于自监督 EEG 表征学习预测 tACS 后卒中上肢比例恢复

## 0. 文档目的

本文件用于指导后续代码开发与论文方法学实现。项目目标是基于卒中患者 **tACS 治疗前的基线 EEG**，预测患者完成 tACS 治疗后是否达到 **FMA-UE 比例恢复（proportional recovery）**。

本项目参考 Lin et al. 的 transferable deep learning prognosis model 设计思路：使用治疗前临床信息与 EEG 特征预测治疗后恢复结局，并通过模型解释方法寻找关键 EEG biomarker。本项目的主要创新点是加入 **自监督学习（self-supervised learning, SSL）**，利用健康人 EEG、未完整完成实验患者 EEG、天花板效应患者 EEG 以及有标签患者 EEG 进行无标签预训练，再在 19 例有标签患者上进行监督微调和预测。

---

## 1. 研究任务定义

### 1.1 临床问题

卒中后上肢障碍患者在接受 tACS 治疗后恢复程度存在较大个体差异。本项目希望利用治疗前 EEG 与基线临床信息预测患者是否能够达到 FMA-UE 比例恢复，从而为 tACS 疗效预测和个体化康复决策提供依据。

### 1.2 预测任务

二分类任务：

- 类别 1：比例恢复组（Proportional Recovery）
- 类别 0：恢复不良组（Poor Recovery）

模型输出：

```text
P(Proportional Recovery | baseline EEG, baseline clinical features)
```

---

## 2. 数据概况与样本划分

### 2.1 有标签监督训练数据

监督训练仅使用 **19 例可打标签患者**。

说明：

- 这 19 例患者具有治疗前和治疗后 FMA-UE，可计算比例恢复 residual。
- 另外 2 例患者因存在天花板效应，无法合理打比例恢复标签，不进入监督训练。
- 监督学习阶段必须以“患者”为单位划分训练集和测试集，禁止将同一患者的 EEG 片段同时放入训练集和测试集。

### 2.2 无标签自监督预训练数据

自监督预训练阶段可使用以下 EEG 数据：

- 19 例有标签患者的基线 EEG；
- 13 例健康人 EEG；
- 未完整完成实验患者 EEG；
- 2 例天花板效应患者 EEG。

注意数据泄漏控制：

- 若采用严格的 leave-one-subject-out cross-validation，当前测试折中被留出的患者 EEG 不应参与该折的自监督预训练。
- 如果使用所有无标签 EEG 预训练，包括测试患者 EEG，则应在论文中明确描述为半监督或转导式设置，并单独报告。

---

## 3. tACS 干预方案

所有患者接受一致的 tACS 干预方案。

| 项目 | 设置 |
|---|---|
| 刺激靶点 | 患手对侧 M1 运动皮层 |
| 患手右侧 | 刺激 C3 |
| 患手左侧 | 刺激 C4 |
| 刺激频率 | 20 Hz |
| 刺激强度 | 1000 μA |
| 单次刺激时长 | 20 min |
| 总次数 | 14 次 |
| 疗程 | 每日 1 次，连续 14 天，共 2 周 |
| 结局评估时间 | 最后一次 tACS 后立即评估 |

---

## 4. 结局指标与标签定义

### 4.1 临床结局

使用 FMA-UE（Fugl-Meyer Assessment of Upper Extremity）作为主要临床结局。

- FMA-UE 满分：66 分
- 治疗前评分：`FMA_pre`
- 治疗后评分：`FMA_post`

### 4.2 比例恢复预测改善量

```text
Delta_FMA_pred = 0.7 * (66 - FMA_pre)
```

### 4.3 实际改善量

```text
Delta_FMA_obs = FMA_post - FMA_pre
```

### 4.4 比例恢复残差

```text
Residual = Delta_FMA_pred - Delta_FMA_obs
```

解释：

- Residual 越小，说明实际恢复越接近或超过 70% 比例恢复预期。
- Residual 越大，说明恢复低于比例恢复预期。

### 4.5 标签划分

使用 19 例有标签患者 residual 的中位数作为阈值。

当前估计：

```text
Median(Residual) ≈ 1.5
```

标签定义：

```text
if Residual <= median_residual:
    label = 1  # Proportional Recovery
else:
    label = 0  # Poor Recovery
```

预计类别分布：

| 类别 | 数量 |
|---|---|
| Proportional Recovery | 约 9 例 |
| Poor Recovery | 约 10 例 |

注意：

- 不使用之前表格中的 `Delta_FMA-UE >= 5` 标签。
- 本项目主标签只使用比例恢复 residual 中位数划分。

---

## 5. EEG 数据输入

### 5.1 使用的 EEG 状态

主模型只使用治疗前基线 EEG 的两个状态：

1. 睁眼静息态 EEG（Eyes Open, EO）
2. 闭眼静息态 EEG（Eyes Closed, EC）

不使用运动想象和抓握任务作为主模型输入。

### 5.2 EEG 基本参数

| 项目 | 设置 |
|---|---|
| 通道数 | 64 通道 |
| 采样率 | 250 Hz |
| 数据类型 | 预处理后 EEG |
| 主模型状态 | EO + EC |
| 主要 EEG 特征 | PSD + FC |

---

## 6. 左右翻转与空间对齐

### 6.1 目的

因为患者存在左、右患手差异，且 tACS 刺激位置与患手有关：

- 患手右侧 → 刺激 C3
- 患手左侧 → 刺激 C4

为了让模型学习“患侧相关半球 / 刺激侧半球”相关模式，而不是学习简单的左右差异，需要在计算 PSD 和 FC 之前进行 EEG 左右翻转，使所有患者的患侧相关半球统一到同一侧。

### 6.2 推荐统一方向

建议统一为：

```text
将所有患者统一到“患手右侧 / 刺激 C3 / 左半球为刺激侧”的空间表示。
```

也就是说：

- 患手右侧患者：不翻转；
- 患手左侧患者：左右通道翻转，使 C4 对应到 C3 的统一位置。

### 6.3 通道翻转映射

需要根据实际 64 通道 montage 建立完整左右镜像表。

常见示例：

| 左侧 | 右侧 |
|---|---|
| Fp1 | Fp2 |
| AF3 | AF4 |
| F7 | F8 |
| F5 | F6 |
| F3 | F4 |
| F1 | F2 |
| FC5 | FC6 |
| FC3 | FC4 |
| FC1 | FC2 |
| T7 | T8 |
| C5 | C6 |
| C3 | C4 |
| C1 | C2 |
| CP5 | CP6 |
| CP3 | CP4 |
| CP1 | CP2 |
| P7 | P8 |
| P5 | P6 |
| P3 | P4 |
| P1 | P2 |
| PO7 | PO8 |
| PO5 | PO6 |
| PO3 | PO4 |
| O1 | O2 |

中线通道不变：

```text
Fpz, AFz, Fz, FCz, Cz, CPz, Pz, POz, Oz
```

注意：

- 以上映射需根据实际 EEG 通道命名校验。
- 翻转应在 PSD 与 FC 计算之前完成。
- FC 计算后不建议再翻转，因为连接边编号会变复杂且容易出错。

---

## 7. EEG 特征计算

## 7.1 PSD 特征

### 7.1.1 单状态 PSD

每个状态单独计算 PSD：

```text
PSD_EO: 64 x 90
PSD_EC: 64 x 90
```

其中：

- 64：EEG 通道数
- 90：频率 bin 数
- 频率范围：0.5–45 Hz
- 频率分辨率：0.5 Hz

### 7.1.2 频段定义

与参考文献保持一致，定义 6 个频段：

| 频段 | 范围 |
|---|---|
| Delta | 1–3 Hz |
| Theta | 4–7 Hz |
| Alpha | 8–13 Hz |
| Beta Low | 13–18 Hz |
| Beta Medium | 18–21 Hz |
| Beta High | 21–30 Hz |

由于 tACS 频率为 20 Hz，应重点关注：

```text
Beta Medium: 18–21 Hz
```

### 7.1.3 PSD 输入形式

主模型采用双状态输入：

```text
PSD_EO: 64 x 90
PSD_EC: 64 x 90
```

同时保留一个对照开发方案：

```text
PSD_3D: 2 x 64 x 90
```

用于测试早期状态融合是否优于双状态共享 encoder。

---

## 7.2 功能连接 FC 特征

### 7.2.1 FC 指标

需要尝试两种功能连接指标：

1. wPLI（weighted Phase Lag Index）
2. imaginary coherence

原因：

- 二者均可一定程度减少体积传导影响；
- 适合 EEG 静息态脑网络分析；
- 可比较两种 FC 指标对预测性能和解释稳定性的影响。

### 7.2.2 单状态 FC 维度

64 通道两两连接数：

```text
64 * 63 / 2 = 2016
```

每个状态、每种 FC 指标得到：

```text
FC_EO: 2016 x 6
FC_EC: 2016 x 6
```

其中：

- 2016：通道对数量；
- 6：频段数量。

### 7.2.3 FC 输入形式

若使用 wPLI：

```text
wPLI_EO: 2016 x 6
wPLI_EC: 2016 x 6
```

若使用 imaginary coherence：

```text
iCoh_EO: 2016 x 6
iCoh_EC: 2016 x 6
```

若两者共同输入，可表示为：

```text
FC_all: 2 x 2 x 2016 x 6
```

维度含义：

```text
state x metric x edge x band
```

建议开发时分三种实验：

| 实验 | FC 指标 |
|---|---|
| FC-wPLI | 只使用 wPLI |
| FC-iCoh | 只使用 imaginary coherence |
| FC-both | wPLI + imaginary coherence |

---

## 8. 临床特征输入

扩展模型允许加入基线临床变量。

建议候选变量：

- 年龄；
- 性别；
- 病程；
- 患手侧；
- FMA-UE_pre；
- MBI_pre；
- 其他可用基线临床量表。

注意：

- 临床变量应只使用治疗前基线信息。
- 不应加入治疗后信息或由治疗后信息计算得到的变量。
- 分类标签由 FMA-UE_pre 和 FMA-UE_post 计算，但模型输入不能包含 FMA-UE_post。

---

## 9. 模型路线总览

本项目建议实现两条深度学习路线：

1. 主模型：双状态输入 + 共享权重 encoder
2. 对照模型：3D 状态融合输入

两者都需要进行开发和比较。

---

# 10. 主模型：双状态输入 + 共享权重 encoder

## 10.1 设计理由

睁眼和闭眼作为两个独立输入，可以提升解释性：

- 可分别查看 EO 和 EC 的关键通道；
- 可分别查看 EO 和 EC 的关键频段；
- 可比较 EO 与 EC 对预测的贡献；
- 与消融实验“睁眼 vs 闭眼 vs 两状态融合”自然对应。

为了避免小样本中过多参数导致过拟合，EO 和 EC 通过同一个 encoder 提取特征，即共享权重。

---

## 10.2 PSD 分支

输入：

```text
PSD_EO: 64 x 90
PSD_EC: 64 x 90
```

共享 encoder：

```text
Shared_PSD_Encoder
```

计算：

```text
z_PSD_EO = Shared_PSD_Encoder(PSD_EO)
z_PSD_EC = Shared_PSD_Encoder(PSD_EC)
```

推荐 encoder 结构：

```text
Input 64 x 90
Conv2D -> BatchNorm -> ReLU -> Dropout
Conv2D -> BatchNorm -> ReLU -> Dropout
Conv2D -> BatchNorm -> ReLU
GlobalAveragePooling
Dense
Output embedding
```

---

## 10.3 FC 分支

输入：

```text
FC_EO: 2016 x 6
FC_EC: 2016 x 6
```

共享 encoder：

```text
Shared_FC_Encoder
```

计算：

```text
z_FC_EO = Shared_FC_Encoder(FC_EO)
z_FC_EC = Shared_FC_Encoder(FC_EC)
```

推荐 encoder 结构：

```text
Input 2016 x 6
Conv1D -> BatchNorm -> ReLU -> Dropout
Conv1D -> BatchNorm -> ReLU -> Dropout
GlobalAveragePooling1D
Dense
Output embedding
```

说明：

- FC 可以将 edge 维度视作序列维度；
- 频段作为 channel 维度；
- 也可将 FC reshape 为 `2016 x 6 x 1` 用 Conv2D，但 Conv1D 更轻量。

---

## 10.4 状态融合模块

对于 PSD：

```text
z_PSD = StateFusion(z_PSD_EO, z_PSD_EC)
```

对于 FC：

```text
z_FC = StateFusion(z_FC_EO, z_FC_EC)
```

建议实现两种融合方式：

### 10.4.1 Concatenation Fusion

```text
z_PSD = concat(z_PSD_EO, z_PSD_EC)
z_FC  = concat(z_FC_EO, z_FC_EC)
```

优点：

- 简单；
- 稳定；
- 适合 baseline deep learning。

### 10.4.2 Gated / Attention Fusion

```text
w_EO, w_EC = softmax(MLP([z_EO, z_EC]))
z = w_EO * z_EO + w_EC * z_EC
```

优点：

- 可以输出状态级权重；
- 可解释 EO 和 EC 哪个贡献更大；
- 适合论文展示。

建议：

- 默认先实现 concat；
- 再实现 gated fusion；
- 若 gated fusion 性能相近或更好，则作为主模型；
- 若 gated fusion 不稳定，则用 concat 作为主模型，gated 权重作为补充分析。

---

## 10.5 Clinical 分支

输入：

```text
clinical_features: n_features
```

结构：

```text
Dense -> BatchNorm -> ReLU -> Dropout
Dense -> ReLU
Output clinical embedding
```

---

## 10.6 最终融合分类器

融合：

```text
z_all = concat(z_PSD, z_FC, z_clinical)
```

分类器：

```text
Dense -> ReLU -> Dropout
Dense -> ReLU -> Dropout
Dense -> Sigmoid
```

输出：

```text
P(Proportional Recovery)
```

损失函数：

```text
Binary Cross Entropy
```

---

# 11. 对照模型：3D 状态融合输入

## 11.1 设计目的

虽然双状态输入解释性更好，但 3D 融合模型可能在早期卷积阶段自动学习 EO/EC 交互模式，从而获得更好的性能。因此开发时需要保留 3D 融合模型作为对照。

---

## 11.2 PSD 3D 输入

输入：

```text
PSD_3D: 2 x 64 x 90
```

可以实现为：

```text
channels/state dimension = 2
height = 64
width = 90
```

根据框架选择输入格式：

PyTorch 推荐：

```text
batch x 2 x 64 x 90
```

TensorFlow/Keras 推荐：

```text
batch x 64 x 90 x 2
```

结构：

```text
Conv2D -> BatchNorm -> ReLU -> Dropout
Conv2D -> BatchNorm -> ReLU -> Dropout
Conv2D -> BatchNorm -> ReLU
GlobalAveragePooling
Dense
```

---

## 11.3 FC 3D 输入

单一 FC 指标时：

```text
FC_3D: 2 x 2016 x 6
```

可实现为：

PyTorch：

```text
batch x 2 x 2016 x 6
```

TensorFlow/Keras：

```text
batch x 2016 x 6 x 2
```

如果 wPLI 和 iCoh 同时输入：

```text
FC_4D: 2 x 2 x 2016 x 6
```

建议为了简化，优先做：

```text
wPLI_3D: 2 x 2016 x 6
iCoh_3D: 2 x 2016 x 6
```

分别训练模型，再尝试双指标融合。

---

## 11.4 3D 融合模型的解释性

3D 输入仍可做解释性分析。

PSD attribution 输出：

```text
A_PSD: 2 x 64 x 90
```

其中：

- `A_PSD[0, :, :]` 为睁眼重要性；
- `A_PSD[1, :, :]` 为闭眼重要性。

FC attribution 输出：

```text
A_FC: 2 x 2016 x 6
```

其中：

- `A_FC[0, :, :]` 为睁眼连接重要性；
- `A_FC[1, :, :]` 为闭眼连接重要性。

注意：

- 3D 模型的早期卷积可能混合 EO 和 EC 信息；
- 因此其状态级解释性不如双输入共享 encoder 清晰；
- 但仍可作为重要对照模型。

---

# 12. 自监督学习设计

## 12.1 目标

利用无标签 EEG 学习可迁移 EEG 表征，缓解 19 例有标签样本导致的过拟合问题。

## 12.2 自监督输入

自监督输入可以使用：

```text
EO EEG segments
EC EEG segments
```

预训练输入可以是：

- 原始时间序列片段；
- PSD 矩阵；
- 或时频图。

推荐优先使用原始时间序列或时频图进行自监督预训练，之后迁移 encoder 到下游预测任务。

如果开发周期有限，可以先在 PSD 输入上做自监督对比学习。

---

## 12.3 推荐自监督任务

### 12.3.1 Contrastive Learning

正样本：

- 同一受试者；
- 同一状态；
- 不同增强片段。

负样本：

- 不同受试者；
- 或不同状态/不同片段。

增强方式：

- 时间裁剪；
- 高斯噪声；
- 幅值缩放；
- 通道 dropout；
- 频带遮挡；
- 时间遮挡。

损失：

```text
NT-Xent loss
```

### 12.3.2 Masked EEG Reconstruction

随机遮挡：

- 时间片段；
- 通道；
- 频段；
- 时频 patch。

模型任务：

```text
重建被遮挡部分
```

损失：

```text
MSE loss
```

### 12.3.3 Combined SSL Loss

可组合：

```text
L_SSL = L_contrastive + lambda * L_reconstruction
```

建议初始：

```text
lambda = 0.5 或 1.0
```

---

## 12.4 微调策略

预训练后迁移 encoder 到 19 例有标签患者。

推荐两阶段微调：

1. 冻结 encoder，仅训练分类头；
2. 解冻 encoder 最后若干层，用较小学习率微调。

建议学习率：

```text
classifier lr = 1e-3
encoder fine-tune lr = 1e-4 或 1e-5
```

---

# 13. 动态学习率策略

## 13.1 自监督预训练阶段

推荐：

```text
Warmup + Cosine Annealing
```

示例：

```text
warmup_epochs = 5
max_lr = 1e-3
min_lr = 1e-6
```

## 13.2 监督训练 / 微调阶段

推荐：

```text
ReduceLROnPlateau
```

示例：

```text
initial_lr = 1e-3
factor = 0.5
patience = 10
min_lr = 1e-6
monitor = validation_loss
```

## 13.3 Early Stopping

由于样本较小，必须使用 early stopping。

推荐：

```text
patience = 20
restore_best_weights = True
monitor = validation_loss
```

---

# 14. 传统机器学习 baseline

## 14.1 目的

为深度学习模型提供对比基线，证明深度模型和自监督预训练的增益。

## 14.2 输入特征

传统 ML 不直接输入完整高维矩阵，应进行特征压缩。

### PSD 特征

将 PSD 转换为通道级频段功率：

```text
state x channel x band = 2 x 64 x 6 = 768 features
```

### FC 特征

原始 FC：

```text
state x edge x band = 2 x 2016 x 6 = 24192 features
```

维度过高，必须降维或特征选择。

推荐方法：

1. 每个训练折内进行单变量特征筛选；
2. 或 PCA；
3. 或保留与运动区相关的边；
4. 或每个频段保留 top-k 边。

重要：

```text
所有特征选择、PCA、标准化都必须只在训练折内 fit，再应用到测试折。
```

不能在 19 例全体样本上提前做特征筛选，否则会数据泄漏。

## 14.3 Baseline 模型

建议实现：

| 模型 | 说明 |
|---|---|
| Logistic Regression L1/L2 | 线性可解释 baseline |
| SVM Linear | 小样本线性 baseline |
| SVM RBF | 小样本非线性强 baseline |
| Random Forest | 非线性树模型 |
| XGBoost 或 LightGBM | 表格特征强 baseline |
| kNN | 简单非参数 baseline |
| Gaussian Naive Bayes | 简单概率 baseline |

建议重点报告：

- Logistic Regression；
- SVM-RBF；
- Random Forest；
- XGBoost/LightGBM；
- Deep learning without SSL；
- Deep learning with SSL。

---

# 15. 训练与验证方案

## 15.1 主验证方法

使用：

```text
Leave-One-Subject-Out Cross-Validation, LOSO-CV
```

流程：

```text
for each subject i in 19 labeled patients:
    test_subject = subject_i
    train_subjects = all other 18 subjects

    fit scaler / feature selector / PCA only on train_subjects
    train model on train_subjects
    predict test_subject
```

## 15.2 患者级预测

EEG 片段预测后，需要聚合为患者级结果。

流程：

```text
segment_probs = model(all_segments_of_test_subject)
subject_prob = mean(segment_probs)
subject_label_pred = subject_prob >= 0.5
```

也可测试：

- mean probability；
- median probability；
- majority voting。

主结果建议使用：

```text
mean probability
```

## 15.3 类别不平衡处理

类别大约 9 vs 10，基本平衡。可以不使用复杂重采样。

可选：

```text
class_weight = inverse_class_frequency
```

---

# 16. 评价指标

需要报告：

- Accuracy
- Balanced Accuracy
- Sensitivity
- Specificity
- Precision
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Bootstrap 95% CI
- Permutation test p-value

由于样本量很小，重点报告：

```text
Balanced Accuracy + ROC-AUC + Bootstrap 95% CI
```

---

# 17. 消融实验设计

## 17.1 状态消融

| 实验 | 输入 |
|---|---|
| EO only | 只用睁眼 |
| EC only | 只用闭眼 |
| EO + EC 双输入共享 encoder | 主模型 |
| EO + EC 3D fusion | 对照模型 |

## 17.2 特征消融

| 实验 | 输入 |
|---|---|
| PSD only | PSD_EO + PSD_EC |
| FC only | FC_EO + FC_EC |
| PSD + FC | EEG-only 主模型 |
| Clinical only | 只用基线临床变量 |
| EEG + Clinical | 临床扩展模型 |

## 17.3 FC 指标消融

| 实验 | 输入 |
|---|---|
| wPLI only | wPLI_EO + wPLI_EC |
| imaginary coherence only | iCoh_EO + iCoh_EC |
| wPLI + iCoh | 双 FC 指标融合 |

## 17.4 自监督消融

| 实验 | 设置 |
|---|---|
| Without SSL | 随机初始化 encoder |
| With SSL | 自监督预训练 encoder 后微调 |
| Frozen encoder | 只训练分类头 |
| Fine-tuned encoder | 解冻最后几层微调 |

---

# 18. 可解释性分析

本项目采用三层解释。

---

## 18.1 第一层：PSD 通道-频率解释

对象：

```text
PSD_EO
PSD_EC
```

方法：

- Integrated Gradients
- Gradient SHAP
- Grad-CAM
- Permutation importance

输出：

- EO 关键通道；
- EC 关键通道；
- EO 关键频段；
- EC 关键频段；
- EO vs EC 总贡献；
- channel-frequency heatmap；
- scalp topography。

重点关注：

- C3/C4；
- FC3/FC4；
- C1/C2；
- CP3/CP4；
- Alpha；
- Beta；
- Beta Medium 18–21 Hz；
- 20 Hz 附近功率。

---

## 18.2 第二层：FC 网络解释

对象：

```text
FC_EO
FC_EC
```

分别对 wPLI 和 imaginary coherence 进行解释。

方法：

- SHAP / Gradient SHAP
- edge masking
- permutation edge importance

输出：

- EO 重要连接；
- EC 重要连接；
- 重要频段；
- 重要跨半球连接；
- 患侧 M1 与健侧 M1 连接；
- 患侧 M1 与额叶、顶叶、中央区连接。

重点关注：

- 感觉运动网络；
- 双侧 M1 连接；
- M1-顶叶连接；
- M1-额叶连接；
- Beta 频段连接。

---

## 18.3 第三层：临床相关验证

将解释得到的关键 EEG 特征与临床恢复指标相关。

临床指标：

- Delta_FMA_obs；
- Residual；
- FMA-UE_pre；
- FMA-UE_post；
- MBI_pre；
- 病程。

统计方法：

- Spearman 相关；
- Pearson 相关；
- Mann–Whitney U test；
- t-test；
- FDR 校正；
- effect size。

重点验证：

```text
关键 EEG 特征是否与 Residual 或 Delta_FMA_obs 显著相关。
```

---

# 19. 代码开发建议结构

推荐项目目录：

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── metadata/
│
├── configs/
│   ├── config_ssl.yaml
│   ├── config_supervised.yaml
│   ├── config_baseline.yaml
│   └── channel_mapping.yaml
│
├── src/
│   ├── preprocessing/
│   │   ├── hemisphere_flip.py
│   │   ├── epoching.py
│   │   └── normalization.py
│   │
│   ├── features/
│   │   ├── psd.py
│   │   ├── connectivity_wpli.py
│   │   ├── connectivity_icoh.py
│   │   └── feature_utils.py
│   │
│   ├── models/
│   │   ├── psd_encoder.py
│   │   ├── fc_encoder.py
│   │   ├── dual_state_model.py
│   │   ├── fusion_3d_model.py
│   │   ├── ssl_model.py
│   │   └── clinical_mlp.py
│   │
│   ├── training/
│   │   ├── train_ssl.py
│   │   ├── train_supervised_loso.py
│   │   ├── train_baselines.py
│   │   ├── lr_schedulers.py
│   │   └── metrics.py
│   │
│   ├── explainability/
│   │   ├── explain_psd.py
│   │   ├── explain_fc.py
│   │   ├── shap_utils.py
│   │   ├── integrated_gradients.py
│   │   └── visualization.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── io.py
│
├── scripts/
│   ├── 01_prepare_metadata.py
│   ├── 02_compute_psd.py
│   ├── 03_compute_fc.py
│   ├── 04_train_ssl.py
│   ├── 05_train_supervised_loso.py
│   ├── 06_train_ml_baselines.py
│   └── 07_run_explainability.py
│
├── results/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   └── explainability/
│
└── README.md
```

---

# 20. 开发优先级

## Phase 1：数据与标签

1. 读取患者信息表；
2. 计算 `Delta_FMA_pred`；
3. 计算 `Delta_FMA_obs`；
4. 计算 `Residual`；
5. 以 residual 中位数打标签；
6. 排除 2 例天花板效应患者；
7. 固定 19 例监督训练患者列表。

## Phase 2：EEG 特征

1. 建立 64 通道左右翻转映射；
2. 对患手左侧患者进行 EEG 左右翻转；
3. 分别提取 EO 和 EC；
4. 计算 PSD_EO 和 PSD_EC；
5. 计算 wPLI_EO 和 wPLI_EC；
6. 计算 iCoh_EO 和 iCoh_EC；
7. 保存患者级 feature 文件。

## Phase 3：传统 ML baseline

1. 构建 PSD band power 特征；
2. 构建 FC 降维特征；
3. 实现 LOSO-CV；
4. 训练 Logistic Regression、SVM、RF、XGBoost；
5. 输出患者级预测和 metrics。

## Phase 4：深度学习模型

1. 实现双状态共享 encoder 模型；
2. 实现 3D fusion 对照模型；
3. 实现 clinical 分支；
4. 实现动态学习率；
5. 实现 LOSO-CV；
6. 输出 metrics。

## Phase 5：自监督学习

1. 实现 EEG 数据增强；
2. 实现 contrastive learning；
3. 实现 masked reconstruction；
4. 预训练 encoder；
5. 微调比例恢复分类器；
6. 与无 SSL 模型对比。

## Phase 6：解释性分析

1. PSD 重要性；
2. FC 重要性；
3. EO vs EC 状态贡献；
4. 关键通道 topography；
5. 关键连接图；
6. 与临床指标相关分析。

---

# 21. 推荐最终模型比较表

最终论文建议报告以下模型：

| 模型编号 | 模型 | 输入 | SSL | 说明 |
|---|---|---|---|---|
| M1 | Logistic Regression | PSD + FC + Clinical | No | 线性 baseline |
| M2 | SVM-RBF | PSD + FC + Clinical | No | 小样本 baseline |
| M3 | Random Forest | PSD + FC + Clinical | No | 非线性 baseline |
| M4 | XGBoost/LightGBM | PSD + FC + Clinical | No | 表格强 baseline |
| M5 | Dual-state DL | PSD + FC | No | EEG-only 深度模型 |
| M6 | Dual-state DL | PSD + FC + Clinical | No | 临床扩展深度模型 |
| M7 | Dual-state DL | PSD + FC + Clinical | Yes | 主创新模型 |
| M8 | 3D-fusion DL | PSD + FC + Clinical | Yes/No | 状态融合对照模型 |

---

# 22. 推荐输出文件

开发完成后建议保存：

```text
results/predictions/loso_predictions.csv
results/metrics/model_comparison.csv
results/metrics/bootstrap_ci.csv
results/metrics/permutation_test.csv
results/figures/roc_curves.png
results/figures/confusion_matrix.png
results/explainability/psd_topomap_eo.png
results/explainability/psd_topomap_ec.png
results/explainability/fc_network_eo.png
results/explainability/fc_network_ec.png
results/explainability/state_importance.csv
```

---

# 23. 关键注意事项

1. 所有模型划分必须以患者为单位。
2. 严禁片段级随机划分造成数据泄漏。
3. 标准化、PCA、特征选择必须只在训练折内 fit。
4. 左右翻转必须在 PSD 和 FC 计算之前完成。
5. 主标签使用 proportional recovery residual，不使用 `Delta_FMA >= 5`。
6. 主模型使用 EO 和 EC 双输入共享 encoder。
7. 必须实现 3D fusion 对照模型，因为它可能性能更好。
8. wPLI 和 imaginary coherence 都需要测试。
9. 监督训练患者数只有 19 例，模型必须轻量化。
10. 结果报告必须包含置信区间和 permutation test。
11. 解释性结果必须分别展示 EO 和 EC。
12. tACS 20 Hz 与 Beta Medium 18–21 Hz 的关系是论文解释重点。

---

# 24. 一句话项目摘要

本项目将基于治疗前睁眼与闭眼 64 通道 EEG，提取功率谱密度和功能连接特征，并结合基线临床信息，构建自监督预训练增强的多分支深度学习模型，用于预测卒中上肢障碍患者在接受 14 次 20 Hz、1000 μA、M1-tACS 治疗后是否达到 FMA-UE 比例恢复；同时通过 PSD 通道-频率解释、FC 网络解释和临床相关验证，识别与 tACS 疗效相关的 EEG biomarker。
