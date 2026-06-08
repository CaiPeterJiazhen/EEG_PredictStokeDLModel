# 基线 EEG 预测 tACS 后卒中上肢比例恢复的论文大纲

## 拟定题目

**Residual-aware self-supervised EEG learning predicts post-tACS proportional upper-limb recovery after stroke**

中文工作题目：**基于残差感知自监督 EEG 表征学习预测卒中患者 tACS 后上肢比例恢复**

## 中心论点

本研究只回答一个核心问题：**治疗前基线静息态 EEG 能否预测卒中患者接受 tACS 后的上肢恢复结局属于比例恢复组还是恢复不良组。**

论文主线不以临床变量作为模型输入，也不设置 clinical-only 或 clinical + EEG 增量模型作为正式结果。临床变量只用于描述队列、定义 FMA-UE 比例恢复残差，以及说明受试者基线状态。

核心叙事顺序：

1. 传统 EEG-ML 在小样本、高维 PSD/WPLI 输入下区分能力有限。
2. CNN 可以保留完整 PSD channel-frequency 矩阵和 WPLI connectivity 矩阵，比展平后筛选特征更适合 EEG 结构化输入。
3. SSL 利用监督队列外本来不能进入二分类建模的 EEG 数据学习更稳定的表征，从而扩大可用于表征学习的样本量；但单独 SSL 不应被夸大为稳定提升性能。
4. residual-aware training 将连续恢复残差信息纳入训练，比只使用二分类标签更符合比例恢复问题。
5. 最终比较应以跨 seed 的 mean accuracy、min accuracy、accuracy SD、ROC-AUC、PR-AUC、Brier score 为主，不把 `seedmean10` 作为临床主指标。
6. 可解释性分析用 PSD topomap 和 WPLI connectivity 图定位通道、频段和连接边，作为机制假设而不是因果证明。

## 图表总安排

Introduction 不放图。三张技术框架图全部放在 **Materials and Methods** 中，并分别对应方法小标题。

主图建议 6 张：

- **Figure 1 Overall framework**：总体技术框架图。放在 Materials and Methods 的 **Study Design** 或 **Workflow** 小标题下。内容包括患者流程、supervised labelled cohort、unlabeled/self-supervised EEG pool、baseline EEG、tACS、FMA-UE 结局、标签定义、特征构建、模型比较和 patient-level validation。
- **Figure 2 SSL framework**：自监督学习框架图。放在 **Self-supervised Learning** 小标题下。内容包括监督队列外 EEG 数据如何进入 unlabeled EEG pool、two augmented views、shared encoder、projection head、cross-correlation matrix、Barlow-style SSL loss。
- **Figure 3 CNN and residual-aware architecture**：CNN 结构 + residual-aware training 图。放在 **CNN and Residual-Aware Learning** 小标题下。内容包括 EO/EC、PSD/WPLI 多分支输入、gated fusion、binary classification head、residual regression/signed-distance head、ranking/soft-label auxiliary heads。
- **Figure 4 Performance**：模型性能图。放在 Results。内容包括多种传统 ML、CNN/SSL/residual-aware 模型的主指标、ROC curve、PR curve、calibration curve、confusion matrix。
- **Figure 5 Stability and Ablation**：seed 稳定性与消融图。放在 Results。内容包括 10-seed accuracy 分布、min/std、SSL 消融、residual-aware 消融、特征/状态/频段消融。
- **Figure 6 Explainability**：可解释性图。放在 Results。内容包括 branch/state occlusion、PSD heatmap、MNE topomap、WPLI connectivity/connectome。

主表建议 5 张：

- **Table 1 Cohort**：患者基线特征与比例恢复/恢复不良分组统计。
- **Table 2 ML Baselines**：多个传统机器学习模型结果。
- **Table 3 Deep Models**：no-SSL CNN、SSL-CNN、no-SSL + residual-aware CNN、SSL + residual-aware CNN 的跨 seed 主结果。
- **Table 4 Ablations**：SSL、residual-aware、PSD/WPLI、EO/EC、频段和连接特征消融。
- **Table 5 Biomarkers**：关键 PSD 通道/频段和 WPLI 连接边解释性特征。

---

## 1. Introduction

建议长度：英文约 **900 words**；中文初稿约 1,500-1,800 字。Introduction 中不放任何图片。

### 1.1 Stroke Recovery

要写的内容：

- 卒中后上肢恢复存在显著个体差异，即使接受相同康复或神经调控方案，恢复轨迹也可能不同。
- FMA-UE 是常用的上肢运动功能指标，比例恢复框架用 baseline impairment 与 expected improvement 描述恢复潜力。
- tACS 可调节运动皮层节律活动，但临床上仍缺少治疗前预测工具来识别可能达到比例恢复的患者。
- 引出本文问题：是否可以用治疗前 baseline EEG 预测 tACS 后的比例恢复类别。

### 1.2 EEG Biomarkers

要写的内容：

- EEG 具有无创、低成本、可重复采集、适合康复场景的优势。
- PSD 反映频段功率，WPLI/FC 反映脑网络连接，二者都可能与卒中后运动网络状态和恢复潜力相关。
- EO 和 EC 静息态可以提供不同状态下的神经生理信息。
- 说明本研究关注完整 EEG 结构化特征，而不是单一通道、单一频段或临床量表。

### 1.3 Method Gap

要写的内容：

- 传统 ML 通常需要展平高维 EEG 特征并进行筛选，容易丢失 channel-frequency 和 connectivity structure。
- 深度学习可以利用完整矩阵结构，但小样本会导致 seed 不稳定和过拟合。
- 卒中 tACS 队列中常有仅完成部分随访或仅有部分时间点 EEG 的患者，这些数据不能用于有标签监督训练，却仍可为自监督表征学习提供 baseline EEG 信息。
- 二分类比例恢复标签便于临床解释，但 residual 的连续信息会被丢弃。
- 因此需要结合自监督表征学习和 residual-aware training 的 EEG 预测框架。

### 1.4 Present Study

要写的内容：

- 本研究开发 residual-aware SSL-CNN，用 baseline EO/EC EEG 的 PSD 和 WPLI 预测 tACS 后比例恢复分组。
- 自监督学习用于利用监督标签之外的 EEG 记录，包括只完成基线记录或未形成完整比例恢复标签的患者数据，从而增加表征学习阶段的可用样本量。
- 系统比较多个传统 ML、no-SSL CNN、SSL-CNN、no-SSL + residual-aware CNN 和 SSL + residual-aware CNN。
- 使用 patient-level LOSO、10-seed 稳定性、ROC-AUC、PR-AUC、Brier score、bootstrap/permutation 和 confusion matrix 报告性能。
- 通过 topomap 和 connectivity 解释模型依赖的 EEG 特征。
- 结尾要限定：这是小样本内部验证研究，目标是建立可验证的预测框架和 EEG biomarker 假设，而不是立即临床部署。

---

## 2. Materials and Methods

建议长度：英文 1,900-2,300 words；中文初稿约 3,500-4,500 字。

### 2.1 Study Design

要写的内容：

- 单中心卒中 tACS 康复预测建模研究。
- 数据流程：baseline EEG acquisition -> tACS intervention -> FMA-UE and MBI assessment after 14 treatment sessions -> residual-defined label -> EEG feature construction -> ML/CNN model comparison -> patient-level validation。
- 说明所有模型输入均来自治疗前 baseline EEG。
- 说明 clinical variables 不作为本文主要模型输入。
- 伦理批准号和知情同意在正式稿中写为代填格式：`This study was approved by [IRB name and approval number to be inserted]. Written informed consent was obtained from all participants or their legally authorized representatives.` 中文稿写为：`本研究经[伦理委员会名称及批准号待填]批准，所有患者或其法定代理人均签署知情同意书。`

图表：

- **Figure 1 Overall framework**：总体技术框架图，应放在本小节。

### 2.2 Participants

要写的内容：

- 数据来源：M1 临床工作簿 29 例，EEG-indexed 28 例，最终监督建模 19 例。
- 纳入：卒中后上肢障碍，接受 tACS，治疗前 EEG，治疗前后 FMA-UE 完整。
- 未进入监督训练原因：未做完整实验或无法形成完整比例恢复监督标签。可公开分为只具有基线数据、具有基线和即时数据、具有基线/即时/最终数据但未进入当前监督标签分析等类型；具体逐例原因以项目统计表记录为准。
- 监督队列外患者的 baseline EEG 仍可进入自监督学习阶段，用于扩大无标签 EEG 表征学习样本量；但不能被当作有标签监督样本。
- 分组：proportional recovery 10 例，poor recovery 9 例。
- 患者医院均知情，数据可用于本研究；正式投稿时伦理批准号按代填占位补入。

统计分析：

- **Table 1**：All、Proportional recovery、Poor recovery 三列。
- 连续变量：mean (SD) 或 median (IQR)。
- 分类变量：n (%)。
- 小样本下建议报告 standardized mean difference，p 值只作描述。

### 2.3 Outcome Labels

要写的内容：

- 主结局为 FMA-UE 比例恢复状态。
- 标签定义必须写在 Methods，而不是只写在 Results。
- 公式写清楚：

```text
Expected improvement_i = 0.7 x (66 - FMA_pre_i)

Observed improvement_i = FMA_post_i - FMA_pre_i

Residual_i = Expected improvement_i - Observed improvement_i

tau = median(Residual_i) in the supervised cohort

y_i = 1, if Residual_i <= tau
y_i = 0, if Residual_i > tau
```

- `y_i = 1` 表示 proportional recovery，`y_i = 0` 表示 poor recovery。
- 可再定义 residual-aware training 使用的 signed distance：

```text
d_i = tau - Residual_i
```

- `d_i > 0` 表示更接近或超过比例恢复；`d_i < 0` 表示低于比例恢复。

图表：

- **Supplementary Figure S1**：Residual 分布和 median threshold。

### 2.4 tACS Protocol

要写的内容：

- 患手对侧 M1 运动皮层刺激。
- 右手受累刺激 C3，左手受累刺激 C4。
- tACS 设备：Neuroscan 1x1 经颅电刺激器 **DC-STIMULATOR PLUS**。
- 20 Hz，1000 microampere，20 min/session，每日 1 次，连续 14 天。
- 阻抗控制：刺激阻抗低于 30 kOhm。
- FMA-UE 和 MBI 在治疗前和 14 次 tACS 治疗结束后评估。
- 评估者和盲法信息在正式稿中代填：`FMA-UE and MBI were assessed by [assessor qualification/blinding status to be inserted] before treatment and after completion of 14 tACS sessions.`
- 若可补充：电极大小和安全性记录。

### 2.5 EEG Preprocessing

要写的内容：

- EEG 采集使用 Compumedics Neuroscan SynAmps2 64 通道 EEG 系统，采用标准 10-20 电极布局。
- 使用治疗前 baseline 静息态 EEG，包括 eyes-open (EO) 和 eyes-closed (EC)。
- EEG 预处理使用 EEGLAB。
- 参考方式：average reference。
- 滤波：high-pass 0.5 Hz，low-pass 45 Hz。
- 工频干扰：去除 50 Hz line noise。
- ICA：使用 EEGLAB 进行 ICA。
- 坏道和伪迹：使用 EEGLAB 进行人工手动剔除。
- 当前项目分析文件为预处理后的 EEGLAB `.set/.fdt` 文件；如分析时保留 62 通道，需要说明 64 通道采集后 M1/M2 或其他参考通道在特征分析中未作为 EEG 通道输入。
- 受累侧对齐：左手受累患者进行左右镜像，使 EEG 表征统一到刺激侧/受累侧约定，避免模型学习简单左右差异。

图表：

- 可在 **Figure 1 Overall framework** 中简要显示 preprocessing 和 affected-side alignment；本小节不需要单独主图。

### 2.6 EEG Features

要写的内容：

- PSD：Welch 方法，0.5-45 Hz，按 channel-frequency 构成矩阵。
- WPLI：按通道连接边和频段构成 connectivity 特征，减少体积传导导致的零相位连接偏差。
- 频段：delta、theta、alpha、beta-low、beta-medium、beta-high。
- 特征分别按 EO 和 EC 构建。
- 输入为完整 EEG PSD/WPLI 结构，不加入 clinical variables。

### 2.7 Machine Learning

要写的内容：

- 本小节单独描述 traditional EEG-ML。
- 输入：展平后的 PSD+WPLI EO+EC 特征。
- 模型：至少包括 Logistic L1、Logistic L2、Logistic L2 + SelectK=100、SVM RBF；若结果表中保留 RF、kNN、decision tree、naive Bayes 等，也应一起列入 Table 2。
- 特征选择必须在 LOSO 训练折内完成，不能使用测试受试者信息。
- 目的：作为传统 EEG-ML baseline，展示高维 EEG 展平特征在小样本下的性能边界。

图表：

- 不放技术框架图。
- 结果放 **Table 2 ML Baselines** 和 **Figure 4 Performance**。

### 2.8 Self-supervised Learning

要写的内容：

- 本小节单独描述 SSL。
- 说明 SSL 使用未标注 EEG pool 学习 patient-level EEG representation，不使用 recovery label。
- 需要强调 SSL 的临床数据价值：监督学习只能使用具有完整治疗前后 FMA-UE 且能计算 residual 标签的 19 例；自监督学习可利用本来不能进入监督训练的 EEG 数据，包括只具有基线数据、具有基线和即时数据、具有基线/即时/最终数据但未形成当前监督标签的患者记录。这使模型在不泄漏 recovery label 的前提下增加表征学习阶段的样本量。
- 说明 two-view augmentation：从同一患者 EEG 特征生成两个增强视图，输入共享 encoder。
- 使用 Barlow-style redundancy-reduction objective。建议写入公式：

```text
z_i^A = g(f(x_i^A)),  z_i^B = g(f(x_i^B))

C_jk =
sum_b z^A_bj z^B_bk /
sqrt(sum_b (z^A_bj)^2) sqrt(sum_b (z^B_bk)^2)

L_SSL = sum_j (1 - C_jj)^2 + lambda sum_j sum_{k != j} C_jk^2
```

- 解释公式含义：
  - diagonal term 让两个增强视图的同一维表征一致；
  - off-diagonal term 降低不同维度之间的冗余；
  - SSL 目标是获得更稳定的 EEG representation，而不是直接预测标签。
- 说明 pretraining 后再进行 supervised fine-tuning。

图表：

- **Figure 2 SSL framework**：必须放在本小节。

### 2.9 CNN and Residual-Aware Learning

要写的内容：

- 本小节单独描述 CNN 结构 + residual-aware。
- CNN 输入：EO/EC 两个状态下的 PSD 矩阵和 WPLI connectivity 矩阵。
- 结构：PSD branch、WPLI branch、state-specific branch、gated fusion、shared embedding、classification head。
- residual-aware heads：
  - binary classification head 预测 proportional recovery；
  - residual/signed-distance head 使用 `d_i = tau - Residual_i`；
  - pairwise ranking 或 soft-label head 使用 residual 顺序信息。
- 总损失可写成：

```text
L_total = L_BCE(y_i, p_i)
        + alpha L_residual(d_i, d_hat_i)
        + beta L_rank
        + gamma L_soft
```

- 说明最终推理只使用 binary classification head；residual-aware heads 只在训练阶段作为辅助监督，避免测试阶段使用 post-treatment 信息。
- 描述比较模型：
  - no-SSL CNN；
  - SSL-CNN；
  - no-SSL + residual-aware CNN；
  - SSL + residual-aware CNN。

图表：

- **Figure 3 CNN and residual-aware architecture**：必须放在本小节。

### 2.10 Validation

要写的内容：

- 主验证：patient-level leave-one-subject-out cross-validation。
- EEG segment 不作为独立样本，所有指标按患者级别计算。
- CNN 训练重复 10 个 random seeds。
- 主比较指标：
  - mean seed accuracy；
  - min accuracy；
  - accuracy SD；
  - ROC-AUC；
  - PR-AUC；
  - Brier score。
- 统计方法：
  - subject-level bootstrap CI；
  - paired bootstrap；
  - permutation test；
  - McNemar test；
  - confusion matrix；
  - calibration curve。
- 避免把 `seedmean10` 作为主指标。若保留 seed-average 或 ensemble prediction，只能作为补充敏感性分析或描述性 locked prediction。

### 2.11 Explainability

要写的内容：

- 本小节只写用了什么方法、做了什么分析，不写任何结果。
- 方法包括：
  - SmoothGrad integrated gradients；
  - branch/state occlusion；
  - PSD channel-frequency attribution heatmap；
  - MNE topomap；
  - WPLI edge attribution；
  - WPLI connectivity/connectome visualization；
  - seed-to-seed attribution stability。
- 说明所有解释性分析都是 model-dependent exploratory analyses，用于定位模型依赖的 EEG 特征，不用于因果推断。

### 2.12 Software

要写的内容：

- Python、PyTorch、scikit-learn、MNE-Python、EEGLAB。
- 报告遵循 TRIPOD/TRIPOD+AI 和 PROBAST 风险意识。
- 数据可用性：derived tables、locked predictions、figure source data 可以公开；原始 EEG 和可识别临床数据按伦理和隐私要求限制访问。

---

## 3. Results

建议长度：英文 1,400-1,800 words；中文初稿约 2,600-3,300 字。小标题尽量短，每节第一句先给结论，再给数据。

### 3.1 Cohort

要写的内容：

- 最终监督队列 n = 19；proportional recovery n = 10，poor recovery n = 9。
- 报告年龄、性别、病程、患侧、FMA_pre、FMA_post、Delta_FMA_obs、Residual、MBI_pre、MBI_post。
- 说明 residual median threshold 和标签分布。

图表：

- **Table 1 Cohort**。
- Supplementary Figure S1 可显示 residual threshold。

### 3.2 ML Baselines

要写的内容：

- 展示多个 traditional EEG-ML 结果，而不是只写一个 logistic baseline。
- 当前可展示的核心行包括：
  - Logistic L1：accuracy 0.737，balanced accuracy 0.733，ROC-AUC 0.711，PR-AUC 0.775，Brier 0.208；
  - Logistic L2：accuracy 0.684，balanced accuracy 0.689，ROC-AUC 0.778，PR-AUC 0.840，Brier 0.221；
  - Logistic L2 + SelectK=100：accuracy 0.684，balanced accuracy 0.678，ROC-AUC 0.767，PR-AUC 0.793，Brier 0.207；
  - SVM RBF + SelectK=100：accuracy 0.684，balanced accuracy 0.694，ROC-AUC 0.778，PR-AUC 0.771，Brier 0.219。
- 如果完整表中还有 RF、kNN、decision tree、naive Bayes 等，应一起列入 Table 2，并在正文中概括范围。
- 结果写法：传统 ML 可以产生一定排序能力，但 hard-label accuracy 和 calibration 不稳定，支持后续 CNN 结构化输入的必要性。

图表：

- **Table 2 ML Baselines**。
- **Figure 4A**：ML 与 deep models 的主性能条形图。

### 3.3 CNN and SSL

要写的内容：

- no-SSL CNN 说明 CNN 架构相对传统 ML 能更完整利用 PSD/WPLI 矩阵。
- 但 no-SSL CNN seed 波动较大：当前 10-seed mean accuracy 约 0.795，accuracy SD 约 0.072，min accuracy 约 0.632。
- SSL-CNN without residual-aware heads 可显示更低 seed 波动，但单独 SSL 不应写成稳定显著优于 no-SSL CNN。
- 正确结论：SSL 有助于 representation stability，但不是单独解决小样本预测的充分条件。

图表：

- **Table 3 Deep Models**。
- **Figure 5A**：10-seed accuracy distribution。

### 3.4 Residual-Aware

要写的内容：

- residual-aware auxiliary supervision 是本文技术贡献的核心。
- no-SSL + residual-aware CNN 和 SSL + residual-aware CNN 用连续 residual 信息训练，而最终仍输出 binary recovery probability。
- 当前最终 SSL + residual-aware CNN 的 10-seed mean accuracy 约 0.837，accuracy SD 约 0.039，min accuracy 约 0.789，ROC-AUC 约 0.860，PR-AUC 约 0.858，Brier 约 0.142。
- 结果写法：残差感知训练提高了稳定性和排序/概率质量，但小样本下仍需保守表述。

图表：

- **Table 3 Deep Models**。
- **Figure 5B-C**：SSL/residual-aware 消融。

### 3.5 Final Model

要写的内容：

- 报告最终 residual-aware SSL-CNN 的 patient-level LOSO 预测性能。
- 必须包含 confusion matrix。
- 如果使用当前 locked hard prediction，可在图中显示：

```text
TP = 10, FN = 0
FP = 3,  TN = 6
```

- 该混淆矩阵对应 sensitivity 1.000、specificity 0.667、accuracy 0.842。若最终改用跨 seed 单模型主口径，则混淆矩阵应使用预先指定 seed 或 locked prediction，并在图注中说明。
- 同时报告 ROC curve、PR curve、calibration curve 和 Brier score。
- 强调 19 例小样本下一个病例会明显影响 accuracy，因此所有性能结论需配合 bootstrap/permutation 和稳定性分析。

图表：

- **Figure 4B**：ROC curve。
- **Figure 4C**：PR curve。
- **Figure 4D**：confusion matrix。
- **Figure 4E**：calibration curve。

### 3.6 Ablations

要写的内容：

- 展示 feature/state/band 消融：
  - PSD-only；
  - WPLI-only；
  - PSD+WPLI；
  - EO-only；
  - EC-only；
  - beta-medium、beta-high、motor WPLI edges 等。
- 写法要谨慎：这些结果用于说明 EEG 成分的信息贡献，不作为独立确认性生物标志物。

图表：

- **Figure 5D**：feature/state/band ablation。
- **Table 4 Ablations**。

### 3.7 Explainability

要写的内容：

- 结果部分才写解释性结果。
- 汇报 branch/state occlusion 显示模型主要依赖哪些输入分支。
- 汇报 PSD attribution heatmap 和 MNE topomap，定位通道和频段。
- 汇报 WPLI connectivity/connectome，定位关键连接边。
- 与文献对照时只写“consistent with prior reports”，不写因果结论。

图表：

- **Figure 6 Explainability**。
- **Table 5 Biomarkers**。

---

## 4. Discussion

建议长度：英文 1,000-1,300 words；中文初稿约 1,800-2,400 字。

### 4.1 Main Finding

要写的内容：

- 基线 EEG 能为 tACS 后比例恢复分组提供预测信息。
- 传统 ML 表现有限，CNN 可利用完整 EEG 矩阵。
- residual-aware training 是本文最重要的模型设计。
- 结果应定位为 pilot evidence。

### 4.2 Residual Learning

要写的内容：

- 比例恢复本质上是连续 residual 问题，二分类只是临床表达方式。
- residual-aware heads 让模型在训练中利用 residual magnitude 和 ranking。
- 这解释为什么 residual-aware 比单纯 binary CNN 更适合本任务。

### 4.3 Structured EEG

要写的内容：

- 传统 ML 展平和筛选特征可能损失空间、频率和连接结构。
- CNN 输入完整 PSD/WPLI 矩阵有方法学优势。
- 但这种优势仍需更大样本和外部验证。

### 4.4 SSL Role

要写的内容：

- SSL 的价值是利用未标注 EEG 提供稳定表征。
- 这部分要特别强调：SSL 能把监督队列外、本来因随访不完整或标签不可用而无法用于二分类训练的 EEG 数据转化为表征学习资源，从而缓解有标签样本过少的问题。
- 当前结果不能夸大为 SSL 单独显著提升性能。
- 未来应扩大 unlabeled EEG pool，并区分 inductive 和 transductive 设置。

### 4.5 EEG Biomarkers

要写的内容：

- 讨论 PSD topography、beta-band connectivity、central/frontal/temporal network 与卒中运动恢复文献的一致性。
- 强调 explainability 是探索性、模型相关分析。
- 后续正式写作需要补充针对性文献检索：post-stroke upper-limb recovery EEG、beta rhythm、resting-state connectivity、tACS response biomarkers、WPLI。

### 4.6 Limitations

要写的内容：

- n = 19，无外部验证。
- residual threshold 是队列内 median，需要外部验证或预注册。
- 小样本 seed stability 和 bootstrap 不能替代 prospective validation。
- explainability 不证明因果机制。
- EEG 增量价值相对临床变量尚未证明；本论文主张应限定为 EEG-based prediction framework and hypothesis-generating biomarkers。

---

## 5. Conclusion

建议长度：英文 120-180 words；中文初稿约 200-300 字。

要写的内容：

- 治疗前 baseline EEG 通过 residual-aware SSL-CNN 可为 tACS 后上肢比例恢复提供 patient-level 预测信息。
- 连续 residual 监督、完整 PSD/WPLI 矩阵输入和 patient-level validation 是本文关键方法学贡献。
- 结果需要在更大、多中心、前瞻性队列中验证后才能进入临床决策。

---

## 补充材料建议

- **Supplementary Figure S1**：Residual 分布与标签阈值。
- **Supplementary Figure S2**：完整 MNE PSD topomap contact sheet。
- **Supplementary Figure S3**：完整 MNE WPLI connectivity contact sheet。
- **Supplementary Figure S4**：performance precision / one-subject-change sensitivity。
- **Supplementary Table S1**：EEG recording metadata and QC。
- **Supplementary Table S2**：model hyperparameters and training commands。
- **Supplementary Table S3**：all model metrics across seeds。
- **Supplementary Table S4**：feature/state/band ablation。
- **Supplementary Table S5**：permutation and bootstrap results。
- **Supplementary Table S6**：explainability top-k PSD and WPLI features。
- **Supplementary Table S7**：participant flow source notes，包含监督队列外患者只基线、基线+即时、基线+即时+最终等可公开记录类型。
- **Supplementary Table S8**：PROBAST/TRIPOD+AI risk audit。
- **Supplementary Table S9**：performance precision audit。
- **Supplementary Table S10**：claim-strength audit。
- **Supplementary Table S11**：model reporting card。

不建议纳入：

- clinical-only 或 clinical + EEG incremental model 作为正式主文或补充图；
- `seedmean10` 作为主性能指标；
- 主文 3.9 错误病例分析小节；
- 任何把 topomap 或 connectivity 解释写成因果机制的图题或结论。

## 参考文献策略

优先主线引用：

- Lassi et al., 2026：EEG + upper-limb recovery prediction。
- Mane et al., 2019：上肢卒中康复 EEG biomarker。
- Lin et al., 2022：卒中康复深度学习预后模型。
- White et al., 2024：卒中恢复、深度学习、可解释 AI。
- Tozlu et al., 2020：个体化上肢治疗后恢复预测。
- AlArfaj et al., 2022 和 Singh et al., 2024：EEG/deep learning stroke feasibility 辅助背景。

谨慎使用：

- Hasanzadeh 2024、Shahabi 2023、Zhao 2025、Li 2026、Olbrich 2026：可作为 broader EEG treatment-response modelling 类比，不作为卒中 tACS 恢复预测的直接证据。
- Nielsen 2018：卒中深度学习背景，但终点是 tissue outcome，不进入主线。

## 仍需作者确认

1. 伦理批准号具体编号和伦理委员会全称，当前正式稿先写代填。
2. tACS 电极大小和安全性记录。
3. FMA-UE 和 MBI 评估者身份及是否盲法，当前正式稿先写代填；评估时间已确定为治疗前和 14 次 tACS 后。
4. 监督队列外患者逐例分类从项目统计表提取后整理为 Supplementary Table S7。
