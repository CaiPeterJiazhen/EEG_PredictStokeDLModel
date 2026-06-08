# 基于残差感知自监督 EEG 学习预测卒中患者 tACS 后上肢比例恢复

## 摘要

卒中后上肢恢复预测有助于在神经调控和康复治疗前进行患者分层，但 EEG 预后建模常受到标注样本量小、高维特征不稳定以及验证方案易发生信息泄漏等限制。本研究开发了一种残差感知自监督卷积神经网络（residual-aware SSL-CNN），用于预测卒中患者接受经颅交流电刺激（tACS）后是否达到上肢比例恢复。模型输入仅来自治疗前基线静息态 EEG，包括睁眼和闭眼状态下的功率谱密度（PSD）以及加权相位滞后指数（WPLI）连接特征。监督学习队列包括 19 例具有完整治疗前和治疗后上肢 Fugl-Meyer 评定（FMA-UE）数据的患者。另有 9 例 EEG 索引患者因未形成完整监督标签而不能进入二分类训练，但其 EEG 数据被保留用于自监督表征学习，从而使本来不能用于监督建模的数据仍能参与编码器预训练。

最终残差感知 SSL-CNN 在患者层面留一受试者交叉验证中达到 10 个随机种子的平均准确率 0.837，最低准确率 0.789，准确率标准差 0.039，平均 ROC-AUC 0.860，平均 PR-AUC 0.858，平均 Brier 分数 0.142。在锁定的患者层面预测中，模型正确识别 10/10 例比例恢复患者和 6/9 例恢复不良患者，混淆矩阵为 TP = 10、FN = 0、FP = 3、TN = 6。传统 EEG 机器学习模型在硬标签区分方面表现有限，结构相同但未使用残差感知训练的 CNN 具有更大的种子间波动。可解释性分析提示，模型主要依赖状态相关 PSD 模式和 beta 频段 WPLI 连接。

这些结果表明，残差感知 EEG 表征学习能够在保留二分类临床终点的同时利用连续恢复残差信息，并可从基线 EEG 中提取具有生理解释价值的候选恢复特征。由于监督队列规模有限且尚缺乏外部验证，模型性能、概率校准和候选 PSD/WPLI 生物标志物仍需在更大规模、前瞻性 tACS 队列中确认。

## 引言

卒中后上肢功能障碍是影响生活质量和康复结局的重要问题。即使患者具有相似的基线运动损伤程度并接受相近的康复或神经调控治疗，其恢复轨迹仍可能存在明显差异。比例恢复框架将基线损伤程度与后续运动改善联系起来，为解释个体恢复差异提供了定量基础 [1,2]。PREP2 等生物标志物算法进一步表明，结合行为学指标和神经生理信息有助于提高上肢恢复预测能力 [3]。然而，比例恢复并不能完全解释卒中后运动结局，部分患者会偏离预期恢复轨迹。对于 tACS 等特定神经调控干预，治疗前识别哪些患者更可能达到有意义的上肢恢复仍然是一个尚未解决的问题。

tACS 是一种非侵入性脑刺激方法，可通过频率特异性方式调节运动网络振荡活动。既往研究已经在慢性卒中和运动网络相关情境中考察了不同频率 tACS 的神经调控效应 [22]。如果能够在治疗前识别更可能达到比例恢复的患者，tACS 可能更容易被纳入个体化康复分层。临床量表对于结局评估不可替代，但它们不能直接描述受损运动系统的振荡状态和网络连接状态。因此，可重复、低成本、适合康复场景的治疗前神经生理标志物具有重要价值。

静息态 EEG 符合这一需求。EEG 无创、采集成本相对较低，并可在康复环境中反复测量。PSD 特征反映频率特异性神经活动，功能连接特征反映不同脑区之间的相互作用。卒中后运动恢复不太可能由单一电极或单一频段完全解释，因此同时建模局部振荡活动和跨区域连接具有合理性。既往研究已将 EEG 生物标志物、静息态 EEG 参数以及 beta 频段活动与卒中上肢康复和运动恢复联系起来 [7-9]。慢性卒中患者静息态皮层 EEG 节律和网络特征也被报道与上肢运动功能相关 [26]，运动网络与非运动网络之间的功能连接则与卒中后运动恢复变化相关 [16]。WPLI 能够降低头皮 EEG 中零相位滞后连接、体积传导和样本量偏差对相位同步估计的影响，适合作为连接特征 [10]。近期机器学习和深度学习研究进一步支持使用临床、影像或 EEG 特征进行个体化卒中恢复预测的可行性 [4-6,23-25]。不过，使用治疗前基线 EEG 预测 tACS 后上肢比例恢复这一问题仍缺乏系统研究。

本研究受三点方法学限制推动。第一，传统 EEG 机器学习通常将高维 PSD 和连接特征展平为向量，再依赖特征选择或正则化。这种方式可能丢失通道-频率矩阵和连接矩阵的结构信息，并使模型性能高度依赖于折内筛选出的特征。第二，CNN 能够更好地保留 EEG 结构，但卒中标注队列往往较小，模型估计容易受随机种子、折划分和预处理细节影响。若在 EEG 片段层面而非患者层面验证，还可能发生受试者信息泄漏，导致性能被高估。第三，比例恢复标签虽然适合二分类推断，但其本质来源于预期改善和实际改善之间的连续残差。仅使用二分类标签会使阈值附近患者和远离阈值患者被同等对待，丢失残差距离信息。

自监督学习为减少对完整结局标签的依赖提供了途径。tACS 队列中常存在仅具有基线 EEG、仅完成部分随访或未形成完整比例恢复标签的患者。这些数据不能作为有标签监督样本，但可在避免结局泄漏的前提下用于学习患者层面 EEG 表征。残差感知训练则处理另一个互补问题：在推断阶段保留二分类比例恢复终点，同时在训练阶段利用连续残差距离和恢复排序信息。

本研究开发了一种残差感知 SSL-CNN，用治疗前 EO/EC 静息态 EEG 的 PSD 和 WPLI 特征预测 tACS 后上肢比例恢复分组。本文的主要贡献包括四点：第一，在患者层面 LOSO 框架内系统比较传统 EEG-ML、无 SSL CNN、无残差感知头 SSL-CNN、无 SSL 残差感知 CNN 和最终残差感知 SSL-CNN；第二，使用 Barlow Twins 自监督预训练纳入无法形成完整监督标签但具有基线 EEG 的患者，从而扩大编码器可见的患者层面 EEG 分布；第三，在二分类比例恢复标签之外引入连续残差距离、成对排序和软标签辅助目标，使训练信号更贴近比例恢复终点的连续结构；第四，通过 integrated gradients、SmoothGrad、遮挡、PSD topomap 和 WPLI connectivity 分析，将模型预测与可检验的 EEG 候选特征相连接。评估指标包括 10 个随机种子的稳定性、ROC-AUC、PR-AUC、Brier 分数、bootstrap 不确定性、置换检验和最终混淆矩阵。

## 材料与方法

### 研究设计

本研究为单中心卒中 tACS 预后建模研究，目标是用治疗前基线静息态 EEG 预测治疗后上肢比例恢复状态。整体流程包括：基线 EEG 采集、EEG 预处理和受累侧对齐、PSD/WPLI 特征提取、患者层面 Barlow SSL 预训练、残差感知监督微调、最终二分类推断以及患者层面 LOSO 评估。所有主要模型输入均来自治疗前 EEG。临床变量仅用于描述队列和定义 FMA-UE 比例恢复残差，不作为本文主要模型输入。

本研究伦理声明保留正式占位：本研究经[伦理委员会名称及批准号待填]批准，所有患者或其法定代理人均签署知情同意书。

![图1 总体技术框架](../results/figures/revised_initial/figure1_overall_framework_provided.png)

**图1｜残差感知 SSL-CNN 预测 tACS 后恢复的总体流程。** 图中展示了基线 EEG 采集、预处理、PSD/WPLI 特征提取、自监督预训练、残差感知监督微调、最终推断和患者层面评估流程。如图1所示，治疗后 FMA-UE 仅用于定义比例恢复标签，而全部模型输入均限定为治疗前 EEG；自监督池和监督验证队列在患者层面分开使用，以减少结局信息泄漏。

### 研究对象

M1 临床源工作簿包含 29 例患者记录，其中 28 例在当前 EEG 数据目录中有索引。最终 19 例患者具有基线 EEG 以及完整治疗前和治疗后 FMA-UE 数据，可定义监督比例恢复标签，因此构成患者层面 LOSO 验证队列。该队列中 10 例被定义为比例恢复，9 例被定义为恢复不良。

监督队列外患者不作为有标签结局样本。其未进入监督训练的原因包括未完成完整实验或无法形成完整比例恢复标签。根据作者提供的分类，这些记录可公开归为仅基线数据、基线加即时数据、基线加即时加最终数据但未进入当前监督标签分析等类型。这些患者的基线 EEG 仍可进入自监督学习池，用于增加编码器表征学习阶段可用的患者 EEG 数量。

表1汇总全部 29 例 M1 临床源记录，并将 19 例监督标签队列与 9 例额外 EEG-indexed SSL 池分开列示。连续变量以 mean (±SD) 表示，分类变量以百分比表示。P 值比较监督标签队列与额外 EEG-indexed SSL 池，使用 Welch t 检验、Mann-Whitney U 检验或 Fisher exact 检验。这些检验用于描述数据来源和缺失结构，不作为确认性基线推断。

**表1｜患者信息。** 数值分别报告全部 M1 临床源记录、监督标签队列和额外 EEG-indexed SSL 池。P 值比较监督标签队列与额外 EEG-indexed SSL 池。1 例临床源记录当前无 EEG 索引，仅计入全部临床记录列。连续变量为 mean (±SD)；仅 1 个可用治疗后数值时显示为 value (n=1)。1 例病程以“天”记录，已按 days/30 换算为月。

| 指标 | 全部临床记录 | 监督标签队列 | 额外 EEG-indexed SSL 池 | P |
|:--|:--:|:--:|:--:|:--:|
| 例数 | 29 | 19 | 9 |  |
| **人口学资料** |  |  |  |  |
| 女性 | 48% | 58% | 33% | 0.42 |
| 年龄，岁 | 63.72 (±8.94) | 64.74 (±6.49) | 60.11 (±12.07) | 0.31 |
| 病程，月 | 37.69 (±21.59) | 36.26 (±17.76) | 42.46 (±29.33) | 0.86 |
| **临床评估** |  |  |  |  |
| 左侧上肢受累 | 55% | 58% | 56% | 1 |
| 右侧上肢受累 | 45% | 42% | 44% |  |
| 治疗前 FMA-UE | 38.66 (±24.19) | 40.47 (±23.83) | 38.11 (±25.53) | 0.98 |
| 14 次治疗后 FMA-UE | 44.86 (±23.72) | 45.47 (±23.23) | 66.00 (n=1) |  |
| FMA-UE 观察改善 | 4.67 (±4.03) | 5.00 (±4.07) | 0.00 (n=1) |  |
| 治疗前 MBI | 56.55 (±21.26) | 55.53 (±19.92) | 62.22 (±22.93) | 0.46 |
| 14 次治疗后 MBI | 76.19 (±22.58) | 76.84 (±21.49) | 100.00 (n=1) |  |
| **数据可用性** |  |  |  |  |
| 基线 EEG 已索引 | 97% | 100% | 100% |  |
| 具备治疗后 FMA-UE | 72% | 100% | 11% | <0.001 |
| 具备完整监督标签 | 66% | 100% | 0% | design |

### 结局标签

主要结局为 tACS 后 FMA-UE 比例恢复状态。第 \(i\) 位患者的预期改善定义为：

```text
Expected improvement_i = 0.7 x (66 - FMA_pre_i)
```

观察到的改善定义为：

```text
Observed improvement_i = FMA_post_i - FMA_pre_i
```

比例恢复残差定义为：

```text
Residual_i = Expected improvement_i - Observed improvement_i
```

监督队列残差中位数作为队列特异性阈值：

```text
tau = median(Residual_i)
```

二分类标签定义为：

```text
y_i = 1, if Residual_i <= tau
y_i = 0, if Residual_i > tau
```

其中 \(y_i = 1\) 表示比例恢复，\(y_i = 0\) 表示恢复不良。残差感知辅助训练使用的有符号残差距离为：

```text
d_i = tau - Residual_i
```

\(d_i > 0\) 表示更接近或超过比例恢复阈值，\(d_i < 0\) 表示低于比例恢复阈值。残差距离、排序和软标签目标仅用于训练，不作为测试时模型输入。

### tACS 方案

tACS 作用于受累手对侧初级运动皮层。右手受累刺激 C3，左手受累刺激 C4。刺激设备为 Neuroscan 1x1 经颅电刺激器 DC-STIMULATOR PLUS。刺激频率为 20 Hz，强度为 1000 microampere，每次 20 min，每日 1 次，共 14 次。刺激阻抗控制在 30 kOhm 以下。FMA-UE 和 MBI 在治疗前以及完成 14 次 tACS 后评估。正式稿中评估者和盲法信息保留占位：FMA-UE 和 MBI 由[评估者资质及盲法状态待填]在治疗前和 14 次 tACS 结束后完成评估。

### EEG 采集与预处理

治疗前 EEG 使用 Compumedics Neuroscan SynAmps2 64 通道 EEG 系统采集，采用标准 10-20 电极布局。静息态记录包括睁眼（EO）和闭眼（EC）条件。预处理在 EEGLAB 中完成，包括平均参考、0.5 Hz 高通滤波、45 Hz 低通滤波、去除 50 Hz 工频干扰、ICA，以及人工剔除坏道和伪迹片段/成分。

当前分析使用预处理后的 EEGLAB `.set/.fdt` 文件。特征提取脚本检查连续数据、通道顺序和最小记录长度。模型输入保留 62 个 EEG 通道，参考通道不作为 EEG 预测变量。提取 PSD 和连接特征前，对 EEG 特征进行受累侧对齐。左手受累患者进行左右镜像，使模型在共同的刺激侧/受累侧约定下学习特征，降低简单左右差异带来的干扰。

### EEG 特征

EO 和 EC 条件分别构建 EEG 特征。PSD 使用 Welch 方法计算 0.5-45 Hz 范围内的频谱功率，并组织为通道-频率矩阵。WPLI 连接按通道对和频段构建，用于减少体积传导和零相位滞后同步的影响 [10]。频段包括 delta、theta、alpha、beta-low、beta-medium 和 beta-high。主要 EEG 输入为 EO/EC 条件下的 PSD 和 WPLI，不包含临床变量。

### 传统机器学习

传统 EEG 机器学习基线使用展平后的 PSD+WPLI EO+EC 特征。模型包括 L1 正则化逻辑回归、L2 正则化逻辑回归，以及折内 SelectK=100 特征选择后的 RBF 核 SVM。所有标准化和特征选择均在 LOSO 训练折内完成，避免测试受试者信息泄漏。这些模型用于定义传统高维 EEG 展平特征建模的性能边界。

### 自监督学习

自监督学习用于在不使用恢复标签的情况下学习患者层面 EEG 表征。该阶段允许监督队列之外、无法形成完整比例恢复标签的 EEG 记录参与编码器预训练。本研究选择 Barlow Twins 而不是依赖负样本的对比学习或重建式自编码预训练，主要基于三个考虑。首先，患者层面 EEG 队列较小，显式负样本或 memory queue 容易把具有相近神经生理状态的患者当作负例；Barlow Twins 通过同一患者两个增强视图之间的交叉相关矩阵学习不变性，不需要大型负样本集合。其次，冗余降低项同时约束表征维度之间的去相关，可降低小样本高维 EEG 表征坍塌或维度冗余。第三，与重建原始高维 EEG 特征相比，Barlow 目标更强调患者层面判别表征的一致性，适合随后进行比例恢复分类微调。对于同一患者 EEG 特征，生成两个增强视图，并输入共享编码器 \(f(\cdot)\) 和投影头 \(g(\cdot)\)：

```text
z_i^A = g(f(x_i^A)),  z_i^B = g(f(x_i^B))
```

两个投影批次之间的交叉相关矩阵为：

```text
C_jk =
sum_b z^A_bj z^B_bk /
sqrt(sum_b (z^A_bj)^2) sqrt(sum_b (z^B_bk)^2)
```

Barlow 风格的冗余降低损失为：

```text
L_SSL = sum_j (1 - C_jj)^2 + lambda sum_j sum_{k != j} C_jk^2
```

对角项鼓励同一患者两个增强视图在同一表征维度上一致；非对角项降低不同维度之间的冗余 [17]。SSL 预训练后，再在 19 例有标签患者上进行监督微调。

![图2 患者层面 Barlow SSL 预训练](../results/figures/revised_initial/figure2_ssl_framework_provided.png)

**图2｜患者层面 Barlow SSL 预训练框架。** 图中展示两个增强视图、共享 EEG 编码器、投影头、交叉相关矩阵和 Barlow Twins 损失。如图2所示，每个 LOSO 折中被留出的测试患者在任何标准化、增强和 SSL 编码器训练之前均被排除，自监督学习只利用训练池内的无标签 EEG 特征。

### CNN 与残差感知学习

深度模型为多分支 CNN。PSD 分支处理 EO/EC 条件下的通道-频率矩阵，WPLI 分支处理 EO/EC 条件下的连接边-频段特征。各状态嵌入通过 gated EO/EC fusion 形成 PSD 嵌入和 WPLI 嵌入，再拼接为 64 维多模态 EEG 表征。二分类头输出比例恢复概率。残差回归头、排序头和软标签辅助损失在训练阶段利用连续残差距离和恢复顺序信息，但测试阶段仅使用二分类头进行最终推断。如图3所示，PSD 和 WPLI 分支先分别提取 EO/EC 状态表征，再在患者层面融合为同一个多模态 EEG embedding，使最终二分类推断与训练期残差辅助目标共享底层编码器。

总损失可写为：

```text
L_total = L_BCE(y_i, p_i)
        + alpha L_residual(d_i, d_hat_i)
        + beta L_rank
        + gamma L_soft
```

其中 \(L_BCE\) 为比例恢复标签的二分类交叉熵，\(L_residual\) 为有符号残差距离损失，\(L_rank\) 为成对恢复排序目标，\(L_soft\) 为基于残差接近程度的软标签监督。推断时只使用二分类头，残差和排序头仅作为训练正则化。

![图3 残差感知患者层面 Barlow SSL-CNN](../results/figures/revised_initial/figure3_cnn_residual_aware_provided.png)

**图3｜残差感知患者层面 Barlow SSL-CNN 结构。** 图中展示 PSD/WPLI 多分支输入、共享编码器、EO/EC gated fusion、PSD+WPLI 拼接、多任务监督头以及最终二分类推断。该结构保证测试阶段输出为单一、可解释的比例恢复概率，同时在训练阶段充分利用连续残差和排序信息。

### 验证与统计分析

主要验证方案为患者层面 LOSO。任何 EEG 片段、增强视图或种子级预测均不作为独立患者。CNN 模型使用 10 个随机种子训练。主要稳定性指标包括平均种子准确率、最低种子准确率和准确率标准差。其他指标包括平衡准确率、敏感度、特异度、ROC-AUC、PR-AUC 和 Brier 分数。

不确定性评估采用受试者层面 bootstrap 区间。配对比较采用受试者层面重采样。硬预测的配对比较采用 McNemar 检验。置换检验使用患者层面标签置换。校准性能通过 Brier 分数和校准曲线评估。最终锁定的患者层面预测用于生成混淆矩阵。

### 可解释性分析

模型解释在训练和验证完成后进行。Integrated gradients 与 SmoothGrad 用于估计 PSD 通道-频段特征和 WPLI 边特征对比例恢复概率的归因 [18,19]。分支/状态遮挡用于量化移除 PSD、WPLI、EO 或 EC 输入后预测损失的变化。PSD 归因按频段映射到 62 通道头皮拓扑图，WPLI 归因按频段筛选绝对归因最高的连接边并绘制 scalp-connectivity 图；两类图均使用 MNE-Python 渲染 [11,12]。可解释性分析报告模型依赖的 EEG 特征分布，统计解释限定为探索性候选生物标志物。

### 软件

分析使用 Python、PyTorch、scikit-learn、MNE-Python 和 EEGLAB [11,12,20,21]。报告原则参考 TRIPOD、TRIPOD+AI 和 PROBAST 对临床预测模型的要求 [13-15]。

## 结果

### 队列特征

最终监督队列包含 19 例患者。平均年龄为 64.74 岁（SD 6.49），11 例为女性。8 例右侧上肢受累，11 例左侧上肢受累。基线 FMA-UE 为 40.47（SD 23.83），治疗 14 次后 FMA-UE 为 45.47（SD 23.23），观察到的 FMA-UE 改善为 5.00 分（SD 4.07）。全队列比例恢复残差为 12.87（SD 16.23）。基线 MBI 为 55.53（SD 19.92），治疗后 MBI 为 76.84（SD 21.49）。

比例恢复组和恢复不良组在年龄（64.10 vs 65.44 岁，p = 0.658）、病程（35.40 vs 37.22 月，p = 1.000）、性别构成（p = 1.000）和受累侧（p = 1.000）方面相近。与残差定义相一致，两组在基线运动功能和日常生活能力上存在差异：比例恢复组治疗前 FMA-UE 更高（59.60 vs 19.22，p < 0.001），治疗后 FMA-UE 更高（64.10 vs 24.78，p < 0.001），治疗前 MBI 更高（68.00 vs 41.67，p = 0.002），治疗后 MBI 也更高（93.00 vs 58.89，p < 0.001）。两组 FMA-UE 观察改善量差异不显著（4.50 vs 5.56，p = 0.480），但比例恢复残差明显分离两组（-0.02 vs 27.19，p < 0.001）。这说明本文二分类标签并不是简单的绝对改善量标签，而是反映患者实际改善是否与其剩余恢复潜力相匹配。

### 传统 EEG-ML 基线

传统 EEG-ML 基线对比例恢复的硬标签区分能力有限。PSD+WPLI L1 逻辑回归达到准确率 0.737、平衡准确率 0.733、ROC-AUC 0.711、PR-AUC 0.775、Brier 分数 0.208。L2 逻辑回归达到准确率 0.684、平衡准确率 0.689、ROC-AUC 0.778、PR-AUC 0.840、Brier 分数 0.221。SelectK=100 后的 RBF SVM 达到准确率 0.684、平衡准确率 0.694、ROC-AUC 0.778、PR-AUC 0.771、Brier 分数 0.219。这些结果提示，传统 EEG-ML 模型保留了一定排序信息，但在小样本高维 EEG 设置下难以形成稳定的硬标签分类和概率校准。模型指标对比图进一步显示，传统模型没有在 accuracy、balanced accuracy、sensitivity、specificity、ROC-AUC、PR-AUC 和 Brier 分数上形成同时占优的表现。

**表2｜传统 EEG 机器学习基线。** 来源：`results/tables/table2_main_model_performance.csv`。

| 模型 | 特征选择 | Accuracy | Balanced accuracy | Sensitivity | Specificity | ROC-AUC | PR-AUC | Brier |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| Logistic L1 | None | 0.737 | 0.733 | 0.800 | 0.667 | 0.711 | 0.775 | 0.208 |
| Logistic L2 | None | 0.684 | 0.689 | 0.600 | 0.778 | 0.778 | 0.840 | 0.221 |
| SVM RBF + SelectK=100 | LOSO 训练折内 SelectK=100 | 0.684 | 0.694 | 0.500 | 0.889 | 0.778 | 0.771 | 0.219 |

### CNN、SSL 与残差感知训练

CNN 能更好地利用完整 EEG 矩阵，但无残差感知训练时仍存在种子敏感性。在同一 paired 10 种子口径下（seed = 0、1、2、3、4、5、6、7、8、13），结构相同的 no-SSL CNN 达到平均准确率 0.753、平衡准确率 0.743、ROC-AUC 0.783、PR-AUC 0.769；最低准确率为 0.579，准确率标准差为 0.094。无残差感知头的患者层面 Barlow SSL-CNN 平均准确率同为 0.753，平衡准确率为 0.742、ROC-AUC 0.739、PR-AUC 0.719；准确率标准差为 0.053，最低准确率为 0.684。因此，在该 paired comparison 中，SSL 单独使用并未提高平均准确率或排序指标，但降低了种子间波动并提高了最差种子表现。

残差感知辅助监督在当前消融结构中提供了更稳定的训练信号。无 SSL 但带残差感知头的 CNN 达到平均准确率 0.816、平衡准确率 0.812、ROC-AUC 0.899、PR-AUC 0.908、Brier 分数 0.130。最终患者层面 Barlow SSL 加残差感知模型达到平均准确率 0.837、平衡准确率 0.831、ROC-AUC 0.860、PR-AUC 0.858、Brier 分数 0.142。其最低准确率为 0.789，准确率标准差为 0.039。总体而言，最终模型具有最有利的稳定性表现，但当前数据不能证明 SSL 相对于残差感知监督具有独立稳定增益。

**表3｜深度模型稳定性与残差感知消融。** 来源：`results/metrics/core_ablation_10seed_summary.csv`。No-SSL CNN 和无残差头 SSL-CNN 两行使用 `results/metrics/updated_sub05_sub28_10seed_no_ssl_barlow_cnn_summary.csv` 中同一 paired seed 口径（0、1、2、3、4、5、6、7、8、13）。NA 表示该 paired source file 未提供 Brier 分数。

| 模型 | Mean accuracy | Min accuracy | Accuracy SD | Balanced accuracy | ROC-AUC | PR-AUC | Brier |
|:--|--:|--:|--:|--:|--:|--:|--:|
| No-SSL CNN | 0.753 | 0.579 | 0.094 | 0.743 | 0.783 | 0.769 | NA |
| SSL-CNN, no residual heads | 0.753 | 0.684 | 0.053 | 0.742 | 0.739 | 0.719 | NA |
| No-SSL CNN + residual-aware heads | 0.816 | 0.737 | 0.045 | 0.812 | 0.899 | 0.908 | 0.130 |
| Residual-aware SSL-CNN | 0.837 | 0.789 | 0.039 | 0.831 | 0.860 | 0.858 | 0.142 |

### 最终模型表现

最终残差感知 SSL-CNN 支持患者层面比例恢复预测，并采用保守的不确定性报告。在锁定的硬预测中，模型正确分类 10/10 例比例恢复患者和 6/9 例恢复不良患者。混淆矩阵为 TP = 10、FN = 0、FP = 3、TN = 6，对应敏感度 1.000、特异度 0.667、准确率 0.842。锁定预测的平衡准确率为 0.833、ROC-AUC 0.844、PR-AUC 0.836、Brier 分数 0.126。由于 19 例 LOSO 队列中 1 例患者预测改变即可使准确率移动 5.3 个百分点，这些结果必须与 10 种子稳定性指标共同解读。

在不同模型的患者层面指标对比中，最终模型保留了锁定预测中的高敏感度，并相对于传统 EEG-ML 基线改善了概率校准。与同结构 no-SSL CNN 相比，最终模型的锁定 accuracy 和 balanced accuracy 相同，但 ROC-AUC 更高（0.844 vs 0.811），PR-AUC 更高（0.836 vs 0.808），且 Brier 分数更低（0.126）。因此，最终模型的优势不应只理解为某一个硬标签指标的提高，而应理解为整体排序、校准和稳定性更均衡。

![图4A 最终模型 ROC 曲线](../results/figures/revised_initial/figure4a_final_model_roc.png)

**图4A｜最终残差感知 SSL-CNN 的 ROC 曲线。** 单独展示锁定 seedmean10 患者层面预测的 ROC 曲线，ROC-AUC = 0.844。由于当前验证队列仅包含 19 例患者，ROC 曲线由有限患者排序点构成，呈阶梯状而不是大样本平滑曲线。如图4A所示，模型在较低假阳性率区间已经获得较高真阳性率，说明概率排序能够较早识别比例恢复患者。

![图4B 最终模型混淆矩阵](../results/figures/revised_initial/figure4b_final_model_confusion_matrix.png)

**图4B｜最终残差感知 SSL-CNN 的混淆矩阵。** 固定阈值 0.5 下，TP = 10、FN = 0、FP = 3、TN = 6；敏感度 1.00、特异度 0.67、准确率 0.84。如图4B所示，最终模型没有漏判比例恢复患者，但仍有 3 例恢复不良患者被预测为比例恢复，提示当前阈值更偏向敏感度而非特异度。

![图4C-a 不同模型指标直方图](../results/figures/revised_initial/figure4c_a_model_metric_histogram.png)

**图4C-a｜不同模型的患者层面指标直方图。** 分组柱状图比较传统 EEG-ML、no-SSL CNN、Barlow CNN 和最终残差感知 SSL-CNN 的 accuracy、balanced accuracy、sensitivity、specificity、ROC-AUC 和 PR-AUC。颜色分别表示不同模型：L1 逻辑回归为蓝色，L2 逻辑回归为青绿色，SVM RBF 为红色，no-SSL CNN 为橙色，Barlow CNN 为紫色，最终残差感知 SSL-CNN 为绿色。如图4C-a所示，最终模型在敏感度、ROC-AUC 和 PR-AUC 上保持较高水平，而 specificity 仍是主要限制。

![图4C-b 不同模型 Brier 校准误差](../results/figures/revised_initial/figure4c_b_brier_calibration.png)

**图4C-b｜不同模型的 Brier 校准误差。** 棒棒糖图展示各模型 Brier 分数，数值越低表示概率校准越好。如图4C-b所示，最终模型的 Brier 分数最低，提示其患者层面概率输出相对于传统 EEG-ML、no-SSL CNN 和 Barlow CNN 更保守且校准误差更小。

![图4D 最终模型损失曲线](../results/figures/revised_initial/figure4d_final_model_loss_curves.png)

**图4D｜最终残差感知 SSL-CNN 的训练和验证损失曲线。** 曲线汇总 10 个随机种子和 19 个 LOSO 折的平均训练过程，阴影表示均值标准误，虚线表示 SWA 起始 epoch。如图4D所示，训练总损失由 epoch 1 的 1.115 降至 epoch 50 的 0.201，并在 epoch 100 稳定于 0.195；验证总损失由 0.927 降至 0.700，并在后期平台化于 0.684 左右。验证分类 BCE 从 0.693 降至 0.460，验证残差项的加权贡献从 0.164 降至 0.117。加权 soft-label 项由 0.069 升至约 0.107，但其权重较低，主要反映软标签约束与硬分类收敛之间的张力，并未导致总验证损失发散。

### 特征与频段消融

特征族消融显示，PSD 特征保留了有用区分信息。PSD-only 特征达到准确率 0.789、平衡准确率 0.789、ROC-AUC 0.811、PR-AUC 0.840、Brier 分数 0.193。WPLI-only 特征达到准确率 0.632、ROC-AUC 0.689、PR-AUC 0.751。这说明 PSD 是当前队列中最强的单一特征族，而 WPLI 单独用于硬标签分类时较弱，但仍保留一定恢复排序信息。

状态和频段消融提供了更细的线索。EO-only PSD+WPLI 表现较差，EC-only PSD+WPLI 保留了较好的排序信息。Beta-medium-only 特征达到准确率 0.737、ROC-AUC 0.711，优于 beta-high-only。运动相关 WPLI 边虽然硬标签准确率不高，但 ROC-AUC 达到 0.733，PR-AUC 达到 0.816。如图5a所示，残差感知 SSL-CNN 的 10 种子准确率分布最集中；如图5b所示，残差感知辅助目标是核心消融中最稳定的性能来源；如图5c和图5d所示，PSD-only 与 EC-only 输入在较低维度下保留了较好的排序能力，而高维 WPLI-only 输入单独使用时并未带来更好的硬标签分类。这些消融结果提示 PSD、EC 状态和 beta 相关连接可作为候选 EEG 信息来源，但仍需独立队列验证。

**表4｜特征、状态和频段消融。** 来源：`results/tables/table3_ablation.csv`。

| 消融 | 输入特征数 | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Brier |
|:--|--:|--:|--:|--:|--:|--:|
| PSD only | 744 | 0.789 | 0.789 | 0.811 | 0.840 | 0.193 |
| WPLI only | 22,692 | 0.632 | 0.628 | 0.689 | 0.751 | 0.232 |
| PSD + WPLI | 23,436 | 0.684 | 0.678 | 0.767 | 0.793 | 0.207 |
| EO only | 11,718 | 0.316 | 0.317 | 0.300 | 0.467 | 0.490 |
| EC only | 11,718 | 0.632 | 0.633 | 0.789 | 0.816 | 0.223 |
| Beta medium only | 3,906 | 0.737 | 0.733 | 0.711 | 0.776 | 0.232 |
| Beta high only | 3,906 | 0.579 | 0.578 | 0.567 | 0.561 | 0.317 |
| Motor WPLI edges | 6,780 | 0.632 | 0.628 | 0.733 | 0.816 | 0.261 |

![图5 稳定性与消融](../results/figures/revised_initial/figure5_stability_ablation.png)

**图5｜稳定性与消融。** a，核心深度模型的 10 种子准确率分布；b，SSL 和残差感知核心消融的指标热图，颜色表示各指标列内归一化表现，Brier 分数已反向处理；c，特征、状态和频段消融的 accuracy 与 ROC-AUC 排名；d，输入特征维度与 ROC-AUC 的关系，点大小表示 PR-AUC。该图用于展示稳定性、校准和信息效率，而不只是单一柱状图。

### 可解释性结果

模型解释将预测信息定位到状态相关 PSD 模式和 beta 频段连接。门控融合权重提示 PSD 与 WPLI 分支存在状态分工：PSD 分支中 EO 嵌入平均权重更高，WPLI 分支中 EC 嵌入占主导。遮挡分析也支持这种状态依赖性，移除 EC 或 WPLI 特征引起的平均损失扰动更大，而 PSD 归因仍集中在 EO 通道-频段图中。这说明最终模型并不是简单依赖单一状态或单一特征族，而是在 EO PSD 与 EC 连接之间形成互补利用。

PSD 归因主要集中在 EO 条件下的颞部、额部和围中央区域。排名靠前的 PSD 特征包括 TP7、C5、F5、FPZ 和 FC5，在 gamma、beta-high 和 delta 范围内均有贡献。如图6A所示，delta 至 gamma 七个频段均可见 EO 行更清晰的空间归因模式，局部高值分布于额部、颞部和中央区，而非单一孤立电极；EC 行归因整体较弱但仍保留部分中央和后部频段信息。名义验证分析提示 EO F5 delta 归因与残差距离存在关联，但由于样本量小且多重检验严格，该结果只能作为候选假设。总体上，PSD 结果与既往将静息态 EEG 节律、beta 活动和卒中运动恢复联系起来的研究一致 [7-9,26]。

WPLI 归因集中在 EC beta 频段连接。排名靠前的边包括 F8-CP1、FT7-P2、C5-P2、F4-C5 和 F8-FC4，涉及额叶、中央区、顶叶以及跨半球连接。如图6B所示，EC 行的 beta-medium 和 beta-high 连接比 EO 行更集中，主要连接额-中央、中央-顶和双侧通道区域；红蓝边同时出现，提示模型既利用有利于比例恢复概率的连接模式，也利用指向恢复不良的连接模式。网络层面摘要提示 beta 频段额-中央、运动邻近和半球间连接与残差距离存在名义关联，但未能作为确认性多重检验结果。该模式与 Lin 等预后模型中对额叶、中央区和顶叶分布式神经生理特征的解释方式相似，即将其视为候选恢复因素而非单一因果机制 [4]。这些结果也与卒中后运动恢复和皮层连接改变的研究相容 [16,26]。

**表5｜EEG 解释性候选特征。** 来源：`results/explainability/psd_channel_band_importance.csv` 和 `results/explainability/wpli_top_edges.csv`。

| 特征族 | 状态 | 频段 | 通道/边 | Signed attribution | Abs attribution |
|:--|:--|:--|:--|--:|--:|
| PSD | EO | Gamma | TP7 | -0.000892 | 0.001330 |
| PSD | EO | Gamma | C5 | -0.000846 | 0.001245 |
| PSD | EO | Beta High | TP7 | -0.000749 | 0.001228 |
| PSD | EO | Beta High | FPZ | -0.000943 | 0.001159 |
| PSD | EO | Delta | F5 | -0.000817 | 0.001137 |
| WPLI | EC | Beta High | F8-CP1 | 0.000154 | 0.002128 |
| WPLI | EC | Beta Medium | FT7-P2 | -0.001331 | 0.002119 |
| WPLI | EC | Beta Medium | C5-P2 | -0.000529 | 0.002111 |
| WPLI | EC | Beta High | F4-C5 | -0.000062 | 0.002097 |
| WPLI | EC | Beta High | F8-FC4 | 0.000994 | 0.002095 |

![图6A PSD 频段 topomap](../results/figures/revised_initial/figure6a_psd_topomap_bands.png)

**图6A｜PSD 频段归因 topomap。** 展示 EO 和 EC 条件下 delta、theta、alpha、beta-low、beta-medium、beta-high 和 gamma 七个频段的 signed PSD attribution。每一列为一个频段，第一行为睁眼状态，第二行为闭眼状态。

![图6B WPLI 频段 connectivity](../results/figures/revised_initial/figure6b_wpli_connectivity_bands.png)

**图6B｜WPLI/FC 频段连接归因图。** 展示 EO 和 EC 条件下六个频段的 top-20 WPLI 边。红色表示正向 signed attribution，蓝色表示负向 signed attribution，线宽表示绝对归因强度。

## 讨论

本研究显示，治疗前基线静息态 EEG 可以在患者层面为卒中患者 tACS 后上肢比例恢复提供可建模的预后信号。最终残差感知 SSL-CNN 在 10 种子 LOSO 验证中取得了当前核心深度模型中最稳定的准确率分布，并在锁定预测中保持较高敏感度和较好的概率排序。传统 EEG-ML 基线保留了一定排序信息，但硬标签分类和校准能力较弱。结合消融和可解释性结果，本文提出的主要观点是：严格患者层面验证下的结构化 EEG 表征学习可以同时服务于恢复预测和候选 PSD/WPLI 生物标志物生成。

队列统计有助于界定模型预测任务的临床背景。比例恢复组和恢复不良组在年龄、病程、性别和受累侧方面相近，但在基线 FMA-UE 和 MBI 上存在差异。这一结果与比例恢复残差的定义一致，因为残差终点来源于基线损伤、预期改善和实际改善。本文将临床变量用于描述队列和定义恢复残差，主要模型输入则限定为治疗前 EEG。因此，当前结果回答的是 EEG-only 特征能否预测一个由临床量表定义的恢复表型；临床-only、EEG-only 和临床+EEG 联合模型之间的相对价值，需要在后续外部验证设计中并行比较。

残差感知训练是本文最重要的方法学贡献。比例恢复可被转化为二分类临床终点，但其底层信息是连续残差。阈值附近患者和远离阈值患者携带的恢复信息并不相同。模型若仅用二分类标签训练，就会丢失残差距离和恢复排序信息。本文在训练阶段引入有符号残差距离、成对排序和残差接近程度软标签，同时在推断阶段保留简单的二分类头，使训练目标更接近临床结局结构。这种设计比单纯二分类监督更适合比例恢复这类由连续临床量表构造出的终点。

结构化 CNN 输入解决了传统 EEG 预后管线的另一个问题。传统模型通常需要将 PSD 和 WPLI 展平为高维向量，再依赖折内特征选择或正则化；这种做法容易丢失通道-频率图和连接矩阵的结构。本文 CNN 保留了 PSD 的通道-频率结构和 WPLI 的连接-频段结构，并在 EO/EC 状态之间进行门控融合。这并不意味着深度学习在所有小样本 EEG 队列中都一定优于传统模型，而是说明在严格避免信息泄漏、报告随机种子稳定性并进行消融验证的前提下，结构化模型可以更合理地利用完整 EEG 特征。

Barlow SSL 在本文中的价值主要体现在患者层面表征学习和数据利用效率。与需要大量负样本的对比学习相比，Barlow Twins 通过两个增强视图的交叉相关矩阵实现不变性学习和冗余降低，更适合患者数有限、患者间神经生理差异较大的 EEG 队列。与重建式自监督相比，该目标不要求模型复原高维 EEG 输入，而是鼓励学习可迁移到比例恢复分类的低维表征。更重要的是，患者层面 Barlow SSL 使无法形成完整监督比例恢复标签的患者 EEG 也能参与编码器预训练。康复和神经调控研究中常见未完成治疗、随访缺失或时间点组合不完整的情况，这些数据不能贡献监督标签，但仍可能帮助模型学习更稳定的 EEG 表征。在 paired seed 比较中，无残差感知头的 SSL-CNN 降低了种子间波动并提高了最差种子准确率，但没有提高平均准确率、ROC-AUC 或 PR-AUC。因此，当前证据支持将 SSL 视为纳入未标注 EEG 和稳定表征学习的组件，其独立增益仍需更大队列检验。

可解释性分析提供了神经生理层面的解释线索。PSD 归因主要出现在 EO 条件下的额叶、颞部和围中央通道；WPLI 归因主要集中在 EC beta 频段的额-中央、额-顶、中央-顶和半球间连接。这些发现与既往卒中 EEG 研究中静息态节律、beta 振荡以及运动网络连接和上肢恢复之间的关系相容 [7-9,16,26]。这些结果也类似 Lin 等迁移深度学习预后模型中的解释方式：分布式神经生理特征被用作候选恢复因子，而不是被解释为单一电极或单一连接的因果作用 [4]。因此，本文的 topomap 和 connectivity 图适合用于提出未来预先指定的假设，例如 EO 额/颞 PSD 或 EC beta 额-中央连接是否能在更大 tACS 队列中改善患者分层。

本研究存在多项局限。首先，有标签监督队列仅 19 例，单例患者预测变化即可使准确率改变 5.3 个百分点。因此，本文同时报告锁定预测、10 种子汇总、种子间波动和保守不确定性解释。其次，残差阈值由当前队列中位数定义，未来研究应在训练队列中预先指定或在外部队列中验证。第三，额外 EEG 池可以支持 SSL 预训练，但不能替代有标签外部验证。第四，可解释性结果具有模型依赖性，且并非所有探索性关联都能通过多重检验校正，因此不能作为因果机制或治疗靶点。第五，伦理批准号、评估者资质、盲法状态、电极尺寸、不良事件记录和数据仓库编号仍需正式投稿前补齐。

总体而言，本研究建立了一个 EEG-only 框架，用于预测 tACS 后上肢比例恢复并生成 PSD/WPLI 候选生物标志物。当前结果支持开展更大规模、前瞻性、外部验证研究，尤其是比较临床-only、EEG-only 和联合模型的研究，并在独立队列中预先检验本文提出的 EO PSD 与 EC beta 连接假设。

## 结论

治疗前基线静息态 EEG 结合残差感知自监督 CNN 建模，可在卒中 tACS 队列中进行上肢比例恢复的患者层面预测。该方法利用完整 PSD 和 WPLI EEG 结构，通过 SSL 纳入无完整监督标签的 EEG 记录，并在训练阶段保留连续残差信息。当前发现为 EEG 引导 tACS 反应预测提供了可复现的建模框架和候选神经生理假设。

## 数据可用性

支持本文结果的去标识化派生数据将在最终投稿或发表前存入可引用的数据仓库。拟公开的数据应包括患者层面队列表、锁定 LOSO 预测、10 种子模型汇总、bootstrap 和置换输出、参与者流程源注释、图源数据汇总、可解释性表格、验证审计、模型报告卡和图件 manifest。原始 EEG、最低限度处理 EEG 文件以及可能直接识别患者身份的临床源记录不在本初稿中公开，因为这些数据属于人类受试者数据，可能受到伦理审批、知情同意、隐私和数据使用限制。受限原始或最低限度处理数据的访问应由负责机构审查。合格研究者可在获得伦理批准并签署数据使用协议后，向通讯作者或机构数据访问委员会提出申请，具体取决于原始知情同意和机构限制。仓库 DOI、许可证、访问委员会名称和最终数据版本待填。

## 代码可用性

建模、统计验证、MNE topomap/connectivity 渲染、图件生成和稿件表格生成脚本将在最终投稿或发表前存入公共代码仓库或链接代码归档。最终归档应包括精确 commit hash 或版本标签、环境元数据、随机种子、锁定预测 CSV，以及从存储的派生数据复现稿件表格和图件所需的脚本。

## 图例汇总

**图1｜总体技术框架。** 文件：`results/figures/revised_initial/figure1_overall_framework_provided.png`。

**图2｜患者层面 Barlow SSL 预训练。** 文件：`results/figures/revised_initial/figure2_ssl_framework_provided.png`。

**图3｜残差感知患者层面 Barlow SSL-CNN。** 文件：`results/figures/revised_initial/figure3_cnn_residual_aware_provided.png`。

**图4A｜最终残差感知 SSL-CNN 的 ROC 曲线。** 单独展示锁定 seedmean10 患者层面预测的 ROC 曲线，ROC-AUC = 0.844。文件：`results/figures/revised_initial/figure4a_final_model_roc.png`。

**图4B｜最终残差感知 SSL-CNN 的混淆矩阵。** 固定阈值 0.5 下，TP = 10、FN = 0、FP = 3、TN = 6；敏感度 = 1.00，特异度 = 0.67，准确率 = 0.84。文件：`results/figures/revised_initial/figure4b_final_model_confusion_matrix.png`。

**图4C-a｜不同模型的患者层面指标直方图。** 比较传统 EEG-ML、no-SSL CNN、Barlow CNN 和最终残差感知 SSL-CNN 的分类与排序指标。文件：`results/figures/revised_initial/figure4c_a_model_metric_histogram.png`。

**图4C-b｜不同模型的 Brier 校准误差。** 比较传统 EEG-ML、no-SSL CNN、Barlow CNN 和最终残差感知 SSL-CNN 的患者层面概率校准误差，Brier 分数越低越好。文件：`results/figures/revised_initial/figure4c_b_brier_calibration.png`。

**图4D｜最终残差感知 SSL-CNN 的训练和验证损失曲线。** 汇总 10 个随机种子和 19 个 LOSO 折的总损失、分类损失和加权残差感知辅助损失。文件：`results/figures/revised_initial/figure4d_final_model_loss_curves.png`。

**图5｜稳定性与消融。** 展示核心深度模型的种子级准确率分布、SSL/残差感知消融指标热图、特征/状态/频段消融排名，以及输入维度与 ROC-AUC/PR-AUC 的信息效率关系。热图颜色为列内归一化表现，Brier 分数已反向处理。文件：`results/figures/revised_initial/figure5_stability_ablation.png`。

**图6A｜PSD 频段归因 topomap。** EO 和 EC 两行展示 delta、theta、alpha、beta-low、beta-medium、beta-high 和 gamma 七个频段的 signed PSD attribution。文件：`results/figures/revised_initial/figure6a_psd_topomap_bands.png`。

**图6B｜WPLI/FC 频段连接归因图。** EO 和 EC 两行展示六个频段的 top-20 WPLI 边，红色表示正向归因，蓝色表示负向归因，线宽表示绝对归因强度。文件：`results/figures/revised_initial/figure6b_wpli_connectivity_bands.png`。

## 参考文献

1. Prabhakaran S, Zarahn E, Riley C, Speizer A, Chong JY, Lazar RM, et al. Inter-individual variability in the capacity for motor recovery after ischemic stroke. Neurorehabil Neural Repair. 2008;22:64-71. doi:10.1177/1545968307305302
2. Byblow WD, Stinear CM, Barber PA, Petoe MA, Ackerley SJ. Proportional recovery after stroke depends on corticomotor integrity. Ann Neurol. 2015;78:848-859. doi:10.1002/ana.24472
3. Stinear CM, Byblow WD, Ackerley SJ, Smith M, Borges VM, Barber PA. PREP2: a biomarker-based algorithm for predicting upper limb function after stroke. Ann Clin Transl Neurol. 2017;4:811-820. doi:10.1002/acn3.488
4. Lin PJ, Zhai X, Li W, Li T, Cheng D, Li C, et al. A transferable deep learning prognosis model for predicting stroke patients' recovery in different rehabilitation trainings. IEEE J Biomed Health Inform. 2022;26:6003-6011. doi:10.1109/JBHI.2022.3205436
5. White A, Saranti M, d'Avila Garcez A, Hope TMH, Price CJ, Bowman H. Predicting recovery following stroke: deep learning, multimodal data and feature selection using explainable AI. NeuroImage Clin. 2024;43:103638. doi:10.1016/j.nicl.2024.103638
6. Lassi M, Dalise S, Privitera L, Giannini N, Mancuso M, Azzollini V, et al. Enhancing upper limb motor recovery prediction after acute stroke using EEG and subacute data. APL Bioeng. 2026;10:016108. doi:10.1063/5.0287165
7. Mane R, Chew E, Phua KS, Ang KK, Robinson N, Vinod AP, et al. Prognostic and monitory EEG-biomarkers for BCI upper-limb stroke rehabilitation. IEEE Trans Neural Syst Rehabil Eng. 2019;27:1654-1664. doi:10.1109/TNSRE.2019.2924742
8. Saes M, Meskers CGM, Daffertshofer A, van Wegen EEH, Kwakkel G. Are early measured resting-state EEG parameters predictive for upper limb motor impairment six months poststroke? Clin Neurophysiol. 2021;132:56-62. doi:10.1016/j.clinph.2020.09.031
9. Tang CW, Hsiao FJ, Lee PL, Tsai YA, Hsu YF, Chen WT, et al. Beta-oscillations reflect recovery of the paretic upper limb in subacute stroke. Neurorehabil Neural Repair. 2020;34:450-462. doi:10.1177/1545968320913502
10. Vinck M, Oostenveld R, van Wingerden M, Battaglia F, Pennartz CMA. An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. NeuroImage. 2011;55:1548-1565. doi:10.1016/j.neuroimage.2011.01.055
11. Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci. 2013;7:267. doi:10.3389/fnins.2013.00267
12. Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, et al. MNE software for processing MEG and EEG data. NeuroImage. 2014;86:446-460. doi:10.1016/j.neuroimage.2013.10.027
13. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. Ann Intern Med. 2015;162:55-63. doi:10.7326/M14-0697
14. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378
15. Wolff RF, Moons KGM, Riley RD, Whiting PF, Westwood M, Collins GS, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. Ann Intern Med. 2019;170:51-58. doi:10.7326/M18-1376
16. Zhang Y, Gong G, Liu G, Xu S, Zeng J. Functional connectivity between non-motor and motor networks predicts motor recovery changes after stroke. Sci Rep. 2025;15:41448. doi:10.1038/s41598-025-19860-4
17. Zbontar J, Jing L, Misra I, LeCun Y, Deny S. Barlow Twins: self-supervised learning via redundancy reduction. Proceedings of the 38th International Conference on Machine Learning. 2021. https://proceedings.mlr.press/v139/zbontar21a.html
18. Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. Proceedings of the 34th International Conference on Machine Learning. 2017. https://proceedings.mlr.press/v70/sundararajan17a.html
19. Smilkov D, Thorat N, Kim B, Viegas F, Wattenberg M. SmoothGrad: removing noise by adding noise. arXiv. 2017. https://arxiv.org/abs/1706.03825
20. Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: an imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems. 2019. https://papers.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
21. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
22. Yuan K, Chen C, Lou WT, Khan A, Ti ECH, Lau CCY, et al. Differential effects of 10 and 20 Hz brain stimulation in chronic stroke: a tACS-fMRI study. IEEE Trans Neural Syst Rehabil Eng. 2022;30:455-464. doi:10.1109/TNSRE.2022.3153353
23. Tozlu C, Edwards D, Boes A, Labar D, Tsagaris KZ, Silverstein J, et al. Machine learning methods predict individual upper-limb motor impairment following therapy in chronic stroke. Neurorehabil Neural Repair. 2020;34:428-439. doi:10.1177/1545968320909796
24. AlArfaj AA, Hosni Mahmoud HA, Hafez AM. A deep learning model for stroke patients' motor function prediction. Appl Bionics Biomech. 2022;2022:1-9. doi:10.1155/2022/8645165
25. Singh S, Dawar D, Mehmood E, Pandian JD, Sahonta R, Singla S, Amit Batra, Cheruvu S, et al. Determining diagnostic utility of EEG for assessing stroke severity using deep learning models. Biomed Eng Adv. 2024;7:100121. doi:10.1016/j.bea.2024.100121
26. Zhang JJ, Bai Z, Fong KNK. Resting-state cortical electroencephalogram rhythms and network in patients after chronic stroke. J NeuroEngineering Rehabil. 2024;21:32. doi:10.1186/s12984-024-01328-7
