# 卒中后比例恢复的残差感知 EEG 学习

## 摘要

准确预测卒中后上肢恢复有助于更早进行康复分层，但 EEG 预后模型常受限于标注队列较小以及验证流程容易发生信息泄漏。我们开发了一种残差感知自监督卷积神经网络，整合经颅交流电刺激前基线静息态睁眼和闭眼 EEG 的功率谱密度以及加权相位滞后指数连接特征。监督学习队列包括 19 例同时具备基线和治疗后上肢 Fugl-Meyer 评定分数的患者。标签由预期运动改善与实际运动改善之间的残差定义。模型性能采用患者层面的留一受试者交叉验证、10 个随机种子、受试者层面 bootstrap 区间、置换检验和配对比较进行评估。

该模型的准确率为 84.2%，平衡准确率为 83.3%，ROC-AUC 为 0.844，PR-AUC 为 0.836，Brier 分数为 0.126。与传统 PSD+WPLI 逻辑回归 EEG 基线相比，该模型在 ROC-AUC 上数值更高、Brier 分数更低；与结构匹配但未使用自监督预训练的 CNN 相比，其排序和校准指标也呈有利方向变化。不过，这些配对差异未达到统计学上的确定性。消融分析提示，残差感知辅助监督是当前队列中最稳定的训练信号，而自监督预训练未显示稳定的独立增益。模型解释结果突出显示了状态和频段特异性的 PSD 模式以及 beta 频段连接。

残差感知 EEG 学习能够在保留二分类预后推断的同时利用连续恢复信息。这些发现仍属探索性结果，在临床使用前需要更大规模的前瞻性外部验证。

## 引言

卒中后上肢恢复在患者之间存在显著差异，即使基线运动损伤程度相似，恢复轨迹也可能不同。比例恢复框架及相关生物标志物算法表明，这种差异在很大程度上可以由基线损伤、皮质脊髓束完整性以及后续运动改善之间的关系来概括 [1-3]。经颅交流电刺激是一种正在发展的非侵入性神经调控方法，可对卒中后运动网络活动产生频率特异性影响 [22]。然而，许多康复研究仍缺乏可在床旁重复采集、并可整合进机器学习预测模型的可扩展生物标志物。

静息态 EEG 适合用于这一目的，因为它具有非侵入、成本较低、并且对卒中后振荡功率和功能连接变化敏感等特点。既往研究已将 EEG 生物标志物与运动损伤和康复反应联系起来，包括 beta 频段振荡活动、静息态连接，以及脑机接口康复场景中使用的 EEG 特征 [7-9,16]。WPLI 可降低电生理记录中零相位滞后耦合对连接估计的膨胀作用，因此适合作为头皮 EEG 预测研究中的连接特征 [10]。近期卒中预后研究还提示，机器学习、深度学习和可解释多模态模型能够从神经影像、临床变量和 EEG 特征中提取具有临床意义的信息 [4-6,23,24]。EEG 深度学习模型也已用于卒中严重程度评估，这支持了 EEG 派生神经特征的总体可行性，但治疗反应预后的问题仍未解决 [25]。

尽管前景明确，EEG 预后研究仍面临三类反复出现的方法学风险。第一，卒中标注队列通常较小，容易导致过拟合和估计不稳定。第二，如果模型评估不是在患者层面进行，片段级 EEG 样本可能造成虚假的样本量扩大。第三，从比例恢复导出的二分类标签可能丢失预期运动增益和实际运动增益之间残差所包含的连续信息。因此，我们设计了一种残差感知自监督 CNN：先利用未标注 EEG 预训练患者层面表征，再同时使用二分类恢复标签和辅助残差排序信息监督网络。

本研究的目标是检验残差感知自监督多模态 EEG 学习能否在卒中后试点队列中支持患者层面的上肢比例恢复预测。我们根据预测模型报告规范报告模型，包括受试者层面交叉验证、面向校准的评分、bootstrap 不确定性、置换检验、消融实验和模型解释 [13-15]。

## 结果

### 队列和结局定义

最终监督学习队列包括 19 例具有基线 EEG 以及完整基线和随访运动评估的患者。平均年龄为 64.7 岁（标准差 6.5），11 例为女性，11 例为左侧上肢受累。基线 FMA 为 40.5（标准差 23.8），随访 FMA 为 45.5（标准差 23.2），观察到的 FMA 改善为 5.0 分（标准差 4.1）。预测和观察到的 FMA 改善之间的残差中位数为 1.5 分。10 例患者被分入比例恢复组，9 例被分入恢复不良组。描述性队列统计见表 1。

全体患者 EEG 池包含 28 例患者，其中 19 例纳入监督留一受试者交叉验证评估，9 例仅用于无监督或自监督学习以及描述性分析。28 例 EEG 索引患者中有 20 例具备随访 FMA 和改良 Barthel 指数（MBI）数值。

### 残差感知 SSL-CNN 的性能

在患者层面的 LOSO 评估中，最终残差感知 SSL-CNN 的准确率为 0.842，平衡准确率为 0.833，敏感度为 1.000，特异度为 0.667，ROC-AUC 为 0.844，PR-AUC 为 0.836，Brier 分数为 0.126（表 2；图 2）。无 SSL 的 CNN 参照模型具有相同的硬标签准确率和平衡准确率，但 ROC-AUC 较低（0.811）、PR-AUC 较低（0.808）、Brier 分数较高（0.177）。传统 PSD+WPLI 逻辑回归 EEG 基线的准确率为 0.737，平衡准确率为 0.733，ROC-AUC 为 0.711，PR-AUC 为 0.775，Brier 分数为 0.208。

配对 bootstrap 比较量化了有利的数值差异，但在这一小队列中不支持确定性的优越性结论。相对于无 SSL 的 CNN，残差感知 SSL-CNN 的准确率和平衡准确率相同。ROC-AUC 高 0.033（95% bootstrap 区间 -0.144 至 0.214；双侧 p = 0.788）。PR-AUC 高 0.028（-0.170 至 0.224；p = 0.824），Brier 分数低 0.051（-0.121 至 0.004；p = 0.078）。相对于逻辑回归 EEG 基线，准确率高 0.105（-0.105 至 0.316；p = 0.423），ROC-AUC 高 0.133（-0.216 至 0.476；p = 0.458），Brier 分数低 0.082（-0.189 至 0.037；p = 0.169）。硬预测的 McNemar 检验未显示残差感知 SSL-CNN 与逻辑回归 EEG 基线之间存在显著差异（不一致计数为 1 对 3；p = 0.625）。残差感知模型和无 SSL CNN 模型之间没有不一致的硬预测。补充表 9 和补充图 3 的性能精度审计进一步显示，一个硬预测发生改变即可使准确率移动 5.3 个百分点。最终模型在准确率、ROC-AUC、PR-AUC 和 Brier 分数上的 bootstrap 区间仍然较宽。

基于受试者层面标签的置换检验支持最终模型具有高于机会水平的区分能力，其中准确率 p = 0.004，平衡准确率 p = 0.003，ROC-AUC p = 0.005，PR-AUC p = 0.015。Brier 分数通过描述性方式和配对 bootstrap 比较解释，因为较低数值代表更好性能，而已存储的置换表采用了统一的上尾方向。

### 探索性临床和增量分析

基线临床变量本身在该队列中已经具有较强信息量。仅使用基线年龄、性别、病程、受累侧、FMA 和 MBI 的临床逻辑回归模型达到 ROC-AUC 0.911、PR-AUC 0.899 和 Brier 分数 0.105。在配对 bootstrap 分析中，将 EEG 特征加入临床变量并未相对于最佳临床-only 模型提供稳定的增量收益（补充表 4；补充图 4）。这些分析被作为探索性支持分析报告，因为本研究的主要建模问题是 EEG 预测，而且当前样本量不足以对临床模型、EEG-only 模型和临床加 EEG 多模态模型进行确定性比较。

### 消融和稳健性分析

消融实验区分了残差感知辅助监督和自监督预训练各自的贡献。带残差感知辅助头、但不使用 SSL 的匹配 CNN 在当前消融表中达到种子均值准确率 0.895、平衡准确率 0.889、ROC-AUC 0.922、PR-AUC 0.927 和 Brier 分数 0.110。最终患者层面的 Barlow SSL 加残差感知模型达到准确率 0.842、平衡准确率 0.833、ROC-AUC 0.844、PR-AUC 0.836 和 Brier 分数 0.126。未使用残差感知辅助头的患者层面 Barlow SSL 模型稳定性较差，其集成 ROC-AUC 为 0.700，PR-AUC 为 0.631。总体而言，这些消融结果支持在监督微调阶段使用连续残差信息；但它们不能证明 Barlow 风格自监督预训练在这一小队列中带来稳定的独立性能增益（表 3；图 3）。

特征族消融显示，PSD-only 模型保留了有用的区分能力（ROC-AUC 0.811），而 WPLI-only 和受限连接特征集准确率较低，但在部分配置中仍携带排序信息。在当前特征选择分析中，闭眼特征比仅睁眼特征更有信息量。由于这些消融基于较小 LOSO 折和多个特征子集，因此应解释为机制支持，而不是独立验证性检验。

### 模型解释和 EEG 生物标志物定位

模型解释结合了 SmoothGrad 平滑的 integrated gradients、分支/状态遮挡、EEG 拓扑图、WPLI connectome 摘要以及网络层面归因摘要（图 4；补充图 2）。最突出的 PSD 贡献涉及睁眼低频额中央通道以及较高频段的围中央或颞部通道。连接解释强调闭眼 beta 频段 WPLI 边，包括额叶、中央区和颞部节点对。若干排名靠前的 WPLI 边在多重比较校正前显示出归因幅度与残差相关距离指标之间的关联，但错误发现率校正后的支持较弱。

这些发现与既往关于卒中后运动恢复和振荡性 beta 活动、静息态 EEG 特征以及运动网络功能连接之间关系的报道一致 [7-9,16]。不过，解释分析具有模型依赖性和关联性。它们应被视为假设生成型生物标志物，而不是某一特定通道、频段或网络边的因果证据。

## 讨论

本试点研究检验了一种残差感知自监督多模态 EEG 模型，用于预测卒中后上肢比例恢复。最终模型在 LOSO 评估中取得了有希望的患者层面区分能力，并且相对于更新后的 PSD+WPLI 逻辑回归 EEG 基线，在排序和校准导向评分上具有数值优势。与结构匹配的无 SSL CNN 相比，残差感知 SSL 并未提高硬标签准确率，但使 ROC-AUC、PR-AUC 和 Brier 分数向有利方向移动。配对比较未达到确定性显著，这与 19 例监督队列的规模相符，因此解释时必须保持谨慎。

本研究的主要方法学贡献是使用残差感知监督。二分类比例恢复标签便于临床解释，但可能丢弃偏离预期恢复的幅度和方向。通过训练辅助残差和排序目标，网络能够在保留二分类推断终点的同时利用连续恢复信息。消融结果提示，这一残差感知组件是当前数据集中最稳定的训练信号。相反，自监督预训练的独立获益仍未解决，需要在更大的 EEG 池和外部验证中检验。

我们采用自监督患者层面表征学习，以减少对标注结局的依赖，并利用未标注患者 EEG。不同于片段级增强，整个交叉验证中的评估单位始终是患者。这一点在 EEG 预后中至关重要，因为如果在片段层面进行训练-测试划分，受试者特异性 EEG 结构泄漏可能显著夸大性能。因此，我们的统计流程对所有报告指标、bootstrap 区间、置换检验和配对比较均使用受试者层面 LOSO 预测。

探索性临床分析显示，基线临床变量在该队列中具有很强预测能力。这与更广泛的恢复文献一致，在这些文献中，基线损伤及相关临床预测因子是恢复预后的核心 [1-3]。这也与既往机器学习研究一致，即临床和多模态变量可预测卒中治疗后的上肢损伤或恢复 [4-6,23,24]。当前数据不能证明 EEG 相对于临床变量 alone 增加了稳健的增量价值。因此，EEG 模型更应被视为一种候选神经生理预后标志物，需要更大规模的前瞻性测试、多模态临床整合和外部验证。

可解释性分析提示，网络依赖于具有生理合理性的 EEG 特征，包括状态依赖的 PSD 模式和 beta 频段连接。这与既往 EEG 和连接研究的收敛性提高了生物学合理性，但不能替代验证。显著性图和遮挡分析可能受到模型结构、预处理和相关输入的影响。在本研究中，它们最强的价值是为未来队列定义可检验的神经生理假设。

本研究有若干局限。监督样本量较小，仅有 19 例标注患者，且没有外部验证队列。临床终点和残差阈值均由当前可用队列导出，未来研究应预先指定最终标签定义。临床-only 模型表现较强，因此当前数据不支持 EEG 增量价值的主张。解释性分析仍为探索性，并且多重比较校正强度不足以支持确定性的通道层面或边层面生物标志物。最后，自监督学习池包含额外的 EEG 索引患者，但这些记录的范围和时间需要在最终投稿中完整报告。参与者、预测因子、结局、分析、缺失数据、解释性和可重复性风险，以及上述局限，均在保守的 PROBAST/TRIPOD+AI 导向审计中总结于补充表 8。补充表 10 的论断强度审计将核心稿件陈述映射到支持证据、可接受表述和应避免的过度主张。补充表 11 的 AI 模型报告卡总结了预期用途、验证保护措施、可重复性行动和不支持的临床用途。

总之，残差感知自监督 EEG 学习为卒中后比例上肢恢复的患者层面预测提供了一个紧凑且可解释的框架。该模型在试点 LOSO 队列中获得了有希望的区分能力和 Brier 分数表现，同时保持了保守的受试者层面验证。这些结果支持进一步前瞻性验证，而不是立即临床部署。

## 方法

### 研究队列

监督学习队列包括 19 例卒中患者，均具有基线静息态 EEG 以及完整的基线和治疗后 FMA-UE 测量。项目记录显示，所有患者采用共同的 tACS 方案。刺激靶点为受累手对侧的初级运动皮层，右手受损时刺激 C3，左手受损时刺激 C4。刺激频率为 20 Hz，强度为 1000 微安，每次 20 分钟，每日 1 次，连续 14 天，共 2 周。治疗后结局评估记录于最后一次 tACS 后立即进行。M1 临床源工作簿包含 29 例患者记录；其中 28 例在当前 EEG 目录中有索引，19 例构成最终监督标注队列。9 例 EEG 索引患者仅保留用于描述性或自监督分析，1 例临床工作簿患者未在当前 EEG 数据目录中索引。非监督或非索引条目的源工作簿备注包括刺激后 EEG 或随访数据缺失、未治疗或出院、依从性或认知交流困难、EEG 帽热不适、治疗后不适，以及一条治疗后不适并于次日出现高血压的备注。这些备注被视为参与者流程源注释，而不是完整的不良事件监测数据集。

### 结局定义

目标终点为基于预期 FMA-UE 改善和实际 FMA-UE 改善之间残差的比例恢复状态。预期改善按比例恢复规则计算：

`预测 Delta FMA = 0.7 x (66 - 基线 FMA-UE)`。

观察改善计算为：

`观察 Delta FMA = 治疗后 FMA-UE - 基线 FMA-UE`。

`残差 = 预测 Delta FMA - 观察 Delta FMA`。

残差小于或等于监督队列残差中位数阈值 1.5 分的患者被分入比例恢复组，残差高于该阈值的患者被分入恢复不良组。最终标签分布为 10 例比例恢复、9 例恢复不良。该中位数导出的阈值被报告为队列特异性建模终点，而不是经外部验证的临床截断值。

### EEG 预处理和特征提取

静息态 EEG 特征分别从 tACS 前采集的睁眼和闭眼记录中提取。分析从项目提供的预处理 EEGLAB `.set/.fdt` 文件开始。当前可用分析材料未包含原始采集日志或预处理方案。因此，采集硬件、在线参考、滤波、重参考、伪迹剔除以及导出前坏道处理仍是投稿前需要作者确认的字段。在本文的特征流程中，加载提供的预处理文件后未再进行额外时间滤波、伪迹剔除、通道插值或坏道移除。特征脚本仅在提取 PSD 和连接前验证有限的 62 通道连续数组、固定通道顺序和最小窗口长度。源项目记录显示，预处理 EEGLAB 文件包含每个文件一个 trial 的连续数据。这些文件采样率为 250 Hz，去除 M1 和 M2 后保留 62 个通道。38 个监督基线 EO/EC 文件中，记录时长平均为 188.4 秒，范围为 101.0 至 247.8 秒。文件名状态规则将基线 `*1.set` 文件分配为睁眼记录，将基线 `*2.set` 文件分配为闭眼记录。固定 62 通道顺序根据项目通道图进行检查。

PSD 特征在受累侧对齐后采用 Welch 方法计算，使用 Hann 窗、0.5 Hz 频率分辨率、50% 重叠、密度缩放和常数去趋势。固定 PSD 网格包含每个状态下 62 个通道的 0.5 至 45 Hz 共 90 个频率 bin。功能连接在相同受累侧对齐后，使用 2 秒 Hann 窗和 50% 重叠的短时傅里叶变换计算。WPLI 和 imaginary coherence 按确定性的通道对上三角列表计算。每个状态由此产生 1,891 条边乘以 6 个频段：delta（1-3 Hz）、theta（4-7 Hz）、alpha（8-13 Hz）、beta-low（13-18 Hz）、beta-medium（18-21 Hz）和 beta-high（21-30 Hz）。选择 WPLI 是因为它可以降低相位同步估计中体积传导和样本量偏倚的影响 [10]。在适用环节，EEG 预处理和拓扑可视化使用 MNE-Python [11,12]。

在 PSD 和连接计算前，特征矩阵被对齐到共同的受累手约定。右手受损患者保持不变，左手受损患者进行镜像，使表示对应于右手受损、C3 刺激和左半球刺激侧约定。PSD 和 WPLI 分支在训练折内标准化，基线模型的所有特征选择均在 LOSO 训练折内部进行，以避免测试折泄漏。

### 模型结构

最终模型为多模态 CNN，包含来自睁眼和闭眼状态的 PSD 与 WPLI 输入的独立分支。分支嵌入通过门控表示层融合为 32 维嵌入。患者层面 Barlow Twins 自监督学习用于在不使用监督标签的情况下从患者 EEG 中学习冗余降低的 EEG 表征 [17]。监督微调阶段将二分类损失与残差感知辅助损失结合，包括有符号残差距离和成对恢复排序目标。有符号距离定义为 `1.5 - residual`，正值表示恢复更接近或超过中位数比例恢复终点，负值表示恢复不良。在流程指定的地方，最终训练使用了随机权重平均。推断时，预测仅由分类头生成；残差头和排序头用于正则化训练，但不用于设定事后测试阈值。

基线模型包括传统 PSD+WPLI 机器学习分类器、具有相同核心结构的无 SSL CNN，以及不含残差感知辅助头的消融 CNN 变体。探索性临床模型使用基线-only 变量：年龄、性别、病程、受累侧、基线 FMA 和基线 MBI。治疗后结局、观察改善、预测改善、残差和标签均不作为任何模型输入。

### 交叉验证和统计分析

主要验证方案为患者层面的 LOSO 交叉验证。对于 CNN 模型，训练在 10 个随机种子上重复，面向稿件的预测被汇总为锁定结果表中预先指定的种子均值或种子集成输出。任何片段级行或种子级行均未被视为独立患者。

主要指标包括准确率、平衡准确率、敏感度、特异度、ROC-AUC、PR-AUC 和 Brier 分数。不确定性估计在可用情况下使用 5,000 次受试者层面 bootstrap 重采样。配对模型比较也按受试者重采样。McNemar 检验用于配对硬预测。置换检验使用受试者层面标签置换和 5,000 次置换。报告遵循 TRIPOD/TRIPOD+AI 原则，并承认样本量、结局定义和验证设计中的 PROBAST 相关偏倚风险 [13-15]。我们完成了保守的 PROBAST/TRIPOD+AI 导向风险审计，以便在投稿前明确偏倚、适用性和剩余作者确认要求。

### 模型解释

模型解释使用 SmoothGrad 平滑的 integrated gradients 进行特征归因 [18,19]。其他支持分析包括分支/状态遮挡、通道-频率归因的拓扑投影、WPLI 边层面摘要以及网络层面聚合。归因分析基于训练后的模型输出，被解释为模型依赖关联，而非因果神经生理机制。

### 软件

分析使用 Python、PyTorch、scikit-learn、MNE-Python 和项目特定脚本 [11,12,20,21]。本文图形生成使用 `scripts/45_make_nature_manuscript_figures.py`。生成的图形清单位于 `results/figures/nature/figure_manifest.csv`。

## 数据可用性

支持本文的去标识化派生数据将在最终期刊投稿或发表前存入可引用仓储。拟存入内容包括受试者层面分析表、锁定 LOSO 模型预测、bootstrap、置换和配对比较输出、参与者流程源注释、图形源摘要，以及验证和工件质量审计表。还将包括保守的 PROBAST/TRIPOD+AI 风险审计、性能精度审计、论断强度审计和数值论断来源追踪审计。其他存入材料包括 AI 模型报告卡、源工作簿作者元数据审计、作者字段替换图、患者病历 PDF 文本层审计和源数据工作簿。当前工作簿包含 36 个 worksheet，其中包括覆盖 37 个派生 CSV 或图形清单文件的 446 字段数据字典。本草稿不公开原始 EEG 记录、最低限度处理后的 EEG 文件、可识别临床源记录以及任何可直接链接到个体参与者的源文件。这些材料包含人类参与者数据，可能受机构审查委员会、知情同意、隐私和数据使用限制约束。受限原始或最低限度处理数据的访问应由责任机构审查。符合条件的研究者可在伦理审批和数据使用协议完成后，依据原始知情同意和伦理审批中的限制，向通讯作者或机构数据访问委员会申请访问。

## 代码可用性

建模、统计验证、源数据组装、图形生成、稿件生成、审计、模型卡和投稿包组装脚本将在最终期刊投稿或发表前随派生数据包存入仓储，或存入关联的公开代码仓储。最终代码归档应包括精确 commit hash 或版本标签、环境元数据、随机种子和锁定预测 CSV。还应包括能够从存入派生数据中复现稿件表格、图形、源数据工作簿和审计报告的脚本。

## 表格

表 1. 队列特征。源文件：`results/tables/table1_cohort_characteristics.csv`。

表 2. 主要模型性能。源文件：`results/tables/table2_main_model_performance.csv`。

表 3. 消融分析。源文件：`results/tables/table3_ablation.csv`。

表 4. 可解释性生物标志物。源文件：`results/tables/table4_explainability_biomarkers.csv`。

## 图注

图 1. 参与者流程、研究设计和残差感知 SSL-CNN 结构。M1 临床源工作簿包含 29 例患者记录，其中 28 例在当前 EEG 目录中有索引，19 例形成最终标注 LOSO 队列。参与者流程面板还显示了监督结局建模之外保留的 9 例 EEG 索引患者、当前 EEG 目录中未索引的 1 例临床源记录，以及 10/9 的比例恢复与恢复不良终点分布。模型整合了睁眼和闭眼 PSD 与 WPLI 分支、患者层面 Barlow 自监督预训练、残差感知辅助监督和分类头推断。图形文件：`results/figures/nature/figure1_study_design_model.png`。

图 2. 主要性能、不确定性和校准。相对于更新后的 PSD+WPLI 逻辑回归 EEG 基线，残差感知 SSL-CNN 显示出数值更高的 ROC-AUC 和 PR-AUC 以及更低的 Brier 分数。尽管硬标签准确率相同，它相对于无 SSL CNN 也显示出有利的评分指标。所有不确定性估计均使用受试者层面重采样。图形文件：`results/figures/nature/figure2_performance_calibration.png`。

图 3. 稳健性、消融和阈值敏感性。消融分析比较了传统 PSD+WPLI 机器学习、无 SSL CNN、不含残差感知头的患者层面 Barlow SSL、无 SSL 残差感知 CNN，以及组合残差感知 SSL-CNN。消融支持残差感知辅助监督作为有用训练信号，但不证明自监督预训练在该队列中带来独立性能增益。特征族分析总结了状态、模态、频段和连接贡献。图形文件：`results/figures/nature/figure3_robustness_ablation.png`。

图 4. 可解释性和神经生理解释。SmoothGrad 平滑的 integrated gradients、遮挡、拓扑图和 WPLI connectome 摘要将模型贡献定位到状态依赖 PSD 模式和 beta 频段功能连接。结果被解释为模型依赖和假设生成。图形文件：`results/figures/nature/figure4_explainability_neurophysiology.png`。

补充图 1. 重复错误受试者分析。事后受试者层面错误摘要识别在不同模型变体或随机种子中反复误分类的患者。该分析为探索性，应当用于指导未来队列复核，而不是事后修改标签。图形文件：`results/figures/nature/supplementary_error_subjects.png`。

补充图 2. MNE 渲染的 WPLI 连接归因图。使用固定 62 通道头皮布局，为每个 EEG 状态和频段渲染排名前 20 的 WPLI 归因边。红色和蓝色边表示平均归因符号，线宽按平均绝对归因幅度缩放。这些图是模型解释结果的探索性可视化摘要，不应解释为经独立验证的网络生物标志物。图形文件：`results/figures/explainability/mne_wpli_connectivity/mne_wpli_connectivity_contact_sheet.png`。

补充图 3. 性能精度和验证边界审计。面板 a 显示最终模型主要性能指标的 bootstrap 区间。面板 b 显示相对于无 SSL CNN 和逻辑回归 EEG 基线的配对 bootstrap 差异。零表示无差异，Brier 分数负差异代表最终模型更有利。面板 c 显示 19 例 LOSO 队列中一个病例变动对准确率、敏感度和特异度的影响。图形文件：`results/figures/nature/supplementary_performance_precision.png`。

补充图 4. 探索性临床基线和 EEG 增量价值审计。面板 a 比较选定的临床-only、EEG 加临床以及 EEG-only 模型指标。面板 b 显示 EEG 加临床候选模型相对于临床-only 逻辑模型的配对 bootstrap 差异；零表示无差异，Brier 分数的正差异代表临床-only 模型更有利。面板 c 在将每个指标转换为正值有利于候选模型后，总结所有探索性 EEG-only 和 EEG 加临床候选模型的方向性点估计。面板 d 显示相对于临床-only 模型的 ROC-AUC 获益和 Brier 分数获益。图形文件：`results/figures/nature/supplementary_clinical_incremental_value.png`。

## 参考文献

1. Prabhakaran S, Zarahn E, Riley C, Speizer A, Chong JY, Lazar RM, et al. Inter-individual variability in the capacity for motor recovery after ischemic stroke. Neurorehabil Neural Repair. 2008;22:64-71. doi:10.1177/1545968307305302
2. Byblow WD, Stinear CM, Barber PA, Petoe MA, Ackerley SJ. Proportional recovery after stroke depends on corticomotor integrity. Ann Neurol. 2015;78:848-859. doi:10.1002/ana.24472
3. Stinear CM, Byblow WD, Ackerley SJ, Smith M, Borges VM, Barber PA. PREP2: a biomarker-based algorithm for predicting upper limb function after stroke. Ann Clin Transl Neurol. 2017;4:811-820. doi:10.1002/acn3.488
4. Lin PJ, Zhai X, Li W, Li T, Cheng D, Li C, et al. A transferable deep learning prognosis model for predicting stroke patients' recovery in different rehabilitation trainings. IEEE J Biomed Health Inform. 2022;26:6003-6011. doi:10.1109/JBHI.2022.3205436
5. White A, Saranti M, d'Avila Garcez A, Hope TM, Price CJ, Bowman H. Predicting recovery following stroke: deep learning, multimodal data and feature selection using explainable AI. NeuroImage Clin. 2024;43:103638. doi:10.1016/j.nicl.2024.103638
6. Lassi M, Dalise S, Privitera L, Giannini N, Mancuso M, Azzollini V, et al. Enhancing upper limb motor recovery prediction after acute stroke using EEG and subacute data. APL Bioeng. 2026;10:016108. doi:10.1063/5.0287165
7. Mane R, Chew E, Phua KS, Ang KK, Robinson N, Vinod AP, et al. Prognostic and monitory EEG-biomarkers for BCI upper-limb stroke rehabilitation. IEEE Trans Neural Syst Rehabil Eng. 2019;27:1654-1664. doi:10.1109/TNSRE.2019.2924742
8. Saes M, Meskers CGM, Daffertshofer A, van Wegen EEH, Kwakkel G. Are early measured resting-state EEG parameters predictive for upper limb motor impairment six months poststroke? Clin Neurophysiol. 2021;132:56-62. doi:10.1016/j.clinph.2020.09.031
9. Tang CW, Hsiao FJ, Lee PL, Tsai YA, Hsu YF, Chen WT, et al. Beta-oscillations reflect recovery of the paretic upper limb in subacute stroke. Neurorehabil Neural Repair. 2020;34:450-462. doi:10.1177/1545968320913502
10. Vinck M, Oostenveld R, van Wingerden M, Battaglia F, Pennartz CMA. An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. NeuroImage. 2011;55:1548-1565. doi:10.1016/j.neuroimage.2011.01.055
11. Gramfort A. MEG and EEG data analysis with MNE-Python. Front Neurosci. 2013;7:267. doi:10.3389/fnins.2013.00267
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
25. Singh S, Dawar D, Mehmood E, Pandian JD, Sahonta R, Singla S, et al. Determining diagnostic utility of EEG for assessing stroke severity using deep learning models. Biomed Eng Adv. 2024;7:100121. doi:10.1016/j.bea.2024.100121
