# White et al. 2024 结构化阅读稿

Source PDF: `C:\Users\HPGZZ\Desktop\预后模型相关文献\White 等 - 2024 - Predicting recovery following stroke Deep learning, multimodal data and feature selection using exp.pdf`

## 阅读目的

该文作为本文第二个写作参照，重点用于学习卒中恢复预测论文中 Methods、Experiments/Results、Explainable AI 和 Discussion 的组织方式。本文不复述全文，而是保留页码、章节和图表锚点，便于将参考论文的写法转化为当前 EEG-tACS 论文的正文结构。

## 章节写作功能

### 2.1 Dataset

先交代队列来源、时间范围、纳入/排除和标签，再说明不同输入数据集的构造方式。对本文的启发是：研究对象小节不仅列样本量，还要解释哪些患者进入监督训练、哪些患者只能进入无标签 SSL 池，以及这些数据可用性如何影响模型问题。

### 2.2 CLEAR image explainable AI system

方法部分只定义可解释性工具、输入和输出，不把解释结果提前写进去。对本文的启发是：可解释性方法应写成归因、遮挡、topomap 和 connectivity 的分析流程，解释性结论放在 Results。

### 3 Experiments and inference

按模型家族逐一说明 baseline、轻量网络、深度网络、多模态模型和 DAFT，并为每个实验说明比较目的。对本文的启发是：传统 ML、CNN、SSL、残差感知和最终模型应分段写清楚各自解决什么问题，而不是只罗列模型名。

### 4 Results

先报告模型比较，再报告特征选择与可解释性输出，表格和图件紧跟支撑的结论。对本文的启发是：最终模型表现应包含 ROC、混淆矩阵、模型指标对比和损失曲线，并在正文中逐图解释。

### 5 Discussion and future work

围绕中心发现、相对已有方法的意义、可解释性发现、数据限制和后续验证展开。对本文的启发是：讨论要解释为什么 EEG-only、SSL 和残差感知各自有价值，同时明确小样本内部验证的边界。

## 主要图表锚点

- **C001 (p.3)** Fig. 1. Left: An example of a stitched MRI consisting of sixty-four axial cross-sectional slices from an MRI scan. Right: An ROI Image consisting of the 12 key (most predictive) ROIs (see Section 3.2). The dotted red lines have been added to this figure for visual clarity, demarcating the boundaries of the left superior temporal gyrus, middle temporal gyrus and inferior frontal gyrus-triangular. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)
  - 中文用途：展示不同影像输入构造方式，用于说明模型输入和特征选择方案。
- **C002 (p.5)** Fig. 2. Plots of how balanced validation loss and balanced test accuracy vary with the number of ROIs displayed in ROI Images. The balanced validation loss was used to determine that 8 ROIs should be included in each ROI Image. Note that the balanced test accuracy only varies slightly with number of ROIs, achieving > 0.79 with only three ROIs.
  - 中文用途：展示不同影像输入构造方式，用于说明模型输入和特征选择方案。
- **C003 (p.5)** Fig. 3. Left: A Hybrid stitched MRI, after pre-processing which reshapes it to 256 x 256. Right: A hybrid ROI image consisting of twelve ROIs plus the symbols for initial severity (normal for this patient), left lesion size and recovery time.
  - 中文用途：展示不同影像输入构造方式，用于说明模型输入和特征选择方案。
- **C004 (p.6)** Fig. 4. Example of a Clear Image output. This explains the classification probability determined by ResNet-18 for the stitched MRI of patient 108. CLEAR Image estimates the feature importance scores that the ResNet-18 has used in determining the classification probability. CLEAR Image also shows the logistic regression equation it generated for this stitched MRI (top left), some counterfactuals and fidelity errors – these are explained in White et al. (2023).
  - 中文用途：展示不同影像输入构造方式，用于说明模型输入和特征选择方案。
- **C005 (p.7)** Table 1 Accuracy results on the lock box test data. Accuracies and confidence intervals are calculated across the four folds of our cross validation. (I = uses image data, T = uses tabular data.).
  - 中文用途：汇总模型性能或特征选择结果，用于支撑结果段落中的定量比较。
- **C006 (p.7)** Table 2 Area under the ROC curve and F1-scores for the test dataset. Accuracies and confidence intervals are calculated across the four folds of our cross validation. (I = uses image data, T = uses tabular data).
  - 中文用途：汇总模型性能或特征选择结果，用于支撑结果段落中的定量比较。
- **C007 (p.8)** Table 3 Comparison of unbalanced accuracy for different cutoff thresholds (confidence intervals are not shown for ease of reading). Accuracies are calculated across the four folds of our cross validation.
  - 中文用途：汇总模型性能或特征选择结果，用于支撑结果段落中的定量比较。
- **C008 (p.8)** Table 4 Accuracy results for modified versions of the Hybrid ROI images. For example, ‘Initial severity & Left lesion size’ refers to experiments carried out with a dataset of images each displaying seven ROIs plus the symbols representing the initial severity and left lesion size features, but without recovery time. Accuracies and confidence intervals are calculated across the four folds of our cross validation.
  - 中文用途：展示不同影像输入构造方式，用于说明模型输入和特征选择方案。
- **C009 (p.8)** Table 5 Accuracy results for ResNet models of different depths, trained on Hybrid ROI dataset. The number at the end of ResNet is the number of layers in the network. These are the four smallest Pytorch ResNet models for which ImageNet weights are available. Accuracies and confidence intervals are calculated across the four folds of our cross validation.
  - 中文用途：展示不同影像输入构造方式，用于说明模型输入和特征选择方案。
- **C010 (p.8)** Table 1′s mean accuracy results point to the Hybrid ROIs and ROIs datasets having greater prognostic information than their respective Stitched MRI datasets; and in the former case, Hybrid-ROIs w/ResNet-18 vs Stitched MRI w/ResNet18, we could show a statistic difference (see appendix A2). As Fig. 2 illustrates, this is the case even when the number of ROIs being displayed is only four, highlighting the benefit feature selection through explainable AI can bring. Importantly, the goal of the work reported here was to obtain high classification accuracy and look at which tabular data improved the classifications. This was done in the context of assessing the effective­ ness of deep learning, applied to MRI stroke data. In particular, in this paper, we are not illuminating the key, but difficult, question of explaining how the classifier has used the features available to it – either those in the MRI scans or the tabular features. Accordingly, we are not providing an exact description of how different regions enable good performance. This is consistent with prior studies, which have typically not been able to explain the critical com­ bination of damage behind their predictions. Thus, we are looking at the combination of features, not which ones are dominating.
  - 中文用途：展示不同影像输入构造方式，用于说明模型输入和特征选择方案。
- **C011 (p.12)** Fig. A1. Results of false positive test. All tests were two-tailed. Mostly paired tests are presented, but we include independent tests for standard t, Wilcoxon and permutation. This is because the paired Wilcoxon fails as a result of insufficient orderings of four items (in fact, permutation-paired faces a similar shortage of orderings, but manages to provide p-values by virtue of the variably introduced by the Monte-Carlo resampling). We wanted to make clear that the problem being considered is not due to loss of normality, i.e. Wilcoxon-independent, as well as the permutation procedures (none of which make normality assumptions), also exhibit very substantial inflation of the false positive rate. t-paired-bast1, t-paired-bast2 and t-paired-bast3 are the new method with corrected degrees of freedom. These are scaled down to three different levels downScale1 = 0.45, downScale2 = 0. 435 and downScale3 = 0.4. t-paired-bast1 has the closest type-I error rate to alpha; it is 4.65 %, i.e. just below 5 % = alpha × 100.
  - 中文用途：展示可解释人工智能输出如何将模型概率与关键特征或图像区域对应。
- **C012 (p.13)** Table A2.1 Statistical inference on across folds balanced accuracies of Hybrid ROIs w/ResNet-18 against all other models (with each row this contrast for one model). Table 1 in main body of paper shows the descriptive statistics for the same models. Unlike in Table 1, here, models are ordered from largest (adjusted) t-value to smallest, which also sorts p-values smallest to largest. Mean accuracies and standard deviations are as presented in Table 1. t-stat (adj) is the t-statistic introduced in appendix A1, with degrees of freedom adjusted using the scaling factor 0.45. p-value is the corresponding p-value, calculated from t-stat (adj), with adjusted degrees of freedom. BH threshold is the False Discovery Rate (FDR) adjustment of a 0.05 statistical threshold, with nine comparisons, using the Benjamini–Hochberg procedure (Benjamini and Hochberg, 1995). BH p-adj is the adjusted p-value implied by the Benjamini–Hochberg procedure; accordingly, these p-values can be considered relative to a 0.05 threshold.
  - 中文用途：展示不同影像输入构造方式，用于说明模型输入和特征选择方案。

## 页面图像

页面级图像位于 `assets/page_XX.png`，用于人工核对图表和正文位置。