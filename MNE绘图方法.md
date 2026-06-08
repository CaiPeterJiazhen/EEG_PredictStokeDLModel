你在项目 F:\CJZProjectFile\EEG_PredictStokeDLModel 中工作。请用 MNE 重新绘制高质量 EEG topomap 和 WPLI connectivity 图，要求如下：

1. 坐标布局
- 使用 configs/paths.example.yaml 里的 standard_1005_ced 文件。
- 不要使用 MNE standard_1005 三维 montage 直接投影，也不要用 .ced 的 X/Y/Z 列来做 2D topomap。
- 必须使用 .ced 文件中的 EEGLAB 2D 坐标：theta/radius。
- 2D 转换方式：
  x = radius * sin(theta)
  y = radius * cos(theta)
- 将所有电极坐标整体缩放，使最大半径约为 0.96，保证电极点填满头模型但仍在头模型内。
- 头模型使用 MNE plot_topomap 的 outlines="head"，sphere=(0,0,0,1)。

2. Topomap 数据
- 读取 results/explainability/psd_channel_band_importance.csv。
- PSD topomap 使用 mean_signed_attribution，不是原始 PSD power。
- 生成 EO/EC × Delta, Theta, Alpha, Beta Low, Beta Medium, Beta High 共 12 张。
- 使用 mne.viz.plot_topomap(data, pos)，根据修正后的电极位置重新插值热图。
- cmap 用 RdBu_r，色标正负对称。
- 输出 PNG 和 SVG 到 results/figures/explainability/mne_topomaps。
- 默认不要显示通道名，避免标签重叠；可以支持 --show-names。

3. WPLI connectivity 数据
- 读取 results/explainability/wpli_top_edges.csv。
- 这是 WPLI 边的解释性重要性，不是重新从原始 EEG 计算 WPLI。
- 生成 EO/EC × Delta, Theta, Alpha, Beta Low, Beta Medium, Beta High 共 12 张。
- 每张取 mean_abs_attribution 排名前 20 的边。
- 节点位置使用和 topomap 完全相同的 EEGLAB theta/radius 2D 坐标。
- 红色边表示 mean_signed_attribution >= 0，蓝色边表示 < 0。
- 线宽按 mean_abs_attribution 缩放。
- 图形风格参考“头模型 + 电极点/标签 + 红蓝连接线”的 EEG scalp connectivity 图。
- 输出 PNG 和 SVG 到 results/figures/explainability/mne_wpli_connectivity。

4. 质量要求
- 图不能空白。
- 电极点必须填满头模型，最大半径接近 0.96。
- Fpz 在前方上侧，Oz 在后方下侧，T7 在左，T8 在右，CB1/CB2 在枕后下方。
- 不要把电极缩在头模型中心，也不要让电极点跑到头模型外。
- 如果 import mne 卡住，先设置 NUMBA_CACHE_DIR 为项目内 .numba_cache。

5. 验证
- 增加或运行测试，至少验证：
  - Fpz/Oz/T7/T8/CB1/CB2 方位正确。
  - 最大电极半径在 0.94 到 0.98 之间。
  - WPLI top edge 筛选按 state/band 过滤，并按 mean_abs_attribution 降序取 top N。
- 运行：
  python -m pytest tests\test_mne_topomap_coordinates.py tests\test_mne_wpli_connectivity.py -q
  python -m py_compile scripts\45_make_mne_explainability_topomaps.py scripts\46_make_mne_wpli_connectivity.py
- 生成后检查 manifest 行数、PNG/SVG 数量和图片非空。
- 最后预览一张 topomap 和一张 connectivity 代表图。