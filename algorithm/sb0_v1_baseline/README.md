# SB0 第一版分项众数基线

训练折每个分项的众数；并列时选较低等级。不使用视频特征。

[model.py](model.py) 保存该版本的实际算法入口，[config.json](config.json) 记录方法与配置。

`fit(labels)` 拟合训练分项众数，`predict(grades, count)` 输出指定数量的评分。

公共实现见[训练与验证](../shared/fphelper/training.py)、[特征提取](../shared/fphelper/video.py)和[序数估计](../shared/fphelper/estimation.py)。本版本通过[统一评估入口](../../tools/benchmark/README.md)参加固定五折比较。

版本标识为 `sb0_v1_baseline`。后续改进另建新版本文件夹，保留本版本以复现实验。
