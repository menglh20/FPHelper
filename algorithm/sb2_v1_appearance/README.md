# SB2 第一版外观特征模型

冻结 ImageNet ResNet18 的全脸和局部外观特征，经训练折标准化和 PCA 后输入13个有序回归头。

[model.py](model.py) 保存该版本的实际算法入口，[config.json](config.json) 记录方法与配置。

`inputs_for` 选择本版本特征；`fit` 和 `predict` 执行本版本的拟合与预测。输入为公共训练模块已投影的六阶段特征列表。

公共实现见[训练与验证](../shared/fphelper/training.py)、[特征提取](../shared/fphelper/video.py)和[序数估计](../shared/fphelper/estimation.py)。本版本通过[统一评估入口](../../tools/benchmark/README.md)参加固定五折比较。

版本标识为 `sb2_v1_appearance`。后续改进另建新版本文件夹，保留本版本以复现实验。
