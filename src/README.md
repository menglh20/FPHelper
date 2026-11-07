# Python 导入兼容层

本目录保留 fphelper 的兼容导入入口，供已有 PYTHONPATH=src 命令使用。

实际公共代码位于[algorithm/shared/fphelper](../algorithm/shared/README.md)，模型实现位于 algorithm 中各自的版本目录。安装包通过 pyproject.toml 直接从这些目录构建。

新增算法实现放入 algorithm 对应版本目录，公共实现放入 algorithm/shared/fphelper。已有 fphelper.training.SunnybrookModel 等模型模块名继续可用。
