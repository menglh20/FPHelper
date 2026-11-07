# 公共算法模块

公共模块位于 fphelper，供 SB0 至 SB3 的版本实现与评估工具使用。

| 模块 | 用途 |
| --- | --- |
| [versions.py](fphelper/versions.py) | 显式注册版本文件夹与方法标识 |
| [training.py](fphelper/training.py) | 训练折标准化、PCA、嵌套验证和模型序列化接口 |
| [estimation.py](fphelper/estimation.py) | 13个分项的公共拟合与预测 |
| [ordinal.py](fphelper/ordinal.py) | 有序逻辑回归实现 |
| [video.py](fphelper/video.py) | 视频解码、几何和冻结 ResNet18 特征提取 |
| [data.py](fphelper/data.py) | 医生标注解析、合法范围、分折和总分公式 |
| [metrics.py](fphelper/metrics.py) | 训练流程的评分指标 |
| [score_contract.py](fphelper/score_contract.py) | 动作核验与可评分状态 |
| [group_protocol.py](fphelper/group_protocol.py) | 患者分组协议 |
| [cli.py](fphelper/cli.py) | 原有训练和推理命令行入口 |

src/fphelper 保留兼容导入层，使用 PYTHONPATH=src 的原命令继续可用。安装后的 Python 包直接从这里和各版本文件夹构建，并保留历史模型的模块名。
