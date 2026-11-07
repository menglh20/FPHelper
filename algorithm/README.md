# FPHelper 算法版本

算法实现按方法和版本分别保存。SB0 至 SB3 表示方法类型，v1 表示该方法的实现版本。当前版本共用训练、特征提取和评分基础模块。

```text
algorithm/
├── legacy_v0/             原版规则评分及视频取帧
├── sb0_v1_baseline/       分项众数基线
├── sb1_v1_geometry/       几何特征及有序回归
├── sb2_v1_appearance/     外观特征及有序回归
├── sb3_v1_fusion/         几何与外观融合
└── shared/fphelper/       公共特征、训练、指标和协议
```

| 文件夹 | 算法入口 | 内容 |
| --- | --- | --- |
| [legacy_v0](legacy_v0/README.md) | Detect.py 与 Recognize.py | 原版六照片评分和全视频极值选帧 |
| [sb0_v1_baseline](sb0_v1_baseline/README.md) | model.py | 分项众数基线 |
| [sb1_v1_geometry](sb1_v1_geometry/README.md) | model.py | 几何特征模型 |
| [sb2_v1_appearance](sb2_v1_appearance/README.md) | model.py | 外观特征模型 |
| [sb3_v1_fusion](sb3_v1_fusion/README.md) | model.py | 融合模型 |
| [shared](shared/README.md) | fphelper/training.py 与 video.py | 公共训练、特征提取、序数估计和统一评分协议 |

## 使用

在仓库根目录使用原有导入方式即可，src/fphelper 作为兼容入口指向这里的公共模块和版本目录：

```sh
PYTHONPATH=src python -m fphelper.cli --help
```

按方法取得当前明确注册的版本实现：

```python
from fphelper.versions import get_model
sb1 = get_model("SB1")
print(sb1.VERSION)  # sb1_v1_geometry
```

[统一评估工具](../tools/benchmark/README.md)调用上述版本实现。已有模型中的 fphelper.training.SunnybrookModel 和 fphelper.ordinal.OrderedLogit 导入路径继续可用。

版本目录保存代码、配置和说明；训练权重、特征缓存和患者数据衍生物统一保存在已被 Git 忽略的 artifacts 目录。旧版自带的 MediaPipe 检测权重保留在 legacy_v0。

## 后续版本

新增实现使用独立目录，例如 sb1_v2_manual_actions，并在 shared/fphelper/versions.py 中显式注册新的方法标识。保留已有版本目录及注册标识，便于历史模型复现。

共享特征、序数估计或评分协议发生变化时，记录共享代码指纹并重新进行统一评估。各版本的 config.json 描述当前实现；实验参数和实际执行代码指纹由评估 protocol.json 保存。

本次整理经过原固定五折预测和旧模型加载的一致性核对，评价结果仍见[统一评价报告](../docs/experiments/unified-evaluation.md)。
