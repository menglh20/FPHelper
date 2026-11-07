# FPHelper

面瘫视频标注、Sunnybrook 算法研究及历史 Django 后端代码。算法版本、本地工具、自动化测试和实验报告按用途分别维护。

## 目录导航

| 目录 | 用途 | 入口 |
| --- | --- | --- |
| algorithm/ | 旧版与 SB0 至 SB3 的版本实现、公共训练和特征模块 | [算法版本](algorithm/README.md) |
| backend/ | 历史 Django 服务，包含用户、记录与旧检测接口 | [后端说明](backend/README.md) |
| tools/ | 本地网页标注和统一算法评价工具 | [工具说明](tools/README.md) |
| tests/algorithm/ | 算法、评分协议和版本兼容性自动化测试 | [测试说明](tests/README.md) |
| docs/ | 仓库结构说明、实验报告和汇总图 | [文档索引](docs/README.md) |
| requirements/ | 研究环境依赖锁文件 | [依赖说明](requirements/README.md) |
| src/ | 原有 fphelper 导入路径的兼容入口 | [兼容层说明](src/README.md) |
| artifacts/ | 本机标注、特征、模型、评估数据、日志和历史归档，已被 Git 忽略 | [存储规则](docs/repository-layout.md) |

完整的目录职责和新增文件规则见[仓库结构说明](docs/repository-layout.md)。微信小程序和其他后端仓库独立维护。

## 本地视频标注

```sh
python3 tools/annotator/server.py --open
```

也可以在 Finder 中双击 tools/annotator/启动标注工具.command。使用方法、数据目录选择与导出格式见[标注工具说明](tools/annotator/README.md)。

## 算法研究与评价

在 Python 3.12 研究环境中，从仓库根目录运行：

```sh
PYTHONPATH=src python -m fphelper.cli --help
```

[统一评价工具](tools/benchmark/README.md)使用固定五折和同一指标比较旧版规则、SB0 至 SB3 与年轻医生。[当前评价报告](docs/experiments/unified-evaluation.md)包含共同视频上的误差、完成率、13个分项和联动结果。

## 测试

```sh
PYTHONPATH=src python -m unittest discover -s tests -q
```

工具测试保留在各工具目录内，完整命令见[测试说明](tests/README.md)。
