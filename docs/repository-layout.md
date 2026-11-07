# FPHelper 仓库结构

仓库按算法、工程代码、工具、测试、文档和本机数据划分。工作目录为仓库根目录；相对路径命令也从这里执行。

```text
FPHelper/
├── algorithm/                 算法实现与版本
│   ├── legacy_v0/             原版取帧、规则评分及原检测模型
│   ├── sb0_v1_baseline/       分项众数基线
│   ├── sb1_v1_geometry/       几何特征模型
│   ├── sb2_v1_appearance/     外观特征模型
│   ├── sb3_v1_fusion/         融合模型
│   └── shared/fphelper/       公共训练、特征、评分和协议
├── backend/                   历史 Django 服务
│   ├── backend/              Django 项目配置
│   ├── common/               用户和结果等公共数据模型
│   ├── user/                 原有用户接口
│   └── detect/               原有检测接口及旧评分副本
├── tools/
│   ├── annotator/            本地网页标注工具及其测试
│   └── benchmark/            统一算法评价工具及其测试
├── tests/algorithm/           算法自动化测试
├── requirements/training.lock 研究环境依赖锁文件
├── docs/experiments/          匿名汇总报告与图
├── src/fphelper/              兼容导入入口
├── artifacts/                 本机数据与运行产物，Git忽略
├── pyproject.toml             算法包定义和依赖范围
└── README.md                  项目导航及常用入口
```

## 算法实现

每个算法版本保存 model.py、config.json 和说明，旧版保存原有 Detect.py 与 Recognize.py。公共实现集中在 algorithm/shared/fphelper。src/fphelper 保留兼容层，原有 PYTHONPATH=src 命令及已保存模型的模块路径继续可用。

新增方法或版本按[算法版本规则](../algorithm/README.md)创建独立文件夹。共享模块修改后需重新验证受影响版本。

## 工程和本地工具

backend 是历史 Django 服务；其中的 detect 仍包含独立旧评分副本，视频接口仍返回固定满分占位输出。它尚未接入研究模型。各应用和 manage.py 的位置保持现有 Django 结构。

tools/annotator 提供本机网页视频标注，tools/benchmark 提供研究方法统一评价。工具的入口、界面和专用测试放在工具内部，便于单独使用。根目录 tests/algorithm 负责跨版本算法和协议测试。

## 本机数据和产物

artifacts 已被 Git 忽略。现有标注数据库、特征缓存和评估结果保持其工具约定的位置。常见位置包括：

| 路径 | 用途 |
| --- | --- |
| artifacts/annotations/ | 标注工具按数据目录标识保存的 SQLite 数据库，运行时生成 |
| artifacts/first-validation/ | 训练流程默认产物，按实际运行生成 |
| artifacts/unified-evaluation-20261107/ | 本次统一评价的逐视频预测、指标、失败记录和复现信息 |
| artifacts/layout-validation-20261107/ | 算法分版本整理后的预测一致性验证 |
| artifacts/logs/ | 本机运行日志；原 backend/nohup.out 已归档在此 |
| artifacts/archive/repository-layout-20261107/manual-test-data/ | 原根目录 test/ 的手工测试素材 |
| artifacts/archive/repository-layout-20261107/generated/ | 本次整理前的构建文件、包元数据和字节码缓存 |
| artifacts/maintenance/ | 文件归档位置及 SHA-256 核对记录 |

本机原始数据目录通过工具参数指定。人工标注、逐视频预测和原始素材按本机数据管理；docs/experiments 只保存匿名汇总报告和图。

## 新增文件规则

- 算法实现和公共模块放入 algorithm；算法自动化测试放入 tests/algorithm。
- 本地工具及专用测试放入 tools 下对应目录。
- 使用说明放在所属目录 README，跨模块说明放入 docs，并在 docs/README.md 中登记。
- 依赖范围在 pyproject.toml 中维护，研究环境锁文件放入 requirements。
- 运行产物、模型、日志和手工测试素材放入 artifacts。构建工具可能重新生成 build 和 egg-info，它们也已被 Git 忽略。

本次归档清单位于 artifacts/maintenance/repository-layout-20261107.json，记录原位置、归档位置及文件指纹。
