# 自动化测试

算法测试统一放在 algorithm 子目录，工具专用测试跟随工具代码保留。

| 测试位置 | 内容 |
| --- | --- |
| algorithm/test_training.py | 标签范围、有序模型、训练与验证隔离 |
| algorithm/test_score_contract.py | 缺失动作、评分完整性和研究输出协议 |
| algorithm/test_group_protocol.py | 患者映射与分组划分 |
| algorithm/test_versions.py | 各算法版本、特征选择和历史模型导入兼容 |
| ../tools/benchmark/test_benchmark.py | 评价分母、非法评分、配对比较和临时帧清理 |
| ../tools/annotator/tests/ | 标注字段、存储、导入导出、HTTP与前端逻辑 |

在仓库根目录、研究环境中运行算法与评价测试：

```sh
PYTHONPATH=src python -m unittest discover -s tests -q
PYTHONPATH=src python -m unittest discover -s tools/benchmark -p test_benchmark.py -q
```

标注工具测试：

```sh
python3 -m unittest discover -s tools/annotator/tests -q
node --test tools/annotator/tests/core.test.js
```

测试使用合成数组、临时文件和本机回环端口。tools/annotator 的 Node 测试无需安装 npm 依赖。根目录原 test/ 是手工测试素材，现已归档至 artifacts/archive/repository-layout-20261107/manual-test-data。
