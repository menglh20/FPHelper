# Sunnybrook 统一评价工具

比较原版视频选帧及评分、参考窗口加旧规则诊断、SB0 至 SB3 折外预测、doctor02、doctor03 和现有接口的固定满分占位输出。doctor01 的 comment 中 13 个分项为统一参考，总分由同一公式重算。

## 运行

使用 Python 3.12 的研究环境，依赖见仓库 pyproject.toml 和 requirements/training.lock。另需 OpenCV、MediaPipe 和本机 FFmpeg。历史 first-validation 目录需要包含 manifest、特征缓存、提取器配置、折外预测、metrics 及数据审计记录。

在 FPHelper 根目录、已激活研究环境的终端中运行：

```sh
PYTHONPATH=src python tools/benchmark/run.py \
  --source /absolute/path/to/artifacts/first-validation \
  --data-root /absolute/path/to/fp \
  --output artifacts/unified-evaluation-20261107
```

默认两个工作进程，每个旧版视频任务超时300秒。整个旧版处理可能耗时较长。旧版逐帧处理、颜色输入、JPEG 中间文件及评分规则均按原代码执行，没有缩帧或阈值修正。

复现会核对医生 CSV、可访问视频指纹及历史折外预测。当前缺少源文件的视频保留为 source_missing；可访问队列在运行旧版前固定。旧版结果按输入、代码和模型指纹缓存；environment.json 绑定实际依赖版本，环境变化需使用新的输出目录。重复执行默认复用相同任务的成功及失败记录；确认需要重试运行失败时加 --retry-failures。需要复用本输出目录已验证的训练复现文件时加 --reuse-reproduced-training。

## 结果

- docs/experiments/unified-evaluation.md 与同名 PNG、PDF：匿名汇总报告和科学图。
- protocol.json：固定队列、评价规则、折号来源、指纹和运行参数。
- results.json：覆盖率、总分、三大维度、十三个分项、混淆矩阵、联动检出、分层及配对结果。
- matrix.csv：全方法共同数值队列的汇总矩阵。
- predictions.csv 和 predictions.npz：匿名视频标识、真值与原始预测，仅保存在 artifacts。
- status.json 与 legacy-cache：运行状态、失败类别、耗时与按指纹缓存的旧版结果。

原始患者视频与医生 CSV 只读。旧版需要的 JPEG 保存在父进程拥有的临时目录，在成功、失败和工作进程超时后清理。不要将 artifacts 加入 Git 或上传到公共服务。

## 评价约定

完整数值输出、合法分项和合法总分分别计数；失败不补正常分，越界值不裁剪，非法等级不从混淆矩阵消失。对所有方法使用相同视频交集比较，也保存每对方法各自交集上的配对结果。不同交集的误差不能直接进行传递排名。

三大维度的归一化分项 MAE 等权：先按每个分项的合法跨度归一化，再在静态、自主运动和联动内平均，最后三个维度取平均。联动单独报告非零召回率及零分误报率，避免大量零分病例产生误导。

当前固定折属于访视层面开发验证，重复患者可能跨折。视频 bootstrap 区间仅描述现有折外误差，不是新患者泛化或临床等效证明。

## 测试

```sh
PYTHONPATH=src python -m unittest discover -s tests -q
PYTHONPATH=src python -m unittest discover -s tools/benchmark -p test_benchmark.py -v
```

评价测试覆盖非法分项、负总分、配对分母、三维度等权、零分占优和工作进程超时后的临时帧清理。

## 算法代码位置

旧版代码位于 algorithm/legacy_v0，SB0 至 SB3 分别位于 algorithm 下的对应版本文件夹。公共代码位于 algorithm/shared/fphelper，src/fphelper 保留原有兼容导入路径。新运行的 protocol.json 记录版本注册与完整算法代码指纹；已有历史报告和结果保留其运行时记录。
