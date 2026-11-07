# 原版规则算法

本目录归档原始实现及检测模型，文件内容与整理前一致。

- [Detect.py](Detect.py)：六张照片的关键点几何比例、经验阈值及 Sunnybrook 公式评分。
- [Recognize.py](Recognize.py)：整段视频逐帧计算几何量，以极值选择五个动作的代表帧，取帧索引10作为静息帧。
- face_landmarker.task：原版 MediaPipe 人脸检测模型。
- [config.json](config.json)：算法标识及原文件 SHA-256。

函数入口为 Detect.detect 和 Recognize.recognize。原脚本使用相对模型路径，直接调用时应以本目录作为运行目录；脚本 main 中保留原来的测试路径。

[统一评估工具](../../tools/benchmark/README.md)已使用本目录执行原版完整流程和参考窗口取帧诊断，负责临时帧清理及失败记录。

本版本的已测效果见[统一评价报告](../../docs/experiments/unified-evaluation.md)。
