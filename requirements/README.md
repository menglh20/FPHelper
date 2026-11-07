# 依赖与运行环境

[training.lock](training.lock) 保存当前本地 Mac 研究环境的精确依赖版本，算法包的 Python 范围及直接依赖在[pyproject.toml](../pyproject.toml)维护。

算法研究使用 Python 3.12。视频特征提取还使用本机 FFmpeg、MediaPipe、PyTorch、torchvision 和 OpenCV；锁文件记录本次研究环境，不代表已测试所有平台组合。

本地标注服务使用 Python 标准库和已有 ffprobe，前端无需 npm 安装。标注服务的启动说明见[tools/annotator](../tools/annotator/README.md)。

backend 为原 Django 服务，其运行依赖需单独维护；training.lock 用于算法研究环境。原根目录 requirements-training.lock 已移动为本目录的 training.lock，文件内容保持一致。
