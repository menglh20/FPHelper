# 历史 Django 后端

本目录保存原 Django 工程。工程入口为 manage.py，应用及配置按原有结构维护。

| 目录或文件 | 用途 |
| --- | --- |
| manage.py | Django 管理命令入口 |
| backend/ | settings、路由、WSGI 与 ASGI 配置 |
| common/ | 用户、检测结果等公共数据模型和接口 |
| user/ | 原用户接口 |
| detect/ | 原图片评分与视频接口、下载逻辑及旧检测模型 |

detect 内的评分实现是历史副本；视频接口仍返回固定100分占位结果。当前研究算法位于[algorithm](../algorithm/README.md)，研究评价流程位于[tools/benchmark](../tools/benchmark/README.md)。

旧运行日志已归档至 artifacts/logs/legacy-backend-nohup.out。数据库与上传素材继续按现有 Django 配置生成。当前研究环境的依赖锁文件属于算法研究，原后端依赖需按该服务环境单独维护。
