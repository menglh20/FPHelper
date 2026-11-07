"""Aggregate report and plots; no patient images or source identifiers."""
from pathlib import Path
import json
import numpy as np
from .data import ITEMS, totals

CHINESE = ("静态眼", "静态鼻唇沟", "静态嘴", "抬眉自主运动", "轻闭眼自主运动", "微笑自主运动", "呲牙自主运动", "嘟嘴自主运动",
           "抬眉联动", "轻闭眼联动", "微笑联动", "呲牙联动", "嘟嘴联动")


def write_report(result, reference, predictions, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    lines = ["# Sunnybrook 第一次完整训练与验证结果", "",
             "实验编号：first-validation。完成真实视频到 13 个分项、公式总分、嵌套验证和模型导出的首轮研究流程。", "",
             f"纳入 {result['source_count']} 段视频，成功提取 {result['eligible_count']} 段的完整特征，失败或不可评分 {result['source_count'] - result['eligible_count']} 段。训练标签仅来自 doctor01 的 comment，doctor02 和 doctor03 始终作为独立对照。", "",
             "## 总分结果", "", "| 对照 | 折外/对照 MAE | RMSE | 平均偏差 | 绝对一致性 ICC |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, m in metrics.items():
        icc = "不可估计" if m["icc_absolute_single_mixed"] is None else f"{m['icc_absolute_single_mixed']:.3f}"
        lines.append(f"| {name} | {m['total_mae']:.2f} | {m['total_rmse']:.2f} | {m['bias']:+.2f} | {icc} |")
    primary = metrics["SB3"]
    ci = primary["mae_ci95_video_bootstrap"]
    lines += ["", f"预先指定的主模型为 SB3。其总分 MAE 为 {primary['total_mae']:.2f}，按视频重采样的探索性 95% 区间为 [{ci[0]:.2f}, {ci[1]:.2f}]。表中所有算法行均为外层折外预测；医生行在同一批成功提取特征的视频上计算。全部源视频的医生表现另存于本地 metrics.json。", "",
              "| SB3 配对比较 | 算法 MAE 减医生 MAE | 95% 区间 |", "| --- | ---: | --- |"]
    for name, m in result["paired_comparison"].items():
        ci = m["ci95_video_bootstrap"]
        lines.append(f"| {name} | {m['mean_absolute_error_difference']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines += ["", "差值小于 0 表示算法误差较小。重复患者身份尚未确认，区间按视频而非患者组计算，不能解释为已证明对新患者的临床性能。", "",
              "## 分项结果", "", "| 分项 | SB3 MAE | doctor02 MAE | doctor03 MAE | SB3 完全一致率 |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, key in zip(CHINESE, ITEMS):
        sub = primary["subscores"][key]
        lines.append(f"| {label} | {sub['mae']:.3f} | {metrics['doctor02']['subscores'][key]['mae']:.3f} | {metrics['doctor03']['subscores'][key]['mae']:.3f} | {sub['exact_agreement']:.1%} |")
    lines += ["", "## 实验设置", "",
              "- SB0：训练折的各分项多数值，按公式计算总分。",
              "- SB1：关键点几何、通用 blendshape 和跨区域运动摘要；这些系数仅作为输入，不直接对应量表评分。",
              "- SB2：冻结 ImageNet ResNet18 的全脸及局部外观，使用动作起始、峰值及特征差。",
              "- SB3：SB1 与 SB2 融合，13 个独立的有序逻辑回归评分头；总分由分项计算。",
              "- 外层固定五折，内层三折；内层选择 L2 强度 0.01、0.1 或 1。特征标准化与 16 维 PCA 均只拟合当前训练折，模型选择使用三个维度等权的归一化分项 MAE，总分 MAE 用于并列处理。",
              "- 初次运行使用 split/train seed 20261012，无数据增强、无编码器微调、无类别加权；多 seed、消融及前瞻测试留待后续实验。",
              "- 视频由 FFmpeg 依据源 PTS 重采样到 15 fps，MediaPipe 在 VIDEO 模式下跟踪。成片按时间顺序分成六个等比例窗口，分别用于静态和五个动作；每个窗口独立选择动作证据。",
              f"- ResNet18 特征提取设备：{result['device']}；有序分类器在 CPU 上优化。依赖版本、权重指纹、视频指纹、固定折号、逐视频预测、混淆矩阵、等级覆盖与训练记录均保存在本地 artifacts/first-validation。",
              "", "## 覆盖与限制", "",
              f"SB3 的公式原始总分中，超出 0–100 范围的记录为 {primary['invalid_raw_totals']} 条；这些记录保留并计入研究误差，不通过裁剪或剔除改善结果。运行失败的记录保留在质量清单，未参与可评分子集误差计算。", "",
              "动作分段为比例窗口假设，尚未逐视频核实。镜像和患侧映射未充分确认，当前未使用患侧字段作为模型特征，也未做水平翻转增强。推理输出标记 needs_review，数值用于研究及医生复核。", "",
              "本数据某些自主运动等级仅有一个病例，部分联动没有 3 分病例。每折记录训练支持等级及训练未覆盖的测试标签；这类测试病例仍计入结果。所有帧、局部图和特征跟随原视频留在同一折；完全重复的视频按文件指纹分组隔离。不同时间的同一患者仍可能跨折，当前结果属于访视层面的开发验证。", "",
              "## 下一步", "",
              "先核实动作分段和镜像，再复核误差最大的分项与动作证据；优先改善鼻唇沟外观和联动的非目标区域运动特征。保留本次划分和指标规则，新增实验单独编号。最终冻结模型后，用新患者视频做独立前瞻测试。", "",
              "## 诊断图", "", "![总分折外预测与误差诊断](first-validation.png)", "",
              "图中只有评分数据，无患者面部图像或原始文件名。", "",
              "实现参考：[MediaPipe 视频模式](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)、[ResNet18 权重与预处理](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)、[嵌套验证](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。", ""]
    path.write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    truth = totals(reference)
    primary_total = totals(predictions["SB3"])
    axes[0].scatter(truth, primary_total, s=24, alpha=0.75, color="#176b87")
    axes[0].plot([0, 100], [0, 100], "--", color="#666666", linewidth=1)
    axes[0].set(xlabel="Doctor01 reference", ylabel="SB3 out-of-fold prediction", title="Sunnybrook total", xlim=(0, 100), ylim=(-5, 105))
    difference = primary_total - truth
    axes[1].scatter((truth + primary_total) / 2, difference, s=24, alpha=0.75, color="#176b87")
    axes[1].axhline(primary["bias"], color="#176b87")
    for value in primary["bland_altman_limits"]:
        axes[1].axhline(value, color="#a33a3a", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Mean reference/prediction", ylabel="Prediction minus reference", title="Bland-Altman")
    names = list(metrics)
    axes[2].barh(names[::-1], [metrics[n]["total_mae"] for n in names[::-1]], color=["#78838e" if n.startswith("doctor") else "#176b87" for n in names[::-1]])
    axes[2].set(xlabel="Total score MAE", title=f"Same {len(reference)} videos")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)
