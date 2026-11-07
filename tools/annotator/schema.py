"""Versioned local annotations. Unknown and missing values are never scores of zero."""
from __future__ import annotations

import math
import re

PHASES = ("rest", "forehead_wrinkle", "eye_closure", "smile", "snarl", "lip_pucker")
LABELS = ("自然放松", "抬眉", "轻闭眼", "露齿微笑", "呲牙", "嘟嘴")
TIMES = ("start_s", "hold_start_s", "peak_s", "hold_end_s", "end_s")
ITEMS = ("rest_eye", "rest_cheek", "rest_mouth") + tuple("vol_" + p for p in PHASES[1:]) + tuple("syn_" + p for p in PHASES[1:])
LIMITS = dict(zip(ITEMS, ((0, 1), (0, 2), (0, 1)) + ((1, 5),) * 5 + ((0, 3),) * 5))
ISSUES = ("blur", "occlusion", "head_motion", "lighting", "speech", "blink", "wrong_action", "tracking", "other")
LEGACY_PROTOCOL = "relative-six-phases-ffmpeg-pts-v1"
GROUP_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class ValidationError(ValueError):
    pass


def text(value, maximum=2000):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValidationError(f"文字必须为字符串，且不超过 {maximum} 个字符")
    return value.strip()


def flag(value):
    if not isinstance(value, bool):
        raise ValidationError("确认字段必须为布尔值")
    return value


def enum(value, choices):
    if value not in choices or not isinstance(value, str):
        raise ValidationError("存在无法识别的选项")
    return value


def issues(value):
    if not isinstance(value, list) or any(not isinstance(x, str) or x not in ISSUES for x in value):
        raise ValidationError("质量标记无效")
    return sorted(set(value))


def default_record(meta):
    return {"schema_version": 1, "video_id": meta["video_id"], "video_sha256": meta["video_sha256"],
            "file_name": meta["file_name"], "duration_s": meta["duration_s"], "revision": 0,
            "annotator_id": "", "patient_group_id": "", "affected_side": "unknown", "mirrored": "unknown",
            "camera": "unknown", "status": "draft", "notes": "", "quality_flags": [],
            "actions": {p: {"presence": "unreviewed", **{k: None for k in TIMES}, "reviewed": False,
                             "quality_flags": [], "notes": "", "legacy_window_verified": False} for p in PHASES},
            "legacy_verification_method": "",
            "independent_scores": {"enabled": False, "rater_id": "", "notes": "", "values": {k: None for k in ITEMS}}}


def action_errors(action):
    errors = []
    presence = action["presence"]
    if presence == "unreviewed":
        return ["尚未判断动作是否存在"]
    if presence in ("missing", "uncertain"):
        if not action["notes"]:
            errors.append("缺失或无法判断的动作需要填写原因")
        return errors
    start, peak, end = (action[k] for k in ("start_s", "peak_s", "end_s"))
    if start is None or peak is None or end is None:
        errors.append("请设置起始、峰值／代表帧和结束时间")
    elif not start <= peak <= end or start == end:
        errors.append("需要 起始 ≤ 峰值 ≤ 结束，且区间长度大于零")
    hs, he = action["hold_start_s"], action["hold_end_s"]
    if (hs is None) != (he is None):
        errors.append("保持区间的起止需要同时填写")
    elif hs is not None:
        if start is None or end is None or not start <= hs <= he <= end:
            errors.append("保持区间必须位于动作起止之间")
        elif peak is not None and not hs <= peak <= he:
            errors.append("已填写保持区间时，峰值应位于保持区间内")
    return errors


def completion_errors(record):
    errors = []
    if not record["annotator_id"]:
        errors.append("请填写标注人代号")
    for phase, label in zip(PHASES, LABELS):
        action = record["actions"][phase]
        errors.extend(f"{label}：{message}" for message in action_errors(action))
        if not action["reviewed"]:
            errors.append(f"{label}：请确认本动作标注")
    # Ordered actions may overlap at transition boundaries, but not reverse their starting order.
    starts = [record["actions"][p]["start_s"] for p in PHASES if record["actions"][p]["presence"] == "present"]
    if all(s is not None for s in starts) and starts != sorted(starts):
        errors.append("动作起始顺序与六阶段协议不一致，请复核或标记待复核")
    return errors


def normalize_record(value, meta):
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValidationError("不支持的标注格式版本")
    if value.get("video_id") != meta["video_id"] or value.get("video_sha256") != meta["video_sha256"]:
        raise ValidationError("视频标识或文件指纹不一致，未覆盖已有标注")
    result = default_record(meta)
    result["annotator_id"] = text(value.get("annotator_id", ""), 80)
    result["patient_group_id"] = text(value.get("patient_group_id", ""), 64)
    if result["patient_group_id"] and not GROUP_PATTERN.fullmatch(result["patient_group_id"]):
        raise ValidationError("患者组请使用 P001 这类代号：字母或数字开头，可包含点、短横线和下划线")
    for key, choices in (("affected_side", ("unknown", "left", "right", "bilateral", "none")),
                         ("mirrored", ("unknown", "yes", "no")), ("camera", ("unknown", "front", "rear")),
                         ("status", ("draft", "complete", "needs_review"))):
        result[key] = enum(value.get(key, result[key]), choices)
    result["notes"] = text(value.get("notes", ""), 5000)
    result["quality_flags"] = issues(value.get("quality_flags", []))
    result["legacy_verification_method"] = text(value.get("legacy_verification_method", ""), 500)
    actions = value.get("actions")
    if not isinstance(actions, dict) or set(actions) != set(PHASES):
        raise ValidationError("标注需要包含六个阶段")
    for phase in PHASES:
        incoming = actions[phase]
        if not isinstance(incoming, dict):
            raise ValidationError("动作标注格式错误")
        action = result["actions"][phase]
        action["presence"] = enum(incoming.get("presence", "unreviewed"), ("unreviewed", "present", "missing", "uncertain"))
        for key in TIMES:
            number = incoming.get(key)
            if number is not None:
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= meta["duration_s"] + 0.001:
                    raise ValidationError("时间必须为视频范围内的有限秒数；未标注请留空")
                number = round(float(number), 3)
            action[key] = number
        action["notes"] = text(incoming.get("notes", ""), 2000)
        action["quality_flags"] = issues(incoming.get("quality_flags", []))
        action["reviewed"] = flag(incoming.get("reviewed", False))
        action["legacy_window_verified"] = flag(incoming.get("legacy_window_verified", False))
        if action["presence"] in ("missing", "uncertain"):
            for key in TIMES:
                action[key] = None
            action["legacy_window_verified"] = False
        if action["reviewed"] and action_errors(action):
            raise ValidationError("已确认的动作仍有错误：" + "；".join(action_errors(action)))
        if action["legacy_window_verified"] and (action["presence"] != "present" or not action["reviewed"]):
            raise ValidationError("只有已确认存在的动作可核验模型窗口")
    scores = value.get("independent_scores", {})
    if not isinstance(scores, dict):
        raise ValidationError("独立评分格式错误")
    target = result["independent_scores"]
    target["enabled"] = flag(scores.get("enabled", False))
    target["rater_id"] = text(scores.get("rater_id", ""), 80)
    target["notes"] = text(scores.get("notes", ""), 2000)
    supplied = scores.get("values", {})
    if not isinstance(supplied, dict) or set(supplied) - set(ITEMS):
        raise ValidationError("存在未知评分项")
    for key, (low, high) in LIMITS.items():
        number = supplied.get(key)
        if number is not None and (isinstance(number, bool) or not isinstance(number, int) or not low <= number <= high):
            raise ValidationError("评分必须为合法整数；未评分请留空")
        target["values"][key] = number
    if any(v is not None for v in target["values"].values()) and (not target["enabled"] or not target["rater_id"]):
        raise ValidationError("独立评分需要开启评分并填写评分者代号")
    if result["status"] == "complete":
        errors = completion_errors(result)
        if errors:
            raise ValidationError("；".join(errors[:8]))
    return result


def action_evidence(record):
    phases = [p for p in PHASES if record["actions"][p]["legacy_window_verified"]]
    method = record["legacy_verification_method"]
    if not phases or not method:
        raise ValidationError("请先人工核验模型参考窗口，并填写核验方法；时间区间标注不会自动成为模型核验证据")
    return {"schema_version": 1, "video_sha256": record["video_sha256"], "feature_protocol": LEGACY_PROTOCOL,
            "verification_method": method, "verified_phases": phases,
            "annotation_revision": record["revision"], "annotator_id": record["annotator_id"]}
