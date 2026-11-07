"""Read-only video catalog and transactional local annotation/revision storage."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import sqlite3
import subprocess
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from schema import ValidationError, default_record, normalize_record, PHASES


class ConflictError(Exception):
    pass


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ffprobe_path():
    return shutil.which("ffprobe") or next((str(p) for p in (Path('/opt/homebrew/bin/ffprobe'), Path('/usr/local/bin/ffprobe')) if p.is_file()), None)


class Catalog:
    def __init__(self, root, manifest=None):
        root = Path(root).expanduser().resolve(strict=True)
        self.root = (root / "videos" if (root / "videos").is_dir() else root).resolve()
        self.label = root.name
        self.dataset_id = hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        lookup = {}
        if manifest and Path(manifest).is_file():
            for item in json.loads(Path(manifest).read_text()):
                lookup[str(Path(item["video_path"]).resolve())] = item
        self.entries = {}
        self.metadata = {}
        self.lock = threading.RLock()
        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (".mp4", ".mov", ".m4v", ".webm"):
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root):
                continue
            source = lookup.get(str(resolved), {})
            video_id = source.get("video_id", "v_" + hashlib.sha256(path.stem.encode()).hexdigest()[:12])
            if not isinstance(video_id, str) or not re.fullmatch(r"v_[A-Za-z0-9_-]+", video_id):
                raise ValidationError("清单中的 video_id 无效")
            if video_id in self.entries:
                raise ValidationError("发现重复的视频标识，请核对文件名或 manifest")
            self.entries[video_id] = {"path": resolved, "file_name": path.name,
                                      "source_side": source.get("side_metadata", "unknown")}
        if not self.entries:
            raise ValidationError("目录中未找到 MP4、MOV、M4V 或 WebM 视频")
        expected = {row['video_id'] for filename, row in lookup.items() if Path(filename).parent == self.root}
        self.manifest_count = len(expected)
        self.manifest_missing = sorted(expected - set(self.entries))
        self.manifest_extra = sorted(set(self.entries) - expected) if expected else []

    def path(self, video_id):
        if video_id not in self.entries:
            raise KeyError("未知视频")
        path = self.entries[video_id]["path"]
        if not path.resolve().is_relative_to(self.root):
            raise ValidationError("视频路径超出数据目录")
        return path

    def meta(self, video_id):
        with self.lock:
            path = self.path(video_id)
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            cached = self.metadata.get(video_id)
            if cached and cached[0] == signature:
                return dict(cached[1])
            run = subprocess.run([ffprobe_path() or "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=width,height,duration,avg_frame_rate,codec_name:format=duration", "-of", "json", str(path)],
                                 capture_output=True, text=True, timeout=30)
            if run.returncode:
                raise ValidationError("视频无法读取，请检查文件是否已从 OneDrive 下载完成")
            data = json.loads(run.stdout)
            stream = data["streams"][0]
            duration = float(stream.get("duration") or data.get("format", {}).get("duration", 0))
            if duration <= 0:
                raise ValidationError("视频时长无效")
            rate = stream.get("avg_frame_rate", "0/1").split("/")
            fps = float(rate[0]) / max(float(rate[1]) if len(rate) > 1 else 1, 1)
            result = {"video_id": video_id, "file_name": self.entries[video_id]["file_name"],
                      "video_sha256": fingerprint(path), "duration_s": round(duration, 6),
                      "width": stream["width"], "height": stream["height"], "fps_hint": fps,
                      "codec": stream.get("codec_name", "unknown"), "source_side": self.entries[video_id]["source_side"]}
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != signature:
                raise ConflictError("读取过程中视频发生变化，请重试")
            self.metadata[video_id] = (signature, result)
            return dict(result)


class Store:
    def __init__(self, directory, catalog):
        self.catalog = catalog
        self.directory = Path(directory).expanduser().resolve() / catalog.dataset_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "annotations.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS annotations(video_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS history(video_id TEXT NOT NULL, revision INTEGER NOT NULL, document TEXT NOT NULL,
                                                 PRIMARY KEY(video_id,revision));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _read(self, db, meta):
        row = db.execute("SELECT document FROM annotations WHERE video_id=?", (meta["video_id"],)).fetchone()
        result = json.loads(row[0]) if row else default_record(meta)
        if result["video_sha256"] != meta["video_sha256"]:
            raise ConflictError("原视频文件指纹已变化；旧标注已保留，请检查数据来源")
        return result

    def get(self, video_id):
        meta = self.catalog.meta(video_id)
        with self.connect() as db:
            return self._read(db, meta), meta

    def _write(self, db, document, old):
        now = datetime.now(timezone.utc).isoformat()
        document["revision"] = old["revision"] + 1
        document["created_at"] = old.get("created_at", now)
        document["updated_at"] = now
        payload = json.dumps(document, ensure_ascii=False, allow_nan=False)
        db.execute("INSERT OR REPLACE INTO annotations VALUES (?,?,?)", (document["video_id"], document["revision"], payload))
        db.execute("INSERT INTO history VALUES (?,?,?)", (document["video_id"], document["revision"], payload))
        return document

    def save(self, video_id, document, expected_revision):
        meta = self.catalog.meta(video_id)
        normalized = normalize_record(document, meta)
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValidationError("保存请求缺少有效的版本号")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = self._read(db, meta)
            if old["revision"] != expected_revision:
                raise ConflictError("该视频已在另一页面更新；本页草稿已保留，请重新载入或导出本页草稿")
            return self._write(db, normalized, old)

    def documents(self):
        with self.connect() as db:
            return {row[0]: json.loads(row[1]) for row in db.execute("SELECT video_id,document FROM annotations")}

    def listing(self):
        saved = self.documents()
        result = []
        for video_id, entry in self.catalog.entries.items():
            record = saved.get(video_id)
            result.append({"video_id": video_id, "file_name": entry["file_name"],
                           "status": record["status"] if record else "unannotated",
                           "patient_group_id": record["patient_group_id"] if record else "",
                           "reviewed_count": sum(bool(a["reviewed"]) for a in record["actions"].values()) if record else 0,
                           "updated_at": record.get("updated_at") if record else None,
                           "revision": record["revision"] if record else 0})
        return result

    def history(self, video_id):
        self.catalog.path(video_id)
        with self.connect() as db:
            records = [json.loads(row[0]) for row in db.execute("SELECT document FROM history WHERE video_id=? ORDER BY revision DESC LIMIT 100", (video_id,))]
        return [{key: r.get(key) for key in ("revision", "updated_at", "annotator_id", "status")} for r in records]

    def restore(self, video_id, revision, expected_revision):
        with self.connect() as db:
            row = db.execute("SELECT document FROM history WHERE video_id=? AND revision=?", (video_id, revision)).fetchone()
        if not row:
            raise ValidationError("历史版本不存在")
        return self.save(video_id, json.loads(row[0]), expected_revision)

    def import_records(self, records):
        if not isinstance(records, list) or len(records) > len(self.catalog.entries):
            raise ValidationError("导入记录数量无效")
        ids = [r.get("video_id") for r in records if isinstance(r, dict)]
        if len(ids) != len(records) or len(set(ids)) != len(ids):
            raise ValidationError("导入记录存在重复或无效 video_id")
        normalized = [(normalize_record(r, self.catalog.meta(r["video_id"])), self.catalog.meta(r["video_id"])) for r in records]
        imported, skipped = 0, 0
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for record, meta in normalized:
                old = self._read(db, meta)
                if old["revision"]:
                    skipped += 1
                    continue
                self._write(db, record, old)
                imported += 1
        return {"imported": imported, "skipped": skipped}

    def import_groups(self, rows):
        if any(not isinstance(r, dict) or set(r) != {'video_id', 'patient_group_id'} for r in rows):
            raise ValidationError("患者分组每行需要且仅需要 video_id,patient_group_id 两列")
        ids = [r.get("video_id") for r in rows]
        if len(set(ids)) != len(ids) or any(k not in self.catalog.entries for k in ids):
            raise ValidationError("患者分组表存在重复或未知视频")
        prepared = [(r["video_id"], r.get("patient_group_id", "").strip(), self.catalog.meta(r["video_id"])) for r in rows]
        count = 0
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for video_id, group, meta in prepared:
                if not group:
                    continue
                old = self._read(db, meta)
                record = dict(old)
                record["patient_group_id"] = group
                record["status"] = "draft"
                record = normalize_record(record, meta)
                self._write(db, record, old)
                count += 1
        return {"imported": count, "skipped": len(rows) - count}
