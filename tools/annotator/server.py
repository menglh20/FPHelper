#!/usr/bin/env python3
"""Local-only annotation application. No application dependencies or cloud services."""
from __future__ import annotations

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import re
import secrets
import shutil
import sys
from urllib.parse import urlsplit
import webbrowser
import zipfile

from schema import PHASES, LABELS, LIMITS, LEGACY_PROTOCOL, ValidationError, action_evidence
from storage import Catalog, Store, ConflictError, ffprobe_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MAX_BODY = 20 * 1024 * 1024


def parse_range(value, size):
    if not value:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or not any(match.groups()) or size <= 0:
        raise ValidationError("无效的视频字节范围")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValidationError("无效的视频字节范围")
        start, end = max(0, size - length), size - 1
    else:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValidationError("视频字节范围超出文件")
    return start, end, True


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, catalog, store):
        self.catalog, self.store = catalog, store
        self.csrf_token = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "FPHelperLocal/1.0"

    def log_message(self, format, *args):
        # Do not log patient filenames, comments, headers or annotation bodies.
        pass

    def allowed(self):
        port = self.server.server_port
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return False
        if self.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            return False
        origin = self.headers.get("Origin")
        return not origin or origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")

    def send_headers(self, status, content_type, length, attachment=None, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def respond(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_headers(status, "application/json; charset=utf-8", len(payload))
        if self.command != "HEAD":
            self.wfile.write(payload)

    def error(self, message, status):
        self.respond({"error": message}, status)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.allowed():
            return self.error("仅允许本机同源访问", 403)
        path = urlsplit(self.path).path
        try:
            if path == "/api/config":
                return self.respond({"schema_version": 1, "dataset_id": self.server.catalog.dataset_id,
                                     "dataset_label": self.server.catalog.label,
                                     "data_directory": str(self.server.catalog.root),
                                     "annotation_directory": str(self.server.store.directory),
                                     "manifest_count": getattr(self.server.catalog, 'manifest_count', 0),
                                     "manifest_missing": getattr(self.server.catalog, 'manifest_missing', []),
                                     "manifest_extra": getattr(self.server.catalog, 'manifest_extra', []),
                                     "phases": list(zip(PHASES, LABELS)), "score_limits": LIMITS,
                                     "legacy_protocol": LEGACY_PROTOCOL, "csrf_token": self.server.csrf_token})
            if path == "/api/videos":
                return self.respond({"videos": self.server.store.listing()})
            match = re.fullmatch(r"/api/videos/(v_[A-Za-z0-9_-]+)(/history)?", path)
            if match:
                if match[2]:
                    return self.respond({"history": self.server.store.history(match[1])})
                record, meta = self.server.store.get(match[1])
                return self.respond({"record": record, "meta": meta})
            match = re.fullmatch(r"/media/(v_[A-Za-z0-9_-]+)", path)
            if match:
                return self.media(match[1])
            if path.startswith("/api/export/"):
                return self.export(path.removeprefix("/api/export/"))
            assets = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/core.js": "core.js", "/styles.css": "styles.css"}
            if path not in assets:
                return self.error("页面不存在", 404)
            file = HERE / "web" / assets[path]
            payload = file.read_bytes()
            mime = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}[file.suffix]
            self.send_headers(200, mime + "; charset=utf-8", len(payload))
            if self.command != "HEAD":
                self.wfile.write(payload)
        except KeyError:
            self.error("未知视频", 404)
        except ConflictError as exc:
            self.error(str(exc), 409)
        except (ValidationError, ValueError, TimeoutError) as exc:
            self.error(str(exc), 422)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.error("本地读取失败，请检查视频文件及标注目录", 500)

    def media(self, video_id):
        path = self.server.catalog.path(video_id)
        size = path.stat().st_size
        try:
            start, end, partial = parse_range(self.headers.get("Range"), size)
        except ValidationError:
            self.send_headers(416, "text/plain", 0, extra={"Content-Range": f"bytes */{size}"})
            return
        length = end - start + 1
        extra = {"Accept-Ranges": "bytes"}
        if partial:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        mime = {".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}.get(path.suffix.lower(), "video/mp4")
        self.send_headers(206 if partial else 200, mime, length, extra=extra)
        if self.command == "HEAD":
            return
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                block = source.read(min(256 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def export(self, kind):
        docs = self.server.store.documents()
        if kind == "annotations":
            from datetime import datetime, timezone
            value = {"schema_version": 1, "format": "fphelper-video-annotations", "dataset_id": self.server.catalog.dataset_id,
                     "exported_at": datetime.now(timezone.utc).isoformat(), "annotation_protocol": "manual-actions-v1",
                     "records": list(docs.values())}
            payload, mime, name = json.dumps(value, ensure_ascii=False, indent=2).encode(), "application/json", "fp-annotations.json"
        elif kind in ("patient-map", "patient-template"):
            if kind == "patient-map":
                if getattr(self.server.catalog, 'manifest_missing', []) or getattr(self.server.catalog, 'manifest_extra', []):
                    raise ValidationError('当前视频目录与训练清单不一致，请补齐视频或显式选择更新后的 manifest；可先导出分组模板或标注 JSON')
                missing = [i for i in self.server.catalog.entries if not docs.get(i, {}).get("patient_group_id")]
                if missing:
                    raise ValidationError(f"还有 {len(missing)} 段视频未分配患者组；可先导出模板，补齐后再导出完整分组表")
                by_hash = {}
                for video_id in self.server.catalog.entries:
                    current, meta = self.server.store.get(video_id)
                    group = current['patient_group_id']
                    previous = by_hash.setdefault(meta['video_sha256'], group)
                    if previous != group:
                        raise ValidationError("相同视频字节被分配到不同患者组，请先核对映射")
            output = io.StringIO(newline="")
            writer = csv.writer(output)
            writer.writerow(["video_id", "patient_group_id"])
            for video_id in self.server.catalog.entries:
                writer.writerow([video_id, docs.get(video_id, {}).get("patient_group_id", "")])
            payload, mime, name = output.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8", "patient-map.csv"
        elif kind == "action-evidence":
            output = io.BytesIO()
            included = 0
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for video_id, record in docs.items():
                    try:
                        evidence = action_evidence(record)
                    except ValidationError:
                        continue
                    archive.writestr(video_id + ".json", json.dumps(evidence, ensure_ascii=False, indent=2))
                    included += 1
            if not included:
                raise ValidationError("还没有人工核验的模型窗口证据。实际区间标注不会自动确认等分窗口")
            payload, mime, name = output.getvalue(), "application/zip", "action-evidence.zip"
        else:
            raise KeyError("未知导出类型")
        self.send_headers(200, mime, len(payload), attachment=name)
        if self.command != "HEAD":
            self.wfile.write(payload)

    def body(self):
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValidationError("请求需要使用 JSON 格式")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_BODY:
            raise ValidationError("请求为空或超过 20 MB")
        return json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("不接受非有限数值")))

    def do_POST(self):
        if not self.allowed() or not secrets.compare_digest(self.headers.get("X-CSRF-Token", ""), self.server.csrf_token):
            return self.error("本地会话已失效，请刷新页面后恢复草稿", 403)
        try:
            value = self.body()
            path = urlsplit(self.path).path
            match = re.fullmatch(r"/api/videos/(v_[A-Za-z0-9_-]+)(/restore)?", path)
            if match:
                if match[2]:
                    saved = self.server.store.restore(match[1], value.get("revision"), value.get("expected_revision"))
                else:
                    saved = self.server.store.save(match[1], value.get("record"), value.get("expected_revision"))
                return self.respond({"record": saved})
            if path == "/api/import":
                kind = value.get("kind")
                if kind == "annotations":
                    bundle = value.get("bundle", {})
                    if bundle.get("schema_version") != 1 or bundle.get("format") != "fphelper-video-annotations":
                        raise ValidationError("请选择本工具导出的标注 JSON")
                    return self.respond(self.server.store.import_records(bundle.get("records")))
                if kind == "patient-map":
                    reader = csv.DictReader(io.StringIO(value.get("csv", "").lstrip("\ufeff")))
                    if reader.fieldnames != ["video_id", "patient_group_id"]:
                        raise ValidationError("分组 CSV 需要 video_id,patient_group_id 两列")
                    return self.respond(self.server.store.import_groups(list(reader)))
                raise ValidationError("未知导入类型")
            self.error("接口不存在", 404)
        except ConflictError as exc:
            self.error(str(exc), 409)
        except KeyError:
            self.error("存在未知视频，未导入", 404)
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            self.error(str(exc), 422)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.error("本地保存失败。草稿仍留在浏览器，请检查磁盘和目录权限", 500)


def default_data_root():
    candidates = [Path.home() / "Library/CloudStorage/OneDrive-个人/fp", Path.home() / "OneDrive/fp", ROOT / "data"]
    return next((p for p in candidates if p.is_dir()), None)


def main():
    parser = argparse.ArgumentParser(description="FPHelper 本地视频标注工具（仅监听 127.0.0.1）")
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="fp 目录或直接包含视频的目录")
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "artifacts/annotations")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/first-validation/manifest.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="启动后打开系统浏览器")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("未找到 fp 数据目录，请通过 --data-root 指定")
    if not ffprobe_path():
        parser.error("未找到 ffprobe，请先安装 FFmpeg 并确认 ffprobe 在 PATH 中")
    try:
        catalog = Catalog(args.data_root, args.manifest)
        store = Store(args.annotations_dir, catalog)
        server = AnnotationServer(("127.0.0.1", args.port), catalog, store)
    except (OSError, ValidationError) as exc:
        parser.error(str(exc))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"FPHelper 标注工具：{url}\n视频：{len(catalog.entries)} 段\n标注保存：{store.directory}\n按 Ctrl+C 停止。", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
