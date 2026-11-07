"""PTS-based FFmpeg resampling, MediaPipe tracking and frozen image features.

No decoded images are saved. Feature caches remain local and must be protected
as patient-derived data. Six proportional action windows are an explicit first
experiment assumption, not verified clinical action annotations.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "relative-six-phases-ffmpeg-pts-v1"
SIZE = 512
REGIONS = ((55, 130, 265, 285), (60, 30, 260, 150), (65, 85, 255, 175),
           (55, 170, 265, 310), (55, 130, 265, 285), (55, 170, 265, 310))


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height,duration,start_time:format=duration", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True, timeout=30)
    value = json.loads(result.stdout)
    stream = value["streams"][0]
    duration = float(stream.get("duration", value["format"]["duration"]))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid video duration")
    return duration, {"duration_seconds": duration, "width": stream["width"], "height": stream["height"]}


def normalized_geometry(points, blendshapes):
    a = points[[33, 133]].mean(axis=0)
    b = points[[263, 362]].mean(axis=0)
    difference = b - a
    distance = np.linalg.norm(difference)
    if distance < 5:
        raise ValueError("Degenerate face scale")
    axis = difference / distance
    rotation = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
    p = (points - (a + b) / 2) @ rotation.T / distance
    eye_a = p[[144, 145, 153], 1].mean() - p[[158, 159, 160], 1].mean()
    eye_b = p[[373, 374, 380], 1].mean() - p[[385, 386, 387], 1].mean()
    brow_a = p[[33, 133], 1].mean() - p[[105, 66, 52, 65], 1].mean()
    brow_b = p[[263, 362], 1].mean() - p[[296, 334, 295, 282], 1].mean()
    mouth_a, mouth_b = p[61], p[291]
    alar_a, alar_b = p[48] - p[1], p[278] - p[1]
    core = np.array([eye_a, eye_b, brow_a, brow_b, mouth_a[0], mouth_a[1], mouth_b[0], mouth_b[1],
                     alar_a[0], alar_a[1], alar_b[0], alar_b[1],
                     np.linalg.norm(mouth_b - mouth_a), np.linalg.norm(p[13] - p[14]),
                     abs(eye_a - eye_b), abs(brow_a - brow_b), abs(mouth_a[1] - mouth_b[1]),
                     abs(abs(mouth_a[0]) - abs(mouth_b[0])), abs(abs(alar_a[0]) - abs(alar_b[0]))])
    # Preserve both absolute movement and asymmetry; no potentially zero ratios.
    return np.r_[core, blendshapes], (rotation * (120 / distance), (a + b) / 2)


def align_image(rgb, alignment):
    matrix, center = alignment
    offset = np.array([160, 130]) - matrix @ center
    inverse = np.linalg.inv(matrix)
    translation = -inverse @ offset
    coefficients = tuple(np.c_[inverse, translation].ravel())
    return Image.fromarray(rgb).transform((320, 320), Image.Transform.AFFINE, coefficients,
                                          resample=Image.Resampling.BILINEAR)


def peak_score(phase, current, baseline):
    delta = current - baseline
    if phase == 1:
        return float(delta[2:4].mean())
    if phase == 2:
        return float(-delta[:2].mean())
    if phase == 3:
        return float(np.linalg.norm(delta[4:8]) + max(delta[12], 0))
    if phase == 4:
        return float(np.linalg.norm(delta[8:12]))
    return float(max(-delta[12], 0) + abs(delta[13]))


def summarize(sequence, timestamps, baseline):
    value = np.asarray(sequence)
    quantiles = np.percentile(value, [5, 50, 95], axis=0)
    movement = np.percentile(np.abs(value - baseline), 95, axis=0)
    dt = np.diff(timestamps)
    velocity = np.mean(np.abs(np.diff(value, axis=0)) / np.maximum(dt[:, None], 1e-4), axis=0)
    return np.r_[baseline, quantiles.ravel(), quantiles[2] - quantiles[0], value.std(axis=0), movement, velocity]


class VideoExtractor:
    def __init__(self, face_model: Path, weights: Path, fps=15, device="cpu"):
        import mediapipe as mp
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
        from .data import sha256_file
        self.mp = mp
        self.torch = torch
        self.face_model = face_model
        self.fps = fps
        self.device = device
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device is unavailable")
        self.encoder = resnet18(weights=None)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.fc = torch.nn.Identity()
        self.encoder.eval().requires_grad_(False).to(device)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.config = {"version": VERSION, "fps": fps, "decode_size": SIZE, "device": device,
                       "face_model_sha256": sha256_file(face_model), "encoder_sha256": sha256_file(weights),
                       "encoder": "resnet18-imagenet1k-v1-frozen", "minimum_face_coverage": 0.5,
                       "minimum_valid_frames_per_phase": 3,
                       "mediapipe_version": mp.__version__, "torch_version": torch.__version__}
        if not self.config["encoder_sha256"].startswith("f37072fd"):
            raise ValueError("ResNet18 weights do not match the official checksum prefix")

    def cache_key(self, video_hash):
        return hashlib.sha256((video_hash + json.dumps(self.config, sort_keys=True)).encode()).hexdigest()

    def encode(self, images):
        encoded = []
        with self.torch.inference_mode():
            for offset in range(0, len(images), 8):
                batch = self.torch.stack([self.transform(p) for p in images[offset:offset + 8]]).to(self.device)
                encoded.append(self.encoder(batch).cpu().numpy())
        return np.concatenate(encoded)

    def extract(self, path):
        mp = self.mp
        duration, metadata = probe(path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.face_model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2,
            output_face_blendshapes=True, min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5, min_tracking_confidence=0.5)
        command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-an", "-vf",
                   f"fps=fps={self.fps}:start_time=0:round=near,scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease,pad={SIZE}:{SIZE}:(ow-iw)/2:(oh-ih)/2",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        sequences, timestamps = [[] for _ in range(6)], [[] for _ in range(6)]
        start_candidates, peak_candidates = {}, {}
        static_candidates = {}
        sampled, multiple, noface = 0, 0, 0
        names = None
        started = time.perf_counter()
        try:
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                while True:
                    raw = process.stdout.read(SIZE * SIZE * 3)
                    if not raw:
                        break
                    if len(raw) != SIZE * SIZE * 3:
                        raise RuntimeError("Truncated decoded RGB frame")
                    t = sampled / self.fps
                    sampled += 1
                    rgb = np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3)
                    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(t * 1000))
                    if not result.face_landmarks:
                        noface += 1
                        continue
                    if len(result.face_landmarks) != 1:
                        multiple += 1
                        continue
                    points = np.array([[p.x * SIZE, p.y * SIZE] for p in result.face_landmarks[0]])
                    byname = {p.category_name: p.score for p in result.face_blendshapes[0]}
                    current_names = sorted(byname)
                    if names is not None and names != current_names:
                        raise RuntimeError("Blendshape schema changed within a video")
                    names = current_names
                    try:
                        geometry, alignment = normalized_geometry(points, np.array([byname[n] for n in names]))
                    except ValueError:
                        continue
                    if not np.isfinite(geometry).all():
                        continue
                    phase = min(int(t / duration * 6), 5)
                    fraction = t / duration * 6 - phase
                    sequences[phase].append(geometry)
                    timestamps[phase].append(t)
                    if phase == 0:
                        for target in (0.25, 0.5, 0.75):
                            difference = abs(fraction - target)
                            if target not in static_candidates or difference < static_candidates[target][0]:
                                static_candidates[target] = (difference, align_image(rgb, alignment), t)
                        continue
                    baseline = np.median(sequences[0], axis=0) if sequences[0] else geometry
                    difference = abs(fraction - 0.1)
                    if phase not in start_candidates or difference < start_candidates[phase][0]:
                        start_candidates[phase] = (difference, align_image(rgb, alignment), t)
                    if 0.15 <= fraction <= 0.8:
                        score = peak_score(phase, geometry, baseline)
                        if phase not in peak_candidates or score > peak_candidates[phase][0]:
                            peak_candidates[phase] = (score, align_image(rgb, alignment), t)
            process.stdout.close()
            if process.wait(timeout=30) != 0:
                raise RuntimeError("FFmpeg decoding failed")
        finally:
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()
        valid = sum(map(len, sequences))
        quality = {**metadata, "sampled_frames": sampled, "valid_frames": valid,
                   "face_coverage": valid / max(sampled, 1), "multiple_face_frames": multiple,
                   "no_face_frames": noface, "phase_frames": list(map(len, sequences)),
                   "segmentation": "assumed six proportional windows; not manually verified",
                   "mirror_status": "unknown", "device": self.device}
        if quality["face_coverage"] < 0.5 or min(map(len, sequences)) < 3:
            return None, None, {**quality, "status": "ungradable", "reason": "insufficient_face_or_phase_coverage"}
        if len(start_candidates) != 5 or len(peak_candidates) != 5:
            return None, None, {**quality, "status": "ungradable", "reason": "missing_action_keyframes"}
        baseline = np.median(sequences[0], axis=0)
        geometry = np.array([summarize(seq, ts, baseline) for seq, ts in zip(sequences, timestamps)], dtype=np.float32)
        images = []
        for target in (0.25, 0.5, 0.75):
            image = static_candidates[target][1]
            images.extend([image, image.crop(REGIONS[0])])
        evidence = [{"phase": 0, "times": [static_candidates[t][2] for t in (0.25, 0.5, 0.75)]}]
        for phase in range(1, 6):
            for entry in (start_candidates[phase], peak_candidates[phase]):
                images.extend([entry[1], entry[1].crop(REGIONS[phase])])
            evidence.append({"phase": phase, "start": start_candidates[phase][2], "peak": peak_candidates[phase][2]})
        encoded = self.encode(images)
        rest = encoded[:6].reshape(3, 2, 512).mean(axis=0).ravel()
        appearance = [np.r_[rest, rest, np.zeros_like(rest)]]
        for phase in range(5):
            start = encoded[6 + phase * 4:8 + phase * 4].ravel()
            peak = encoded[8 + phase * 4:10 + phase * 4].ravel()
            appearance.append(np.r_[start, peak, peak - start])
        quality.update(status="ok", evidence=evidence, blendshape_names=names,
                       elapsed_seconds=round(time.perf_counter() - started, 3))
        return geometry, np.asarray(appearance, dtype=np.float32), quality

    def cached(self, record, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self.cache_key(record["video_sha256"])
        npz = cache_dir / f"{key}.npz"
        meta = cache_dir / f"{key}.json"
        if meta.exists():
            quality = json.loads(meta.read_text())
            if quality["status"] != "ok":
                return None, None, quality
            with np.load(npz, allow_pickle=False) as data:
                return data["geometry"].copy(), data["appearance"].copy(), quality
        try:
            geometry, appearance, quality = self.extract(record["video_path"])
        except Exception as error:
            geometry, appearance = None, None
            quality = {"status": "failed", "reason": type(error).__name__, "message": str(error)[:250]}
        if geometry is not None:
            np.savez_compressed(npz, geometry=geometry, appearance=appearance)
        meta.write_text(json.dumps(quality, ensure_ascii=False, indent=2))
        return geometry, appearance, quality
