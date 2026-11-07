"""Run a frozen legacy pipeline on one local video; never read any labels."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(job):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    algorithm = Path(job['algorithm']).resolve()
    os.chdir(algorithm)
    started = time.perf_counter()
    result = {'method': job['method'], 'status': 'failed'}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            detect = load_module('legacy_detect', algorithm / 'Detect.py')
            with tempfile.TemporaryDirectory(prefix='fp-legacy-', dir=job.get('temp_root')) as directory:
                folder = directory + os.sep
                if job['method'] == 'LEGACY_E2E':
                    recognize = load_module('legacy_recognize', algorithm / 'Recognize.py')
                    recognize.recognize(job['video_path'], folder, debug=False)
                elif job['method'] == 'LEGACY_WINDOW':
                    # Diagnostic: take the existing research extractor's selected
                    # timestamps. No expert labels or manual annotation are used.
                    names = ['at_rest', 'forehead_wrinkle', 'eye_closure', 'smile', 'snarl', 'lip_pucker']
                    cap = cv2.VideoCapture(job['video_path'])
                    try:
                        for name, stamp in zip(names, job['timestamps']):
                            cap.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)
                            success, frame = cap.read()
                            if not success:
                                raise RuntimeError('selected_frame_unavailable')
                            if not cv2.imwrite(folder + 'pic_' + name + '.jpg', frame):
                                raise RuntimeError('temporary_frame_write_failed')
                    finally:
                        cap.release()
                else:
                    raise ValueError('unknown_legacy_method')
                total, detail = detect.detect(folder)
                if detail is None:
                    raise RuntimeError('legacy_images_unreadable')
                actions = ('forehead_wrinkle', 'eye_closure', 'smile', 'snarl', 'lip_pucker')
                values = [detail['rest symmetry'][k] for k in ('eye', 'cheek', 'mouth')]
                for section in ('voluntary symmetry', 'synkinesis'):
                    group = {k.replace(' ', '_'): v for k, v in detail[section].items()}
                    values.extend(group[k] for k in actions)
                if len(values) != 13 or not np.isfinite(values).all() or not np.isfinite(total):
                    raise FloatingPointError('nonfinite_legacy_scores')
                expected = 4 * sum(values[3:8]) - 5 * sum(values[:3]) - sum(values[8:])
                if total != expected:
                    raise RuntimeError('legacy_total_mismatch')
                result.update(status='ok', prediction=[int(v) for v in values], raw_total=int(total))
    except Exception as error:
        # Error classes are sufficient for failure accounting; omit source paths,
        # filenames, frames and legacy stdout from persistent logs.
        result.update(error_type=type(error).__name__)
    result['elapsed_seconds'] = time.perf_counter() - started
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.job.read_text())), allow_nan=False), flush=True)
