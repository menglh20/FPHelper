"""No patient data: temporary files, fake metadata and ephemeral loopback servers."""
import copy
import http.client
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import PHASES, TIMES, default_record, normalize_record, ValidationError, action_evidence
from storage import Store, ConflictError, Catalog
from server import AnnotationServer, parse_range


class FakeCatalog:
    def __init__(self, root):
        self.root = Path(root)
        self.label = "Synthetic test data"
        self.dataset_id = "test-dataset"
        self.entries = {}
        self.metadata = {}
        for index in (1, 2):
            video_id = f"v_test{index}"
            path = self.root / f"sample{index}.mp4"
            path.write_bytes(bytes(range(256)) * 4)
            self.entries[video_id] = {"path": path, "file_name": path.name}
            self.metadata[video_id] = {"video_id": video_id, "file_name": path.name, "video_sha256": str(index) * 64,
                                       "duration_s": 12.0, "width": 640, "height": 480, "fps_hint": 25, "codec": "h264", "source_side": "unknown"}

    def path(self, video_id):
        return self.entries[video_id]["path"]

    def meta(self, video_id):
        return copy.deepcopy(self.metadata[video_id])


def completed(meta):
    record = default_record(meta)
    record["annotator_id"] = "reviewer01"
    for index, phase in enumerate(PHASES):
        record["actions"][phase].update(presence="present", start_s=index * 2, peak_s=index * 2 + 1,
                                         end_s=(index + 1) * 2, reviewed=True)
    record["status"] = "complete"
    return record


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.meta = {"video_id": "v_one", "video_sha256": "a" * 64, "file_name": "sample.mp4", "duration_s": 12}

    def test_zero_is_real_time_and_missing_is_null(self):
        value = normalize_record(completed(self.meta), self.meta)
        self.assertEqual(value["actions"]["rest"]["start_s"], 0)
        self.assertIsNone(value["actions"]["rest"]["hold_start_s"])
        self.assertTrue(all(v is None for v in value["independent_scores"]["values"].values()))

    def test_invalid_time_order_and_nonfinite_rejected(self):
        for number in (float('nan'), float('inf'), -1, 13, True):
            value = completed(self.meta)
            value["actions"]["rest"]["peak_s"] = number
            with self.assertRaises(ValidationError):
                normalize_record(value, self.meta)
        value = completed(self.meta)
        value["actions"]["rest"]["peak_s"] = 3
        with self.assertRaises(ValidationError):
            normalize_record(value, self.meta)

    def test_partial_timing_can_save_as_draft_but_not_complete(self):
        record = default_record(self.meta)
        record["actions"]["rest"].update(presence="present", start_s=0)
        normalize_record(record, self.meta)
        record["status"] = "complete"
        with self.assertRaises(ValidationError):
            normalize_record(record, self.meta)

    def test_missing_action_never_becomes_normal_or_evidence(self):
        value = completed(self.meta)
        value["actions"]["smile"].update(presence="missing", notes="动作没有执行")
        result = normalize_record(value, self.meta)
        self.assertTrue(all(result["actions"]["smile"][key] is None for key in TIMES))
        with self.assertRaises(ValidationError):
            action_evidence({**result, "revision": 1})

    def test_explicit_model_window_verification_is_separate(self):
        result = normalize_record(completed(self.meta), self.meta)
        with self.assertRaises(ValidationError):
            action_evidence({**result, "revision": 1})
        result["actions"]["rest"]["legacy_window_verified"] = True
        result["legacy_verification_method"] = "人工播放核验对应等分窗口"
        evidence = action_evidence({**result, "revision": 2})
        self.assertEqual(evidence["verified_phases"], ["rest"])
        self.assertEqual(evidence["feature_protocol"], "relative-six-phases-ffmpeg-pts-v1")

    def test_scores_require_integer_ranges_and_rater(self):
        value = default_record(self.meta)
        value["independent_scores"].update(enabled=True, rater_id="doctor-new")
        value["independent_scores"]["values"]["vol_smile"] = 0
        with self.assertRaises(ValidationError):
            normalize_record(value, self.meta)
        value["independent_scores"]["values"]["vol_smile"] = 5
        normalize_record(value, self.meta)
        value["independent_scores"]["rater_id"] = ""
        with self.assertRaises(ValidationError):
            normalize_record(value, self.meta)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = FakeCatalog(self.temp.name)
        self.store = Store(Path(self.temp.name) / "annotations", self.catalog)

    def tearDown(self):
        self.temp.cleanup()

    def test_persistence_conflict_history_and_restore(self):
        record, _ = self.store.get("v_test1")
        record["notes"] = "first"
        self.store.save("v_test1", record, 0)
        with self.assertRaises(ConflictError):
            self.store.save("v_test1", record, 0)
        record["notes"] = "second"
        self.store.save("v_test1", record, 1)
        restored = self.store.restore("v_test1", 1, 2)
        self.assertEqual(restored["revision"], 3)
        self.assertEqual(restored["notes"], "first")
        self.assertEqual(len(self.store.history("v_test1")), 3)
        reopened = Store(Path(self.temp.name) / "annotations", self.catalog)
        self.assertEqual(reopened.get("v_test1")[0]["notes"], "first")

    def test_import_is_transactional_and_never_overwrites(self):
        first = default_record(self.catalog.meta("v_test1"))
        second = default_record(self.catalog.meta("v_test2"))
        second["video_sha256"] = "bad"
        with self.assertRaises(ValidationError):
            self.store.import_records([first, second])
        self.assertEqual(self.store.documents(), {})
        self.assertEqual(self.store.import_records([first]), {"imported": 1, "skipped": 0})
        first["notes"] = "must not overwrite"
        self.assertEqual(self.store.import_records([first])["skipped"], 1)
        self.assertEqual(self.store.get("v_test1")[0]["notes"], "")

    def test_group_import_rolls_back_invalid_second_row(self):
        with self.assertRaises(ValidationError):
            self.store.import_groups([{"video_id": "v_test1", "patient_group_id": "P001"},
                                      {"video_id": "v_test2", "patient_group_id": "=FORMULA"}])
        self.assertEqual(self.store.documents(), {})

    def test_changed_source_does_not_reuse_or_overwrite_annotations(self):
        record, _ = self.store.get("v_test1")
        self.store.save("v_test1", record, 0)
        self.catalog.metadata["v_test1"]["video_sha256"] = "c" * 64
        with self.assertRaises(ConflictError):
            self.store.get("v_test1")
        self.assertEqual(self.store.documents()["v_test1"]["video_sha256"], "1" * 64)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = FakeCatalog(self.temp.name)
        self.store = Store(Path(self.temp.name) / "annotations", self.catalog)
        self.server = AnnotationServer(("127.0.0.1", 0), self.catalog, self.store)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, path, body=None, headers=None, method=None):
        method = method or ("POST" if body is not None else "GET")
        supplied = {"Content-Type": "application/json", "X-CSRF-Token": self.server.csrf_token} if body is not None else {}
        supplied.update(headers or {})
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=supplied)
        response = conn.getresponse()
        status, data, response_headers = response.status, response.read(), dict(response.getheaders())
        conn.close()
        return status, data, response_headers

    def test_range_suffix_head_and_invalid_range(self):
        code, data, headers = self.request('/media/v_test1', headers={"Range": "bytes=2-5"})
        self.assertEqual((code, data), (206, bytes([2, 3, 4, 5])))
        self.assertEqual(headers["Content-Range"], "bytes 2-5/1024")
        self.assertEqual(self.request('/media/v_test1', headers={"Range": "bytes=-4"})[1], bytes([252, 253, 254, 255]))
        code, data, headers = self.request('/media/v_test1', headers={"Range": "bytes=2-5"}, method="HEAD")
        self.assertEqual((code, data, headers["Content-Length"]), (206, b'', "4"))
        for value in ('bytes=1024-', 'bytes=4-2', 'bytes=0-1,3-4', 'bytes=-0'):
            self.assertEqual(self.request('/media/v_test1', headers={"Range": value})[0], 416)

    def test_mutations_and_resources_are_same_origin_only(self):
        record = default_record(self.catalog.meta('v_test1'))
        body = {"record": record, "expected_revision": 0}
        self.assertEqual(self.request('/api/videos/v_test1', body, {"X-CSRF-Token": "wrong"})[0], 403)
        self.assertEqual(self.request('/api/videos/v_test1', body, {"Origin": "https://example.org"})[0], 403)
        self.assertEqual(self.request('/api/config', headers={"Host": "example.org"})[0], 403)
        self.assertEqual(self.request('/media/v_test1', headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.request('/api/videos/v_test1', body)[0], 200)

    def test_exports_preserve_unknown_values_and_require_actual_evidence(self):
        record = completed(self.catalog.meta('v_test1'))
        self.store.save('v_test1', record, 0)
        self.assertEqual(self.request('/api/export/patient-map')[0], 422)
        self.assertEqual(self.request('/api/export/action-evidence')[0], 422)
        code, payload, _ = self.request('/api/export/annotations')
        self.assertEqual(code, 200)
        bundle = json.loads(payload)
        self.assertIsNone(bundle['records'][0]['independent_scores']['values']['vol_smile'])
        self.store.import_groups([{"video_id": 'v_test1', 'patient_group_id': 'P001'}, {"video_id": 'v_test2', 'patient_group_id': 'P001'}])
        code, payload, _ = self.request('/api/export/patient-map')
        self.assertEqual(code, 200)
        self.assertIn('v_test2,P001', payload.decode('utf-8-sig'))
        record, _ = self.store.get('v_test1')
        record['actions']['rest']['legacy_window_verified'] = True
        record['legacy_verification_method'] = '人工逐段检查模型对应窗口'
        self.store.save('v_test1', record, record['revision'])
        code, payload, _ = self.request('/api/export/action-evidence')
        self.assertEqual(code, 200)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            evidence = json.loads(archive.read('v_test1.json'))
            self.assertEqual(evidence['verified_phases'], ['rest'])

    def test_only_registered_media_and_static_assets_are_exposed(self):
        for route in ('/media/v_unknown', '/../../README.md', '/api/videos/v_unknown', '/schema.py'):
            self.assertEqual(self.request(route)[0], 404)
        self.assertEqual(self.request('/')[0], 200)

    def test_patient_export_rejects_identical_bytes_assigned_to_different_patients(self):
        self.catalog.metadata['v_test2']['video_sha256'] = self.catalog.metadata['v_test1']['video_sha256']
        self.store.import_groups([{'video_id': 'v_test1', 'patient_group_id': 'P001'},
                                  {'video_id': 'v_test2', 'patient_group_id': 'P002'}])
        self.assertEqual(self.request('/api/export/patient-map')[0], 422)

    def test_patient_export_does_not_hide_missing_training_videos(self):
        self.catalog.manifest_missing = ['v_missing']
        self.store.import_groups([{'video_id': 'v_test1', 'patient_group_id': 'P001'},
                                  {'video_id': 'v_test2', 'patient_group_id': 'P002'}])
        self.assertEqual(self.request('/api/export/patient-map')[0], 422)
        self.assertEqual(self.request('/api/export/patient-template')[0], 200)


if __name__ == '__main__':
    unittest.main()
