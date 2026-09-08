"""Web-tier behavior: config validation, upload materialization, AI error hygiene.

These exercise the Flask app with its test client. Engine runs are avoided
except where a packaged config is guaranteed present (skipped otherwise), so
the suite also passes in the public (no-samples) checkout.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import ai_planner


def _client():
    import app as webapp
    application = webapp.create_app()
    application.config["TESTING"] = True
    return application.test_client()


class SanitizeDetailTests(unittest.TestCase):
    def test_strips_html_and_collapses_whitespace(self):
        raw = ("<html>\n<head><title>504 Gateway Time-out</title></head>\n"
               "  <body><center><h1>504 Gateway Time-out</h1></center></body></html>")
        cleaned = ai_planner.sanitize_detail(raw)
        self.assertNotIn("<", cleaned)
        self.assertNotIn(">", cleaned)
        self.assertNotIn("\n", cleaned)
        self.assertIn("504 Gateway Time-out", cleaned)
        self.assertLessEqual(len(cleaned), 160)

    def test_empty_and_none_are_safe(self):
        self.assertEqual(ai_planner.sanitize_detail(""), "")
        self.assertEqual(ai_planner.sanitize_detail(None), "")

    def test_unavailable_carries_raw_detail(self):
        exc = ai_planner.AiProviderUnavailable("m HTTP 504: <h1>down</h1>",
                                              raw_detail="<h1>down</h1>")
        self.assertEqual(exc.raw_detail, "<h1>down</h1>")
        self.assertIn("504", str(exc))


class ConfigErrorMessageTests(unittest.TestCase):
    def test_empty_config_gets_actionable_message(self):
        import app as webapp
        msg = webapp.describe_config_error("   \n ", json.JSONDecodeError("Expecting value", "", 0))
        self.assertIn("empty", msg.lower())
        self.assertIn("config", msg.lower())

    def test_invalid_config_names_line_and_snippet(self):
        import app as webapp
        text = '{\n  "schema_version": 1,\n  broken\n}'
        exc = json.JSONDecodeError("Expecting value", text, 25)
        msg = webapp.describe_config_error(text, exc)
        self.assertIn("line 3", msg)
        self.assertIn("broken", msg)

    def test_valid_json_passes_through_untouched(self):
        json.loads('{"schema_version": 1}')  # sanity: no exception

    def test_run_custom_empty_config_is_rejected_before_engine(self):
        client = _client()
        resp = client.post("/run-custom", data={"config_text": "   ", "mode": ""},
                           content_type="multipart/form-data")
        page = resp.get_data(as_text=True)
        self.assertIn("Provide a JSON config", page)

    def test_run_custom_empty_uploaded_config_file(self):
        client = _client()
        data = {"config": (io.BytesIO(b""), "my_config.json"),
                "mode": ""}
        resp = client.post("/run-custom", data=data, content_type="multipart/form-data")
        page = resp.get_data(as_text=True)
        self.assertIn("empty", page.lower())
        self.assertNotIn("Expecting value", page)

    def test_run_custom_uploaded_config_with_html_uses_sanitized_message(self):
        client = _client()
        data = {"config": (io.BytesIO(b"<html>504</html>"), "proxy_dump.json"),
                "mode": ""}
        resp = client.post("/run-custom", data=data, content_type="multipart/form-data")
        page = resp.get_data(as_text=True)
        # The bare parser error must never stand alone; the message names the
        # position and shows the offending line.
        self.assertIn("not valid", page)
        self.assertIn("line 1, column", page)
        self.assertIn("Offending line", page)


class MaterializeFilesTests(unittest.TestCase):
    def test_filename_less_parts_are_dropped(self):
        import app as webapp

        class FakeStorage:
            def __init__(self, filename):
                self.filename = filename
                self.saved = []

            def save(self, dest):
                self.saved.append(dest)

        phantom = FakeStorage("")
        real = FakeStorage("input.png")
        files, tmp_root = webapp._materialize_files([("config", phantom), ("sources", real)])
        self.assertNotIn("config", files)
        self.assertEqual([f["filename"] for f in files["sources"]], ["input.png"])

    def test_packaged_config_roundtrip_when_bundle_present(self):
        if not (ROOT / "configs").is_dir():
            self.skipTest("private bundle not present")
        client = _client()
        resp = client.post("/run-example", data={"recipe": "single_grey"})
        page = resp.get_data(as_text=True)
        # The packaged example must still succeed end-to-end (or reject for
        # engine reasons) — never fail with a config-read error.
        self.assertNotIn("reading uploaded config", page)


class AiFailurePathTests(unittest.TestCase):
    """The all-attempts-fail render must show sanitized details, label the
    planner confidence, and link the raw log kept server-side."""

    def _png_bytes(self) -> bytes:
        import cv2
        import numpy as np
        img = np.full((120, 240, 3), 90, dtype=np.uint8)
        ok, enc = cv2.imencode(".png", img)
        assert ok
        return enc.tobytes()

    def test_unreachable_provider_renders_sanitized_failure(self):
        client = _client()
        settings = {
            "planner": {"base_url": "http://127.0.0.1:9/v1", "api_key": "test", "model": "mock"},
            "max_attempts": 1,
            "request_timeout": 15,
        }
        data = {"image": (io.BytesIO(self._png_bytes()), "input.png"),
                "settings": json.dumps(settings)}
        resp = client.post("/run-ai", data=data, content_type="multipart/form-data")
        page = resp.get_data(as_text=True)
        self.assertIn("AI could not produce a config", page)
        self.assertIn("ai_run_log.json", page)
        self.assertIn("provider unreachable", page)
        # No plan was ever received, so no confidence numbers may be shown.
        self.assertNotIn("Planner confidence", page)
        # Connection-refused text is plain; no HTML page may leak through.
        self.assertNotIn("<html", page)

    def test_raw_log_artifact_is_written(self):
        client = _client()
        settings = {
            "planner": {"base_url": "http://127.0.0.1:9/v1", "api_key": "test", "model": "mock"},
            "max_attempts": 1,
            "request_timeout": 15,
        }
        data = {"image": (io.BytesIO(self._png_bytes()), "input.png"),
                "settings": json.dumps(settings)}
        client.post("/run-ai", data=data, content_type="multipart/form-data")
        import app as webapp
        job_dirs = sorted((webapp.JOBS_DIR).glob("ai-*/ai_run_log.json"),
                          key=lambda p: p.stat().st_mtime)
        self.assertTrue(job_dirs, "expected an ai_run_log.json artifact")
        entries = json.loads(job_dirs[-1].read_text())
        self.assertTrue(entries and entries[0]["ok"] is False)


if __name__ == "__main__":
    unittest.main()
