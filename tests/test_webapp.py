"""Web-tier behavior: config validation, upload materialization, AI error hygiene.

These exercise the Flask app with its test client. Engine runs are avoided
except where a packaged config is guaranteed present (skipped otherwise), so
the suite also passes in the public (no-samples) checkout.
"""
from __future__ import annotations

import io
import json
import os
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


class PublicExposureTests(unittest.TestCase):
    """Guards for the hosted demo deployment (CST_PUBLIC=1)."""

    def test_ssrf_guard_flags_private_endpoints(self):
        import app as webapp
        err = webapp._assert_public_ai_endpoints(
            {"planner": {"base_url": "http://127.0.0.1:8000/v1", "model": "m"}})
        self.assertIsNotNone(err)
        self.assertIn("private", err)
        err2 = webapp._assert_public_ai_endpoints(
            {"planner": {"base_url": "http://localhost:9000/v1", "model": "m"}})
        self.assertIsNotNone(err2)
        self.assertIn("localhost", err2)

    def test_ssrf_guard_allows_missing_reviewer(self):
        import app as webapp
        self.assertIsNone(webapp._assert_public_ai_endpoints(
            {"planner": {"base_url": "", "model": ""}, "reviewer": {}}))

    def test_run_custom_rejects_path_outside_uploads(self):
        client = _client()
        data = {"config_text": json.dumps({
                    "schema_version": 1, "mode": "single", "height": 64,
                    "sources": {"main": {"path": "../../../etc/passwd"}},
                    "containers": []}),
                "mode": ""}
        resp = client.post("/run-custom", data=data, content_type="multipart/form-data")
        page = resp.get_data(as_text=True)
        self.assertIn("must stay inside the uploaded files folder", page)

    def test_recipe_availability_filter(self):
        import app as webapp
        ok = webapp._BASE_EXAMPLE_RECIPES[0]
        missing = dict(ok)
        missing["config"] = ok["config"].parent / "no_such_recipe.json"
        # The private bundle ships the sample photos; the public checkout does
        # not, so the filter must agree with whichever environment it runs in.
        expected = (webapp.BASE_DIR / "sources" / "single_grey.png").is_file()
        self.assertEqual(webapp._recipe_available(ok), expected)
        self.assertFalse(webapp._recipe_available(missing))

    def test_example_thumb_falls_back_to_demo_source(self):
        """Seeded demo recipes have no examples/<key>/result.png; the thumb
        endpoint must serve their main source instead of 404."""
        import app as webapp
        demo_cfg = webapp.BASE_DIR / "configs" / "demo_horizontal.json"
        if not demo_cfg.is_file():
            self.skipTest("demo seed not generated in this environment")
        source = webapp.BASE_DIR / "sources" / "demo_horizontal.png"
        if not source.is_file():
            # Public checkouts ship the deterministic seeder; Docker images
            # run it at build time. Generate the same pixels either way.
            import demo_seed
            demo_seed.main()
        application = webapp.create_app()
        application.config["TESTING"] = True
        client = application.test_client()
        resp = client.get("/example-thumb/demo_horizontal")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "image/png")

    def test_example_thumb_unknown_key_404(self):
        import app as webapp
        application = webapp.create_app()
        application.config["TESTING"] = True
        client = application.test_client()
        self.assertEqual(client.get("/example-thumb/no_such_demo").status_code, 404)


class AuthTests(unittest.TestCase):
    """Password gate (CST_PASSWORD) — deployed instances require a login."""

    def _client(self, password: str):
        import app as webapp
        saved = os.environ.get("CST_PASSWORD")
        os.environ["CST_PASSWORD"] = password
        try:
            application = webapp.create_app()
        finally:
            if saved is None:
                os.environ.pop("CST_PASSWORD", None)
            else:
                os.environ["CST_PASSWORD"] = saved
        application.config["TESTING"] = True
        return application.test_client()

    def test_no_password_means_no_gate(self):
        client = self._client("")
        self.assertEqual(client.get("/").status_code, 200)

    def test_password_redirects_to_login(self):
        client = self._client("secret123")
        resp = client.get("/")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_login_flow(self):
        client = self._client("secret123")
        page = client.get("/login").get_data(as_text=True)
        self.assertIn("Password", page)
        bad = client.post("/login", data={"password": "wrong"})
        self.assertIn("Incorrect password", bad.get_data(as_text=True))
        ok = client.post("/login", data={"password": "secret123"})
        self.assertEqual(ok.status_code, 302)
        self.assertEqual(client.get("/").status_code, 200)

    def test_logout_clears_session(self):
        client = self._client("secret123")
        client.post("/login", data={"password": "secret123"})
        self.assertEqual(client.get("/").status_code, 200)
        client.post("/logout")
        self.assertEqual(client.get("/").status_code, 302)

    def test_open_redirect_blocked(self):
        client = self._client("secret123")
        client.post("/login", data={"password": "secret123"})
        resp = client.post("/login?next=//evil.example.com", data={"password": "secret123"})
        self.assertNotIn("evil.example.com", resp.headers.get("Location", ""))


class ServerAiDefaultsTests(unittest.TestCase):
    def _with_env(self, **env):
        import app as webapp
        keys = list(env)
        saved = {k: os.environ.get(k) for k in keys}
        os.environ.update(env)
        try:
            return webapp.server_ai_defaults()
        finally:
            for k in keys:
                if saved[k] is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = saved[k]

    def test_absent_when_unset(self):
        self.assertIsNone(self._with_env(CST_AI_PLANNER_BASE="", CST_AI_PLANNER_MODEL=""))

    def test_present_with_env(self):
        defaults = self._with_env(
            CST_AI_PLANNER_BASE="https://api.example.com/v1",
            CST_AI_PLANNER_KEY="test-key-value",
            CST_AI_PLANNER_MODEL="test-model",
            CST_AI_REVIEWER_BASE="https://api.example.com/v1",
            CST_AI_REVIEWER_KEY="test-key-value",
            CST_AI_REVIEWER_MODEL="reviewer-model")
        self.assertTrue(defaults["server_defaults"])
        self.assertEqual(defaults["planner"]["model"], "test-model")
        self.assertEqual(defaults["reviewer"]["model"], "reviewer-model")

    def test_api_never_exposes_secrets(self):
        client = _client()
        keys = ("CST_AI_PLANNER_BASE", "CST_AI_PLANNER_KEY", "CST_AI_PLANNER_MODEL")
        saved = {k: os.environ.get(k) for k in keys}
        os.environ.update({"CST_AI_PLANNER_BASE": "https://api.example.com/v1",
                           "CST_AI_PLANNER_KEY": "TEST-SECRET-VALUE",
                           "CST_AI_PLANNER_MODEL": "test-model"})
        try:
            page = client.get("/api/ai-defaults").get_data(as_text=True)
            self.assertIn("configured", page)
            self.assertIn("test-model", page)
            self.assertNotIn("TEST-SECRET-VALUE", page)
            self.assertNotIn("api.example.com", page)
        finally:
            for k in keys:
                if saved[k] is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = saved[k]


if __name__ == "__main__":
    unittest.main()
