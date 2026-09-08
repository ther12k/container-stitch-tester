"""Fixed-camera profile fixtures: calibration reuse, size/pin enforcement.

Profiles package reviewed corners for one fixed camera. Conversion is
deterministic and refuses reinterpreting corners for a differently-sized frame
or (when pinned) a different physical frame.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import container_stitch as cs
import camera_profiles as cp


def textured_plane(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    im = rng.integers(40, 215, (240, 720, 3), dtype=np.uint8)
    im = cv2.GaussianBlur(im, (3, 3), .65)
    for _ in range(90):
        x, y = int(rng.integers(10, 710)), int(rng.integers(10, 230))
        cv2.circle(im, (x, y), int(rng.integers(3, 10)), (int(rng.integers(20, 235)),) * 3, -1)
    return im


class CameraProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        # calibrated frame + capture from the same fixed camera
        self.frame = textured_plane(5)
        self.capture = self.frame.copy()
        cs.save_image(self.base / "capture.png", self.capture)

    def tearDown(self):
        self.tmp.cleanup()

    def profile(self, **overrides):
        p = {
            "schema_version": 1,
            "profile_key": "crane_a_north",
            "camera": {"label": "Crane A north camera", "expected_size_wh": [720, 240]},
            "job_defaults": {"mode": "single", "direction": "horizontal",
                             "cross_size_px": 240, "gap_px": 12},
            "containers": [{
                "key": "c1", "method": "overlap",
                "matching": {"min_inliers": 20, "contrast_threshold": 0.04,
                             "ransac_px": 3.0, "feather_px": 24},
                # absolute-space quads over the shared strip of the two halves
                "regions": [
                    {"quad": [[0, 0], [449, 0], [449, 239], [0, 239]]},
                    {"quad": [[250, 0], [699, 0], [699, 239], [250, 239]]},
                ],
            }],
        }
        p.update(overrides)
        return p

    def run_profile(self, profile, capture, name="run"):
        cfg = cp.profile_to_config(profile, capture, source_path=str(capture))
        cpath = self.base / f"{name}.json"
        cpath.write_text(json.dumps(cfg))
        return cs.run_job(cpath, self.base / name), cfg

    def test_01_profile_generates_config_and_stitches(self):
        report, cfg = self.run_profile(self.profile(), self.base / "capture.png")
        self.assertEqual(report["status"], "created_requires_review")
        self.assertEqual(report["quality_state"], "overlap_requires_visual_review")
        r1, r2 = cfg["containers"][0]["regions"]
        self.assertEqual(r1["view_box"], [0, 0, 450, 240])
        self.assertEqual(r2["view_box"], [250, 0, 700, 240])
        self.assertEqual(r2["quad"][0], [0, 0])
        self.assertEqual(r2["quad"][1], [449, 0])
        self.assertTrue((self.base / "run" / "metrics.json").is_file())

    def test_02_wrong_size_capture_rejected(self):
        other = np.zeros((300, 800, 3), np.uint8)
        cs.save_image(self.base / "other_size.png", other)
        with self.assertRaises(cs.ProcessingError) as ctx:
            cp.profile_to_config(self.profile(), self.base / "other_size.png")
        self.assertIn("does not match the calibrated camera", str(ctx.exception))

    def test_03_enforced_pin_rejects_different_frame(self):
        different = textured_plane(17)
        cs.save_image(self.base / "different.png", different)
        import hashlib
        pin_sha = hashlib.sha256((self.base / "capture.png").read_bytes()).hexdigest()
        profile = self.profile(pin={"reference_sha256": pin_sha, "enforce": True})
        # identical capture passes
        cp.profile_to_config(profile, self.base / "capture.png")
        # different frame, same size: rejected
        with self.assertRaises(cs.ProcessingError) as ctx:
            cp.profile_to_config(profile, self.base / "different.png")
        self.assertIn("pinned reference frame", str(ctx.exception))

    def test_04_invalid_profiles_rejected(self):
        bad = self.profile(containers=[{"key": "c1", "method": "rectify", "regions": [
            {"quad": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"quad": [[20, 0], [30, 0], [30, 10], [20, 10]]},  # rectify allows only 1
        ]}])
        with self.assertRaises(cs.ProcessingError):
            cp.validate_profile(bad)
        with self.assertRaises(cs.ProcessingError):
            cp.validate_profile(self.profile(schema_version=2))
        with self.assertRaises(cs.ProcessingError):
            cp.validate_profile({"schema_version": 1, "unknown_field": True})

    def test_05_metrics_json_summarizes_the_run(self):
        report, _ = self.run_profile(self.profile(), self.base / "capture.png")
        metrics = json.loads((self.base / "run" / "metrics.json").read_text())
        self.assertEqual(metrics["job_verdict"], "overlap_requires_visual_review")
        self.assertEqual(metrics["container_count"], 1)
        c = metrics["containers"][0]
        self.assertEqual(c["method"], "overlap")
        self.assertGreaterEqual(c["inliers"], 20)
        self.assertGreater(c["overlap_ratio"], 0.3)
        self.assertTrue(c["sanity_checks"]["scale_bounds"])

    def test_06_rejected_run_still_writes_metrics(self):
        # independent planes -> engine rejection; metrics must reflect it
        cs.save_image(self.base / "a.png", textured_plane(11))
        cs.save_image(self.base / "b.png", textured_plane(23))
        cfg = {
            "schema_version": 2, "mode": "single", "direction": "horizontal",
            "cross_size_px": 240, "gap_px": 12,
            "sources": {"a": {"path": str(self.base / "a.png"), "expected_size_wh": [450, 240]},
                        "b": {"path": str(self.base / "b.png"), "expected_size_wh": [450, 240]}},
            "containers": [{"key": "c1", "method": "overlap", "same_surface_confirmed": True,
                            "matching": {"min_inliers": 20, "contrast_threshold": 0.04,
                                         "ransac_px": 3.0, "feather_px": 24},
                            "regions": [
                                {"source": "a", "quad": [[0, 0], [449, 0], [449, 239], [0, 239]]},
                                {"source": "b", "quad": [[0, 0], [449, 0], [449, 239], [0, 239]]}]}],
        }
        cpath = self.base / "reject.json"
        cpath.write_text(json.dumps(cfg))
        with self.assertRaises(cs.ProcessingError):
            cs.run_job(cpath, self.base / "reject_out")
        metrics = json.loads((self.base / "reject_out" / "metrics.json").read_text())
        self.assertEqual(metrics["job_verdict"], "rejected")
        self.assertTrue(metrics["rejection_reason"])


if __name__ == "__main__":
    unittest.main()
