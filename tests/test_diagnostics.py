"""Debug-overlay diagnostics fixtures: pass, weak-match warning, wrong-container
rejection, degenerate-homography rejection.

The overlay must make each case visually obvious and deterministic: it draws
decisions already made by the engine and never re-evaluates quality. Synthetic
scenes provide controlled correspondence only.
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


def textured_plane(seed: int, size=(240, 720), blobs=90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    im = rng.integers(40, 215, (*size, 3), dtype=np.uint8)
    im = cv2.GaussianBlur(im, (3, 3), .65)
    h, w = size
    for _ in range(blobs):
        x, y = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        v = int(rng.integers(20, 235))
        cv2.circle(im, (x, y), int(rng.integers(3, 10)), (v, v, v), -1)
    return im


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def overlap_cfg(self, left: Path, right: Path, *, min_inliers=20,
                    contrast=0.04, seedA=5, seedB=17):
        self.counter += 1
        left_img, right_img = cv2.imread(str(left)), cv2.imread(str(right))
        return {
            "schema_version": 2, "mode": "single", "direction": "horizontal",
            "cross_size_px": 240, "gap_px": 12,
            "sources": {
                "a": {"path": str(left), "expected_size_wh": [left_img.shape[1], left_img.shape[0]]},
                "b": {"path": str(right), "expected_size_wh": [right_img.shape[1], right_img.shape[0]]},
            },
            "containers": [{
                "key": "c1", "method": "overlap", "same_surface_confirmed": True,
                "matching": {"min_inliers": min_inliers, "contrast_threshold": contrast,
                             "ransac_px": 3.0, "feather_px": 24},
                "regions": [
                    {"source": "a", "quad": [[0, 0], [left_img.shape[1] - 1, 0],
                                             [left_img.shape[1] - 1, left_img.shape[0] - 1], [0, left_img.shape[0] - 1]]},
                    {"source": "b", "quad": [[0, 0], [right_img.shape[1] - 1, 0],
                                             [right_img.shape[1] - 1, right_img.shape[0] - 1], [0, right_img.shape[0] - 1]]},
                ],
            }],
        }

    def run_job(self, cfg, name="run"):
        self.counter += 1
        cpath = self.base / f"{name}.json"
        cpath.write_text(json.dumps(cfg))
        out = self.base / name
        report = cs.run_job(cpath, out)
        return report, out

    def test_01_clean_pass_writes_deterministic_overlay_and_json(self):
        plane = textured_plane(5)
        cs.save_image(self.base / "left.png", plane[:, :450])
        cs.save_image(self.base / "right.png", plane[:, 250:])
        cfg = self.overlap_cfg(self.base / "left.png", self.base / "right.png")
        report, out = self.run_job(cfg, "clean")

        self.assertEqual(report["quality_state"], "overlap_requires_visual_review")
        overlay = out / "debug_overlay.png"
        diag_json = out / "diagnostics.json"
        self.assertTrue(overlay.is_file(), "root debug_overlay.png missing")
        self.assertTrue(diag_json.is_file(), "root diagnostics.json missing")
        self.assertTrue((out / "containers" / "c1" / "debug_overlay.png").is_file())

        d = json.loads(diag_json.read_text())
        self.assertEqual(d["method"], "overlap")
        self.assertEqual(d["quality_state"], "overlap_requires_visual_review")
        self.assertIsNone(d["rejection_reason"])
        self.assertGreaterEqual(d["inliers"], 20)
        self.assertGreater(d["overlap_ratio"], 0.3)
        self.assertLess(d["median_reprojection_error_px"], 3.0)
        self.assertTrue(d["sanity_checks"]["scale_bounds"])
        self.assertEqual(len(d["detected_corners"]), 2)
        self.assertTrue(d["overlap_polygon_in_output"], "overlap polygon missing")
        self.assertTrue(d["seam_points_in_output"], "seam polyline missing")

        # deterministic: rerun → byte-identical overlay
        report2, out2 = self.run_job(cfg, "clean2")
        self.assertEqual((out / "debug_overlay.png").read_bytes(),
                         (out2 / "debug_overlay.png").read_bytes())

    def test_02_weak_match_warning_is_flagged(self):
        rng = np.random.default_rng(9)
        # four sparse distinctive blobs in the shared strip: a few genuine,
        # non-collinear matches -> passes min_inliers but lands under the
        # 20-inlier "limited feature support" warning threshold
        im = np.full((240, 700, 3), 110, np.uint8)
        for i in range(4):
            x = 260 + i * 45
            y = 30 + (i * 55) % 180
            cv2.circle(im, (x, y), 9, (int(rng.integers(0, 255)),) * 3, -1)
        im = cv2.GaussianBlur(im, (3, 3), .6)
        cs.save_image(self.base / "left.png", im[:, :450])
        cs.save_image(self.base / "right.png", im[:, 250:])
        cfg = self.overlap_cfg(self.base / "left.png", self.base / "right.png",
                               min_inliers=8, contrast=0.02)
        report, out = self.run_job(cfg, "weak")
        container = report["containers"][0]
        d = container["diagnostics"]
        self.assertEqual(report["quality_state"], "overlap_requires_visual_review")
        self.assertLess(d["inliers"], 20, "fixture should produce limited feature support")
        self.assertGreaterEqual(d["inliers"], 8)
        self.assertTrue(any("Limited feature support" in w for w in container["warnings"]))
        self.assertTrue((out / "debug_overlay.png").is_file())

    def test_03_wrong_container_rejection_writes_reason_overlay(self):
        # two INDEPENDENT planes: no shared surface at all
        cs.save_image(self.base / "left.png", textured_plane(11))
        cs.save_image(self.base / "right.png", textured_plane(23))
        cfg = self.overlap_cfg(self.base / "left.png", self.base / "right.png")
        cpath = self.base / "wrong.json"
        cpath.write_text(json.dumps(cfg))
        out = self.base / "wrong"
        with self.assertRaises(cs.ProcessingError):
            cs.run_job(cpath, out)

        self.assertEqual(json.loads((out / "report.json").read_text())["status"], "rejected")
        d = json.loads((out / "diagnostics.json").read_text())
        self.assertEqual(d["quality_state"], "rejected")
        self.assertTrue(d["rejection_reason"], "rejection reason must be recorded")
        overlay = cv2.imread(str(out / "debug_overlay.png"))
        self.assertIsNotNone(overlay, "rejection overlay missing")
        self.assertGreater(overlay.shape[1], 200)

    def test_04_degenerate_homography_rejection(self):
        # pure vertical stripes: structure is collinear/ambiguous along y
        stripes = np.zeros((240, 700, 3), np.uint8)
        stripes[:, ::14] = 235
        stripes = cv2.GaussianBlur(stripes, (3, 3), .8)
        cs.save_image(self.base / "left.png", stripes[:, :450])
        cs.save_image(self.base / "right.png", stripes[:, 250:])
        cfg = self.overlap_cfg(self.base / "left.png", self.base / "right.png",
                               contrast=0.01)
        cpath = self.base / "degenerate.json"
        cpath.write_text(json.dumps(cfg))
        out = self.base / "degenerate"
        try:
            cs.run_job(cpath, out)
        except cs.ProcessingError:
            pass
        else:
            self.fail("degenerate fixture unexpectedly passed")
        d = json.loads((out / "diagnostics.json").read_text())
        self.assertEqual(d["quality_state"], "rejected")
        self.assertRegex(d["rejection_reason"],
                         r"matches|inliers|homography|scale|concentrated|residuals|extend")
        self.assertTrue((out / "debug_overlay.png").is_file())

    def test_05_overlay_renders_without_matches_or_result(self):
        diag = cs.StitchDiagnostics(
            method="edge", status="manual_edge_join_unverified",
            quality_state="unverified_edge_composite", direction="horizontal",
            source_size_wh=[[100, 50]], detected_corners=[[[0, 0], [99, 0], [99, 49], [0, 49]]],
            rejection_reason=None, seam_points=[[50, 0], [50, 49]], feather_px=8)
        img = np.full((50, 100, 3), 90, np.uint8)
        cs.render_debug_overlay([(img, [0, 0, 100, 50])], diag, None, self.base / "mini.png")
        out = cv2.imread(str(self.base / "mini.png"))
        self.assertIsNotNone(out)
        payload = diag.to_payload()
        json.dumps(payload)  # must be JSON-serializable


if __name__ == "__main__":
    unittest.main()
