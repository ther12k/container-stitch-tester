"""Feature-measured edge seam refinement (edge_alignment: "measure").

A declared-quad edge join butts the two warped strips; when the planner's
corners are slightly off, content is duplicated or shifted at the seam.
"measure" cross-matches the warped strips (SIFT + RANSAC): when the strips
demonstrably share coverage the pair is first promoted to the verified
overlap method, and only when that proof cannot be met does the engine apply
a clamped translation correction to the edge join itself. Fixtures here pin
the promotion, the translation math, the untouched default ("butt"), and the
refusal path.
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


def textured_plane(seed: int, w: int = 1600, h: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    im = rng.integers(30, 225, (h, w, 3), dtype=np.uint8)
    im = cv2.GaussianBlur(im, (3, 3), .8)
    for _ in range(220):
        x, y = int(rng.integers(4, w - 4)), int(rng.integers(4, h - 4))
        r = int(rng.integers(2, 9))
        cv2.circle(im, (x, y), r, (int(rng.integers(0, 255)),) * 3, -1)
    return im


def write_config(base: Path, name: str, edge_alignment: str | None) -> Path:
    """Two-region edge config over one source.

    Truth: the surface spans source columns 100..1100. Region A quad covers
    cols 100..700 rows 50..350. Region B is declared 8 px low and starting at
    col 560 instead of the correct 700, so the warped strips duplicate
    ~140 px of content and sit ~8 px apart vertically before correction.
    """
    cfg: dict = {
        "schema_version": 2, "mode": "single", "direction": "horizontal",
        "cross_size_px": 300, "gap_px": 8,
        "sources": {"main": {"path": "input.png", "expected_size_wh": [1600, 400]}},
        "containers": [{
            "key": "container_1", "method": "edge", "same_surface_confirmed": True,
            "regions": [
                {"source": "main", "view_box": [0, 0, 800, 400],
                 "quad": [[100, 50], [700, 50], [700, 350], [100, 350]]},
                {"source": "main", "view_box": [0, 0, 1600, 400],
                 "quad": [[560, 58], [1060, 58], [1060, 358], [560, 358]]},
            ],
        }],
    }
    if edge_alignment:
        cfg["edge_alignment"] = edge_alignment
    path = base / name
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def warped_strips(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two synthetic strips with a known 140 px duplicated band and 8 px offset."""
    rng = np.random.default_rng(seed)
    left = rng.integers(40, 215, (300, 600, 3), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), .8)
    # strip B: source truth continues from column 560, i.e. its first 140 px
    # duplicate A's last 140 px; the whole strip sits 8 px low.
    dup = left[:, 460:600]
    truth = rng.integers(40, 215, (300, 360, 3), dtype=np.uint8)
    truth = cv2.GaussianBlur(truth, (3, 3), .8)
    b = np.concatenate([dup, truth], axis=1)
    b_padded = np.zeros((308, 500, 3), np.uint8)
    b_padded[8:308, :b.shape[1]] = b
    vb = np.zeros((308, 500), bool)
    vb[8:308, :b.shape[1]] = True
    return left, np.ones(left.shape[:2], bool), b_padded, vb


class MeasureStripAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.a, self.va, self.b, self.vb = warped_strips(7)

    def test_measures_known_overlap_and_offset(self):
        m = cs.measure_strip_alignment(self.a, self.va, self.b, self.vb, "horizontal")
        self.assertTrue(m["applied"], m.get("reason"))
        self.assertAlmostEqual(m["measured_overlap_px"], 140, delta=8)
        # B was built 8 px low, so mapping B->A measures -8
        self.assertAlmostEqual(m["cross_offset_px"], -8, delta=3)
        self.assertGreaterEqual(m["inliers"], 12)
        self.assertLessEqual(m["median_reprojection_error_px"], 2.0)
        self.assertEqual(m["trim_px"], max(0, m["measured_overlap_px"]))
        self.assertTrue(m["promotion_candidate"])  # overlap clearly present

    def test_realigned_strip_trims_and_shifts(self):
        m = cs.measure_strip_alignment(self.a, self.va, self.b, self.vb, "horizontal")
        b2, vb2 = cs._realigned_strip(self.b, self.vb, m, "horizontal")
        self.assertEqual(b2.shape[1], self.b.shape[1] - m["trim_px"])
        # valid pixels shifted up by the cross offset; nothing invented below
        self.assertLess(int(vb2[-1].sum()), int(self.vb[-1].sum()))

    def test_flat_strips_are_refused(self):
        flat = np.zeros((300, 600, 3), np.uint8)
        valid = np.ones((300, 600), bool)
        m = cs.measure_strip_alignment(flat, valid, flat, valid, "horizontal")
        self.assertFalse(m["applied"])
        self.assertFalse(m.get("promotion_candidate", False))
        self.assertIn("reason", m)


class EdgeAlignmentJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        cs.save_image(self.base / "input.png", textured_plane(11))

    def test_rich_texture_promotes_to_verified_overlap(self):
        out = self.base / "out_measure"
        report = cs.run_job(write_config(self.base, "cfg.json", "measure"), out)
        container = report["containers"][0]
        self.assertEqual(container["method"], "overlap")
        self.assertIn("promoted_from_edge", container)
        self.assertEqual(container["status"], "overlap_estimated_requires_visual_review")
        self.assertEqual(report["quality_state"], "overlap_requires_visual_review")
        alignment = container["promoted_from_edge"]["seam_alignment"]
        self.assertAlmostEqual(alignment["measured_overlap_px"], 140, delta=10)
        w = report["output_size_wh"][0]
        self.assertAlmostEqual(w, 600 + 500 - 140, delta=10)  # A + B - shared band

    def test_butt_default_is_unchanged(self):
        out = self.base / "out_butt"
        report = cs.run_job(write_config(self.base, "cfg.json", None), out)
        container = report["containers"][0]
        self.assertIsNone(container.get("seam_alignment"))
        self.assertNotIn("promoted_from_edge", container)
        self.assertEqual(container["status"], "manual_edge_join_unverified")
        self.assertEqual(report["quality_state"], "unverified_edge_composite")
        self.assertAlmostEqual(report["output_size_wh"][0], 1100, delta=4)

    def test_invalid_edge_alignment_value_rejected(self):
        path = write_config(self.base, "cfg.json", "magic")
        with self.assertRaises(cs.ProcessingError):
            cs.run_job(path, self.base / "out_bad")

    def test_diagnostics_payload_carries_alignment(self):
        out = self.base / "out_diag"
        cs.run_job(write_config(self.base, "cfg.json", "measure"), out)
        diag = json.loads((out / "diagnostics.json").read_text())
        self.assertTrue(diag.get("seam_alignment"))
        per_container = json.loads(
            (out / "containers" / "container_1" / "diagnostics.json").read_text())
        self.assertTrue(per_container.get("seam_alignment"))


if __name__ == "__main__":
    unittest.main()
