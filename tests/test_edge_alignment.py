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
        # Acceptance is scored under the rendered trim+shift transform.
        self.assertLessEqual(m["rendered_translation_error_px"], 2.0)
        self.assertEqual(m["trim_px"], max(0, m["measured_overlap_px"]))
        self.assertTrue(m["promotion_candidate"])  # overlap clearly present

    def test_realigned_strip_trims_and_shifts(self):
        m = cs.measure_strip_alignment(self.a, self.va, self.b, self.vb, "horizontal")
        b2, vb2 = cs._realigned_strip(self.b, self.vb, m, "horizontal")
        self.assertEqual(b2.shape[1], self.b.shape[1] - m["trim_px"])
        # shift -8 -> canvas grows by 8 (the caller pads strip A to match);
        # every valid pixel that survives the deliberate overlap trim is kept.
        self.assertEqual(b2.shape[0], self.b.shape[0] + 8)
        self.assertEqual(int(vb2.sum()), int(self.vb[:, m["trim_px"]:].sum()))
        self.assertTrue(bool(vb2.sum()))

    def test_realigned_strip_preserves_unique_content(self):
        # A marker filling the bottom ten rows survives a +10 px cross shift:
        # the canvas grows instead of clipping content off the far edge.
        marked = np.zeros((100, 100, 3), np.float32)
        marked[-10:, :, 2] = 255
        valid = np.ones((100, 100), bool)
        for direction, axis in (("horizontal", 0), ("vertical", 1)):
            moved, mv = cs._realigned_strip(
                marked.copy(), valid.copy(), {"trim_px": 0, "shift_px": 10}, direction)
            self.assertEqual(moved.shape[axis], 110, direction)
            self.assertEqual(int(np.count_nonzero(moved[..., 2] > 0)), 1000, direction)
            self.assertEqual(int(mv.sum()), 10_000, direction)

    def test_fractional_shift_is_quantized_without_darkening(self):
        white = np.full((32, 32, 3), 255.0, np.float32)
        valid = np.ones((32, 32), bool)
        for direction in ("horizontal", "vertical"):
            moved, mv = cs._realigned_strip(
                white.copy(), valid.copy(), {"trim_px": 0, "shift_px": 1.25}, direction)
            self.assertEqual(moved.shape[0 if direction == "horizontal" else 1], 33, direction)
            self.assertEqual(float(moved[mv].min()), 255.0, direction)

    def test_scale_mismatch_declines_translation_correction(self):
        # Known B->A relationship includes 10% scale: the similarity fit is
        # excellent, but the translation-only render path cannot honour it, so
        # the correction must be declined instead of scored on the fitted model.
        rng = np.random.default_rng(8743)
        world = cv2.GaussianBlur(
            rng.integers(0, 256, (640, 850, 3), dtype=np.uint8), (3, 3), 0.8)
        a = world[:, :400]
        b = cv2.warpAffine(world, np.float32([[1 / 1.1, 0, -377 / 1.1], [0, 1 / 1.1, 0]]),
                           (400, 640))
        valid = np.ones(a.shape[:2], bool)
        m = cs.measure_strip_alignment(
            a.astype(np.float32), valid, b.astype(np.float32), np.ones(b.shape[:2], bool),
            "horizontal")
        self.assertFalse(m["applied"], m)
        self.assertGreater(m["scale_measured"], 1.05)
        self.assertFalse(m["sanity_checks"]["rendered_error"])
        self.assertGreater(m["rendered_translation_error_px"], 2.0)
        self.assertIn("model_mismatch_note", m)
        self.assertNotIn("trim_px", m)

    def test_flat_strips_are_refused(self):
        flat = np.zeros((300, 600, 3), np.uint8)
        valid = np.ones((300, 600), bool)
        m = cs.measure_strip_alignment(flat, valid, flat, valid, "horizontal")
        self.assertFalse(m["applied"])
        self.assertFalse(m.get("promotion_candidate", False))
        self.assertIn("reason", m)


class DirectionHintTests(unittest.TestCase):
    """Repacking identical views into a different collage layout must not
    change their physical stitching interpretation: an explicit direction is
    the operator's physical claim and overrides the input-packaging hint."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        cs.save_image(self.base / "input.png", textured_plane(13, 800, 400))

    def _config(self, name: str, stacked: bool) -> Path:
        boxes = ([(0, 0, 800, 200), (0, 200, 800, 400)] if stacked
                 else [(0, 0, 400, 400), (400, 0, 800, 400)])
        regions = [
            {"source": "main", "view_box": [x0, y0, x1, y1],
             "quad": [[4, 4], [x1 - x0 - 4, 4], [x1 - x0 - 4, y1 - y0 - 4], [4, y1 - y0 - 4]]}
            for x0, y0, x1, y1 in boxes
        ]
        cfg = {
            "schema_version": 2, "mode": "single", "direction": "horizontal",
            "cross_size_px": 200,
            "sources": {"main": {"path": "input.png", "expected_size_wh": [800, 400]}},
            "containers": [{
                "key": "container_1", "method": "edge", "same_surface_confirmed": True,
                "regions": regions,
            }],
        }
        path = self.base / name
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return path

    def test_explicit_direction_wins_over_packaging(self):
        # The same two views, stacked in the uploaded image: the hint says
        # vertical, the recipe says horizontal. Formerly a hard rejection.
        config, _, _, _, warnings = cs.load_job(self._config("stacked.json", stacked=True))
        self.assertEqual(config["direction"], "horizontal")
        self.assertTrue(any("packaging" in w for w in warnings), warnings)

    def test_matching_packaging_stays_warning_free(self):
        config, _, _, _, warnings = cs.load_job(self._config("side.json", stacked=False))
        self.assertEqual(config["direction"], "horizontal")
        self.assertFalse(any("packaging" in w for w in warnings), warnings)

    def test_auto_still_resolves_from_layout(self):
        cfg_path = self._config("auto.json", stacked=True)
        cfg = json.loads(cfg_path.read_text())
        cfg["direction"] = "auto"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        config, _, _, _, warnings = cs.load_job(cfg_path)
        self.assertEqual(config["direction"], "vertical")
        self.assertTrue(any("auto-resolved" in w for w in warnings), warnings)


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
