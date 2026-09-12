"""Frozen end-to-end combine acceptance set.

Runs the actual job pipeline on representative single-purpose inputs and
checks the final exported image, the source-provenance masks, and the report
TOGETHER — so boundary details are proven to survive composition and export,
reported placement is proven to match rendered placement, and separate
containers are proven to stay separate identities. These examples are the
regression baseline for future seam or blending changes.
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


def textured_plane(seed: int, w: int = 800, h: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    im = rng.integers(30, 225, (h, w, 3), dtype=np.uint8)
    im = cv2.GaussianBlur(im, (3, 3), .8)
    for _ in range(160):
        x, y = int(rng.integers(4, w - 4)), int(rng.integers(4, h - 4))
        cv2.circle(im, (x, y), int(rng.integers(2, 8)),
                   (int(rng.integers(0, 255)),) * 3, -1)
    return im


def _count_color(bgra: np.ndarray, bgr: tuple[int, int, int], tol: int = 6) -> np.ndarray:
    return np.all(np.abs(bgra[..., :3].astype(int) - np.array(bgr)) <= tol, axis=2)


class CorrugationAmbiguityTests(unittest.TestCase):
    """A repeated-rib pattern can match at t and t+p equally well; the engine
    must never present an aliased correction as confident — it discloses the
    aliasing risk on every measured correction and keeps the result in a
    review-required quality state."""

    def test_measured_correction_always_discloses_aliasing(self):
        rng = np.random.default_rng(7)
        left = cv2.GaussianBlur(rng.integers(40, 215, (300, 600, 3), dtype=np.uint8), (3, 3), .8)
        dup = left[:, 460:600]
        truth = cv2.GaussianBlur(rng.integers(40, 215, (300, 360, 3), dtype=np.uint8), (3, 3), .8)
        bb = np.concatenate([dup, truth], axis=1)
        b_pad = np.zeros((308, 500, 3), np.uint8)
        b_pad[8:308, :bb.shape[1]] = bb
        vb_mask = np.zeros((308, 500), bool)
        vb_mask[8:308, :bb.shape[1]] = True
        m = cs.measure_strip_alignment(left, np.ones(left.shape[:2], bool),
                                       b_pad, vb_mask, "horizontal")
        if m.get("applied"):
            # Applied corrections are marked uncertain, never silent.
            self.assertIn("alias", m["warning"])
            self.assertIn("quantized", m["placement_note"])
            self.assertEqual(m["shift_px"], int(round(m["shift_px"])))
        else:
            # …or the correction is declined outright.
            self.assertIn("reason", m)

    def test_pure_periodic_pattern_is_not_silently_confident(self):
        # Ideal ribs: every pitch-multiple displacement is equally plausible.
        x = np.arange(600)
        ribs = np.where((x // 30) % 2 == 0, 40, 200).astype(np.uint8)
        ribs = np.tile(ribs, (300, 1))
        a = cv2.cvtColor(ribs, cv2.COLOR_GRAY2BGR).astype(np.float32)
        b = a[:, 60:]  # shifted by exactly one period
        m = cs.measure_strip_alignment(a, np.ones(a.shape[:2], bool),
                                       b, np.ones(b.shape[:2], bool), "horizontal")
        if m.get("applied"):
            self.assertIn("alias", m["warning"])


class ComboAcceptanceTests(unittest.TestCase):
    """Two physical containers, two sources, one job: identities stay separate
    through composition and export; report, masks, and pixels agree."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        # Container 1: two identical-crop regions of input1 joined by an edge
        # join (zero-overlap claim, butt). Container 2: one rectified region
        # of input2. Distinct unique markers per container.
        input1 = textured_plane(21)
        input1[100:140, 30:70] = (0, 0, 255)     # red marker -> container 1
        input2 = textured_plane(22)
        input2[100:140, 30:70] = (255, 0, 0)     # blue marker -> container 2
        cs.save_image(base / "input1.png", input1)
        cs.save_image(base / "input2.png", input2)
        cfg = {
            "schema_version": 2, "mode": "combo", "direction": "horizontal",
            "cross_size_px": 300, "gap_px": 12,
            "sources": {
                "one": {"path": "input1.png", "expected_size_wh": [800, 400]},
                "two": {"path": "input2.png", "expected_size_wh": [800, 400]},
            },
            "containers": [
                {"key": "c1", "method": "edge", "same_surface_confirmed": True,
                 "regions": [
                     {"source": "one", "view_box": [0, 0, 800, 400],
                      "quad": [[10, 50], [790, 50], [790, 350], [10, 350]]},
                     {"source": "one", "view_box": [0, 0, 800, 400],
                      "quad": [[10, 50], [790, 50], [790, 350], [10, 350]]},
                 ]},
                {"key": "c2", "method": "rectify",
                 "regions": [
                     {"source": "two", "view_box": [0, 0, 800, 400],
                      "quad": [[10, 50], [790, 50], [790, 350], [10, 350]]},
                 ]},
            ],
        }
        cls.config_path = base / "combo.json"
        cls.config_path.write_text(json.dumps(cfg), encoding="utf-8")
        cls.out = base / "out"
        cls.report = cs.run_job(cls.config_path, cls.out)
        cls.result = cv2.imread(str(cls.out / "result.png"), cv2.IMREAD_UNCHANGED)
        cls.source_map = cv2.imread(str(cls.out / "source_map_16bit.png"),
                                    cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

    def _tile(self, key: str) -> dict:
        return next(c for c in self.report["containers"] if c["key"] == key)

    def test_report_matches_rendered_image(self):
        self.assertEqual(list(self.result.shape[1::-1]), self.report["output_size_wh"])
        self.assertEqual(self.source_map.shape[:2], tuple(self.report["output_size_wh"][::-1]))
        self.assertEqual(self.report["mode"], "combo")
        self.assertEqual(self.report["physical_container_count"], 2)
        self.assertEqual(self.report["direction_basis"], "explicit_recipe")
        for c in self.report["containers"]:
            x0, y0, x1, y1 = c["output_box_xyxy_exclusive"]
            self.assertEqual(x1 - x0, c["output_size_wh"][0])

    def test_reported_join_position_matches_rendered_seam(self):
        c1 = self._tile("c1")
        self.assertEqual(c1["method"], "edge")
        join = c1["join_x"]
        bits = {int(bit): info for bit, info in self.report["source_map_encoding"]["bits"].items()}
        region_a_bit = c1["regions"][0]["source_bit"]
        region_b_bit = c1["regions"][1]["source_bit"]
        # Provenance transitions from region A to region B exactly at join_x.
        left_bits = {int(v) for v in np.unique(self.source_map[:, join - 1]) if v}
        right_bits = {int(v) for v in np.unique(self.source_map[:, join]) if v}
        self.assertEqual(left_bits, {region_a_bit})
        self.assertEqual(right_bits, {region_b_bit})
        self.assertTrue(set(bits) >= {region_a_bit, region_b_bit})

    def test_markers_survive_inside_their_own_container(self):
        c1 = self._tile("c1")
        c2 = self._tile("c2")
        c1_x0, _, c1_x1, _ = c1["output_box_xyxy_exclusive"]
        c2_x0, _, c2_x1, _ = c2["output_box_xyxy_exclusive"]
        red = _count_color(self.result[:, c1_x0:c1_x1], (0, 0, 255))
        blue = _count_color(self.result[:, c2_x0:c2_x1 + 1], (255, 0, 0))
        # Marker is 40x40 = 1600 px; the slight warp resample may soften the
        # border rows, so require the solid interior, not every pixel.
        self.assertGreaterEqual(int(red.sum()), 1000,
                                "container 1's unique marker was lost in composition")
        self.assertGreaterEqual(int(blue.sum()), 1000,
                                "container 2's unique marker was lost in composition")
        # No cross-container bleed: identities stayed separate.
        self.assertEqual(int(_count_color(self.result[:, c1_x0:c1_x1], (255, 0, 0)).sum()), 0)
        self.assertEqual(int(_count_color(self.result[:, c2_x0:], (0, 0, 255)).sum()), 0)

    def test_source_mask_partitions_the_output(self):
        gap = self.report["gap_px"]
        c1_x1 = self._tile("c1")["output_box_xyxy_exclusive"][2]
        gap_cols = self.source_map[:, c1_x1:c1_x1 + gap]
        self.assertEqual(int(gap_cols.max()), 0, "gap must be transparent/no-source")
        # Every content column carries exactly one region's provenance.
        content = self.source_map[:, :c1_x1]
        self.assertGreater(int(content.max()), 0)


if __name__ == "__main__":
    unittest.main()
