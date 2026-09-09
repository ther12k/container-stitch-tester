"""AI plan coverage consistency: combo scenes must yield combo results.

The planner used to be free to plan one group for a two-container scene,
silently dropping a container (the "it's combo but the result is single"
report). The contract is now enforced deterministically: one group per
visible physical container unless the plan records exclusions with reasons.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

import ai_planner
import container_stitch as cs


def region(x0, y0, x1, y1, qx0=None, qy0=None, qx1=None, qy1=None):
    qx0 = x0 + 0.05 if qx0 is None else qx0
    qy0 = y0 + 0.05 if qy0 is None else qy0
    qx1 = x1 - 0.05 if qx1 is None else qx1
    qy1 = y1 - 0.05 if qy1 is None else qy1
    return {"view_box_normalized": [x0, y0, x1, y1],
            "quad_normalized": [[qx0, qy0], [qx1, qy0], [qx1, qy1], [qx0, qy1]]}


def group(key, method, regions):
    return {"key": key, "method": method, "regions": regions}


class CoverageConsistencyTests(unittest.TestCase):
    def test_two_containers_one_group_without_exclude_is_rejected(self):
        plan = {"scene": {"physical_containers": 2},
                "containers": [group("c1", "rectify", [region(0.4, 0.1, 0.6, 0.9)])],
                "exclude": []}
        with self.assertRaises(ValueError) as ctx:
            ai_planner.validate_plan(plan)
        self.assertIn("coverage mismatch", str(ctx.exception))
        self.assertIn("2 physical container", str(ctx.exception))

    def test_two_containers_two_groups_pass(self):
        plan = {"scene": {"physical_containers": 2},
                "containers": [group("c1", "rectify", [region(0.1, 0.1, 0.4, 0.9)]),
                               group("c2", "rectify", [region(0.6, 0.1, 0.9, 0.9)])]}
        ai_planner.validate_plan(plan)

    def test_left_out_container_requires_named_exclusion(self):
        plan = {"scene": {"physical_containers": 2},
                "containers": [group("c1", "rectify", [region(0.1, 0.1, 0.4, 0.9)])],
                "exclude": ["container 2 fully occluded by the spreader"]}
        ai_planner.validate_plan(plan)

    def test_more_groups_than_containers_is_rejected(self):
        plan = {"scene": {"physical_containers": 1},
                "containers": [group("c1", "rectify", [region(0.1, 0.1, 0.4, 0.9)]),
                               group("c2", "rectify", [region(0.6, 0.1, 0.9, 0.9)])],
                "exclude": []}
        with self.assertRaises(ValueError) as ctx:
            ai_planner.validate_plan(plan)
        self.assertIn("at most one group per container", str(ctx.exception))

    def test_missing_scene_count_is_tolerated(self):
        plan = {"containers": [group("c1", "rectify", [region(0.1, 0.1, 0.9, 0.9)])]}
        ai_planner.validate_plan(plan)

    def test_plan_to_config_yields_combo_for_two_groups(self):
        cfg = ai_planner.plan_to_config(
            {"containers": [group("c1", "rectify", [region(0.1, 0.1, 0.4, 0.9)]),
                            group("c2", "rectify", [region(0.6, 0.1, 0.9, 0.9)])]}, 1920, 2160)
        self.assertEqual(cfg["mode"], "combo")


class SeparationWarningTests(unittest.TestCase):
    def test_far_apart_quads_get_disclosed_warning(self):
        import cv2
        import numpy as np
        rng = np.random.default_rng(3)
        img = rng.integers(30, 225, (400, 1600, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cs.save_image(base / "input.png", img)
            cfg = {
                "schema_version": 2, "mode": "single", "direction": "horizontal",
                "cross_size_px": 300, "gap_px": 8,
                "sources": {"main": {"path": "input.png", "expected_size_wh": [1600, 400]}},
                "containers": [{
                    "key": "container_1", "method": "edge", "same_surface_confirmed": True,
                    "regions": [
                        {"source": "main", "view_box": [0, 0, 800, 400],
                         "quad": [[50, 50], [400, 50], [400, 350], [50, 350]]},
                        {"source": "main", "view_box": [800, 0, 1600, 400],
                         "quad": [[250, 60], [600, 60], [600, 340], [250, 340]]},
                    ],
                }],
            }
            (base / "cfg.json").write_text(json.dumps(cfg))
            report = cs.run_job(base / "cfg.json", base / "out")
            warnings = " ".join(report["containers"][0]["warnings"])
            self.assertIn("separated by", warnings)
            self.assertIn("one container, not two", warnings)
            diag = json.loads((base / "out" / "containers" / "container_1"
                               / "diagnostics.json").read_text())
            self.assertIn("quad_separation_px", diag["sanity_checks"])

    def test_adjacent_quads_do_not_warn(self):
        import numpy as np
        rng = np.random.default_rng(5)
        img = rng.integers(30, 225, (400, 1600, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cs.save_image(base / "input.png", img)
            cfg = {
                "schema_version": 2, "mode": "single", "direction": "horizontal",
                "cross_size_px": 300, "gap_px": 8,
                "sources": {"main": {"path": "input.png", "expected_size_wh": [1600, 400]}},
                "containers": [{
                    "key": "container_1", "method": "edge", "same_surface_confirmed": True,
                    "regions": [
                        {"source": "main", "view_box": [0, 0, 800, 400],
                         "quad": [[100, 50], [780, 50], [780, 350], [100, 350]]},
                        {"source": "main", "view_box": [800, 0, 1600, 400],
                         "quad": [[10, 55], [700, 55], [700, 345], [10, 345]]},
                    ],
                }],
            }
            (base / "cfg.json").write_text(json.dumps(cfg))
            report = cs.run_job(base / "cfg.json", base / "out")
            warnings = " ".join(report["containers"][0]["warnings"])
            self.assertNotIn("separated by", warnings)


class AiRunCoverageRetryTests(unittest.TestCase):
    """End-to-end app flow: a 1-group plan for a 2-container scene is retried
    into a 2-group combo, using a scripted planner (no network)."""

    def _png(self) -> bytes:
        import cv2
        import numpy as np
        img = np.full((400, 800, 3), 90, dtype=np.uint8)
        ok, enc = cv2.imencode(".png", img)
        assert ok
        return enc.tobytes()

    def test_retry_upgrades_single_to_combo(self):
        import app as webapp
        application = webapp.create_app()
        application.config["TESTING"] = True
        client = application.test_client()

        bad_plan = json.dumps({
            "scene": {"view_layout": "horizontal_pair", "physical_containers": 2},
            "target": {"container_index": 0, "surface": "side"},
            "containers": [group("container_1", "rectify", [region(0.1, 0.1, 0.4, 0.9)])],
            "exclude": [], "reason": "one container only",
            "confidence": {"direction": 0.9, "container_grouping": 0.9, "same_surface": 0.9},
        })
        good_plan = json.dumps({
            "scene": {"view_layout": "horizontal_pair", "physical_containers": 2},
            "target": {"container_index": 0, "surface": "side"},
            "containers": [group("container_1", "rectify", [region(0.1, 0.1, 0.4, 0.9)]),
                           group("container_2", "rectify", [region(0.6, 0.1, 0.9, 0.9)])],
            "exclude": [], "reason": "both containers",
            "confidence": {"direction": 0.9, "container_grouping": 0.95, "same_surface": 0.9},
        })
        replies = iter([bad_plan, good_plan])
        with mock.patch.object(ai_planner, "chat_completion", side_effect=lambda *a, **k: next(replies)):
            data = {"image": (io.BytesIO(self._png()), "input.png"),
                    "settings": json.dumps({
                        "planner": {"base_url": "http://example.invalid/v1",
                                    "api_key": "k", "model": "mock"},
                        "max_attempts": 2, "request_timeout": 15})}
            resp = client.post("/run-ai", data=data, content_type="multipart/form-data")
        page = resp.get_data(as_text=True)
        # Engine accepted attempt 2 with a combo config.
        self.assertIn("combo", page.lower())
        self.assertIn("coverage mismatch", page)  # disclosed in the attempt log


if __name__ == "__main__":
    unittest.main()
