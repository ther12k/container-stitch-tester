"""Direction/layout, real roof examples, provenance and safe batch processing.

Known-translation fixtures test numerical behavior only. They are not evidence
that arbitrary camera images share a physical plane or correct correspondence.
"""
from __future__ import annotations
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import container_stitch as cs

# Tests further down use the real roof recipes; skip the class when the private
# sample bundle is absent (the synthetic-fixture tests remain available in the
# private checkout).
PRIVATE_BUNDLE = (ROOT / 'sources').is_dir()


class VerticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.fixtures = Path(cls.shared.name)
        rng = np.random.default_rng(35)
        im = cv2.GaussianBlur(rng.integers(40, 215, (700, 220, 3), dtype=np.uint8), (3, 3), .65)
        for _ in range(60):
            x, y = int(rng.integers(10, 210)), int(rng.integers(10, 690))
            v = int(rng.integers(10, 240))
            cv2.circle(im, (x, y), int(rng.integers(3, 10)), (v,)*3, -1)
        cs.save_image(cls.fixtures/'top.png', im[:450])
        cs.save_image(cls.fixtures/'bottom.png', im[250:])
        cs.save_image(cls.fixtures/'blank.png', np.full((450, 220, 3), 127, np.uint8))
        cls.roof_runs = {}
        if not PRIVATE_BUNDLE:
            raise unittest.SkipTest(
                'private sample bundle (sources/, examples/) is not part of this repository')
        for name in ('vertical_blue_single', 'vertical_blue_combo', 'vertical_red_single'):
            out = cls.fixtures / name
            report = cs.run_job(ROOT/'configs'/f'{name}.json', out)
            cls.roof_runs[name] = (report, out)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def roof(self, name='vertical_blue_single'):
        return self.roof_runs[name]

    def cfg(self, space='source', method='overlap'):
        q = [[0, 0], [219, 0], [219, 449], [0, 449]]
        c = {'schema_version': 2, 'mode': 'single', 'direction': 'vertical', 'layout': 'vertical',
             'cross_size_px': 220, 'gap_px': 16,
             'sources': {n: {'path': str(self.fixtures/f'{n}.png'), 'expected_size_wh': [220, 450]}
                         for n in ('top', 'bottom')},
             'containers': [{'key': 'roof', 'method': method, 'coverage': 'unverified',
                             'same_surface_confirmed': True,
                             'regions': [{'source': n, 'quad': q} for n in ('top', 'bottom')]}]}
        if method == 'overlap':
            c['containers'][0]['matching'] = {'space': space, 'min_inliers': 20, 'feather_px': 16}
        if method == 'rectify':
            c['containers'][0]['regions'] = c['containers'][0]['regions'][:1]
        return c

    def write(self, c):
        self.counter += 1
        path = self.base/f'cfg_{self.counter}.json'
        path.write_text(json.dumps(c))
        return path, self.base/f'out_{self.counter}'

    def process(self, c, **kwargs):
        path, out = self.write(c)
        return cs.run_job(path, out, **kwargs), out

    def reject(self, c, contains=None, **kwargs):
        path, out = self.write(c)
        with self.assertRaises(cs.ProcessingError) as exc:
            cs.run_job(path, out, **kwargs)
        if contains:
            self.assertIn(contains, str(exc.exception))
        self.assertEqual({p.name for p in out.iterdir()}
                         - {'debug_overlay.png', 'diagnostics.json', 'containers'},
                         {'report.json'})
        r = json.loads((out/'report.json').read_text())
        self.assertFalse(r['result_created'])
        self.assertFalse(r['fallback_to_edge'])
        return r

    def test_01_blue_input_is_exact_uploaded_file(self):
        r, _ = self.roof()
        self.assertEqual(r['source_records'][0]['sha256'], 'abcbfa387cc4699873b249b7bf0e941307859b29c73df1de09fbeaff9284ccdb')
        self.assertEqual(r['source_records'][0]['size_wh'], [1820, 2048])

    def test_02_blue_roof_fit_and_configured_seam(self):
        r, _ = self.roof()
        g = r['containers'][0]
        self.assertEqual(r['output_size_wh'], [360, 756])
        self.assertEqual(g['matching']['inliers'], 27)
        self.assertEqual(g['matching']['mutual_unique_candidates'], 49)
        self.assertAlmostEqual(g['matching']['median_reprojection_error_px'], .8241767, places=4)
        self.assertEqual(g['seam']['position_px'], 696)
        self.assertEqual(g['seam']['axis'], 'y')
        self.assertFalse(g['independently_verified_overlap'])

    def test_03_red_partial_and_limited_support_are_reported(self):
        r, _ = self.roof('vertical_red_single')
        g = r['containers'][0]
        self.assertEqual(g['coverage'], 'partial')
        self.assertEqual(g['matching']['inliers'], 10)
        self.assertEqual(g['matching']['mutual_unique_candidates'], 16)
        self.assertTrue(any('fewer than 20' in w for w in g['warnings']))
        self.assertEqual(g['seam']['selection'], 'automatic_pixel_difference')
        self.assertFalse(r['independently_verified_overlap'])

    def test_04_combo_retains_independent_partial_second_container(self):
        r, _ = self.roof('vertical_blue_combo')
        self.assertEqual(r['physical_container_count'], 2)
        self.assertEqual([g['method'] for g in r['containers']], ['overlap', 'rectify'])
        self.assertEqual(r['containers'][1]['coverage'], 'partial')
        self.assertFalse(r['cross_container_blending'])
        self.assertEqual(r['output_size_wh'], [360, 1411])

    def test_05_combo_upper_matches_single_result_pixels(self):
        _, s = self.roof()
        r, out = self.roof('vertical_blue_combo')
        im = cv2.imread(str(out/'result.png'), -1)
        x0, y0, x1, y1 = r['containers'][0]['output_box_xyxy_exclusive']
        self.assertTrue(np.array_equal(im[y0:y1, x0:x1], cv2.imread(str(s/'result.png'), -1)))

    def test_06_vertical_gutter_transparent_and_never_cross_blended(self):
        r, out = self.roof('vertical_blue_combo')
        im = cv2.imread(str(out/'result.png'), -1)
        sm = cv2.imread(str(out/'source_map_16bit.png'), -1)
        cm = cv2.imread(str(out/'container_map.png'), -1)
        a = r['containers'][0]['output_box_xyxy_exclusive'][3]
        b = r['containers'][1]['output_box_xyxy_exclusive'][1]
        self.assertEqual(b-a, 16)
        self.assertTrue(np.all(im[a:b, :, 3] == 0))
        self.assertTrue(np.all(sm[a:b] == 0))
        self.assertTrue(np.all(cm[a:b] == 0))
        self.assertEqual(set(np.unique(sm)), {0, 1, 2, 3, 4})
        self.assertTrue(np.all((sm > 0) == (im[..., 3] > 0)))

    def test_07_source_selected_no_old_view_side_strips(self):
        r, out = self.roof()
        g = r['containers'][0]
        sm = cv2.imread(str(out/'source_map_16bit.png'), -1)
        self.assertTrue(np.all((sm[701:] & 1) == 0))
        self.assertTrue(np.all((sm[:692] & 2) == 0))
        self.assertGreater(g['seam']['union_pixels_omitted_by_selection'], 0)

    def test_08_weights_reconstruct_rounded_source_colors(self):
        _, out = self.roof()
        sub = out/'containers'/'blue_upper'
        a = cv2.imread(str(sub/'warped_first.png'), -1)[..., :3].astype(float)
        b = cv2.imread(str(sub/'warped_second.png'), -1)[..., :3].astype(float)
        t = cv2.imread(str(sub/'second_weight_16bit.png'), -1).astype(float)/65535
        im = cv2.imread(str(sub/'result.png'), -1)
        recomposed = np.rint(a*(1-t[..., None]) + b*t[..., None])
        self.assertLessEqual(np.abs(recomposed-im[..., :3])[im[..., 3] > 0].max(), 1)

    def test_09_final_homographies_include_vertical_group_offset(self):
        r, _ = self.roof('vertical_blue_combo')
        g = r['containers'][1]
        for region in g['regions']:
            H = np.array(region['input_to_final_homography'])
            q = np.array(region['quad_tl_tr_br_bl_in_view']) + np.array(region['view_box_xyxy_exclusive'][:2])
            transformed = cv2.perspectiveTransform(q[None].astype(np.float64), H)[0]
            self.assertAlmostEqual(float(transformed[:, 1].min()), g['output_box_xyxy_exclusive'][1], places=4)
            self.assertAlmostEqual(float(transformed[:, 1].max()), g['output_box_xyxy_exclusive'][3]-1, places=4)

    def test_10_source_vertical_overlap_recovers_known_translation(self):
        r, out = self.process(self.cfg())
        m = r['containers'][0]['matching']
        H = np.array(m['homography_second_view_to_first_view'])
        self.assertAlmostEqual(H[1, 2], 250, delta=.15)
        self.assertAlmostEqual(H[0, 2], 0, delta=.15)
        self.assertLess(m['median_reprojection_error_px'], .15)
        self.assertEqual(r['direction'], 'vertical')
        self.assertTrue((out/'containers/roof/second_weight_16bit.png').is_file())

    def test_11_rectified_vertical_overlap_recovers_known_translation(self):
        r, _ = self.process(self.cfg('rectified'))
        H = r['containers'][0]['matching']['homography_second_view_to_first_view']
        self.assertAlmostEqual(H[1][2], 250, delta=.15)

    def test_12_vertical_edge_join_is_not_reported_as_overlap(self):
        r, out = self.process(self.cfg(method='edge'))
        g = r['containers'][0]
        self.assertEqual(g['join_y'], 450)
        self.assertEqual(r['output_size_wh'], [220, 900])
        self.assertFalse(g['overlap_alignment_applied'])
        self.assertEqual(g['status'], 'manual_edge_join_unverified')
        self.assertTrue(np.array_equal(cv2.imread(str(out/'geometry_only.png'), -1), cv2.imread(str(out/'result.png'), -1)))

    def test_13_vertical_exposure_and_no_balance_override(self):
        c = self.cfg(method='edge')
        c['containers'][0]['exposure'] = {'enabled': True}
        r, _ = self.process(c)
        self.assertTrue(r['containers'][0]['exposure']['enabled'])
        r, out = self.process(c, no_balance=True)
        self.assertFalse(r['containers'][0]['exposure']['enabled'])
        self.assertTrue(np.array_equal(cv2.imread(str(out/'geometry_only.png'), -1), cv2.imread(str(out/'result.png'), -1)))

    def test_14_vertical_rectify_respects_width(self):
        r, _ = self.process(self.cfg(method='rectify'))
        self.assertEqual(r['output_size_wh'], [220, 450])

    def test_15_layout_is_independent_of_stitch_direction(self):
        c = self.cfg(method='rectify'); c['mode'] = 'combo'; c['layout'] = 'horizontal'
        second = copy.deepcopy(c['containers'][0]); second['key'] = 'second'
        second['regions'][0]['source'] = 'bottom'
        c['containers'].append(second)
        r, _ = self.process(c)
        self.assertEqual(r['direction'], 'vertical')
        self.assertEqual(r['layout'], 'horizontal')
        self.assertEqual(r['output_size_wh'], [456, 450])
        self.assertEqual(r['containers'][1]['output_box_xyxy_exclusive'], [236, 0, 456, 450])

    def test_16_reversed_vertical_order_rejected(self):
        c = self.cfg(); c['containers'][0]['regions'].reverse()
        self.reject(c, 'extend downward')

    def test_17_bad_direction_and_conflicting_cli_rejected(self):
        c = self.cfg(); c['direction'] = 'auto'; self.reject(c, 'direction')
        self.reject(self.cfg(), 'conflicts', direction='horizontal')

    def test_18_schema_two_requires_direction(self):
        c = self.cfg(); del c['direction']; self.reject(c, 'explicit direction')

    def test_19_height_alias_not_ambiguous_in_vertical(self):
        c = self.cfg(); c['height'] = 220; self.reject(c, 'legacy alias')

    def test_20_cross_size_must_match_explicit_rectification(self):
        c = self.cfg(); c['containers'][0]['regions'][0]['rectified_size_wh'] = [219, 450]
        self.reject(c, 'cross-axis')

    def test_21_empty_feature_interior_rejected(self):
        c = self.cfg('rectified'); c['containers'][0]['matching']['border_cross_px'] = 200
        self.reject(c, 'too little interior')

    def test_22_source_matching_does_not_ignore_rectified_options(self):
        c = self.cfg(); c['containers'][0]['matching']['feature_channel'] = 'green'
        self.reject(c, 'require matching.space')

    def test_23_unsupported_seam_positions_rejected(self):
        c = self.cfg(); c['containers'][0]['seam'] = {'policy': 'source_selected', 'position_px': 10}
        self.reject(c, 'overlap')
        c['containers'][0]['seam']['position_px'] = 10000
        self.reject(c, 'outside')

    def test_24_position_requires_source_selected_policy(self):
        c = self.cfg(); c['containers'][0]['seam'] = {'position_px': 320}
        self.reject(c, 'requires source_selected')

    def test_25_seam_not_allowed_on_rectify(self):
        c = self.cfg(method='rectify'); c['containers'][0]['seam'] = {'policy': 'source_selected'}
        self.reject(c, 'only valid for method overlap')

    def test_26_hard_seam_has_no_mixed_pixels(self):
        c = self.cfg(); g = c['containers'][0]
        g['matching']['feather_px'] = 0; g['seam'] = {'policy': 'source_selected', 'position_px': 350}
        _, out = self.process(c)
        sm = cv2.imread(str(out/'source_map_16bit.png'), -1)
        self.assertTrue(set(np.unique(sm)).issubset({0, 1, 2}))

    def test_27_rectified_horizontal_is_supported_too(self):
        # Rotate exact test crops, not the user images. Stitch axis becomes horizontal.
        c = self.cfg('rectified'); c['direction'] = c['layout'] = 'horizontal'
        for name, info in c['sources'].items():
            im = cv2.imread(info['path'])
            path = self.base/f'{name}.png'; cs.save_image(path, cv2.transpose(im))
            info['path'] = str(path); info['expected_size_wh'] = [450, 220]
        for reg in c['containers'][0]['regions']:
            reg['quad'] = [[0, 0], [449, 0], [449, 219], [0, 219]]
        r, _ = self.process(c)
        self.assertAlmostEqual(r['containers'][0]['matching']['homography_second_view_to_first_view'][0][2], 250, delta=.15)

    def test_28_batch_success_and_rejection_are_both_reported(self):
        c1, _ = self.write(self.cfg(method='rectify'))
        bad = self.cfg()
        for info in bad['sources'].values():
            info['path'] = str(self.fixtures/'blank.png')
        c2, _ = self.write(bad)
        manifest = self.base/'batch.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'jobs': [{'key': 'good', 'config': str(c1)},
                                                                     {'key': 'bad', 'config': str(c2)}]}))
        out = self.base/'batch'
        r = cs.run_batch(manifest, out)
        self.assertEqual((r['jobs_created'], r['jobs_rejected']), (1, 1))
        self.assertTrue((out/'good/result.png').is_file())
        self.assertFalse((out/'bad/result.png').exists())
        self.assertTrue((out/'batch_report.json').is_file())
        self.assertEqual({p.name for p in (out/'bad').iterdir()}
                         - {'debug_overlay.png', 'diagnostics.json', 'containers'},
                         {'report.json'})

    def test_29_duplicate_batch_key_rejected_before_output_claim(self):
        cfg, _ = self.write(self.cfg())
        manifest = self.base/'batch.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'jobs': [{'key': 'a', 'config': str(cfg)}]*2}))
        with self.assertRaisesRegex(cs.ProcessingError, 'unique'):
            cs.run_batch(manifest, self.base/'out')
        self.assertFalse((self.base/'out').exists())

    def test_30_cli_batch_rejects_per_job_overrides(self):
        args = [sys.executable, str(ROOT/'container_stitch.py'), '--batch', str(ROOT/'configs/all_examples.json'),
                '--out', str(self.base/'out'), '--direction', 'vertical']
        proc = subprocess.run(args, text=True, capture_output=True, timeout=20)
        self.assertEqual(proc.returncode, 2)
        self.assertIn('per-job overrides', json.loads(proc.stderr)['reason'])
        self.assertFalse((self.base/'out').exists())

    def test_31_resolved_config_can_be_rerun(self):
        _, out1 = self.process(self.cfg(method='rectify'))
        out2 = self.base/'rerun'
        cs.run_job(out1/'resolved_config.json', out2)
        self.assertEqual((out1/'result.png').read_bytes(), (out2/'result.png').read_bytes())

    def test_32_invalid_coverage_and_layout_rejected(self):
        c = self.cfg(); c['layout'] = 'automatic'; self.reject(c, 'layout')
        c = self.cfg(); c['containers'][0]['coverage'] = 'perfect'; self.reject(c, 'coverage')


if __name__ == '__main__':
    unittest.main(verbosity=2)
