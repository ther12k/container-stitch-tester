# Unified container stitching — v2.0.0

One self-contained **Python/OpenCV program** for **horizontal or vertical**, **single or combo** container images. It incorporates the previously separate roof workflow and preserves the earlier horizontal recipes. It does not import or require the old programs or their ZIP files.

The included package contains **five original input images and six freshly processed outputs**: the blue roof input has both an upper-only result and a two-container result. All image content comes from the supplied photographs. There is no generative model, inpainting, OCR, replacement lettering, or invented container end.

**Manual configuration remains necessary.** This release does not detect containers, count them automatically, verify identity from pixels, calibrate cameras, or establish physical dimensions.

## Install and regenerate everything

Use Python 3.11 or newer. Open a terminal in the extracted `unified_container_stitch_v2` folder. A clean virtual environment is recommended; install only one OpenCV distribution.

```bash
python -m pip install -r requirements.txt
python container_stitch.py --batch configs/all_examples.json --out runs/all
```

The manifest runs all six field recipes. Each result is at `runs/all/<recipe>/result.png`. The final `runs/all/batch_report.json` records every job's status, dimensions and output paths. A rejected fit is **not** converted to an edge join. Other batch jobs continue, and a batch with any rejections returns exit code 2.

**Output directories must not already exist.** This prevents accidental overwrites and confusion between stale and current results. Use a new job directory for a rerun.

## Individual runs

```bash
# Horizontal, one physical container, unverified edge join
python container_stitch.py --mode single --direction horizontal --config configs/single_grey.json --out runs/grey

# Horizontal, two physical containers, independently rectified
python container_stitch.py --mode combo --config configs/combo_1.json --out runs/combo_1
python container_stitch.py --mode combo --config configs/combo_2.json --out runs/combo_2

# Vertical, upper central blue container only
python container_stitch.py --mode single --direction vertical --config configs/vertical_blue_single.json --out runs/blue_upper

# Vertical, upper roof plus the separate, partially visible lower roof
python container_stitch.py --mode combo --direction vertical --config configs/vertical_blue_combo.json --out runs/blue_combo

# Vertical, red roof, with its unseen far end still missing
python container_stitch.py --mode single --direction vertical --config configs/vertical_red_single.json --out runs/red
```

`--mode` and `--direction` are optional **assertions**, not overrides: they must agree with the recipe. Selecting a direction does not silently rotate the source or reinterpret corner coordinates. Do not pass per-job mode/direction/source overrides with `--batch`.

## Results from this release

| Recipe | Output pixels, width × height | What it represents |
|---|---:|---|
| `single_grey` | 702 × 256 | Two selected sections of one side joined edge-to-edge; overlap unverified |
| `combo_1` | 1467 × 320 | Two different containers, kept separate |
| `combo_2` | 1378 × 320 | The second horizontal sample, retaining reversed container order |
| `vertical_blue_single` | 360 × 756 | Upper central blue roof estimated from two views |
| `vertical_blue_combo` | 360 × 1411 | Same upper roof, then a separate **partial** lower roof |
| `vertical_red_single` | 364 × 1298 | Stitched visible red roof; far end **outside the original frame** |

The three horizontal results are **pixel-for-pixel identical to v1**. The blue roof registration reproduces **27 inliers / 49 candidates**, with approximately **0.8242 rectified-pixel median fitting residual**. Its manually configured seam is at row 696, with an 8-pixel transition.

The red roof uses the earlier manual corners with a reviewed rectified cross-axis match gate. It has **10 inliers / 16 candidates**, approximately **0.8172 rectified-pixel median fitting residual**, an automatically selected seam at row 552 and a 24-pixel transition. The recipe explicitly permits a minimum of 8 inliers; the report flags its limited feature support. This is not enough field validation for unattended inspection.

These residuals measure fit to accepted feature points, **not accuracy of all corrugations, scratches, edges, or physical dimensions**. Shadows, glare, seam differences and viewpoint-dependent shading remain. The updated roof renderer masks unselected pixels before final interpolation, so roof outputs need not be pixel-identical to the old standalone roof renderer.

## Three independent choices

**Mode** describes how many physical containers are selected: `single` requires one group; `combo` requires two or more. It does not count image halves or cameras. The blue roof combo selects the two central containers; the unrelated left-lane container is excluded.

**Direction** describes the order of views within each group: `horizontal` = first/left then second/right; `vertical` = first/top then second/bottom. It also determines the cross-axis presentation size.

**Layout** describes where independently processed containers are placed in the final image. It defaults to the direction, but is independent: vertical roofs can also be displayed side-by-side. Groups are top/left aligned, without post-stitch scaling. The transparent gutter is a presentation separator, **not a measured physical gap**.

Each group uses one method:

| Method | Selected regions | Behavior |
|---|---:|---|
| `rectify` | 1 | Perspective-correct the visible panel; no new coverage |
| `edge` | 2 | Rectify and place sections adjacent; optional exposure balancing; **no overlap established or removed** |
| `overlap` | 2 | Estimate and validate correspondence, warp original pixels, then blend accepted overlap |

All three methods work in both directions. A combo can mix methods, as demonstrated by the blue roof combo (`overlap` + `rectify`). Feature matching and blending never cross container groups.

## Configuration and compatibility

See [CONFIGURATION.md](CONFIGURATION.md). The supplied recipes are locked to the original files using dimensions and SHA-256. Replacing a file with another frame—even one of the same dimensions—does not make its old corners valid.

Schema-1 horizontal jobs remain supported, including their `height` setting. New recipes use schema 2, explicit `direction`, and `cross_size_px` (height for horizontal, width for vertical). Earlier `left_quad/right_quad` and horizontal `panels[]` configurations are still adapted with `--input`. The standalone roof `view_a/view_b` settings have been migrated into the supplied schema-2 recipes; their old CLI flags are **not** accepted by this program.

Single files containing stacked/side-by-side views use separate `view_box` selections. Two separate camera files use two named sources. Image orientation is the encoded pixel orientation; EXIF auto-rotation is not applied.

Paths inside recipes are relative to the recipe, not your terminal directory. Override the sole source with `--input FILE`, or repeat `--source NAME=FILE` for named sources. Overrides still undergo dimensions/hash checks. Create a reviewed recipe for new images rather than bypassing checks on an old recipe.

## Outputs and provenance

Every successful job produces:

```text
result.png                       final BGRA image
geometry_only.png                no exposure adjustment; warping and overlap blending remain
source_map_16bit.png              per-pixel source-region bitmask
container_map.png                per-pixel physical-container group index
selected_regions/<source>.jpg    original with configured corners annotated
resolved_config.json             normalized configuration used by the run
report.json                      completion marker, methods, transforms, hashes and warnings
containers/<key>/                individual result, report and diagnostics
```

An overlap group also saves `feature_matches.jpg`, `hard_seam.png`, `warped_first.png`, `warped_second.png`, and `second_weight_16bit.png`. A rectified matching group saves its two feature-working images. The final output is warped directly from the originals, **not from those intermediate working images**. Horizontal overlap jobs retain `right_weight_16bit.png` as a compatibility alias.

Read 16-bit maps with `cv2.IMREAD_UNCHANGED`. Source values are bitmasks: 1, 2, 4, etc., one bit per selected region. A pixel blending regions 1 and 2 has value 3; 0 means transparent. The job report maps each bit to the source, view and physical container. These maps identify pixel samples; exposure gains can additionally depend on statistics from both sections within that group.

Divide `second_weight_16bit.png` by 65535 to obtain the second source's blend weight, **only where output alpha is nonzero**. `source_selected` seams deliberately omit unsupported side strips after the seam instead of retaining doubled edges from an earlier view. Missing support is transparent. No gap is filled with new image content.

## Python integration

```python
from pathlib import Path
from container_stitch import ProcessingError, run_job, run_batch

try:
    report = run_job(
        config_path="configs/vertical_blue_combo.json",
        out="runs/job_2141",
        mode="combo",
        direction="vertical",
    )
    result = Path("runs/job_2141") / report["result_file"]
except ProcessingError as exc:
    print(f"Rejected: {exc}")  # Review rather than publishing stale imagery.
```

`run_job` publishes only after every selected container succeeds. A failed job leaves a rejection report, not a partial final image. A successful report has `status: created_requires_review`; this is a processing status, not certification of physical accuracy. The report is committed last. A batch is not an all-or-nothing transaction across jobs; its summary is committed after every job finishes.

Use separate processes for parallel jobs because OpenCV RNG/thread settings are process-global. Local paths/configurations are trusted input, not a security sandbox. A web service needs access control, path isolation and upload/resource limits. A crash or full disk still needs job-level cleanup and retry handling.

## Validation

```bash
python -m unittest discover -s tests -v
```

The included log records **80 passing tests**: all 48 earlier tests plus 32 new vertical/batch tests. Coverage includes all three horizontal regressions, both uploaded roof frames, the blue partial combo, known horizontal/vertical translations in both feature domains, source/alpha/weight consistency, direction and layout independence, exposure override, invalid configurations, unsupported seams, reversed order, rejected-fit publication and batch success/failure reporting.

Controlled textured fixtures and overlapping crops exercise code behavior. They do not establish reliability across independent cameras, varying lighting, motion, occlusion or repeated markings. Keep the original captures and review the diagnostics before relying on a composite.

## OpenCV references

- [Geometric transformations](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)
- [Feature matching and homography](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html)

These explain the underlying operations, not the physical correctness of these particular stitches.

## Simple HTMX web tester

A minimal Flask + HTMX frontend is included in `app.py`. It is intended for local testing of reviewed configurations, not for an untrusted public deployment.

### Run the web UI

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000/`.

### What the UI supports

- Run any packaged example recipe.
- Run the packaged batch manifest.
- Upload a custom config JSON and the image files it references.
- Inspect the result PNG and the emitted `report.json` / `batch_report.json`.
- Browse and delete recent job outputs.

### Notes

- The HTMX script tag currently uses the public CDN URL `https://unpkg.com/htmx.org@1.9.12`.
- Uploaded custom files are staged under `web_uploads/` and outputs are written to `web_jobs/`.
- The app does **not** add automatic corner selection, OCR, calibration, identity verification, or upload isolation beyond basic path handling.
- For custom jobs, keep the uploaded filenames matching the paths expected by the JSON recipe.
