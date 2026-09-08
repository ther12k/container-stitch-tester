# Container Stitch Tester

A local web tester for a deterministic, non-generative container-photo stitching engine.
**AI proposes — OpenCV proves.** A vision model may plan *how* to interpret a photo
(one container or several, stacked or side-by-side views, which surface to use), but a
Python/OpenCV pipeline alone validates every claim and performs the actual stitch.
No generative model ever touches image pixels.

```
image ──► AI vision planner (optional)  ──► ai_plan (normalized JSON)
                                               │
                                               ▼
                                    config normalizer + validator
                                               │
                                      pass ◄──┴──► rejected
                                        │              │
                                        ▼              ▼
                                 OpenCV stitch     diagnostics back
                                 (only pixel          to the model
                                 manipulation)        (capped retries)
                                        │
                                        ▼
                               result.png + report.json
```

## Highlights

- **Unified engine** (`container_stitch.py`): horizontal/vertical, single/combo,
  three explicit methods — `rectify` (one complete view), `edge` (adjacent views,
  overlap *not* verified), `overlap` (SIFT + RANSAC-verified shared surface).
- **Direction safety** (v2.1): view-box positions are checked against the configured
  direction; obvious contradictions (e.g. stacked views marked `horizontal`) are
  rejected instead of producing misleading composites. `direction: "auto"` resolves
  only unambiguous same-source layouts.
- **Honest reporting**: processing success is separated from stitch quality —
  reports carry `rectified_only`, `unverified_edge_composite`,
  `overlap_requires_visual_review` or `rejected`; the UI colors them accordingly.
- **AI planning (optional, off by default)**: plug in any OpenAI-compatible vision
  endpoint (e.g. GLM, GPT) as planner and an optional second model as reviewer.
  The planner returns an intermediate `ai_plan` with *normalized* coordinates and
  confidence scores; the backend converts it to an engine config. Rejections are
  sent back with diagnostics for a capped number of retries. If the reviewer is
  unreachable the planner takes over automatically without consuming an attempt.
  A provider that fails twice in a row (timeout or 5xx) is skipped for the rest
  of the run, raw provider error pages are sanitized out of the UI (full
  responses stay in a server-side `ai_run_log.json`), and confidence numbers are
  labeled as the *planner's* claim — the engine's verdict is separate.
  **The engine's validation is final — the AI cannot override a rejection.**
- **Measured seam refinement (v2.3, `edge_alignment: "measure"`)**: declared quad corners are
  estimates, so an `edge` join can duplicate a band of content or leave a vertical step at the seam.
  With `measure`, the engine cross-matches the two warped strips (SIFT + RANSAC, deterministic):
  when the strips demonstrably share coverage the pair is first *promoted to the verified overlap
  method* (same fixed proof standards), and otherwise a clamped translation correction trims the
  duplicate band and aligns the seam. Every applied correction is disclosed in `seam_alignment`
  (report + `diagnostics.json`), strong evidence upgrades the quality state, and the AI planner is
  never involved in the measurement.
- **Fixed-camera profiles**: calibrate corners once per camera mount
  (`profiles/*.json`); every capture from that camera then generates a validated
  config. Frame dimensions must match the calibration exactly, and an optional
  pinned reference hash refuses a different physical frame.
- **metrics.json** on every run (success and rejection): a descriptive
  per-container scorecard — inliers, ratios, reprojection error, sanity checks —
  mirroring engine decisions without adding new quality logic.
- **Debug overlay** (`debug_overlay.png` + `diagnostics.json` per job): one sheet
  answers "why did this stitch pass or fail?" — configured corners, feature matches
  (inliers vs rejected), projected quads, overlap polygon, seam + feather band,
  compact metrics, and for rejections the exact reason printed on the image.
  Deterministic engine artifact; it only draws decisions already made and never
  re-evaluates quality.
- **Live console**: every run streams its stages (staging, AI calls, engine
  verdicts, retries, fallbacks) to an in-page console via SSE.
- **Paste-to-stitch**: paste an image anywhere; a default two-half config is
  generated automatically with orientation detected from the aspect ratio.

## Setup

Python 3.11+ (pinned versions in `requirements.txt`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8000/

This is a local development tool: Flask's development server with debug enabled,
bound to localhost. Do not expose it to untrusted users. AI endpoint URLs and API
keys are entered in the UI and stored only in your browser's localStorage; they are
forwarded per-run to this local server, which calls the configured endpoint directly.

## Sample images are not included

This repository intentionally **excludes the original sample photographs** and their
derived outputs (`sources/`, `examples/`, preview collages). They were supplied for
private testing only. `samples_manifest.json` and `sha256_manifest.json` document
what the private bundle contained (filenames, dimensions, checksums) so results can
be audited against it. The packaged JSON recipes in `configs/` reference those
excluded files by relative path + SHA-256 — point them at your own copies of the
images (byte-identical) or write configs for your own photos.

Most tests run without the samples: `python -m unittest discover -s tests`
executes the synthetic-fixture suite (48 tests) and automatically **skips** the 32
recipes that require the private bundle — 80 tests total in a private checkout.

## Method limits (deliberate)

`rectify` straightens one selected region; `edge` joins explicitly selected sections
without proving overlap; `overlap` estimates and checks an operator-confirmed shared
surface. Failed overlap is rejected, never silently replaced by an edge join.
Corner selection and container grouping are operator/AI-proposed inputs, not
detections. No OCR, no inpainting, no learned super-resolution, no reconstruction of
occluded or out-of-frame surfaces. Warping resamples pixels — keep the original
photographs as the authoritative record.

## AI planning rules

1. The model returns a strict intermediate JSON plan (never an engine config):
   scene layout, container grouping, per-region view boxes/quads in **normalized**
   coordinates, preferred method, confidence scores, exclusions, reason.
2. The backend clamps/converts coordinates, injects image dimensions, and emits a
   normal engine config.
3. The engine validates everything again (corner geometry, direction consistency,
   SIFT/RANSAC overlap proof, homography sanity, scale/rotation bounds).
4. Rejections feed back to the model as diagnostics; attempts are capped (2–4).
5. `confidence.same_surface < threshold` plans should not request `overlap`; if they
   do and the geometry disagrees, the engine wins.

## Repository layout

```
app.py                  Flask + HTMX web tester (runs, live console, AI planning)
ai_planner.py           AI vision-planner bridge (plan parse/validate/normalize)
container_stitch.py     The deterministic stitching engine (no network, no AI)
templates/, static/     Tailwind-based UI
configs/                Packaged JSON recipes (reference excluded sample paths)
tests/                  80 unit/regression tests + synthetic fixtures
validation/             Packaging and run-validation logs
```

## Keeping this repository clean

The original photographs and any private endpoint credentials live only in a
separate private workspace. Publishing is guarded at three layers:

1. **One-command sync** — a private-side script copies the workspace in with the
   sample bundle, previews and AI credentials excluded, blanks the seeded API
   defaults, and runs the test suite before you review the diff.
2. **Local pre-push hook** — `.githooks/pre-push` rejects any push whose commits
   add banned paths (`sources/`, `examples/`, previews, runtime dirs), secret-shaped
   strings (`sk-…`), or patterns listed in the untracked
   `.githooks/local-banned.txt` (machine-local regexes, e.g. a private endpoint).
   Enable once per clone: `git config core.hooksPath .githooks`.
3. **CI** — `.github/workflows/security.yml` runs Gitleaks over the full history
   plus an explicit check that no private-asset paths or `sk-…` keys are tracked.

## Status

Development preview. The engine is conservative by design: it refuses to guess
container identity, refuses physically implausible fits, and labels every output
with what was and was not verified. See `CHANGELOG.md`, `ENGINE_README.md` and
`CONFIGURATION.md` for details.
