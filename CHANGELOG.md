# v2.2.0-diagnostics

- Engine-level StitchDiagnostics + deterministic debug overlay (debug_overlay.png, diagnostics.json) for pass, weak-match, and rejection cases; exact rejection reason rendered on the image.
- Rejected jobs now leave diagnostics-only artifacts in addition to report.json; no composites are published.
- 85 tests (5 new diagnostic fixtures: clean pass, weak-match warning, wrong-container rejection, degenerate homography, minimal render).

# v2.1.0

- One program for horizontal/vertical, single/combo processing.
- Direction-aware rectify, edge, and overlap methods.
- Rectified-space SIFT options migrated from the supplied roof prototype.
- Source-selected seams, top/bottom ordering validation, and alpha-aware final source warping.
- Independent layout direction, cross-axis sizing, and explicit partial-coverage metadata.
- Batch manifest CLI/API with per-job failure isolation and a final batch summary.
- Six regenerated field outputs from five original images; previous three horizontal outputs unchanged pixel-for-pixel.
- 80 passing tests (48 retained + 32 added).

No automatic detection, OCR, physical identity proof, camera calibration, or reconstruction of missing content has been added. The grey join remains unverified; the red and lower blue roof coverage remain partial.
