# Configuration — v2.1.0

Use a supplied recipe as the working example. The single-container and combo paths share one implementation; groups are never matched against each other.

## Job settings

| Field | Meaning |
|---|---|
| `schema_version` | `2` for new jobs; `1` remains supported for older horizontal jobs |
| `mode` | `single` (exactly one group) or `combo` (two to eight groups) |
| `direction` | Required in schema 2: `horizontal`, `vertical`, or conservative `auto` for obvious same-source stacked/side-by-side view boxes |
| `cross_size_px` | Target height for horizontal, width for vertical; integer 16–4096, default 320 |
| `layout` | `horizontal` or `vertical`, defaulting to `direction`; independent of within-group alignment |
| `gap_px` | Transparent separation between groups, integer 1–256; default 12; not physical distance |
| `edge_alignment` | Seam handling for `edge` joins: `butt` (default — declared corners joined as-is) or `measure` (feature-measured refinement, see below) |
| `sources` | One to eight named local input images |
| `containers` | Explicitly ordered physical-container groups |
| `notes` | Optional human-readable selection/provenance note |

The schema-1 `height` field is a horizontal alias for `cross_size_px`. Do not combine both fields, or use `height` with a vertical direction. Overlap output bounds can be larger than the target cross size because the second view is transformed rather than stretched to fit.

### `edge_alignment: "measure"` — measured seam refinement

Declared quad corners are estimates; an `edge` join normally butts the warped
strips exactly as declared, so slightly-wrong corners duplicate a band of
content or leave a vertical step at the seam. With `"edge_alignment": "measure"`
the engine cross-matches the two warped strips (SIFT + RANSAC similarity,
deterministic) and:

1. **Promotes to verified overlap** when the strips demonstrably share coverage
   (measured overlap ≥ ~8% of the strip) — the pair is re-run through the
   normal `overlap` method with its *unchanged* proof standards (≥ 20 inliers,
   inlier ratio ≥ 0.35, sanity clamps), just more sensitive feature detection.
   A passing proof yields the violet `overlap_requires_visual_review` result.
2. Otherwise applies a **translation-only correction** to the edge join: trims
   the measured duplicate band and shifts the cross-axis offset, both clamped
   (overlap ≤ 45% of the second strip, offset ≤ 14% of the cross size, scale
   0.85–1.15, rotation ≤ 3°). With ≥ 20 inliers spanning the strip the result
   is upgraded to `edge_join_feature_aligned`; weaker support keeps the amber
   `unverified_edge_composite` state with the correction disclosed in
   `seam_alignment` (report + `diagnostics.json`).

The correction is measured from pixels only — the AI planner is never involved,
and repeated corrugations can still alias a measurement by one period, so the
seam remains subject to visual review.

A minimal configuration section for a vertical combo is:

```json
{
  "schema_version": 2,
  "mode": "combo",
  "direction": "vertical",
  "cross_size_px": 341,
  "layout": "vertical",
  "gap_px": 16
}
```

This section is not a complete recipe: it also needs `sources` and `containers`, as in `configs/vertical_blue_combo.json`.


### Direction guard and `auto`

Version 2.1 adds a conservative layout guard for two-region groups that use two disjoint `view_box` regions from the same source image. If the boxes are clearly stacked top/bottom while the recipe says `horizontal` (or clearly side-by-side while it says `vertical`), the job is rejected before producing a misleading composite.

`"direction": "auto"` is accepted only when that same-source view-box geometry is unambiguous. It does **not** inspect pixels, infer camera motion, or decide whether two regions truly depict the same physical container. For separate source files, overlapping view boxes, or ambiguous layouts, set the direction explicitly.

This guard catches the common failure where a vertically stacked v1/v2 image is accidentally processed as a horizontal edge join. It does not select the stitch method.

## Sources and coordinate conventions

```json
"sources": {
  "main": {
    "path": "../sources/roof_blue.png",
    "expected_size_wh": [1820, 2048],
    "sha256": "abcbfa387cc4699873b249b7bf0e941307859b29c73df1de09fbeaff9284ccdb"
  }
}
```

`path` is relative to the recipe. `expected_size_wh` is required. `sha256` is optional for new jobs, but omitting it produces a warning: dimensions do not validate identity or corner placement. Supplied examples are hash-locked. Sources must be opaque 8-bit images; grayscale inputs are converted to BGR, transparent inputs are rejected.

A region uses:

```json
{
  "source": "main",
  "view_box": [0, 1024, 1820, 2048],
  "quad": [[831, 0], [1068, 0], [1109, 426], [839, 433]],
  "rectified_size_wh": [341, 580]
}
```

`view_box` is `[x0,y0,x1,y1]` in the full source image, with **exclusive** right/bottom bounds. Omit it to use the whole file. `quad` is four `[x,y]` points **local to that view box**, ordered top-left, top-right, bottom-right, bottom-left. They must be finite, convex, correctly ordered and within the view's pixel bounds. The example above selects a roof portion in the bottom camera view, not the second physical container.

Do not substitute preview/display coordinates for native pixel coordinates. The two supplied roof files are **1820 × 2048**; their split is **y = 1024**. A scaled image shown in chat is not the working coordinate system.

`feature_quad`, when present, is an inner quadrilateral used for feature detection. It must lie inside `quad`. Rendering remains restricted to `quad`.

`rectified_size_wh` optionally specifies a working rectangle. Its cross-axis dimension must equal the job's `cross_size_px`. It is useful for roof feature matching because the two views cover different lengths. These sizes are manual presentation scales, **not physical container dimensions**. Without them, rectangle length is derived from the selected edges' apparent aspect ratio.

For two original files, define `top` and `bottom` sources and reference them in the two regions instead of using two view boxes. Region order is left/right for horizontal, top/bottom for vertical. It is not inferred from filenames.

## Physical-container group

Required: a unique safe `key`, `method`, and `regions`. Optional: `label`, `container_id`, `coverage`, `notes`, method-specific settings below.

`coverage` is one of `visible_panel`, `partial`, or `unverified` (default). This is **operator metadata**, not automatic verification. `visible_panel` describes the selected face, not a full 3D container or complete inspection. Mark cropped ends as `partial`.

`container_id` may also appear on individual regions. Supplied IDs are normalized and checked for conflicts within a group and duplication across physical groups. IDs are **not** read from pixels or independently verified. Never rely on shared branding as evidence of a shared physical surface.

For `edge` or `overlap`, explicitly set `same_surface_confirmed: true`. This records the operator's grouping intention. It does not prove that overlap exists or that a computed homography is correct.

### `rectify`

Exactly one region. Straightens that selected visible panel. No feature matching or exposure correction. The lower blue roof in the combo uses this method and is marked `partial`.

### `edge`

Exactly two regions. Rectifies each and places them adjacent along `direction`, without inferring, deleting or blending overlap. The report remains `manual_edge_join_unverified`. Feature diagnostics are informational only.

Optional `exposure`:

```json
{
  "enabled": true,
  "sample_width": 64,
  "exclude_seam_px": 6,
  "sample_y_fraction": [0.12, 0.87],
  "fade_px": 160,
  "max_gain": 1.6
}
```

These legacy parameter names are interpreted in an axis-normalized coordinate system. `sample_width` measures distance along the join direction, and `sample_y_fraction` selects the cross-axis band: y for horizontal sides, x for vertical roofs. `fade_px` is the distance into each section over which the gain tapers. The paint beside the seam must be comparable; this does not establish correspondence. `--no-balance` disables all configured exposure correction. Exposure correction is supported only for `edge`, never across different groups.

### `overlap`

Exactly two regions. Requires a feature fit, spatially distributed inliers, valid projected boundaries, a non-reflected/non-collapsed transformation, sensible relative scale, extension in the configured direction, and sufficient pixel overlap. No automatic fallback to `edge`.

Common `matching` fields:

| Field | Default / meaning |
|---|---|
| `space` | `source` (match in selected original views) or `rectified` (match in working rectangles) |
| `ratio` | 0.72 in source space, 0.8 rectified; permitted 0.5–0.9 |
| `ransac_px` | 3; in **first matching image** pixel units, not physical units |
| `min_inliers` | 20; minimum allowed configuration 8; low-support accepted fits get a warning |
| `contrast_threshold` | 0.04 source, 0.008 rectified |
| `feather_px` | 32; 0 gives a hard seam; up to 512 |

Both feature paths use symmetric matching and coordinate deduplication. Acceptance also requires at least a 0.35 inlier ratio. A high fit count or low residual is not proof against repeated-rib or repeated-logo false matches.

Rectified-only options (rejected in source space rather than silently ignored):

| Field | Default / meaning |
|---|---|
| `feature_channel` | `gray` or `green`, default `gray`; changes feature detection, not output colors |
| `edge_threshold` | 18, SIFT edge threshold |
| `border_cross_px` | 12-pixel excluded border across the matching rectangle |
| `border_axis_px` | 10-pixel excluded border along the rectangle |
| `max_cross_displacement_px` | Optional gate on matching points' cross-axis displacement, based on manually comparable working widths/heights |

The final image is warped from original pixels using composed transformations; it is not repeatedly resampled from the feature-working images.

Optional `seam`:

```json
"seam": {"policy": "source_selected", "position_px": 696}
```

`union` is the default: select the overlap midpoint on each crossline and preserve all unique support. `source_selected` uses the first region before the seam and the second after it. It omits stale first-view side strips beyond the seam; missing support remains transparent.

`position_px` requires `source_selected`, is measured in the individual container output canvas (x for horizontal, y for vertical), and must have broad overlap support. Omit it to select a low pixel-difference seam in a supported band. There is no claim that this is an optimal seam or that it avoids every defect/marking. `matching.feather_px` sets the transition width. `hard_seam.png` uses the same position without blending.

The blue recipe retains its reviewed row-696 seam. The red recipe selects a seam automatically. Position depends on output geometry; it is not portable to a changed configuration/frame without review.

## Batch manifest

```json
{
  "schema_version": 1,
  "jobs": [
    {"key": "grey", "config": "single_grey.json"},
    {"key": "roofs", "config": "vertical_blue_combo.json"}
  ]
}
```

Paths are relative to the manifest. Up to 64 explicitly named jobs are allowed. Run `--batch ... --out NEW_DIR`; each recipe supplies its own direction, mode and sources. Invalid manifest structure is rejected before claiming output. Runtime job rejections are isolated and recorded. `batch_report.json` is written last and should be required by batch consumers.

## Limits and compatibility

Eight groups, sixteen regions, eight sources per job. Every group has one or two regions according to its method. Direction is job-wide; make separate jobs for mixed horizontal/vertical source surfaces. Layout remains independent.

Schema-1 unified horizontal recipes and the earlier horizontal manual formats are supported. Standalone roof configurations have been migrated into schema-2 examples rather than being accepted unchanged. Old roof CLI options such as `--seam-y` belong in the new recipe's `seam` settings. There is no automatic corner detector or camera-profile reuse guarantee.

Unknown fields, conflicting modes/directions, wrong dimensions/hashes, out-of-view corners, incompatible metadata and unsupported fits are rejected. Existing output directories are never overwritten. This is a trusted-local-input tool, not a sandbox for arbitrary uploaded configurations or file paths.
