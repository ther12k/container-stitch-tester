# All original sample images

**9 unique originals; 5 configured originals with 6 existing outputs; 4 newer raw-only inputs.**

| Sample | Original | Size (W × H) | Readiness |
|---|---|---|---|
| Blue SINOKOR horizontal combo — first view | [combo_1.png](sources/combo_1.png) | 2048 × 569 | [combo_1](configs/combo_1.json) |
| Blue SINOKOR horizontal combo — second view | [combo_2.png](sources/combo_2.png) | 2048 × 448 | [combo_2](configs/combo_2.json) |
| Grey horizontal single container | [single_grey.png](sources/single_grey.png) | 2048 × 486 | [single_grey](configs/single_grey.json) |
| Blue vertical roof view — central combo | [roof_blue.png](sources/roof_blue.png) | 1820 × 2048 | [vertical_blue_single](configs/vertical_blue_single.json), [vertical_blue_combo](configs/vertical_blue_combo.json) |
| Red vertical single-container roof | [roof_red.png](sources/roof_red.png) | 1820 × 2048 | [vertical_red_single](configs/vertical_red_single.json) |
| Blue HEUNG-A side view — cropped/misaligned halves | [side_blue_heung_partial.png](sources/side_blue_heung_partial.png) | 2048 × 576 | **Raw input only — configuration pending** |
| Blue top-down view with vehicle 177 — roof/doors/equipment | [roof_blue_vehicle_177.png](sources/roof_blue_vehicle_177.png) | 1820 × 2048 | **Raw input only — configuration pending** |
| Green HEUNG-A horizontal side view | [side_green_heung.png](sources/side_green_heung.png) | 2048 × 576 | **Raw input only — configuration pending** |
| Blue top-down view with spreader occluding the roof | [roof_blue_spreader.png](sources/roof_blue_spreader.png) | 1820 × 2048 | **Raw input only — configuration pending** |

The four raw-only inputs are intentionally excluded from `configs/all_examples.json`. Do not reuse older corner coordinates without checking the new frame. The archive contains the input photographs, not invented roofs or hidden panels.

Open `sample_gallery.html` to browse all originals. `samples_manifest.json` records original upload filenames, dimensions and checksums. Existing sample outputs are available under `examples/`.

### `vertical_lightblue_stacked.png`
- New regression sample from the failed UI case.
- The source contains two vertically stacked camera views and an overlaid yellow guide line.
- The lower view already contains the target light-blue roof sufficiently, so `vertical_lightblue_best_view.json` uses `rectify` instead of forcing a false stitch.
- The separate dark-blue container below is excluded.
