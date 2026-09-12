#!/usr/bin/env python3
"""One non-generative horizontal/vertical processor for single and combo containers.

A job contains explicitly grouped physical containers. Each group independently
uses one of three methods: rectify (one region), edge (two unverified adjacent
sections), or overlap (two operator-confirmed overlapping views). Groups are then
laid out with a transparent separator. Features NEVER match across groups.

Example:
    python container_stitch.py --mode single --config configs/single_grey.json --out run_single
    python container_stitch.py --mode combo --config configs/combo_1.json --out run_combo
    python container_stitch.py --direction vertical --config configs/vertical_blue_combo.json --out run_roofs
    python container_stitch.py --batch configs/all_examples.json --out run_all

Requires Python >=3.11 with the pinned dependencies, NumPy and OpenCV with SIFT. No network, learned model,
OCR, inpainting, detection, automatic identity inference, or generated pixels.
Image warping resamples pixels; exposure correction and overlap blending change
values. Keep original photographs as the authoritative record.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import cv2
import numpy as np

VERSION = "2.3.2"
MAX_PIXELS = 25_000_000
MAX_DIM = 20_000
MAX_REGIONS = 16




def direction_hint_from_view_boxes(config: dict[str, Any]) -> str | None:
    """Infer only obvious side-by-side / stacked composite-view layouts.

    This is deliberately conservative: it votes only when two regions from the
    same source use disjoint view boxes on one axis and strongly overlap on the
    other. It does not infer camera motion from pixels or filenames.
    """
    votes: list[str] = []
    containers = config.get("containers")
    if not isinstance(containers, list):
        return None
    for c in containers:
        if not isinstance(c, dict) or c.get("method") == "rectify":
            continue
        regions = c.get("regions")
        if not isinstance(regions, list) or len(regions) != 2:
            continue
        a, b = regions
        if not isinstance(a, dict) or not isinstance(b, dict) or a.get("source") != b.get("source"):
            continue
        ba, bb = a.get("view_box"), b.get("view_box")
        if not (isinstance(ba, list) and isinstance(bb, list) and len(ba) == len(bb) == 4):
            continue
        try:
            ax0, ay0, ax1, ay1 = map(float, ba)
            bx0, by0, bx1, by1 = map(float, bb)
        except (TypeError, ValueError):
            continue
        aw, ah, bw, bh = ax1-ax0, ay1-ay0, bx1-bx0, by1-by0
        if min(aw, ah, bw, bh) <= 0:
            continue
        x_overlap = max(0.0, min(ax1, bx1)-max(ax0, bx0))
        y_overlap = max(0.0, min(ay1, by1)-max(ay0, by0))
        x_fraction = x_overlap / min(aw, bw)
        y_fraction = y_overlap / min(ah, bh)
        vertical_disjoint = ay1 <= by0 or by1 <= ay0
        horizontal_disjoint = ax1 <= bx0 or bx1 <= ax0
        if vertical_disjoint and x_fraction >= 0.70:
            votes.append("vertical")
        elif horizontal_disjoint and y_fraction >= 0.70:
            votes.append("horizontal")
    return votes[0] if votes and all(v == votes[0] for v in votes) else None

class ProcessingError(ValueError):
    """Invalid input or an alignment that must not be published."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ProcessingError(f"{name} must be an integer.")
    if not low <= value <= high:
        raise ProcessingError(f"{name} must be {low}..{high}.")
    return int(value)


def number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ProcessingError(f"{name} must be numeric.")
    value = float(value)
    if not np.isfinite(value) or not low <= value <= high:
        raise ProcessingError(f"{name} must be finite and {low}..{high}.")
    return value


def boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProcessingError(f"{name} must be true or false, not a string or number.")
    return value


def object_keys(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProcessingError(f"{name} must be a JSON object.")
    unknown = set(value) - allowed
    if unknown:
        raise ProcessingError(f"Unknown {name} field(s): {', '.join(sorted(unknown))}.")
    return value


def safe_key(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) is None:
        raise ProcessingError(f"{name} must be a safe 1..64 character name using letters, digits, _ or -.")
    return value


def normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProcessingError("Container IDs must be nonempty strings or null.")
    result = "".join(c for c in value.upper() if c.isalnum())
    if not result:
        raise ProcessingError("Container ID must contain letters or digits.")
    return result


def check_size(width: int, height: int) -> None:
    if min(width, height) < 2 or max(width, height) > MAX_DIM or width * height > MAX_PIXELS:
        raise ProcessingError(f"Unsafe image/canvas size: {width} x {height}.")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Commit the report last; a success report is the job completion marker.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ProcessingError(f"Cannot encode output: {path.name}")
    encoded.tofile(str(path))


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ProcessingError(f"Image not found: {path}")
    if path.stat().st_size > 256_000_000:
        raise ProcessingError("Encoded image exceeds the 256 MB input limit.")
    # UNCHANGED uses encoded pixel orientation (no automatic EXIF rotation).
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ProcessingError("Input must decode to an 8-bit grayscale, BGR or opaque BGRA image.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ProcessingError("Unsupported input channel count.")
    if image.shape[2] == 4:
        if np.any(image[..., 3] != 255):
            raise ProcessingError("Transparent source images are not supported. Use opaque originals.")
        image = image[..., :3].copy()
    h, w = image.shape[:2]
    check_size(w, h)
    if min(w, h) < 16:
        raise ProcessingError("Source must be at least 16 pixels on each side.")
    return image


def check_quad(value: Any, width: int, height: int) -> np.ndarray:
    try:
        q = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ProcessingError("quad must contain numeric coordinates.") from exc
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ProcessingError("quad must be four finite [x,y] points in TL, TR, BR, BL order.")
    if (q < 0).any() or (q[:, 0] > width - 1).any() or (q[:, 1] > height - 1).any():
        raise ProcessingError("A corner is outside its source view.")
    if not cv2.isContourConvex(q) or cv2.contourArea(q, oriented=True) < 16:
        raise ProcessingError("quad must be a convex, nondegenerate TL/TR/BR/BL quadrilateral.")
    if q[0, 0] + q[3, 0] >= q[1, 0] + q[2, 0] or q[0, 1] + q[1, 1] >= q[2, 1] + q[3, 1]:
        raise ProcessingError("quad corner order must be TL, TR, BR, BL.")
    return q


def mask_for(shape: tuple[int, int], q: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(q).astype(np.int32), 255)
    return mask


def rectification(q: np.ndarray, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    horizontal = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    vertical = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    width = int(round(float(horizontal / vertical) * (height - 1))) + 1
    check_size(width, height)
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    H = cv2.getPerspectiveTransform(q, dst)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ProcessingError("Degenerate perspective rectification.")
    return H, (width, height)



def rectification_axis(q: np.ndarray, cross_size: int, direction: str,
                       explicit_size: list[int] | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """Output dimensions are a presentation coordinate system, not physical dimensions."""
    if explicit_size is None and direction == "horizontal":
        return rectification(q, cross_size)
    if explicit_size is not None:
        width, height = explicit_size
    else:
        across = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
        along = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
        width, height = cross_size, int(round(float(along / across) * (cross_size - 1))) + 1
    check_size(width, height)
    dst = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    H = cv2.getPerspectiveTransform(q, dst)
    projected_quad(q, H)
    return H, (width, height)

def warp(image: np.ndarray, mask: np.ndarray, H: np.ndarray,
         size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Premultiplied bilinear warp; samples outside the selected face do not bleed in."""
    check_size(*size)
    m = mask.astype(np.float32) / 255.0
    coverage = cv2.warpPerspective(m, H, size, flags=cv2.INTER_LINEAR)
    pixels = cv2.warpPerspective(image.astype(np.float32) * m[..., None], H, size,
                                 flags=cv2.INTER_LINEAR)
    valid = coverage > 1e-6
    pixels[valid] /= coverage[valid, None]
    pixels[~valid] = 0
    return pixels, valid


def bgra(pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.dstack([np.clip(np.rint(pixels), 0, 255).astype(np.uint8), valid.astype(np.uint8) * 255])


@dataclass
class Region:
    source: str
    box: list[int]
    quad: np.ndarray
    feature_quad: np.ndarray
    image: np.ndarray
    bit: int
    config: dict[str, Any]

    @property
    def mask(self) -> np.ndarray:
        return mask_for(self.image.shape[:2], self.quad)

    @property
    def feature_mask(self) -> np.ndarray:
        return mask_for(self.image.shape[:2], self.feature_quad)

    def report(self, H: np.ndarray) -> dict[str, Any]:
        x0, y0, _, _ = self.box
        input_to_view = np.float64([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
        return {
            "source": self.source, "source_bit": self.bit,
            "view_box_xyxy_exclusive": self.box,
            "quad_tl_tr_br_bl_in_view": self.quad.tolist(),
            "feature_quad_in_view": self.feature_quad.tolist(),
            "view_to_container_homography": H.tolist(),
            "input_to_container_homography": (H @ input_to_view).tolist(),
            "supplied_container_id": self.config.get("container_id"),
            "notes": self.config.get("notes", "Visible region only; hidden edges not reconstructed."),
        }


@dataclass
class Group:
    config: dict[str, Any]
    regions: list[Region]


@dataclass
class Tile:
    image: np.ndarray
    geometry: np.ndarray
    sources: np.ndarray
    report: dict[str, Any]
    diagnostics: "StitchDiagnostics | None" = None


def legacy_to_job(config: dict[str, Any], source_path: str) -> dict[str, Any]:
    """Adapt the two earlier manual configuration formats without old modules."""
    size = config.get("expected_size_wh")
    common = {
        "schema_version": 1, "height": config.get("height", 320),
        "sources": {"main": {"path": source_path, "expected_size_wh": size}},
        "notes": "Adapted from an earlier manual configuration; original group selection retained.",
    }
    if "left_quad" in config:
        if not isinstance(size, list) or len(size) != 2:
            raise ProcessingError("Legacy config requires expected_size_wh.")
        w, h = size
        split = config.get("split_x")
        regions = []
        for side, box in [("left", [0, 0, split, h]), ("right", [split, 0, w, h])]:
            item = {"source": "main", "view_box": box, "quad": config.get(f"{side}_quad")}
            if f"{side}_feature_quad" in config:
                item["feature_quad"] = config[f"{side}_feature_quad"]
            regions.append(item)
        exposure = {**config.get("exposure", {}), "enabled": True}
        common.update(mode="single", gap_px=12, containers=[{
            "key": "container_1", "method": "edge", "same_surface_confirmed": True,
            "regions": regions, "exposure": exposure,
        }])
        return common
    if "panels" in config:
        groups = []
        for i, panel in enumerate(config["panels"], 1):
            groups.append({
                "key": f"container_{i}", "label": panel.get("label", f"Container {i}"),
                "method": "rectify", "regions": [{
                    "source": "main", "view_box": panel.get("view_box"), "quad": panel.get("quad"),
                    "notes": panel.get("coverage_note", "Visible panel only."),
                }],
            })
        common.update(mode="single" if len(groups) == 1 else "combo",
                      gap_px=max(1, config.get("gap", 12)), containers=groups)
        return common
    raise ProcessingError("Unrecognized configuration. Use schema_version: 1 or an earlier horizontal config.")


def load_job(config_path: Path, mode: str | None = None, input_path: Path | None = None,
             overrides: dict[str, Path] | None = None, direction: str | None = None
             ) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], list[Group], list[str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ProcessingError("Configuration must be a JSON object.")
    warnings: list[str] = []
    if "schema_version" not in config:
        if input_path is None:
            raise ProcessingError("Earlier configuration formats require --input ORIGINAL_IMAGE.")
        config = legacy_to_job(config, str(input_path.resolve()))
        warnings.append("Legacy config adapted. Physical grouping comes from its manual selections, not detection.")
    object_keys(config, {"schema_version", "mode", "height", "gap_px", "sources", "containers", "notes",
                         "direction", "layout", "cross_size_px", "edge_alignment"}, "job")
    version = integer(config.get("schema_version"), "schema_version", 1, 2)
    if version == 2 and "direction" not in config:
        raise ProcessingError("Schema 2 requires an explicit direction: horizontal, vertical, or auto.")
    requested_direction = config.get("direction", "horizontal")
    if requested_direction not in ("horizontal", "vertical", "auto"):
        raise ProcessingError("direction must be horizontal, vertical, or auto.")
    view_hint = direction_hint_from_view_boxes(config)
    if requested_direction == "auto":
        if view_hint is None:
            raise ProcessingError("direction:auto is ambiguous for these regions. Set horizontal or vertical explicitly.")
        config["direction"] = view_hint
        warnings.append(f"Direction auto-resolved to {view_hint} from obvious same-source view-box layout; pixel content was not inspected.")
    else:
        config["direction"] = requested_direction
        if view_hint is not None and view_hint != requested_direction:
            # View-box arrangement describes how panels were packaged into the
            # uploaded image, not the physical stitch axis; an explicit
            # direction is the operator's physical claim and wins.
            warnings.append(
                f"View boxes are {view_hint}ly arranged in the input image, but the recipe says "
                f"{requested_direction}. Input packaging does not prove the physical stitch axis; "
                f"using the explicit direction '{requested_direction}'.")
    if direction is not None and direction != config["direction"]:
        raise ProcessingError("CLI direction conflicts with resolved configuration direction; refusing to reinterpret corners.")
    config["layout"] = config.get("layout", config["direction"])
    if config["layout"] not in ("horizontal", "vertical"):
        raise ProcessingError("layout must be horizontal or vertical.")
    if "height" in config and (config["direction"] != "horizontal" or "cross_size_px" in config):
        raise ProcessingError("height is a horizontal legacy alias; do not combine it with vertical direction or cross_size_px.")
    if "height" in config:
        integer(config["height"], "height", 16, 4096)
    config["cross_size_px"] = integer(config.get("cross_size_px", config.pop("height", 320)),
                                      "cross_size_px", 16, 4096)
    if config.get("mode") not in ("single", "combo"):
        raise ProcessingError("Config mode must explicitly be single or combo. Automatic mode is not supported.")
    if mode is not None and mode != config["mode"]:
        raise ProcessingError("CLI mode conflicts with configuration mode; refusing to reinterpret grouping.")
    config["gap_px"] = integer(config.get("gap_px", 12), "gap_px", 1, 256)
    if "edge_alignment" in config:
        if config["edge_alignment"] not in ("butt", "measure"):
            raise ProcessingError('edge_alignment must be "butt" (declared corners joined as-is) '
                                  'or "measure" (feature-measured seam refinement).')
    else:
        config["edge_alignment"] = "butt"
    containers = config.get("containers")
    if not isinstance(containers, list) or not 1 <= len(containers) <= 8:
        raise ProcessingError("containers must contain 1..8 physical-container groups.")
    if (config["mode"] == "single" and len(containers) != 1
            or config["mode"] == "combo" and len(containers) < 2):
        raise ProcessingError("single requires exactly one group; combo requires two or more groups.")
    source_cfg = config.get("sources")
    if not isinstance(source_cfg, dict) or not 1 <= len(source_cfg) <= 8:
        raise ProcessingError("sources must name 1..8 input images.")
    overrides = dict(overrides or {})
    if input_path is not None:
        if len(source_cfg) != 1:
            raise ProcessingError("--input is only for a one-source job. Use --source NAME=PATH for multiple sources.")
        name = next(iter(source_cfg))
        if name in overrides:
            raise ProcessingError("Do not override the same source with both --input and --source.")
        overrides[name] = input_path
    if set(overrides) - set(source_cfg):
        raise ProcessingError("A --source override names an unknown source.")
    images: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    total_pixels = 0
    for name, info in source_cfg.items():
        safe_key(name, "source name")
        object_keys(info, {"path", "expected_size_wh", "sha256"}, f"source {name}")
        if not isinstance(info.get("path"), str) or not info["path"]:
            raise ProcessingError("Each source requires a nonempty local path.")
        path = (Path(overrides[name]).resolve() if name in overrides
                else (config_path.parent / info["path"]).resolve())
        im = read_image(path)
        h, w = im.shape[:2]
        expected = info.get("expected_size_wh")
        if not isinstance(expected, list) or len(expected) != 2:
            raise ProcessingError("Each source requires expected_size_wh: [width,height].")
        for value in expected:
            integer(value, "expected image size", 16, MAX_DIM)
        if expected != [w, h]:
            raise ProcessingError(f"Source {name} dimensions do not match the config. Reselect the corners.")
        sha = digest(path)
        pinned = info.get("sha256")
        if pinned is not None and (not isinstance(pinned, str) or re.fullmatch(r"[a-fA-F0-9]{64}", pinned) is None):
            raise ProcessingError("sha256 must be a 64-character hexadecimal string.")
        if pinned is not None and pinned.lower() != sha:
            raise ProcessingError(f"Source {name} SHA-256 mismatch. This recipe belongs to a different frame.")
        if pinned is None:
            warnings.append(f"Source {name} is not hash-locked: matching dimensions do not validate its corners or identity.")
        info["path"] = str(path)
        total_pixels += w * h
        if total_pixels > 60_000_000:
            raise ProcessingError("Combined decoded sources exceed the 60-megapixel job limit.")
        images[name] = im
        records.append({"name": name, "file": path.name, "sha256": sha, "size_wh": [w, h],
                        "sha256_checked_against_config": pinned is not None})
    groups: list[Group] = []
    keys, used_ids = set(), set()
    next_bit = 0
    for c in containers:
        object_keys(c, {"key", "container_id", "label", "method", "regions", "same_surface_confirmed",
                        "exposure", "matching", "notes", "seam", "coverage"}, "container")
        key = safe_key(c.get("key"), "container key")
        if key in keys:
            raise ProcessingError("Container keys must be unique.")
        keys.add(key)
        method = c.get("method")
        if method not in ("rectify", "edge", "overlap"):
            raise ProcessingError(f"Container {key}: method must be rectify, edge or overlap.")
        rcfg = c.get("regions")
        required_count = 1 if method == "rectify" else 2
        if not isinstance(rcfg, list) or len(rcfg) != required_count:
            raise ProcessingError(f"Container {key}: {method} requires exactly {required_count} region(s).")
        if method != "rectify" and c.get("same_surface_confirmed") is not True:
            raise ProcessingError(f"Container {key}: explicitly set same_surface_confirmed to true. "
                                  "The selected sections must depict the same physical side of one container.")
        ids = {v for v in [normalize_id(c.get("container_id"))] if v is not None}
        regions: list[Region] = []
        for r in rcfg:
            object_keys(r, {"source", "view_box", "quad", "feature_quad", "container_id", "notes", "rectified_size_wh"}, "region")
            name = r.get("source")
            if not isinstance(name, str) or name not in images:
                raise ProcessingError("Region references an unknown source.")
            im = images[name]
            box = r.get("view_box", [0, 0, im.shape[1], im.shape[0]])
            if not isinstance(box, list) or len(box) != 4:
                raise ProcessingError("view_box must be [x0,y0,x1,y1] with exclusive x1,y1.")
            for v in box:
                integer(v, "view_box coordinate", 0, MAX_DIM)
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= im.shape[1] and 0 <= y0 < y1 <= im.shape[0]):
                raise ProcessingError("view_box is outside its source image.")
            view = im[y0:y1, x0:x1]
            q = check_quad(r.get("quad"), x1 - x0, y1 - y0)
            fq = check_quad(r.get("feature_quad", q), x1 - x0, y1 - y0)
            if any(cv2.pointPolygonTest(q, tuple(map(float, p)), True) < -1.0 for p in fq):
                raise ProcessingError("feature_quad must lie inside its selected region quad.")
            explicit_size = r.get("rectified_size_wh")
            if explicit_size is not None:
                if not isinstance(explicit_size, list) or len(explicit_size) != 2:
                    raise ProcessingError("rectified_size_wh must be [width,height].")
                for n in explicit_size:
                    integer(n, "rectified_size_wh", 16, MAX_DIM)
                check_size(*explicit_size)
                cross_index = 0 if config["direction"] == "vertical" else 1
                if explicit_size[cross_index] != config["cross_size_px"]:
                    raise ProcessingError("rectified_size_wh cross-axis dimension must equal cross_size_px.")
            if next_bit >= MAX_REGIONS:
                raise ProcessingError("No more than 16 regions per job.")
            regions.append(Region(name, box, q, fq, view, 1 << next_bit, r))
            next_bit += 1
            normalized = normalize_id(r.get("container_id"))
            if normalized is not None:
                ids.add(normalized)
        if len(ids) > 1:
            raise ProcessingError(f"Container {key}: conflicting supplied IDs. Never stitch different containers together.")
        if ids & used_ids:
            raise ProcessingError("The same supplied container ID appears in separate physical-container groups.")
        used_ids |= ids
        if "same_surface_confirmed" in c:
            boolean(c["same_surface_confirmed"], "same_surface_confirmed")
        exposure = object_keys(c.get("exposure", {}), {"enabled", "sample_width", "exclude_seam_px",
                               "sample_y_fraction", "fade_px", "max_gain"}, "exposure")
        enabled = boolean(exposure.get("enabled", False), "exposure.enabled")
        if enabled and method != "edge":
            raise ProcessingError("Exposure balancing is only supported within an edge-joined container.")
        if "matching" in c:
            object_keys(c["matching"], {"ratio", "ransac_px", "min_inliers", "contrast_threshold", "feather_px",
                                       "space", "feature_channel", "edge_threshold", "border_cross_px",
                                       "border_axis_px", "max_cross_displacement_px"}, "matching")
            if method != "overlap":
                raise ProcessingError("matching settings are only valid for method overlap.")
        if c.get("coverage", "unverified") not in ("visible_panel", "partial", "unverified"):
            raise ProcessingError("coverage must be visible_panel, partial or unverified.")
        seam = object_keys(c.get("seam", {}), {"policy", "position_px"}, "seam")
        if seam and method != "overlap":
            raise ProcessingError("seam settings are only valid for method overlap.")
        if seam.get("policy", "union") not in ("union", "source_selected"):
            raise ProcessingError("seam.policy must be union or source_selected.")
        if "position_px" in seam:
            integer(seam["position_px"], "seam.position_px", 0, MAX_DIM)
            if seam.get("policy") != "source_selected":
                raise ProcessingError("An explicit seam position requires source_selected policy.")
        groups.append(Group({**c, "edge_alignment": config["edge_alignment"]}, regions))
    return config, images, records, groups, warnings


def match_features(a: Region, b: Region, ratio: float = 0.72, contrast: float = 0.04
                   ) -> tuple[Any, Any, list[Any]]:
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=contrast)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a.image, cv2.COLOR_BGR2GRAY), a.feature_mask)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b.image, cv2.COLOR_BGR2GRAY), b.feature_mask)
    if da is None or db is None or min(len(da), len(db)) < 2:
        return ka, kb, []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    forward = {m.queryIdx: m for pair in bf.knnMatch(db, da, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < ratio * n.distance}
    reverse = {m.queryIdx: m for pair in bf.knnMatch(da, db, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < ratio * n.distance}
    good, seen_a, seen_b = [], set(), set()
    for m in sorted(forward.values(), key=lambda v: v.distance):
        rev = reverse.get(m.trainIdx)
        pa = tuple(round(v, 1) for v in ka[m.trainIdx].pt)
        pb = tuple(round(v, 1) for v in kb[m.queryIdx].pt)
        if rev is not None and rev.trainIdx == m.queryIdx and pa not in seen_a and pb not in seen_b:
            good.append(m)
            seen_a.add(pa)
            seen_b.add(pb)
    return ka, kb, good


def save_matches(path: Path, a: Region, b: Region, ka: Any, kb: Any,
                 matches: list[Any], status: np.ndarray | None = None) -> None:
    viz = cv2.drawMatches(b.image, kb, a.image, ka, matches, None, flags=2,
                         matchesMask=None if status is None else status.astype(int).tolist())
    save_image(path, viz)


def edge_diagnostic(a: Region, b: Region, out: Path) -> dict[str, Any]:
    tests = []
    for contrast in (0.04, 0.01, 0.005):
        ka, kb, good = match_features(a, b, ratio=0.75, contrast=contrast)
        tests.append({"contrast_threshold": contrast, "keypoints_left_right": [len(ka), len(kb)],
                      "mutual_unique_candidates": len(good)})
        if contrast == 0.01:
            save_matches(out / "match_candidates_unverified.jpg", a, b, ka, kb, good)
    best = max(t["mutual_unique_candidates"] for t in tests)
    return {"kind": "diagnostic_only", "tests": tests, "best_mutual_unique_candidates": best,
            "alignment_applied": False, "overlap_verified": False,
            "note": "Candidate matches do not establish overlap. No inferred overlap is removed in edge mode."}


# Clamps for the feature-measured edge seam refinement. Deliberately tight:
# the correction is translation-only and must look like a mislocated quad,
# not like two different views of the surface. "strong" evidence (>=20
# inliers spanning the strip) upgrades the quality state; weaker evidence
# still applies a disclosed translation fix but keeps the amber state.
SEAM_ALIGN_CLAMP = {
    "min_inliers": 12, "strong_inliers": 20, "max_candidates": 8,
    "max_median_error_px": 2.0, "max_overlap_fraction": 0.45,
    "max_cross_offset_fraction": 0.14, "scale_range": (0.85, 1.15),
    "max_rotation_deg": 3.0, "min_spread_fraction": 0.55,
    "max_measured_gap_px": 4,
    "min_promotion_overlap_px": 24, "max_promotion_median_error_px": 2.5,
}


def measure_strip_alignment(a: np.ndarray, va: np.ndarray, b: np.ndarray, vb: np.ndarray,
                            direction: str) -> dict[str, Any]:
    """Feature-measured rigid offset between two warped edge strips (B -> A).

    SIFT + mutual ratio-test matches + RANSAC similarity, entirely on the
    already-rectified strips: same pixels in, same numbers out. Decides
    whether a clamped translation correction is supportable; applying it is
    the caller's decision. Acceptance is scored under the EXACT translation
    the renderer will apply (trim + quantized cross shift) — a good similarity
    fit alone never approves a correction its own scale/rotation terms forbid.
    This is a measured refinement of declared corners, not an independent
    verification of container identity.
    """
    clamp = SEAM_ALIGN_CLAMP
    out: dict[str, Any] = {"enabled": True, "applied": False, "direction": direction,
                           "note": "SIFT + RANSAC similarity measured on the warped strips."}
    # warp() returns premultiplied float pixels; SIFT needs 8-bit gray. Small
    # cross sizes (e.g. 320 px) starve the detector, so match at 2x and
    # divide the measured geometry back down.
    up = 2 if max(a.shape[0], a.shape[1]) < 2400 else 1
    gray_a = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(np.clip(b, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if up > 1:
        gray_a = cv2.resize(gray_a, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        gray_b = cv2.resize(gray_b, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        va = cv2.resize(va.astype(np.uint8), None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        vb = cv2.resize(vb.astype(np.uint8), None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01)
    ka, da = sift.detectAndCompute(gray_a, va.astype(np.uint8) * 255)
    kb, db = sift.detectAndCompute(gray_b, vb.astype(np.uint8) * 255)
    out["keypoints_first"], out["keypoints_second"] = len(ka), len(kb)
    if da is None or db is None or min(len(da), len(db)) < 2:
        out["reason"] = "too few features on the warped strips"
        return out
    bf = cv2.BFMatcher(cv2.NORM_L2)
    forward = {m.queryIdx: m for pair in bf.knnMatch(db, da, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < 0.80 * n.distance}
    reverse = {m.queryIdx: m for pair in bf.knnMatch(da, db, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < 0.80 * n.distance}
    good, seen_a, seen_b = [], set(), set()
    for m in sorted(forward.values(), key=lambda v: v.distance):
        rev = reverse.get(m.trainIdx)
        pa = tuple(round(v, 1) for v in ka[m.trainIdx].pt)
        pb = tuple(round(v, 1) for v in kb[m.queryIdx].pt)
        if rev is not None and rev.trainIdx == m.queryIdx and pa not in seen_a and pb not in seen_b:
            good.append(m)
            seen_a.add(pa)
            seen_b.add(pb)
    out["matches"] = len(good)
    if len(good) < clamp["max_candidates"]:
        out["reason"] = "fewer than 8 candidate matches between the strips"
        return out
    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / up
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) / up
    model, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0,
                                              maxIters=5000, confidence=0.999)
    if model is None or mask is None:
        out["reason"] = "RANSAC found no consistent model"
        return out
    inlier = mask.astype(bool).ravel()
    out["inliers"] = int(inlier.sum())
    out["inlier_ratio"] = float(inlier.mean())
    if out["inliers"] < 4:
        out["reason"] = "too few RANSAC inliers"
        return out
    residual = np.linalg.norm(cv2.transform(src, model).reshape(-1, 2) - dst.reshape(-1, 2), axis=1)
    median_err = float(np.median(residual[inlier]))
    scale = float(np.hypot(model[0, 0], model[1, 0]))
    rotation = float(np.degrees(np.arctan2(model[1, 0], model[0, 0])))
    tx, ty = float(model[0, 2]), float(model[1, 2])
    out["median_reprojection_error_px"] = round(median_err, 2)
    out["scale_measured"] = round(scale, 4)
    out["rotation_deg"] = round(rotation, 2)
    out["translation_px"] = [round(tx, 2), round(ty, 2)]
    h_a, w_a = a.shape[:2]
    h_b, w_b = b.shape[:2]
    pts_b = src.reshape(-1, 2)[inlier]
    pts_a = dst.reshape(-1, 2)[inlier]
    if direction == "horizontal":
        along = w_a - tx
        spread_pts = pts_b[:, 1]
        spread_ref, limit_ref = float(h_b), float(h_a)
        # Direct translation fit for the cross axis: this — not the similarity
        # model's ty — is the offset the renderer will actually apply.
        cross_fit = float(np.median(pts_a[:, 1] - pts_b[:, 1]))
    else:
        along = h_a - ty
        spread_pts = pts_b[:, 0]
        spread_ref, limit_ref = float(w_b), float(w_a)
        cross_fit = float(np.median(pts_a[:, 0] - pts_b[:, 0]))
    spread = float(spread_pts.max() - spread_pts.min()) if len(spread_pts) else 0.0
    out["measured_overlap_px"] = int(round(along))
    out["cross_offset_px"] = round(cross_fit, 2)
    out["inlier_spread_px"] = round(spread, 1)
    # Score the EXACT transform the renderer will apply — trim + quantized
    # cross translation only. A similarity fit can look perfect while its
    # scale/rotation terms are silently discarded by the render path; residuals
    # must be measured under the rendered model, or acceptance is meaningless.
    trim = max(0, int(round(along)))
    shift_applied = int(round(cross_fit)) if abs(cross_fit) >= 0.5 else 0
    t_render = (np.float64([w_a - trim, shift_applied]) if direction == "horizontal"
                else np.float64([shift_applied, h_a - trim]))
    rendered_err = float(np.median(np.linalg.norm(pts_a - (pts_b + t_render), axis=1)))
    out["rendered_translation_error_px"] = round(rendered_err, 2)
    strip_len = w_b if direction == "horizontal" else h_b
    gates = {
        "min_inliers": out["inliers"] >= clamp["min_inliers"],
        "median_error": median_err <= clamp["max_median_error_px"],
        "rendered_error": rendered_err <= clamp["max_median_error_px"],
        "scale_bounds": clamp["scale_range"][0] <= scale <= clamp["scale_range"][1],
        "rotation_bounds": abs(rotation) <= clamp["max_rotation_deg"],
        "overlap_bounds": (-clamp["max_measured_gap_px"]
                           <= along <= clamp["max_overlap_fraction"] * (w_b if direction == "horizontal" else h_b)),
        "cross_offset_bounds": abs(cross_fit) <= clamp["max_cross_offset_fraction"] * limit_ref,
    }
    out["sanity_checks"] = gates
    # "strong" requires inliers that span the strip, not just one feature band
    # (e.g. lettering), and accuracy under the rendered transform — only strong
    # evidence upgrades the quality state.
    out["strong"] = bool(out["inliers"] >= clamp["strong_inliers"]
                         and rendered_err <= clamp["max_median_error_px"]
                         and spread >= clamp["min_spread_fraction"] * spread_ref)
    out["inlier_spread_note"] = (
        "inliers concentrate in a narrow band; the translation is well-evidenced there but "
        "cross-axis scale drift is not corrected" if spread < clamp["min_spread_fraction"] * spread_ref
        else "inliers span the strip height")
    # Overlap candidate: the strips demonstrably share coverage, so the
    # verified overlap method (with its own fixed proof standards) may accept
    # this pair even when the translation-only evidence is thin. Promotion is
    # gated on the FITTED model because the overlap method applies that full
    # transform — scoring it here against the similarity is correct for that path.
    out["promotion_candidate"] = bool(
        len(good) >= clamp["max_candidates"]
        and median_err <= clamp["max_promotion_median_error_px"]
        and clamp["scale_range"][0] <= scale <= clamp["scale_range"][1]
        and abs(rotation) <= clamp["max_rotation_deg"]
        and along >= max(clamp["min_promotion_overlap_px"], 0.08 * strip_len))
    failed = [name for name, ok in gates.items() if not ok]
    if failed:
        if "rendered_error" in failed and "median_error" not in failed:
            out["model_mismatch_note"] = (
                f"the fitted similarity (scale {scale:.3f}, rotation {rotation:.2f} deg) is not "
                "consistent with the translation-only correction that would be applied; "
                "residuals are scored under the rendered transform")
        out["reason"] = ("measured geometry outside clamps: " + ", ".join(failed)
                         if not out["promotion_candidate"] else
                         "translation refinement declined; overlap promotion is the better fix")
        return out
    out["applied"] = True
    out["trim_px"] = trim
    out["shift_px"] = shift_applied
    out["warning"] = ("Correction is measured from pixels and re-scored under the exact "
                      "translation-only transform that was applied; repeated corrugations can "
                      "alias by one period. Visual review of the seam is still required.")
    return out


def _realigned_strip(b: np.ndarray, vb: np.ndarray, alignment: dict[str, Any],
                     direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Trim the measured overlap and apply the clamped cross-axis shift.

    Coverage policy: the cross shift is quantized to whole pixels and applied
    by growing the strip canvas — no interpolation (so no bilinear darkening
    of valid pixels) and no clipping (unique content is never discarded).
    The caller must pad the opposite strip by |shift_px| along the cross axis
    on the complementary side before concatenating.
    """
    trim = alignment["trim_px"]
    shift = int(round(float(alignment["shift_px"])))
    if direction == "horizontal":
        if trim > 0:
            b, vb = b[:, trim:], vb[:, trim:]
        if shift:
            before, after = max(shift, 0), max(-shift, 0)
            b = np.pad(b, ((before, after), (0, 0), (0, 0)))
            vb = np.pad(vb, ((before, after), (0, 0)), constant_values=False)
    else:
        if trim > 0:
            b, vb = b[trim:, :], vb[trim:, :]
        if shift:
            before, after = max(shift, 0), max(-shift, 0)
            b = np.pad(b, ((0, 0), (before, after), (0, 0)))
            vb = np.pad(vb, ((0, 0), (before, after)), constant_values=False)
    return b, vb


def balance_edges(a: np.ndarray, b: np.ndarray, va: np.ndarray, vb: np.ndarray,
                  cfg: dict[str, Any], enabled: bool
                  ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not enabled:
        return a, b, {"enabled": False}
    limit = min(a.shape[1], b.shape[1])
    strip = integer(cfg.get("sample_width", min(64, limit)), "sample_width", 8, limit)
    margin = integer(cfg.get("exclude_seam_px", min(6, strip - 4)), "exclude_seam_px", 0, strip - 4)
    fade = integer(cfg.get("fade_px", 160), "fade_px", 1, MAX_DIM)
    fractions = cfg.get("sample_y_fraction", [0.12, 0.87])
    if not isinstance(fractions, list) or len(fractions) != 2:
        raise ProcessingError("sample_y_fraction must be [start,end].")
    f0, f1 = [number(v, "sample_y_fraction", 0, 1) for v in fractions]
    if f0 >= f1:
        raise ProcessingError("Exposure band start must precede end.")
    h = a.shape[0]
    y0, y1 = int(h * f0), int(h * f1)
    if y1 - y0 < 4:
        raise ProcessingError("Exposure sample band is too small.")
    sa = (slice(y0, y1), slice(a.shape[1] - strip, a.shape[1] - margin))
    sb = (slice(y0, y1), slice(margin, strip))
    pa, pb = a[sa][va[sa]], b[sb][vb[sb]]
    if min(len(pa), len(pb)) < 32:
        raise ProcessingError("Too few valid exposure samples; disable balancing or adjust sample bands.")
    med_a, med_b = np.median(pa, axis=0), np.median(pb, axis=0)
    if min(float(med_a.min()), float(med_b.min())) < 2:
        raise ProcessingError("Near-black exposure strip; disable balancing or select another region.")
    max_gain = number(cfg.get("max_gain", 1.6), "max_gain", 1, 4)
    target = np.sqrt(med_a * med_b)
    ga = np.clip(target / med_a, 1 / max_gain, max_gain)
    gb = np.clip(target / med_b, 1 / max_gain, max_gain)
    ta = np.clip(1 - np.arange(a.shape[1] - 1, -1, -1, dtype=np.float32) / fade, 0, 1)
    tb = np.clip(1 - np.arange(b.shape[1], dtype=np.float32) / fade, 0, 1)
    ta, tb = ta * ta * (3 - 2 * ta), tb * tb * (3 - 2 * tb)
    aa = a * (1 + ta[None, :, None] * (ga - 1)[None, None, :])
    bb = b * (1 + tb[None, :, None] * (gb - 1)[None, None, :])
    return aa, bb, {
        "enabled": True, "method": "median-channel gain with smoothstep taper inside each section",
        "assumption": "Adjacent strips show comparable paint, NOT confirmed identical scene pixels.",
        "cross_source_pixel_blending": False,
        "left_median_bgr": med_a.tolist(), "right_median_bgr": med_b.tolist(),
        "left_gain_at_seam_bgr": ga.tolist(), "right_gain_at_seam_bgr": gb.tolist(),
        "fade_width_px": fade,
        "clipped_channel_values": int(np.count_nonzero(aa[va] > 255) + np.count_nonzero(bb[vb] > 255)),
    }


def projected_quad(q: np.ndarray, H: np.ndarray) -> np.ndarray:
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ProcessingError("Invalid or singular homography.")
    denominator = np.c_[q, np.ones(4)] @ H[2]
    if not (np.all(denominator > 1e-6) or np.all(denominator < -1e-6)):
        raise ProcessingError("Homography crosses a projective horizon.")
    transformed = cv2.perspectiveTransform(q[None].astype(np.float64), H)[0].astype(np.float32)
    if not np.isfinite(transformed).all() or not cv2.isContourConvex(transformed):
        raise ProcessingError("Folded or invalid projected boundary.")
    if cv2.contourArea(transformed, oriented=True) <= 0:
        raise ProcessingError("Reflected or collapsed projected boundary.")
    return transformed


def blend(a: np.ndarray, va: np.ndarray, b: np.ndarray, vb: np.ndarray,
          feather: int) -> tuple[np.ndarray, np.ndarray]:
    overlap = va & vb
    if np.count_nonzero(overlap) < 512:
        raise ProcessingError("Insufficient valid pixel overlap; no invented join or automatic fallback is allowed.")
    xx = np.arange(va.shape[1], dtype=np.float32)[None, :]
    lo = np.where(overlap, xx, np.inf).min(axis=1)
    hi = np.where(overlap, xx, -np.inf).max(axis=1)
    rows = overlap.any(axis=1)
    mid = np.zeros(va.shape[0], np.float32)
    mid[rows] = (lo[rows] + hi[rows]) / 2
    if feather == 0:
        t = (xx >= mid[:, None]).astype(np.float32)
    else:
        effective = np.ones(va.shape[0], np.float32)
        effective[rows] = np.maximum(1, np.minimum(feather, hi[rows] - lo[rows]))
        t = np.clip((xx - mid[:, None]) / effective[:, None] + 0.5, 0, 1)
    t[~vb] = 0
    t[vb & ~va] = 1
    return a * (1 - t[..., None]) + b * t[..., None], t


def matching_rectified(a: Region, b: Region, cross_size: int, direction: str,
                       cfg: dict[str, Any]) -> tuple[Any, ...]:
    """Rectify for feature estimation only; final image is warped once from source pixels."""
    channel = cfg.get("feature_channel", "gray")
    if channel not in ("gray", "green"):
        raise ProcessingError("feature_channel must be gray or green.")
    contrast = number(cfg.get("contrast_threshold", 0.008), "contrast_threshold", 0.001, 0.1)
    edge = number(cfg.get("edge_threshold", 18), "edge_threshold", 1, 100)
    ratio = number(cfg.get("ratio", 0.8), "ratio", 0.5, 0.9)
    bc = integer(cfg.get("border_cross_px", 12), "border_cross_px", 1, 512)
    ba = integer(cfg.get("border_axis_px", 10), "border_axis_px", 1, 512)
    max_cross = cfg.get("max_cross_displacement_px")
    if max_cross is not None:
        max_cross = number(max_cross, "max_cross_displacement_px", 0.1, 4096)
    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=contrast, edgeThreshold=edge)
    views, transforms, masks, keypoints, descriptors = [], [], [], [], []
    for r in (a, b):
        R, size = rectification_axis(r.quad, cross_size, direction, r.config.get("rectified_size_wh"))
        # This 8-bit representation is ONLY for finding features, never the final stitched pixels.
        im = cv2.warpPerspective(r.image, R, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(r.feature_mask, R, size, flags=cv2.INTER_NEAREST)
        by, bx = (ba, bc) if direction == "vertical" else (bc, ba)
        if min(size[0]-2*bx, size[1]-2*by) < 8:
            raise ProcessingError("Rectified feature-mask borders leave too little interior.")
        mask[:by] = mask[-by:] = 0
        mask[:, :bx] = mask[:, -bx:] = 0
        feature_image = im[..., 1] if channel == "green" else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k, d = sift.detectAndCompute(feature_image, mask)
        views.append(im); transforms.append(R); masks.append(mask)
        keypoints.append(k); descriptors.append(d)
    ka, kb = keypoints
    da, db = descriptors
    matches = []
    if da is not None and db is not None and min(len(da), len(db)) >= 2:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        reverse = {m.queryIdx: m.trainIdx for pair in bf.knnMatch(da, db, k=2) if len(pair) == 2
                   for m, n in [pair] if m.distance < ratio*n.distance}
        candidates = [m for pair in bf.knnMatch(db, da, k=2) if len(pair) == 2
                      for m, n in [pair] if m.distance < ratio*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
        seen_a, seen_b = set(), set()
        cross_axis = 0 if direction == "vertical" else 1
        for m in sorted(candidates, key=lambda v: v.distance):
            pa, pb = ka[m.trainIdx].pt, kb[m.queryIdx].pt
            if max_cross is not None and abs(pa[cross_axis]-pb[cross_axis]) >= max_cross:
                continue
            # Multiple SIFT orientations at one location do not count as independent evidence.
            qa, qb = tuple(np.rint(pa).astype(int)), tuple(np.rint(pb).astype(int))
            if qa not in seen_a and qb not in seen_b:
                seen_a.add(qa); seen_b.add(qb); matches.append(m)
    return ka, kb, matches, transforms, views, masks


def source_selected_blend(a: np.ndarray, va: np.ndarray, b: np.ndarray, vb: np.ndarray,
                          direction: str, feather: int, position: int | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Choose first source before the seam, second after it; leave missing support transparent."""
    # Canonical representation: length runs along array rows, width across columns.
    swap = direction == "horizontal"
    if swap:
        a, va, b, vb = [np.swapaxes(v, 0, 1) for v in (a, va, b, vb)]
    overlap = va & vb
    support = overlap.sum(axis=1)
    threshold = .85 * min(va.sum(axis=1).max(), vb.sum(axis=1).max())
    rows = np.flatnonzero((support >= threshold) & (support > 0))
    half = max(4, feather//2)
    length, width = va.shape
    if len(rows) < max(feather+4, 32):
        raise ProcessingError("Not enough broad overlap for a source-selected seam.")
    def score(at: int) -> float:
        if at < half or at+half >= length:
            raise ProcessingError("Requested seam is outside the supported canvas.")
        valid = overlap[at-half:at+half]
        if valid.mean() < .75:
            raise ProcessingError("Requested seam lacks sufficient overlap.")
        return float(np.abs(a[at-half:at+half][valid]-b[at-half:at+half][valid]).mean())
    if position is None:
        first_end = int(np.flatnonzero(va.any(axis=1))[-1])
        start = max(int(rows[0])+half, int(rows[0]+.40*(first_end-rows[0])))
        end = min(int(rows[-1])-half, first_end-half-4)
        candidates = []
        for pos in range(start, end+1):
            if overlap[pos-half:pos+half].mean() >= .80:
                candidates.append((score(pos), pos))
        if not candidates:
            raise ProcessingError("No supported source-selected seam was found.")
        difference, seam = min(candidates)
    else:
        seam = position
        difference = score(seam)
    along = np.arange(length, dtype=np.float32)[:, None]
    initial_b = ((along >= seam).astype(np.float32) if feather == 0 else
                 np.clip((along-seam)/feather+.5, 0, 1))
    first = (1-initial_b)*va
    second = initial_b*vb
    total = first+second
    valid = total > 1e-6
    t = np.divide(second, total, out=np.zeros_like(second), where=valid)
    pixels = a*(1-t[..., None])+b*t[..., None]
    pixels[~valid] = 0
    t[~valid] = 0
    report = {"policy": "source_selected", "position_px": int(seam),
              "axis": "y" if direction == "vertical" else "x",
              "selection": "automatic_pixel_difference" if position is None else "configured",
              "feather_px": feather, "mean_abs_seam_difference_0_255": difference,
              "union_pixels_omitted_by_selection": int(np.count_nonzero((va | vb) & ~valid)),
              "note": "First source before seam; second after. No stale first-view strips beyond seam."}
    if swap:
        pixels, valid, t = [np.swapaxes(v, 0, 1) for v in (pixels, valid, t)]
    return pixels, valid, t, report


def directional_blend(a: np.ndarray, va: np.ndarray, b: np.ndarray, vb: np.ndarray,
                      direction: str, feather: int, seam_cfg: dict[str, Any]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if seam_cfg.get("policy", "union") == "source_selected":
        return source_selected_blend(a, va, b, vb, direction, feather, seam_cfg.get("position_px"))
    if direction == "vertical":
        p, t = blend(np.swapaxes(a, 0, 1), va.T, np.swapaxes(b, 0, 1), vb.T, feather)
        p, t = np.swapaxes(p, 0, 1), t.T
    else:
        p, t = blend(a, va, b, vb, feather)
    return p, va | vb, t, {"policy": "union", "selection": "per_crossline_overlap_midpoint",
                           "feather_px": feather, "axis": "y" if direction == "vertical" else "x"}


# ------------------------------------------------------------------ diagnostics
# Deterministic "why did this stitch pass or fail" artifacts. The overlay only
# DRAWS decisions the engine already made; it never evaluates quality itself,
# and AI planning never enters this path.


@dataclass
class StitchDiagnostics:
    """Machine-readable record of one container group's stitch decision.

    Coordinates: detected_corners are absolute source-image pixels;
    match_points_* are in the first/second region-view spaces;
    projected_bounds/overlap_polygon/seam_points are in stitched-output space.
    """
    method: str
    status: str
    quality_state: str
    direction: str
    source_size_wh: list[list[int]]
    detected_corners: list[list[list[float]]]
    rejection_reason: str | None = None
    matches_total: int | None = None
    inliers: int | None = None
    inlier_ratio: float | None = None
    match_points_first: list[list[float]] | None = None
    match_points_second: list[list[float]] | None = None
    inlier_mask: list[bool] | None = None
    homography_second_to_first: list[list[float]] | None = None
    projected_bounds: list[int] | None = None
    projected_quad_first: list[list[float]] | None = None
    projected_quad_second: list[list[float]] | None = None
    overlap_polygon: list[list[int]] | None = None
    overlap_ratio: float | None = None
    seam_points: list[list[int]] | None = None
    feather_px: int | None = None
    median_reprojection_error_px: float | None = None
    seam_alignment: dict[str, Any] | None = None
    sanity: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "method": self.method, "status": self.status,
            "quality_state": self.quality_state, "direction": self.direction,
            "source_size_wh": self.source_size_wh, "detected_corners": self.detected_corners,
            "rejection_reason": self.rejection_reason, "matches_total": self.matches_total,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "match_points_first": self.match_points_first,
            "match_points_second": self.match_points_second,
            "inlier_mask": self.inlier_mask,
            "homography_second_view_to_first_view": self.homography_second_to_first,
            "projected_bounds_xyxy": self.projected_bounds,
            "projected_quad_first_in_output": self.projected_quad_first,
            "projected_quad_second_in_output": self.projected_quad_second,
            "overlap_polygon_in_output": self.overlap_polygon,
            "overlap_ratio": self.overlap_ratio,
            "seam_points_in_output": self.seam_points,
            "feather_px": self.feather_px,
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "seam_alignment": self.seam_alignment,
            "sanity_checks": self.sanity or {},
            "note": "Deterministic engine diagnostic. Rendering never re-evaluates quality.",
        }


_DIAG_COLORS = {  # BGR, fixed palette — no randomness enters the overlay.
    "corner": (0, 165, 255), "frame": (200, 200, 200),
    "inlier": (60, 200, 60), "rejected": (60, 60, 230),
    "quad_first": (255, 160, 40), "quad_second": (0, 165, 255),
    "overlap": (60, 200, 60), "seam": (0, 230, 230), "feather": (180, 120, 40),
    "text": (255, 255, 255), "text_bg": (24, 24, 24), "reason": (60, 60, 255),
}


def _fit_text(img: np.ndarray, x: int, y: int, text: str, scale: float,
              color=(255, 255, 255), thickness: int = 1, bg: bool = True) -> None:
    if bg:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base + 2), _DIAG_COLORS["text_bg"], -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


def render_debug_overlay(panels: list[tuple[np.ndarray, list[int]]],
                         diag: StitchDiagnostics, result_image: np.ndarray | None,
                         out: Path, max_width: int = 1500) -> None:
    """Draw one deterministic diagnostic sheet for a container group.

    panels: [(region_image_crop, view_box), ...] in region order;
    result_image: stitched output (BGRA) when one was produced.
    Layout: [panel A | panel B] match sheet on top, output-space sheet below,
    metrics bar at the very top; rejection reason printed in red when rejected.
    """
    crops = []
    for image, box in panels:
        x0, y0, x1, y1 = box
        crops.append(image[y0:y1, x0:x1][:, :, :3].copy() if image.ndim == 3 else image[y0:y1, x0:x1].copy())

    def scaled(img: np.ndarray, target_h: int) -> np.ndarray:
        h = img.shape[0]
        return img if h <= 0 or abs(h - target_h) <= 1 else cv2.resize(
            img, (max(1, round(img.shape[1] * target_h / h)), target_h), interpolation=cv2.INTER_AREA)

    # ── match sheet: regions side by side, quads + matches drawn ──
    band_h = 380
    match_sheet = None
    if crops:
        shown = [scaled(c, band_h) for c in crops]
        gap = 8
        total_w = sum(c.shape[1] for c in shown) + gap * (len(shown) - 1)
        match_sheet = np.full((band_h, total_w, 3), 24, np.uint8)
        offsets, x = [], 0
        for c in shown:
            match_sheet[:, x:x + c.shape[1]] = c
            offsets.append(x)
            x += c.shape[1] + gap
        scale_f = [c.shape[1] / max(1, crops[i].shape[1]) for i, c in enumerate(shown)]
        for i, (crop, (_, box)) in enumerate(zip(crops, panels)):
            x0b, y0b = box[0], box[1]
            fx, fy = scale_f[i], band_h / max(1, crop.shape[0])
            if i < len(diag.detected_corners):
                pts = np.float32(diag.detected_corners[i]) - np.float32([x0b, y0b])
                pts = np.rint(pts * np.float32([fx, fy])).astype(np.int32)
                cv2.polylines(match_sheet, [pts], True, _DIAG_COLORS["corner"], 2, cv2.LINE_AA)
                for j, pt in enumerate(pts):
                    cv2.circle(match_sheet, tuple(pt), 3, _DIAG_COLORS["corner"], -1)
        if diag.match_points_first and diag.match_points_second and len(crops) == 2:
            fx0, fy0 = scale_f[0], band_h / max(1, crops[0].shape[0])
            fx1, fy1 = scale_f[1], band_h / max(1, crops[1].shape[0])
            mask = diag.inlier_mask or []
            inlier_idx = [i for i in range(len(diag.match_points_first))
                          if i < len(mask) and mask[i]]
            # deterministic display cap: dense matches would render as a smear
            shown_inliers = inlier_idx
            if len(shown_inliers) > 120:
                step = len(shown_inliers) / 120.0
                shown_inliers = [inlier_idx[int(i * step)] for i in range(120)]
            shown = set(shown_inliers)
            for idx, (p1, p2) in enumerate(zip(diag.match_points_first, diag.match_points_second)):
                ok = idx in shown
                color = _DIAG_COLORS["inlier"] if ok else _DIAG_COLORS["rejected"]
                a = tuple(np.rint(np.float32(p1) * np.float32([fx0, fy0])).astype(int))
                b = tuple(np.rint(np.float32(p2) * np.float32([fx1, fy1])).astype(int) + np.int32([offsets[1], 0]))
                cv2.circle(match_sheet, a, 3 if ok else 2, color, -1)
                cv2.circle(match_sheet, b, 3 if ok else 2, color, -1)
                if ok:
                    cv2.line(match_sheet, a, b, color, 1, cv2.LINE_AA)
            if len(inlier_idx) > len(shown_inliers):
                _fit_text(match_sheet, offsets[1] + 6, band_h - 8,
                          f"showing {len(shown_inliers)} of {len(inlier_idx)} inliers",
                          0.42, _DIAG_COLORS["inlier"], 1)

    # ── output-space sheet: result + projected quads + overlap + seam ──
    out_sheet = None
    if result_image is not None and result_image.shape[0] > 0:
        base = result_image[:, :, :3].copy() if result_image.ndim == 3 else result_image.copy()
        out_h = 380
        s = out_h / max(1, base.shape[0])
        base = cv2.resize(base, (max(1, round(base.shape[1] * s)), out_h), interpolation=cv2.INTER_AREA)
        overlay = base.copy()
        if diag.projected_quad_first is not None:
            pts = np.rint(np.float32(diag.projected_quad_first) * s).astype(np.int32)
            cv2.polylines(overlay, [pts], True, _DIAG_COLORS["quad_first"], 2, cv2.LINE_AA)
        if diag.projected_quad_second is not None:
            pts = np.rint(np.float32(diag.projected_quad_second) * s).astype(np.int32)
            cv2.polylines(overlay, [pts], True, _DIAG_COLORS["quad_second"], 2, cv2.LINE_AA)
        if diag.overlap_polygon and len(diag.overlap_polygon) >= 3:
            pts = np.rint(np.float32(diag.overlap_polygon) * s).astype(np.int32)
            mask_img = np.zeros(base.shape[:2], np.uint8)
            cv2.fillPoly(mask_img, [pts], 255)
            overlay[mask_img > 0] = np.rint(0.55 * overlay[mask_img > 0]
                                            + 0.45 * np.float32(_DIAG_COLORS["overlap"])).astype(np.uint8)
            cv2.polylines(overlay, [pts], True, _DIAG_COLORS["overlap"], 2, cv2.LINE_AA)
        if diag.seam_points and len(diag.seam_points) >= 2:
            seam = np.float32(diag.seam_points) * s
            if diag.feather_px:
                band = max(1.5, float(diag.feather_px) * s / 2)
                vec = seam[-1] - seam[0]
                norm = np.float32([-vec[1], vec[0]])
                length = float(np.hypot(*norm)) or 1.0
                normal = norm / length
                side_a = np.rint(seam + band * normal).astype(np.int32)
                side_b = np.rint(seam - band * normal).astype(np.int32)
                poly = np.vstack([side_a, side_b[::-1]])
                band_mask = np.zeros(base.shape[:2], np.uint8)
                cv2.fillPoly(band_mask, [poly], 255)
                overlay[band_mask > 0] = np.rint(
                    0.72 * overlay[band_mask > 0]
                    + 0.28 * np.float32(_DIAG_COLORS["feather"])).astype(np.uint8)
            pts = np.rint(seam).astype(np.int32)
            cv2.polylines(overlay, [pts], False, _DIAG_COLORS["seam"], 2, cv2.LINE_AA)
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1),
                      _DIAG_COLORS["frame"], 1)
        out_sheet = overlay

    # ── compose sheets + metrics bar ──
    sheets = [sh for sh in (match_sheet, out_sheet) if sh is not None]
    if not sheets:
        sheets = [np.full((120, 480, 3), 24, np.uint8)]
    max_w = max(sh.shape[1] for sh in sheets)
    padded = []
    for label, sh in zip(
            [lab for lab, sht in (("match sheet", match_sheet), ("stitched output", out_sheet)) if sht is not None],
            sheets):
        left = (max_w - sh.shape[1]) // 2
        canvas = np.pad(sh, ((0, 0), (left, max_w - sh.shape[1] - left), (0, 0)),
                        constant_values=24)
        _fit_text(canvas, 8, 16, label, 0.42, (160, 160, 160), 1)
        padded.append(canvas)
    body = np.vstack(padded)

    metrics = [f"{diag.method}  |  {diag.status}  |  quality: {diag.quality_state}"]
    if diag.matches_total is not None:
        ratio = f" ({100 * diag.inlier_ratio:.0f}%)" if diag.inlier_ratio is not None else ""
        err = f" | median err {diag.median_reprojection_error_px:.2f}px" if diag.median_reprojection_error_px is not None else ""
        metrics.append(f"matches {diag.matches_total} | inliers {diag.inliers}{ratio}{err}")
    extra = []
    if diag.overlap_ratio is not None:
        extra.append(f"overlap {100 * diag.overlap_ratio:.0f}%")
    if diag.feather_px is not None:
        extra.append(f"feather {diag.feather_px}px")
    if diag.projected_bounds is not None:
        extra.append(f"bounds {diag.projected_bounds}")
    if extra:
        metrics.append(" | ".join(extra))

    bar_lines = len(metrics) + (len(_wrap(diag.rejection_reason, 92)) if diag.rejection_reason else 0)
    bar = np.full((22 * (bar_lines + 1) + 8, max_w, 3), 24, np.uint8)
    y = 20
    for line in metrics:
        _fit_text(bar, 10, y, line, 0.5, _DIAG_COLORS["text"], 1)
        y += 22
    if diag.rejection_reason:
        for line in _wrap(f"REJECTED: {diag.rejection_reason}", 92):
            _fit_text(bar, 10, y, line, 0.55, _DIAG_COLORS["reason"], 2)
            y += 22

    sheet = np.vstack([bar, body])
    if sheet.shape[1] > max_width:
        sc = max_width / sheet.shape[1]
        sheet = cv2.resize(sheet, (max_width, max(1, round(sheet.shape[0] * sc))),
                           interpolation=cv2.INTER_AREA)
    save_image(out, cv2.cvtColor(sheet, cv2.COLOR_BGR2BGRA))


def _seam_polyline(t: np.ndarray, valid: np.ndarray, direction: str) -> list[list[int]]:
    """Extract the t=0.5 crossing per cross-line from the blend weight map."""
    pts: list[list[int]] = []
    if direction == "horizontal":
        for y in range(t.shape[0]):
            row = t[y]
            idx = np.flatnonzero((row[:-1] <= .5) & (row[1:] > .5))
            if len(idx):
                pts.append([int(idx[0]), y])
    else:
        for x in range(t.shape[1]):
            col = t[:, x]
            idx = np.flatnonzero((col[:-1] <= .5) & (col[1:] > .5))
            if len(idx):
                pts.append([x, int(idx[0])])
    return pts[:2000]


def _largest_overlap_polygon(va: np.ndarray, vb: np.ndarray) -> list[list[int]] | None:
    overlap = (va & vb).astype(np.uint8)
    contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    return hull.reshape(-1, 2).tolist()


def overlap_group(group: Group, cross_size: int, out: Path, direction: str = "horizontal") -> Tile:
    a, b = group.regions
    cfg = group.config.get("matching", {})
    space = cfg.get("space", "source")
    if space not in ("source", "rectified"):
        raise ProcessingError("matching.space must be source or rectified.")
    rectified_only = {"feature_channel", "edge_threshold", "border_cross_px", "border_axis_px", "max_cross_displacement_px"}
    if space == "source" and rectified_only & set(cfg):
        raise ProcessingError("Rectified feature settings require matching.space: rectified.")
    ratio = number(cfg.get("ratio", 0.8 if space == "rectified" else .72), "ratio", 0.5, 0.9)
    threshold = number(cfg.get("ransac_px", 3.0), "ransac_px", 0.1, 20)
    minimum = integer(cfg.get("min_inliers", 20), "min_inliers", 8, 20000)
    contrast = number(cfg.get("contrast_threshold", .008 if space == "rectified" else .04),
                      "contrast_threshold", .001, .1)
    feather = integer(cfg.get("feather_px", 32), "feather_px", 0, 512)
    if space == "rectified":
        ka, kb, matches, transforms, views, masks = matching_rectified(a, b, cross_size, direction, cfg)
        RA, RB = transforms
        for index, im in enumerate(views, 1):
            save_image(out / f"matching_rectified_{index}.png", im)
    else:
        ka, kb, matches = match_features(a, b, ratio, contrast)
        views, masks = [a.image, b.image], [a.feature_mask, b.feature_mask]
        RA = RB = np.eye(3)
    stats: dict[str, Any] = {"keypoints_first_second": [len(ka), len(kb)],
                            "mutual_unique_candidates": len(matches), "matching_space": space,
                            "residual_coordinate_space": f"first {space} image pixels",
                            "feature_channel": cfg.get("feature_channel", "gray"),
                            "max_cross_displacement_px": cfg.get("max_cross_displacement_px"),
                            "thresholds": {"ratio": ratio, "ransac_px": threshold, "min_inliers": minimum,
                                           "min_inlier_ratio": .35, "contrast_threshold": contrast}}
    if direction == "horizontal":
        stats["keypoints_left_right"] = [len(ka), len(kb)]
    src = np.float32([kb[m.queryIdx].pt for m in matches])
    dst = np.float32([ka[m.trainIdx].pt for m in matches])
    stats.update(match_points_second=src.tolist(), match_points_first=dst.tolist())
    if len(matches) < minimum:
        raise ProcessingError(f"Only {len(matches)} unique symmetric matches; need {minimum} for overlap mode.", stats)
    Hfit, status = cv2.findHomography(src, dst, cv2.RANSAC, threshold,
                                     maxIters=30000 if space == "rectified" else 5000, confidence=.999)
    if Hfit is None or status is None:
        raise ProcessingError("RANSAC could not estimate an overlap homography.", stats)
    good = status.ravel().astype(bool)
    stats.update(inliers=int(good.sum()), inlier_ratio=float(good.mean()), inlier_mask=good.tolist())
    if good.sum() < minimum or good.mean() < .35:
        raise ProcessingError(f"Weak overlap fit: {int(good.sum())}/{len(matches)} inliers.", stats)
    coverage = []
    for pts, mask in [(src[good], masks[1]), (dst[good], masks[0])]:
        area = cv2.contourArea(cv2.convexHull(pts))
        frac = float(area / max(1, np.count_nonzero(mask)))
        coverage.append(frac)
        if frac < .02 or (space == "rectified" and area < 1000):
            raise ProcessingError("Inliers are too concentrated to trust a homography.", stats)
    stats["inlier_hull_area_fraction_second_first"] = coverage
    H = np.linalg.inv(RA) @ Hfit @ RB
    if not np.isfinite(H).all() or abs(float(H[2, 2])) < 1e-12:
        raise ProcessingError("Unstable homography normalization.", stats)
    H /= H[2, 2]
    raw_boundary = projected_quad(b.quad, H)
    scale = cv2.contourArea(raw_boundary)/cv2.contourArea(b.quad)
    if not .25 <= scale <= 4:
        raise ProcessingError("Implausible relative camera scale (area ratio outside 0.25..4).", stats)
    HA = (RA if space == "rectified" else
          rectification_axis(a.quad, cross_size, direction, a.config.get("rectified_size_wh"))[0])
    qa, qb = projected_quad(a.quad, HA), projected_quad(b.quad, HA @ H)
    top = qb[1]-qb[0]
    angle = math.degrees(math.atan2(float(top[1]), float(top[0])))
    if abs(angle) > 20:
        raise ProcessingError("Estimated rectified cross-edge rotation exceeds 20 degrees.", stats)
    axis = 1 if direction == "vertical" else 0
    span = float(qa[:, axis].max()-qa[:, axis].min()+1)
    if (qb[:, axis].mean() <= qa[:, axis].mean()+max(2, .05*span)
            or qb[:, axis].max() <= qa[:, axis].max()+max(2, .02*span)):
        word = "downward" if direction == "vertical" else "rightward"
        raise ProcessingError(f"Second region does not extend {word}; check order and identity.", stats)
    error = np.linalg.norm(cv2.perspectiveTransform(src[good][None], Hfit)[0]-dst[good], axis=1)
    if float(np.median(error)) > threshold or float(np.percentile(error, 95)) > 2*threshold:
        raise ProcessingError("Large reprojection residuals despite an apparent feature fit.", stats)
    corners = np.vstack([qa, qb]).astype(np.float64)
    near_int = np.abs(corners-np.rint(corners)) < .001
    corners[near_int] = np.rint(corners[near_int])
    low, high = np.floor(corners.min(axis=0)).astype(int), np.ceil(corners.max(axis=0)).astype(int)
    width, height = map(int, high-low+1)
    check_size(width, height)
    T = np.float64([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]])
    WA, WB = T @ HA, T @ HA @ H
    aa, va = warp(a.image, a.mask, WA, (width, height))
    bb, vb = warp(b.image, b.mask, WB, (width, height))
    fraction = float(np.count_nonzero(va & vb)/max(1, min(np.count_nonzero(va), np.count_nonzero(vb))))
    if fraction < .05:
        raise ProcessingError("Less than 5% valid selected-surface overlap.", stats)
    seam_cfg = group.config.get("seam", {})
    pixels, valid, t, seam_report = directional_blend(aa, va, bb, vb, direction, feather, seam_cfg)
    hard_cfg = dict(seam_cfg)
    if seam_report["policy"] == "source_selected":
        hard_cfg["position_px"] = seam_report["position_px"]
    hard, hard_valid, _, _ = directional_blend(aa, va, bb, vb, direction, 0, hard_cfg)
    provenance = np.zeros(valid.shape, np.uint16)
    provenance[valid & va & (t < 1)] |= np.uint16(a.bit)
    provenance[valid & vb & (t > 0)] |= np.uint16(b.bit)
    save_image(out / "hard_seam.png", bgra(hard, hard_valid))
    save_image(out / "warped_first.png", bgra(aa, va))
    save_image(out / "warped_second.png", bgra(bb, vb))
    weights = np.rint(t*65535).astype(np.uint16)
    save_image(out / "second_weight_16bit.png", weights)
    if direction == "horizontal":  # v1 diagnostic filename compatibility
        save_image(out / "right_weight_16bit.png", weights)
    viz = cv2.drawMatches(views[1], kb, views[0], ka, matches, None, flags=2,
                          matchesMask=good.astype(int).tolist())
    save_image(out / "feature_matches.jpg", viz)
    stats.update(median_reprojection_error_px=float(np.median(error)), max_reprojection_error_px=float(error.max()),
                 homography_second_view_to_first_view=H.tolist(),
                 homography_second_matching_to_first_matching=Hfit.tolist(),
                 inlier_coordinates_second=src[good].tolist(), inlier_coordinates_first=dst[good].tolist(),
                 overlap_fraction_of_smaller_warp=fraction, feather_px=feather, geometric_checks_passed=True)
    if direction == "horizontal":
        stats["homography_right_view_to_left_view"] = H.tolist()
    warnings = ["Passing geometry checks does not prove correct physical correspondence.",
                "Repeated corrugations and branding can yield convincing false fits.",
                "Review feature_matches.jpg and hard_seam.png; no camera calibration or identity verification was performed."]
    if int(good.sum()) < 20:
        warnings.append("Limited feature support: fewer than 20 inliers. Sample-specific lower minimum was explicitly configured.")
    seam_pts = _seam_polyline(t, valid, direction)
    diagnostics = StitchDiagnostics(
        method="overlap", status="overlap_estimated_requires_visual_review",
        quality_state="overlap_requires_visual_review", direction=direction,
        source_size_wh=[[a.image.shape[1], a.image.shape[0]], [b.image.shape[1], b.image.shape[0]]],
        detected_corners=[(a.quad + np.float32(a.box[:2])).tolist(),
                          (b.quad + np.float32(b.box[:2])).tolist()],
        matches_total=len(matches), inliers=int(good.sum()), inlier_ratio=float(good.mean()),
        match_points_first=dst.tolist(), match_points_second=src.tolist(),
        inlier_mask=good.tolist(), homography_second_to_first=H.tolist(),
        projected_bounds=[int(low[0]), int(low[1]), int(high[0]), int(high[1])],
        projected_quad_first=(qa - low).tolist(), projected_quad_second=(qb - low).tolist(),
        overlap_polygon=_largest_overlap_polygon(va, vb), overlap_ratio=fraction,
        seam_points=seam_pts if seam_report.get("policy") == "union" else
        ([[seam_report["position_px"], 0], [seam_report["position_px"], height - 1]]
         if direction == "horizontal" else
         [[0, seam_report["position_px"]], [width - 1, seam_report["position_px"]]]),
        feather_px=feather, median_reprojection_error_px=float(np.median(error)),
        sanity={"min_inliers": bool(int(good.sum()) >= minimum),
                "inlier_ratio": bool(good.mean() >= .35),
                "scale_bounds": bool(.25 <= scale <= 4),
                "cross_edge_rotation": bool(abs(angle) <= 20),
                "extends_along_axis": True,
                "reprojection_residuals": True})
    report = {"method": "overlap", "status": "overlap_estimated_requires_visual_review",
              "overlap_alignment_applied": True, "independently_verified_overlap": False,
              "same_surface_confirmed_by_operator": True, "matching": stats, "seam": seam_report,
              "exposure": {"enabled": False}, "regions": [a.report(WA), b.report(WB)],
              "warnings": warnings, "diagnostics": diagnostics.to_payload()}
    image = bgra(pixels, valid)
    return Tile(image, image.copy(), provenance, report, diagnostics)


def process_group(group: Group, cross_size: int, out: Path, no_balance: bool,
                  direction: str = "horizontal") -> Tile:
    method = group.config["method"]
    out.mkdir(parents=True, exist_ok=True)
    if method == "overlap":
        return overlap_group(group, cross_size, out, direction)
    warped, valids, transforms = [], [], []
    for index, region in enumerate(group.regions, 1):
        H, size = rectification_axis(region.quad, cross_size, direction, region.config.get("rectified_size_wh"))
        pixels, valid = warp(region.image, region.mask, H, size)
        warped.append(pixels); valids.append(valid); transforms.append(H)
        save_image(out / f"section_{index}_geometry.png", bgra(pixels, valid))
    # Optional feature-measured seam refinement (edge only): measure the true
    # offset between the two warped strips. When the strips demonstrably share
    # coverage, first try promoting the pair to the verified overlap method
    # (same fixed proof standards, more sensitive feature detection); only
    # fall back to a clamped translation correction on the edge join itself.
    alignment: dict[str, Any] = {"enabled": False}
    if len(warped) == 2 and group.config.get("edge_alignment") == "measure":
        alignment = measure_strip_alignment(warped[0], valids[0], warped[1], valids[1], direction)
        if alignment.get("promotion_candidate"):
            matching_cfg = dict(group.config.get("matching") or {})
            matching_cfg.setdefault("contrast_threshold", 0.01)
            promo = Group({**group.config, "method": "overlap", "matching": matching_cfg,
                           "edge_alignment": "butt"}, group.regions)
            try:
                tile = overlap_group(promo, cross_size, out, direction)
                tile.report["promoted_from_edge"] = {
                    "reason": "measured overlap between the declared edge regions; "
                              "promoted to the verified overlap method",
                    "seam_alignment": alignment}
                if tile.diagnostics is not None:
                    tile.diagnostics.seam_alignment = {**alignment, "promotion_used": True}
                return tile
            except ProcessingError as exc:
                alignment["promotion_attempted"] = True
                alignment["promotion_rejected"] = str(exc)[:200]
        if alignment.get("applied"):
            warped[1], valids[1] = _realigned_strip(warped[1], valids[1], alignment, direction)
            # The shifted strip grew by |shift| along the cross axis; pad the
            # other strip on the complementary side so both contents stay
            # co-aligned and nothing is cropped away.
            shift = int(round(float(alignment["shift_px"])))
            if shift:
                before, after = max(-shift, 0), max(shift, 0)
                widths3 = ((before, after), (0, 0), (0, 0)) if direction == "horizontal" \
                    else ((0, 0), (before, after), (0, 0))
                widths1 = ((before, after), (0, 0)) if direction == "horizontal" \
                    else ((0, 0), (before, after))
                warped[0] = np.pad(warped[0], widths3)
                valids[0] = np.pad(valids[0], widths1, constant_values=False)
                alignment["cross_pad_px"] = [before, after]
    array_axis = 0 if direction == "vertical" else 1
    output_length = sum(p.shape[array_axis] for p in warped)
    check_size(cross_size, output_length) if direction == "vertical" else check_size(output_length, cross_size)
    geometry = np.concatenate([bgra(p, v) for p, v in zip(warped, valids)], axis=array_axis)
    if method == "rectify":
        image = geometry.copy()
        provenance = np.where(valids[0], group.regions[0].bit, 0).astype(np.uint16)
        region0 = group.regions[0]
        diagnostics = StitchDiagnostics(
            method="rectify", status="rectified_visible_panel", quality_state="rectified_only",
            direction=direction,
            source_size_wh=[[region0.image.shape[1], region0.image.shape[0]]],
            detected_corners=[(region0.quad + np.float32(region0.box[:2])).tolist()],
            sanity={"no_stitch_needed": True})
        report = {"method": "rectify", "status": "rectified_visible_panel",
                  "overlap_alignment_applied": False, "independently_verified_overlap": False,
                  "exposure": {"enabled": False}, "regions": [region0.report(transforms[0])],
                  "warnings": ["Manually selected visible panel; no new surface coverage was reconstructed."],
                  "diagnostics": diagnostics.to_payload()}
    else:
        a, b = group.regions
        cfg = group.config.get("exposure", {})
        if direction == "vertical":
            args = [np.swapaxes(p, 0, 1) for p in (*warped, *valids)]
            aa, bb, exposure = balance_edges(*args, cfg, enabled=cfg.get("enabled", False) and not no_balance)
            aa, bb = np.swapaxes(aa, 0, 1), np.swapaxes(bb, 0, 1)
            exposure["coordinate_note"] = "Legacy left/right gain names refer to first/top and second/bottom; sample_y_fraction is across the roof width."
        else:
            aa, bb, exposure = balance_edges(*warped, *valids, cfg,
                                             enabled=cfg.get("enabled", False) and not no_balance)
        image = np.concatenate([bgra(aa, valids[0]), bgra(bb, valids[1])], axis=array_axis)
        provenance = np.concatenate([np.where(v, r.bit, 0).astype(np.uint16)
                                     for v, r in zip(valids, group.regions)], axis=array_axis)
        seam = warped[0].shape[array_axis]
        offset = np.float64([[1, 0, seam if direction == "horizontal" else 0],
                             [0, 1, seam if direction == "vertical" else 0], [0, 0, 1]])
        # Transparency: adjacent views of ONE surface sit next to each other
        # in the source frame. Widely separated quads mean the join claim
        # rests entirely on the configuration (and may span two containers).
        qa_abs = a.quad + np.float32(a.box[:2])
        qb_abs = b.quad + np.float32(b.box[:2])
        if direction == "horizontal":
            quad_gap = float(qb_abs[:, 0].min() - qa_abs[:, 0].max())
            view_span = max(a.image.shape[1], b.image.shape[1])
        else:
            quad_gap = float(qb_abs[:, 1].min() - qa_abs[:, 1].max())
            view_span = max(a.image.shape[0], b.image.shape[0])
        separation_warning = None
        if quad_gap > max(12.0, 0.04 * view_span):
            separation_warning = (
                f"Configured regions are separated by {quad_gap:.0f} px of source frame; "
                "adjacent views of one surface are expected to nearly touch. The same-surface "
                "claim rests entirely on the configuration — verify this is one container, not two.")
        diagnostic = edge_diagnostic(a, b, out)
        out_w, out_h = image.shape[1], image.shape[0]
        seam_pts = ([[seam, 0], [seam, out_h - 1]] if direction == "horizontal"
                    else [[0, seam], [out_w - 1, seam]])
        fade = int(exposure.get("fade_px", 0)) if isinstance(exposure, dict) else 0
        strong_alignment = bool(alignment.get("applied") and alignment.get("strong"))
        status = "edge_join_feature_aligned" if strong_alignment else "manual_edge_join_unverified"
        quality = "overlap_requires_visual_review" if strong_alignment else "unverified_edge_composite"
        edge_diag = StitchDiagnostics(
            method="edge", status=status,
            quality_state=quality, direction=direction,
            source_size_wh=[[a.image.shape[1], a.image.shape[0]], [b.image.shape[1], b.image.shape[0]]],
            detected_corners=[(a.quad + np.float32(a.box[:2])).tolist(),
                              (b.quad + np.float32(b.box[:2])).tolist()],
            matches_total=(alignment.get("matches") if alignment.get("enabled")
                           else diagnostic.get("best_mutual_unique_candidates")),
            inliers=(alignment.get("inliers") if alignment.get("enabled") else None),
            inlier_ratio=(alignment.get("inlier_ratio") if alignment.get("enabled") else None),
            median_reprojection_error_px=(alignment.get("median_reprojection_error_px")
                                          if alignment.get("enabled") else None),
            seam_points=seam_pts, feather_px=fade or None,
            seam_alignment=alignment if alignment.get("enabled") else None,
            sanity={"overlap_verified": bool(strong_alignment),
                    "quad_separation_px": round(quad_gap, 1),
                    "note": ("seam translation was feature-measured and applied within clamps"
                             if strong_alignment else
                             "edge adjacency is configured, not feature-proven")})
        warnings = ["No overlap was established or removed; edge adjacency comes from the configuration.",
                    "Physical proportions, surface continuity and corrugation count at the seam are unverified.",
                    "Exposure balancing assumes comparable paint and can alter appearance near the seam."]
        if strong_alignment:
            warnings = ["The seam position and cross-axis offset were measured from pixels (SIFT + RANSAC), "
                        "re-scored under the exact translation-only correction that was applied, and stayed "
                        "within tight clamps; declared corners were refined, not replaced.",
                        "Repeated corrugations can alias the measurement by one period; review the seam visually."]
        elif alignment.get("applied"):
            warnings = ["The seam was refined by a translation measured from pixels with limited feature "
                        "support; no overlap verification passed, so treat the output as unreviewed.",
                        "Repeated corrugations can alias the measurement by one period; review the seam visually."]
        pad_px = alignment.get("cross_pad_px") or [0, 0]
        if alignment.get("applied") and any(pad_px):
            warnings.append(
                f"The cross-axis correction expanded the output by {sum(pad_px)} px instead of cropping; "
                "padded bands are transparent and no source content was discarded.")
        if separation_warning:
            warnings.append(separation_warning)
        report = {"method": "edge", "status": status,
                  "overlap_alignment_applied": bool(alignment.get("applied")),
                  "independently_verified_overlap": False,
                  "same_surface_confirmed_by_operator": True,
                  "join_y" if direction == "vertical" else "join_x": seam,
                  "seam_alignment": alignment if alignment.get("enabled") else None,
                  "exposure": exposure, "overlap_diagnostic": diagnostic,
                  "regions": [a.report(transforms[0]), b.report(offset @ transforms[1])],
                  "warnings": warnings,
                  "diagnostics": edge_diag.to_payload()}
    check_size(image.shape[1], image.shape[0])
    diag_obj = edge_diag if method == "edge" else (diagnostics if method == "rectify" else None)
    return Tile(image, geometry, provenance, report, diag_obj)


def summarize_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Descriptive scorecard over a run: aggregates engine-made decisions.

    Adds no new quality logic — verdict mirrors the report's own status /
    quality_state; container numbers come from StitchDiagnostics / matching
    stats unchanged.
    """
    containers = []
    for c in report.get("containers") or []:
        d = c.get("diagnostics") or {}
        m = c.get("matching") or {}
        containers.append({
            "key": c.get("key"), "method": c.get("method"),
            "quality_state": d.get("quality_state") or c.get("status", "unknown"),
            "matches": d.get("matches_total", m.get("mutual_unique_candidates")),
            "inliers": d.get("inliers", m.get("inliers")),
            "inlier_ratio": d.get("inlier_ratio", m.get("inlier_ratio")),
            "median_reprojection_error_px": d.get("median_reprojection_error_px",
                                                  m.get("median_reprojection_error_px")),
            "overlap_ratio": d.get("overlap_ratio", m.get("overlap_fraction_of_smaller_warp")),
            "sanity_checks": d.get("sanity_checks") or {},
            "warnings": c.get("warnings") or [],
        })
    if report.get("status") == "rejected":
        verdict = "rejected"
    else:
        verdict = report.get("quality_state", "requires_review")
    return {
        "schema_version": 1, "job_verdict": verdict,
        "quality_state": report.get("quality_state") or report.get("status"),
        "rejection_reason": report.get("reason"),
        "container_count": len(containers),
        "containers": containers,
        "note": "Descriptive aggregation of engine decisions; adds no new quality logic.",
    }


def run_job(config_path: Path | str, out: Path | str, *, mode: str | None = None,
            input_path: Path | str | None = None, source_overrides: dict[str, Path] | None = None,
            no_balance: bool = False, direction: str | None = None) -> dict[str, Any]:
    """Process a trusted local JSON job; return the committed report or raise ProcessingError.

    The output path MUST NOT exist. A failed job leaves report.json plus
    diagnostics-only artifacts (debug_overlay.png / diagnostics.json showing the
    exact rejection reason); no final/partial composites are published. Use one
    process per job for parallel workers (OpenCV thread/RNG configuration is global).
    """
    config_path, out = Path(config_path).resolve(), Path(out).resolve()
    if out.exists():
        raise ProcessingError("Output path already exists. Use a new directory to avoid stale results.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()  # Reserve exclusively, so we never overwrite someone else's images or reports.
    stage = Path(tempfile.mkdtemp(prefix=".work-", dir=out))
    published: list[Path] = []
    current = None
    groups: list[Group] = []
    config: dict[str, Any] = {}
    images: dict[str, np.ndarray] = {}
    try:
        boolean(no_balance, "no_balance")
        cv2.setNumThreads(1)
        cv2.setRNGSeed(7)
        config, images, source_records, groups, warnings = load_job(
            config_path, mode, Path(input_path) if input_path is not None else None, source_overrides, direction)
        tiles = []
        cross_size, gap = config["cross_size_px"], config["gap_px"]
        for group in groups:
            current = group.config["key"]
            tile = process_group(group, cross_size, stage / "containers" / current, no_balance, config["direction"])
            tile.report.update(key=current, container_id=group.config.get("container_id"),
                               label=group.config.get("label", current), coverage=group.config.get("coverage", "unverified"),
                               notes=group.config.get("notes", ""), direction=config["direction"],
                               output_size_wh=[tile.image.shape[1], tile.image.shape[0]])
            sub = stage / "containers" / current
            save_image(sub / "result.png", tile.image)
            save_image(sub / "geometry_only.png", tile.geometry)
            save_image(sub / "source_map_16bit.png", tile.sources)
            tiles.append(tile)
        current = None
        horizontal_layout = config["layout"] == "horizontal"
        width = (sum(t.image.shape[1] for t in tiles) + gap * (len(tiles) - 1)
                 if horizontal_layout else max(t.image.shape[1] for t in tiles))
        height = (max(t.image.shape[0] for t in tiles) if horizontal_layout else
                  sum(t.image.shape[0] for t in tiles) + gap * (len(tiles) - 1))
        check_size(width, height)
        canvas, geometry = np.zeros((height, width, 4), np.uint8), np.zeros((height, width, 4), np.uint8)
        source_map, container_map = np.zeros((height, width), np.uint16), np.zeros((height, width), np.uint8)
        x = y = 0
        provenance_keys: dict[str, Any] = {}
        for index, tile in enumerate(tiles, 1):
            h, w = tile.image.shape[:2]
            canvas[y:y+h, x:x+w], geometry[y:y+h, x:x+w] = tile.image, tile.geometry
            source_map[y:y+h, x:x+w] = tile.sources
            container_map[y:y+h, x:x+w] = np.where(tile.image[..., 3] > 0, index, 0)
            L = np.float64([[1, 0, x], [0, 1, y], [0, 0, 1]])
            tile.report["output_box_xyxy_exclusive"] = [x, y, x + w, y + h]
            for r in tile.report["regions"]:
                H = np.asarray(r["input_to_container_homography"], np.float64)
                r["input_to_final_homography"] = (L @ H).tolist()
                provenance_keys[str(r["source_bit"])] = {
                    "container_key": tile.report["key"], "source": r["source"],
                    "view_box_xyxy_exclusive": r["view_box_xyxy_exclusive"],
                }
            write_json(stage / "containers" / tile.report["key"] / "report.json", tile.report)
            group = next(g for g in groups if g.config["key"] == tile.report["key"])
            if tile.diagnostics is not None:
                render_debug_overlay(
                    [(images[r.source], r.box) for r in group.regions],
                    tile.diagnostics, tile.image,
                    stage / "containers" / tile.report["key"] / "debug_overlay.png")
                write_json(stage / "containers" / tile.report["key"] / "diagnostics.json",
                           tile.diagnostics.to_payload())
            if horizontal_layout:
                x += w + gap
            else:
                y += h + gap
        first_diag = next((t for t in tiles if t.diagnostics is not None), None)
        if first_diag is not None:
            shutil.copyfile(stage / "containers" / first_diag.report["key"] / "debug_overlay.png",
                            stage / "debug_overlay.png")
            shutil.copyfile(stage / "containers" / first_diag.report["key"] / "diagnostics.json",
                            stage / "diagnostics.json")
        save_image(stage / "result.png", canvas)
        save_image(stage / "geometry_only.png", geometry)
        save_image(stage / "source_map_16bit.png", source_map)
        save_image(stage / "container_map.png", container_map)
        for name, im in images.items():
            annotated = im.copy()
            for group in groups:
                for i, r in enumerate(group.regions, 1):
                    if r.source != name:
                        continue
                    q = r.quad + np.float32(r.box[:2])
                    cv2.polylines(annotated, [np.rint(q).astype(np.int32)], True, (0, 210, 255), 2)
                    for j, pt in enumerate(q):
                        xx, yy = np.rint(pt).astype(int)
                        cv2.circle(annotated, (int(xx), int(yy)), 4, (0, 210, 255), -1)
                        cv2.putText(annotated, str(j), (int(xx) + 4, int(yy) + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 210, 255), 1, cv2.LINE_AA)
                    xx, yy = np.rint(q[0]).astype(int)
                    cv2.putText(annotated, f"{group.config['key']}/{i}", (int(xx) + 6, int(yy) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 210, 255), 1, cv2.LINE_AA)
            save_image(stage / "selected_regions" / f"{name}.jpg", annotated)
        write_json(stage / "resolved_config.json", config)
        method_statuses = [t.report.get("status", "unknown") for t in tiles]
        if any(v == "edge_join_feature_aligned" for v in method_statuses):
            quality_state = "overlap_requires_visual_review"
            quality_label = "Seam realigned by feature match; visual review required"
        elif any(v == "manual_edge_join_unverified" for v in method_statuses):
            quality_state = "unverified_edge_composite"
            quality_label = "Created, but overlap was not verified"
        elif any(v == "overlap_estimated_requires_visual_review" for v in method_statuses):
            quality_state = "overlap_requires_visual_review"
            quality_label = "Overlap estimated; visual review required"
        elif all(v == "rectified_visible_panel" for v in method_statuses):
            quality_state = "rectified_only"
            quality_label = "Rectified visible panel; no stitch was needed"
        else:
            quality_state = "requires_review"
            quality_label = "Created; review required"
        report = {
            "schema_version": 2, "program_version": VERSION, "status": "created_requires_review",
            "processing_state": "created", "quality_state": quality_state, "quality_label": quality_label,
            "mode": config["mode"], "direction": config["direction"], "layout": config["layout"],
            "cross_size_px": cross_size, "physical_container_count": len(groups),
            "mode_and_identity_source": "explicit configuration/operator input; not automatic detection or OCR",
            "source_records": source_records, "config_sha256": digest(config_path),
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "output_size_wh": [width, height], "result_file": "result.png",
            "gap_px": gap if len(groups) > 1 else 0,
            "gap_meaning": "Transparent layout separator, NOT measured physical space.",
            "same_height_rescaling_after_stitch": False, "rescaling_after_stitch": False, "cross_container_blending": False,
            "independently_verified_overlap": False,
            "source_map_encoding": {"format": "uint16 bitmask", "zero": "no source; transparent",
                                    "bits": provenance_keys,
                                    "combination_rule": "bitwise OR when pixels are weighted from both regions",
                                    "weight_file": "For overlap groups, containers/<key>/second_weight_16bit.png stores second-region weight / 65535, only at valid output pixels."},
            "container_map_values": {"0": "no container/transparent", **{
                str(i): t.report["key"] for i, t in enumerate(tiles, 1)}},
            "containers": [t.report for t in tiles],
            "warnings": warnings + [
                "Manual corners are frame-specific; matching image dimensions alone are insufficient.",
                "Warping resamples pixels. Keep original captures for inspection and audit.",
                "Not calibrated for dimensions, gap size, damage measurement or corrugation count.",
                "No automatic single/combo detection, container identity verification, or correspondence proof.",
                "Direction and combo layout are explicit settings; partial coverage is not reconstructed.",
            ],
            "not_performed": ["generative fill", "inpainting", "OCR", "lettering replacement",
                              "learned super-resolution", "automatic container detection", "cross-container matching"],
            "no_balance_override": bool(no_balance),
        }
        write_json(stage / "metrics.json", summarize_metrics(report))
        # Publish only after EVERY container succeeds. report.json commits last.
        for path in list(stage.iterdir()):
            target = out / path.name
            path.rename(target)
            published.append(target)
        stage.rmdir()
        write_json(out / "report.json", report)
        return report
    except (ProcessingError, OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
        for path in published:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
        details = exc.details if isinstance(exc, ProcessingError) else {}
        rejection = {"status": "rejected", "program_version": VERSION, "reason": str(exc),
                     "container_key": current, "details": details, "result_created": False,
                     "fallback_to_edge": False}
        diag_files: list[str] = []
        # Rejection overlay: same geometry the engine saw, with the exact reason.
        group_in_flight = next((g for g in groups if g.config["key"] == current), None) \
            if current and groups else None
        if group_in_flight is not None:
            try:
                stats = details if isinstance(details, dict) else {}
                regions = group_in_flight.regions
                rejected_diag = StitchDiagnostics(
                    method=group_in_flight.config.get("method", "unknown"),
                    status="rejected", quality_state="rejected",
                    direction=config["direction"] if isinstance(config, dict) else "horizontal",
                    source_size_wh=[[r.image.shape[1], r.image.shape[0]] for r in regions],
                    detected_corners=[(r.quad + np.float32(r.box[:2])).tolist() for r in regions],
                    rejection_reason=str(exc),
                    matches_total=stats.get("mutual_unique_candidates"),
                    inliers=stats.get("inliers"), inlier_ratio=stats.get("inlier_ratio"),
                    match_points_first=stats.get("match_points_first"),
                    match_points_second=stats.get("match_points_second"),
                    inlier_mask=stats.get("inlier_mask"),
                    homography_second_to_first=(stats.get("homography_second_view_to_first_view")
                                                or stats.get("homography_second_matching_to_first_matching")),
                    sanity={"geometric_checks_passed": False,
                            "failing_stage": "validation"})
                diag_dir = out / "containers" / current
                render_debug_overlay([(images[r.source], r.box) for r in regions],
                                     rejected_diag, None, diag_dir / "debug_overlay.png")
                write_json(diag_dir / "diagnostics.json", rejected_diag.to_payload())
                rejection["diagnostics_files"] = [
                    f"containers/{current}/debug_overlay.png", f"containers/{current}/diagnostics.json"]
                shutil.copyfile(diag_dir / "debug_overlay.png", out / "debug_overlay.png")
                shutil.copyfile(diag_dir / "diagnostics.json", out / "diagnostics.json")
            except Exception:
                pass  # diagnostics must never mask the original rejection
        try:
            write_json(out / "metrics.json", summarize_metrics(rejection))
        except OSError:
            pass
        try:
            write_json(out / "report.json", rejection)
        except OSError:
            pass
        raise ProcessingError(str(exc), rejection) from exc


def run_batch(batch_path: Path | str, out: Path | str, *, no_balance: bool = False) -> dict[str, Any]:
    """Run explicit local recipes independently; publish a batch summary last.

    A failed job never becomes an edge join. Other jobs continue. This is not an
    all-or-nothing transaction across jobs: require batch_report.json to know that
    the batch completed, and each job's report.json to know that job's status.
    """
    boolean(no_balance, "no_balance")
    batch_path, out = Path(batch_path).resolve(), Path(out).resolve()
    manifest = json.loads(batch_path.read_text(encoding="utf-8"))
    object_keys(manifest, {"schema_version", "jobs", "notes"}, "batch")
    integer(manifest.get("schema_version"), "batch.schema_version", 1, 1)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 64:
        raise ProcessingError("batch.jobs requires 1..64 named recipes.")
    seen, checked = set(), []
    for item in jobs:
        object_keys(item, {"key", "config"}, "batch job")
        key = safe_key(item.get("key"), "batch job key")
        if key in seen:
            raise ProcessingError("Batch job keys must be unique.")
        seen.add(key)
        if not isinstance(item.get("config"), str) or not item["config"]:
            raise ProcessingError("Each batch job requires a nonempty config path.")
        path = (batch_path.parent / item["config"]).resolve()
        if not path.is_file():
            raise ProcessingError(f"Batch recipe not found: {path}")
        checked.append((key, path))
    if out.exists():
        raise ProcessingError("Output path already exists. Use a new directory to avoid stale results.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    rows = []
    for key, path in checked:
        try:
            report = run_job(path, out / key, no_balance=no_balance)
            rows.append({"key": key, "status": report["status"], "mode": report["mode"],
                         "direction": report["direction"], "container_count": report["physical_container_count"],
                         "size_wh": report["output_size_wh"], "result_file": f"{key}/result.png",
                         "report_file": f"{key}/report.json"})
        except (ProcessingError, OSError) as exc:
            rows.append({"key": key, "status": "rejected", "reason": str(exc),
                         "result_created": False, "report_file": f"{key}/report.json"})
    rejected = sum(r["status"] == "rejected" for r in rows)
    summary = {"program_version": VERSION, "status": "batch_complete_requires_review" if rejected == 0 else "batch_has_rejections",
               "jobs_total": len(rows), "jobs_created": len(rows)-rejected, "jobs_rejected": rejected,
               "manifest_sha256": digest(batch_path), "no_balance_override": no_balance,
               "jobs": rows, "note": "Each job is independent. Completion/fit does not prove physical correctness."}
    write_json(out / "batch_report.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"container_stitch {VERSION}")
    task = p.add_mutually_exclusive_group(required=True)
    task.add_argument("--config", type=Path, help="Unified JSON recipe (schema 1 or 2), or an earlier horizontal config")
    task.add_argument("--batch", type=Path, help="Batch manifest listing named recipes; writes a batch_report.json")
    p.add_argument("--out", type=Path, required=True, help="NEW output directory (must not exist)")
    p.add_argument("--mode", choices=["single", "combo"], help="Assert the recipe's mode; mismatch is an error")
    p.add_argument("--direction", choices=["horizontal", "vertical"], help="Assert recipe direction; does not rotate or reinterpret selections")
    p.add_argument("--input", type=Path, help="Override a sole source; required with an earlier config")
    p.add_argument("--source", action="append", default=[], metavar="NAME=PATH", help="Override a named source; repeatable")
    p.add_argument("--no-balance", action="store_true", help="Disable configured exposure correction; geometry still resamples")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.batch is not None:
            if args.mode is not None or args.direction is not None or args.input is not None or args.source:
                raise ProcessingError("Batch mode uses each recipe's grouping/direction/sources; do not pass per-job overrides.")
            report = run_batch(args.batch, args.out, no_balance=args.no_balance)
            print(json.dumps(report, indent=2))
            return 0 if report["jobs_rejected"] == 0 else 2
        overrides = {}
        for spec in args.source:
            if "=" not in spec:
                raise ProcessingError("--source must use NAME=PATH.")
            name, path = spec.split("=", 1)
            if not name or not path or name in overrides:
                raise ProcessingError("--source names must be nonempty and unique, and paths nonempty.")
            overrides[name] = Path(path)
        report = run_job(args.config, args.out, mode=args.mode, input_path=args.input,
                         source_overrides=overrides, no_balance=args.no_balance, direction=args.direction)
    except (ProcessingError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc), "output": str(args.out)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "mode": report["mode"],
                      "containers": report["physical_container_count"], "direction": report["direction"],
                      "result": str(args.out / "result.png"),
                      "report": str(args.out / "report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
