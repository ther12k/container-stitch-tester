#!/usr/bin/env python3
"""Fixed-camera profiles for the container stitch engine.

A profile is a reviewed calibration for one fixed camera position: the frame
size it produces, optionally a pinned reference frame, and the container
selections (absolute-image-space quads) an operator reviewed once. Converting a
new capture with ``profile_to_config`` emits a normal engine config — the same
deterministic engine then validates and stitches it. Nothing here inspects
pixels or reinterprets corners: frame dimensions must match the calibration
exactly, and a pinned reference hash must match byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from container_stitch import ProcessingError, check_quad

PROFILE_KEYS = {"schema_version", "profile_key", "camera", "pin", "job_defaults", "containers", "notes"}
CONTAINER_KEYS = {"key", "container_id", "label", "method", "regions", "exposure", "matching", "seam", "coverage", "notes"}
REGION_KEYS = {"source", "view_box", "quad", "feature_quad", "notes"}


def _safe_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) is None:
        raise ProcessingError(f"{name} must be a safe 1..64 character name using letters, digits, _ or -.")
    return value


def validate_profile(profile: dict[str, Any]) -> None:
    if not isinstance(profile, dict):
        raise ProcessingError("Profile must be a JSON object.")
    unknown = set(profile) - PROFILE_KEYS
    if unknown:
        raise ProcessingError(f"Unknown profile field(s): {', '.join(sorted(unknown))}.")
    if profile.get("schema_version") != 1:
        raise ProcessingError("Profile schema_version must be 1.")
    _safe_name(profile.get("profile_key"), "profile_key")

    camera = profile.get("camera")
    if not isinstance(camera, dict) or set(camera) - {"label", "expected_size_wh"}:
        raise ProcessingError("profile.camera must contain only label and expected_size_wh.")
    size = camera.get("expected_size_wh")
    if (not isinstance(size, list) or len(size) != 2
            or not all(isinstance(v, int) and v >= 16 for v in size)):
        raise ProcessingError("camera.expected_size_wh must be [width, height] integers (>= 16).")

    pin = profile.get("pin", {})
    if not isinstance(pin, dict) or set(pin) - {"reference_sha256", "enforce"}:
        raise ProcessingError("profile.pin must contain only reference_sha256 and enforce.")
    if "reference_sha256" in pin and not re.fullmatch(r"[a-fA-F0-9]{64}", str(pin["reference_sha256"])):
        raise ProcessingError("pin.reference_sha256 must be a 64-character hexadecimal string.")
    if "enforce" in pin and not isinstance(pin["enforce"], bool):
        raise ProcessingError("pin.enforce must be true or false.")

    defaults = profile.get("job_defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - {"mode", "direction", "layout", "cross_size_px", "gap_px"}:
        raise ProcessingError("job_defaults may contain only mode, direction, layout, cross_size_px, gap_px.")

    containers = profile.get("containers")
    if not isinstance(containers, list) or not 1 <= len(containers) <= 8:
        raise ProcessingError("profile.containers must be a list of 1..8 groups.")
    for c in containers:
        if not isinstance(c, dict) or set(c) - CONTAINER_KEYS:
            raise ProcessingError("Profile container has unknown or missing fields.")
        _safe_name(c.get("key"), "profile container key")
        if c.get("method") not in ("rectify", "edge", "overlap"):
            raise ProcessingError(f"Container {c['key']}: method must be rectify, edge or overlap.")
        regions = c.get("regions")
        want = 1 if c["method"] == "rectify" else 2
        if not isinstance(regions, list) or len(regions) != want:
            raise ProcessingError(f"Container {c['key']}: {c['method']} needs exactly {want} region(s).")
        for r in regions:
            if not isinstance(r, dict) or set(r) - REGION_KEYS:
                raise ProcessingError("Profile region may contain only source, view_box, quad, feature_quad, notes.")
            if "quad" not in r:
                raise ProcessingError("Profile region requires an absolute-space quad.")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_to_config(profile: dict[str, Any], image_path: Path | str,
                      source_name: str = "main", source_path: str = "input.png") -> dict[str, Any]:
    """Build an engine config for a new capture from a calibrated profile.

    Quads in the profile are absolute image-space; each is converted to a
    view_box + view-relative quad pair. Frame dimensions must equal the
    calibration exactly; an enforced pin must match the capture byte-for-byte.
    """
    import cv2

    validate_profile(profile)
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ProcessingError(f"Could not decode the capture: {image_path.name}")
    height, width = image.shape[:2]
    expected = profile["camera"]["expected_size_wh"]
    if [width, height] != expected:
        raise ProcessingError(
            f"Capture size {width}x{height} does not match the calibrated camera "
            f"{expected[0]}x{expected[1]}. Profile quads are frame-specific; re-calibrate.")

    pin = profile.get("pin", {})
    if pin.get("enforce") and "reference_sha256" in pin:
        actual = _digest(image_path)
        if actual != str(pin["reference_sha256"]).lower():
            raise ProcessingError("Capture SHA-256 does not match the profile's pinned reference frame.")

    def absolute_region(region: dict[str, Any]) -> dict[str, Any]:
        quad = check_quad(region["quad"], width, height)
        if "view_box" in region:
            box = region["view_box"]
        else:
            xs, ys = quad[:, 0], quad[:, 1]
            box = [max(0, int(np.floor(xs.min()))), max(0, int(np.floor(ys.min()))),
                   min(width, int(np.ceil(xs.max())) + 1), min(height, int(np.ceil(ys.max())) + 1)]
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ProcessingError(f"Region view_box {box} is outside the calibrated frame.")
        out: dict[str, Any] = {"source": source_name, "view_box": box,
                               "quad": (quad - np.float32([x0, y0])).tolist()}
        if "feature_quad" in region:
            fq = check_quad(region["feature_quad"], width, height)
            out["feature_quad"] = (fq - np.float32([x0, y0])).tolist()
        if "notes" in region:
            out["notes"] = region["notes"]
        return out

    containers_out = []
    for c in profile["containers"]:
        entry: dict[str, Any] = {"key": c["key"], "method": c["method"],
                                 "regions": [absolute_region(r) for r in c["regions"]]}
        if c["method"] != "rectify":
            entry["same_surface_confirmed"] = True  # operator confirmed at calibration time
        for optional in ("container_id", "label", "exposure", "matching", "seam", "coverage", "notes"):
            if optional in c:
                entry[optional] = c[optional]
        containers_out.append(entry)

    defaults = profile.get("job_defaults", {})
    direction = defaults.get("direction", "horizontal")
    config: dict[str, Any] = {
        "schema_version": 2 if direction == "vertical" else 1,
        "mode": defaults.get("mode", "single" if len(containers_out) == 1 else "combo"),
        "direction": direction,
        "cross_size_px": defaults.get("cross_size_px", 320),
        "gap_px": defaults.get("gap_px", 12),
        "sources": {source_name: {"path": source_path, "expected_size_wh": [width, height]}},
        "containers": containers_out,
        "notes": (f"Generated from camera profile '{profile['profile_key']}'. "
                  "Corners were calibrated and reviewed for this fixed camera; "
                  "dimensions matched the calibration exactly."),
    }
    if direction == "vertical":
        config["layout"] = defaults.get("layout", "vertical")
    if defaults.get("layout") and direction != "vertical":
        config["layout"] = defaults["layout"]
    return config


def load_profile(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_profile(data)
    return data
