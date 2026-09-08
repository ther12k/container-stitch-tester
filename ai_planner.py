#!/usr/bin/env python3
"""AI vision-planner bridge for the container stitch web tester.

The AI only proposes. It inspects a photo and returns an intermediate
``ai_plan`` JSON (normalized coordinates, scene analysis, confidence). This
module parses/validates that plan and converts it into a normal
container_stitch engine config. The deterministic engine keeps final
authority: plans run through the exact same validation and rejection paths as
hand-written configs, and no AI output ever touches pixels.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_GLM_BASE = "https://open.bigmodel.cn/api/paas/v4"

PLANNER_SYSTEM = """You are the vision-planning module of a deterministic container-photo \
stitching pipeline. You never edit, generate, or repair pixels; a separate OpenCV program \
does all pixel work and may reject your plan.

Inspect the photo and return ONLY one JSON object, no prose, matching this schema:
{
  "analysis_version": 1,
  "scene": {"view_layout": "vertical_stack|horizontal_pair|single_view|other",
            "physical_containers": <int>},
  "target": {"container_index": 0, "surface": "roof|side|doors|other"},
  "containers": [
    {"key": "container_1",
     "method": "rectify|edge|overlap",
     "regions": [
       {"view_box_normalized": [x0, y0, x1, y1],
        "quad_normalized": [[x,y],[x,y],[x,y],[x,y]]}
     ]}
  ],
  "exclude": ["free text: what you deliberately left out and why"],
  "reason": "one short sentence",
  "confidence": {"direction": 0.0-1.0, "container_grouping": 0.0-1.0, "same_surface": 0.0-1.0}
}

Rules:
- All coordinates are normalized 0..1 relative to the FULL image; quad corner order is TL, TR, BR, BL.
- rectify: ONE view already shows the complete target surface. Preferred when a single view suffices.
- edge: two adjacent views of the same surface, adjacency known, overlap NOT verified.
- overlap: the SAME surface genuinely appears in both views with real shared coverage.
  Choose overlap ONLY when confidence.same_surface >= 0.85.
- One container group per physical container. Never group different containers together.
- view_box_normalized should tightly bound the camera view you are using; quad_normalized must
  sit inside its view box and trace the target panel's corners as precisely you can.
"""

RETRY_INSTRUCTION = """Your previous proposal was REJECTED by the deterministic validator. \
Do not repeat it. Re-evaluate container grouping, orientation, selected regions, and whether \
rectify or edge is more appropriate than overlap. Return a REVISED plan JSON only."""


def build_image_data_uri(path: Path, max_side: int = 900, quality: int = 82) -> str:
    """Downscale + JPEG-encode the staged photo for the vision request."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    scale = max(1.0, max(h, w) / max_side)
    if scale > 1.0:
        img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("cannot encode image for the vision request")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def build_messages(data_uri: str, attempt_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image_url",
                                      "image_url": {"url": data_uri}}]
    if attempt_log:
        content.append({"type": "text", "text": RETRY_INSTRUCTION})
        content.append({"type": "text",
                        "text": "Previous attempts and validator diagnostics:\n"
                                + json.dumps(attempt_log, indent=2, ensure_ascii=False)})
    else:
        content.append({"type": "text",
                        "text": "Analyze this container photo. Return the ai_plan JSON only."})
    return [{"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": content}]


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def sanitize_detail(text: str, limit: int = 160) -> str:
    """Strip HTML tags and collapse whitespace from a provider error body.

    Proxies answer outages with whole nginx HTML pages; those never belong in
    the UI or the console. The untruncated raw body stays available to callers
    via ``AiProviderUnavailable.raw_detail`` for server-side artifacts.
    """
    cleaned = _WS_RE.sub(" ", _TAG_RE.sub(" ", str(text or ""))).strip()
    return cleaned[:limit]


class AiProviderUnavailable(RuntimeError):
    """The endpoint could not be reached or is not usable (down, auth, wrong shape).

    Distinct from a bad model reply: availability failures let the run fall
    back to another provider without consuming a retry attempt, while a bad
    reply (e.g. the model returning prose instead of JSON) counts as a real
    failed attempt. ``raw_detail`` keeps the unsanitized provider response so
    the app can persist it server-side without showing it in the UI.
    """

    def __init__(self, message: str, raw_detail: str = ""):
        super().__init__(message)
        self.raw_detail = raw_detail


def chat_completion(base_url: str, api_key: str, model: str,
                    messages: list[dict[str, Any]], timeout: int = 180,
                    max_tokens: int = 2000) -> str:
    """Minimal OpenAI-compatible chat/completions call (GLM and Luna both speak it).

    Requests non-streaming but tolerates SSE replies: some proxies stream even
    when stream=false, and reasoning models spend tokens before the content.
    """
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.1,
                       "max_tokens": max_tokens, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", "replace")[:2000]
        raise AiProviderUnavailable(
            f"{model} HTTP {exc.code}: {sanitize_detail(raw_body) or 'no response body'}",
            raw_detail=raw_body) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = str(exc) or repr(exc) or exc.__class__.__name__
        raise AiProviderUnavailable(f"{model} unreachable: {sanitize_detail(reason, 120)}",
                                    raw_detail=reason) from exc

    if raw.lstrip().startswith("data:"):
        # Server streamed anyway — reassemble the deltas.
        parts: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            parts.append(delta.get("content") or "")
            msg = choice.get("message") or {}
            parts.append(msg.get("content") or "")
        content = "".join(p for p in parts if p).strip()
        if not content:
            raise AiProviderUnavailable(f"{model} streamed an empty reply")
        return content

    try:
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise AiProviderUnavailable(f"{model} returned an unexpected response shape") from exc


def extract_json(text: str) -> dict[str, Any]:
    """Parse the plan out of a model reply, tolerating code fences and prose."""
    if not text:
        raise ValueError("empty reply")
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in reply")
    plan = json.loads(candidate[start:end + 1])
    if not isinstance(plan, dict):
        raise ValueError("plan is not a JSON object")
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    containers = plan.get("containers")
    if not isinstance(containers, list) or not 1 <= len(containers) <= 8:
        raise ValueError("plan.containers must be a list of 1..8 groups")
    for c in containers:
        if not isinstance(c, dict):
            raise ValueError("plan container is not an object")
        if c.get("method") not in ("rectify", "edge", "overlap"):
            raise ValueError(f"unknown container method: {c.get('method')!r}")
        regions = c.get("regions")
        want = 1 if c["method"] == "rectify" else 2
        if not isinstance(regions, list) or len(regions) != want:
            raise ValueError(f"method {c['method']} needs exactly {want} region(s)")
        for r in regions:
            box = r.get("view_box_normalized")
            quad = r.get("quad_normalized")
            if (not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(v, (int, float)) for v in box)):
                raise ValueError("region needs view_box_normalized [x0,y0,x1,y1]")
            if (not isinstance(quad, list) or len(quad) != 4
                    or not all(isinstance(p, list) and len(p) == 2
                               and all(isinstance(v, (int, float)) for v in p) for p in quad)):
                raise ValueError("region needs quad_normalized [[x,y] x4] TL,TR,BR,BL")


def plan_to_config(plan: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Convert a validated plan (normalized coords) into an engine config.

    Coordinates are clamped into the image; quads are converted to view-relative
    corner order. The engine re-validates everything and may still reject.
    """
    def px(v: float, span: int) -> int:
        return max(0, min(int(round(float(v) * span)), span))

    containers_out: list[dict[str, Any]] = []
    for c in plan.get("containers", []):
        method = c["method"]
        regions_out = []
        for i, r in enumerate(c.get("regions", []), 1):
            bx0, by0, bx1, by1 = [float(v) for v in r["view_box_normalized"]]
            x0, x1 = sorted((px(bx0, width), px(bx1, width)))
            y0, y1 = sorted((px(by0, height), px(by1, height)))
            # keep view boxes workable
            x1 = max(x1, min(x0 + 16, width))
            y1 = max(y1, min(y0 + 16, height))
            vw, vh = x1 - x0, y1 - y0
            quad = []
            for qx, qy in r["quad_normalized"]:
                qx_abs = max(0, min(int(round(float(qx) * width)) - x0, vw - 1))
                qy_abs = max(0, min(int(round(float(qy) * height)) - y0, vh - 1))
                quad.append([qx_abs, qy_abs])
            regions_out.append({"source": "main", "view_box": [x0, y0, x1, y1], "quad": quad})

        container: dict[str, Any] = {"key": str(c.get("key") or f"container_{len(containers_out) + 1}")[:64],
                                     "method": method, "regions": regions_out}
        if method != "rectify":
            # The plan asserts the two regions show the same physical surface;
            # the engine still verifies geometry on its own terms.
            container["same_surface_confirmed"] = True
        containers_out.append(container)

    direction = (plan.get("suggested_processing") or {}).get("direction")
    if direction not in ("horizontal", "vertical"):
        direction = "vertical" if height > width else "horizontal"

    # The planner's quad corners are estimates; let the engine measure the
    # seam between edge-joined strips from pixels (SIFT + RANSAC) and apply a
    # clamped correction instead of butting possibly-mislocated quads.
    edge_methods = {c.get("method") for c in plan.get("containers", [])}
    edge_alignment = "measure" if "edge" in edge_methods else "butt"

    config = {
        "schema_version": 2,
        "mode": "combo" if len(containers_out) > 1 else "single",
        "direction": direction,
        "cross_size_px": 320,
        "gap_px": 12,
        "edge_alignment": edge_alignment,
        "sources": {"main": {"path": "input.png", "expected_size_wh": [width, height]}},
        "containers": containers_out,
        "notes": (f"AI-proposed plan ({plan.get('reason', 'no reason given')}). "
                  "AI proposed; the deterministic engine validated. No AI pixel editing."),
    }
    return config
