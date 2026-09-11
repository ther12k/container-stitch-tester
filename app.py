#!/usr/bin/env python3
from __future__ import annotations

import copy
import ipaddress
import json
import os
import queue as queue_mod
import shutil
import socket
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, Response, abort, render_template, request, send_file, url_for

from container_stitch import ProcessingError, VERSION, run_batch, run_job
import ai_planner
import camera_profiles

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "web_jobs"
UPLOADS_DIR = BASE_DIR / "web_uploads"
JOBS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Default config shown in the JSON editor on first load.
# References input.png so a pasted/dropped image (auto-named input.png)
# can be stitched without uploading a config. The browser fills in
# expected_size_wh from the actual image dimensions before submitting.
# ------------------------------------------------------------------
DEFAULT_CONFIG_TEMPLATE = """{
  "schema_version": 1,
  "mode": "single",
  "height": 320,
  "gap_px": 8,
  "edge_alignment": "measure",
  "sources": {
    "main": {
      "path": "input.png"
    }
  },
  "containers": [
    {
      "key": "container_1",
      "method": "edge",
      "same_surface_confirmed": true,
      "regions": [
        {
          "source": "main",
          "view_box": [0, 0, 1024, 512],
          "quad": [[0, 0], [1024, 0], [1024, 512], [0, 512]]
        },
        {
          "source": "main",
          "view_box": [1024, 0, 2048, 512],
          "quad": [[0, 0], [1024, 0], [1024, 512], [0, 512]]
        }
      ]
    }
  ]
}"""

_BASE_EXAMPLE_RECIPES = [
    {
        "key": "single_grey",
        "label": "Grey single container (horizontal edge join)",
        "short": "Grey single",
        "config": BASE_DIR / "configs" / "single_grey.json",
        "kind": "job",
    },
    {
        "key": "combo_1",
        "label": "Blue combo 1 (horizontal, two containers)",
        "short": "Blue combo",
        "config": BASE_DIR / "configs" / "combo_1.json",
        "kind": "job",
    },
    {
        "key": "combo_2",
        "label": "Blue combo 2 (horizontal, two containers)",
        "short": "Blue combo 2",
        "config": BASE_DIR / "configs" / "combo_2.json",
        "kind": "job",
    },
    {
        "key": "vertical_blue_single",
        "label": "Blue roof upper only (vertical overlap)",
        "short": "Blue single",
        "config": BASE_DIR / "configs" / "vertical_blue_single.json",
        "kind": "job",
    },
    {
        "key": "vertical_blue_combo",
        "label": "Blue roof combo (upper roof + lower partial)",
        "short": "Blue roof combo",
        "config": BASE_DIR / "configs" / "vertical_blue_combo.json",
        "kind": "job",
    },
    {
        "key": "vertical_red_single",
        "label": "Red roof single (vertical overlap)",
        "short": "Red single",
        "config": BASE_DIR / "configs" / "vertical_red_single.json",
        "kind": "job",
    },
    {
        "key": "vertical_lightblue_best_view",
        "label": "Light-blue stacked views (best complete vertical view)",
        "short": "Light blue",
        "config": BASE_DIR / "configs" / "vertical_lightblue_best_view.json",
        "kind": "job",
    },
    {
        "key": "all_examples",
        "label": "Run all packaged examples (batch)",
        "short": "All examples",
        "config": BASE_DIR / "configs" / "all_examples.json",
        "kind": "batch",
    },
]
def _recipe_available(item: dict) -> bool:
    """True when every source file the recipe needs is actually present.

    The public deployment ships without the private sample photographs; those
    recipes are hidden instead of failing at run time. Demo recipes generated
    by demo_seed.py (containers) are always available."""
    try:
        cfg = json.loads(item["config"].read_text(encoding="utf-8"))
    except Exception:
        return False
    if item.get("kind") == "batch":
        try:
            names = [j["config"] for j in cfg.get("jobs", [])]
        except Exception:
            return False
        paths = [item["config"].parent / n for n in names]
    else:
        paths = [item["config"]]
    for path in paths:
        try:
            job_cfg = json.loads(path.read_text(encoding="utf-8"))
            for info in (job_cfg.get("sources") or {}).values():
                if not (path.parent / info.get("path", "")).resolve().is_file():
                    return False
        except Exception:
            return False
    return True


_DEMO_LABELS = {
    "demo_horizontal": ("Demo — synthetic side view (horizontal edge join)", "Demo side"),
    "demo_vertical": ("Demo — synthetic roof (vertical edge join)", "Demo roof"),
    "demo_rectify": ("Demo — synthetic single view (rectify)", "Demo rectify"),
}


def _demo_recipes() -> list[dict]:
    """Example entries for seeded synthetic demo samples (demo_seed.py)."""
    recipes = []
    for cfg_path in sorted((BASE_DIR / "configs").glob("demo_*.json")):
        try:
            json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label, short = _DEMO_LABELS.get(cfg_path.stem, (cfg_path.stem, cfg_path.stem))
        recipes.append({
            "key": cfg_path.stem,
            "label": label,
            "short": short,
            "config": cfg_path,
            "kind": "job",
        })
    return recipes


EXAMPLE_RECIPES = [r for r in _BASE_EXAMPLE_RECIPES if _recipe_available(r)] + _demo_recipes()
EXAMPLE_BY_KEY = {item["key"]: item for item in EXAMPLE_RECIPES}


def _recipe_direction(item: dict) -> str:
    try:
        cfg = json.loads(item["config"].read_text(encoding="utf-8"))
        return cfg.get("direction", "horizontal")
    except Exception:
        return "horizontal"


def _example_cards() -> list[dict]:
    return [
        {
            "key": item["key"],
            "short": item["short"],
            "label": item["label"],
            "direction": _recipe_direction(item).capitalize(),
            "thumb_url": url_for("example_thumb", key=item["key"]),
        }
        for item in EXAMPLE_RECIPES if item["kind"] == "job"
    ]

DOCS = {
    "readme": BASE_DIR / "README.md",
    "engine": BASE_DIR / "ENGINE_README.md",
    "configuration": BASE_DIR / "CONFIGURATION.md",
    "samples": BASE_DIR / "SAMPLES.md",
}


def _sweep_old_jobs(ttl_seconds: float, interval_seconds: float = 3600) -> None:
    """Background retention: delete job/upload dirs older than the TTL.

    A public tester otherwise grows web_jobs/ without bound."""
    while True:
        cutoff = time.time() - ttl_seconds
        for base in (JOBS_DIR, UPLOADS_DIR):
            for path in base.iterdir():
                try:
                    if path.is_dir() and path.stat().st_mtime < cutoff:
                        shutil.rmtree(path, ignore_errors=True)
                except OSError:
                    pass
        time.sleep(interval_seconds)


def server_ai_defaults() -> dict | None:
    """Operator-configured AI providers (CST_AI_* env vars) for hosted demos.

    Keys never reach the browser: /api/ai-defaults only exposes whether
    defaults exist plus the model names for display. Returns a settings-shaped
    dict, or None when not configured."""
    p_base = os.environ.get("CST_AI_PLANNER_BASE", "")
    p_model = os.environ.get("CST_AI_PLANNER_MODEL", "")
    if not (p_base and p_model):
        return None
    r_base = os.environ.get("CST_AI_REVIEWER_BASE", "")
    r_model = os.environ.get("CST_AI_REVIEWER_MODEL", "")
    return {
        "planner": {"base_url": p_base, "api_key": os.environ.get("CST_AI_PLANNER_KEY", ""),
                    "model": p_model},
        "reviewer": ({"base_url": r_base, "api_key": os.environ.get("CST_AI_REVIEWER_KEY", ""),
                      "model": r_model} if (r_base and r_model) else {}),
        "server_defaults": True,
    }


def create_app() -> Flask:
    app = Flask(__name__)
    # 64 MB covers a multi-photo custom run; larger bodies get a 413.
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    app.config["CST_PUBLIC"] = os.environ.get("CST_PUBLIC", "") == "1"
    password = os.environ.get("CST_PASSWORD", "")
    app.config["CST_PASSWORD"] = password
    if password:
        # Stable session secret derived from the password so restarts do not
        # log users out; override with CST_SECRET for multi-instance setups.
        import hashlib
        app.secret_key = os.environ.get("CST_SECRET") or hashlib.sha256(
            b"cst-session:" + password.encode()).digest()
    if app.config["CST_PUBLIC"]:
        threading.Thread(target=_sweep_old_jobs, args=(7 * 24 * 3600,), daemon=True).start()

    @app.before_request
    def _require_login():
        if not password:
            return None
        if request.path == "/login" or request.path.startswith("/static/"):
            return None
        from flask import session, redirect
        if not session.get("auth"):
            return redirect(url_for("login", next=request.path))
        return None

    @app.route("/login", methods=["GET", "POST"])
    def login():
        from flask import session, redirect
        error = ""
        if request.method == "POST":
            if request.form.get("password", "") == password:
                session["auth"] = True
                session.permanent = True
                target = request.args.get("next") or "/"
                if not target.startswith("/") or target.startswith("//"):
                    target = "/"
                return redirect(target)
            error = "Incorrect password."
        return render_template("login.html", error=error)

    @app.post("/logout")
    def logout():
        from flask import session, redirect
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    def index() -> str:
        jobs = list_recent_jobs()
        n_sources = len([p for p in (BASE_DIR / "sources").glob("*") if p.is_file()])
        runtimes = [j["runtime_ms"] for j in jobs if j.get("runtime_ms")]
        avg = f"{sum(runtimes) / len(runtimes) / 1000:.1f}s" if runtimes else "~3s"
        return render_template(
            "index.html",
            examples=EXAMPLE_RECIPES,
            example_cards=_example_cards(),
            version=VERSION,
            jobs=jobs,
            default_config=DEFAULT_CONFIG_TEMPLATE,
            n_examples=len([e for e in EXAMPLE_RECIPES if e["kind"] == "job"]),
            n_sources=n_sources,
            avg_runtime=avg,
            auth_enabled=bool(password),
        )

    # ---- shared run logic (direct routes = silent; /runs/stream = live events) ----

    def _example_job(form: dict, files: dict, emit) -> str:
        key = form.get("recipe", "")
        no_balance = form.get("no_balance") == "on"
        example = EXAMPLE_BY_KEY.get(key)
        if example is None:
            emit("error", "Unknown example recipe.")
            return render_result_error("Unknown example recipe.")

        emit("run", f"packaged example: {example['label']}")
        job_id = make_job_id(key)
        out_dir = JOBS_DIR / job_id
        started = utc_now()
        t0 = time.time()
        try:
            if example["kind"] == "batch":
                emit("run", f"engine running batch manifest ({len(json.loads(example['config'].read_text(encoding='utf-8'))['jobs'])} jobs)…")
                summary = run_batch(example["config"], out_dir, no_balance=no_balance)
                write_meta(out_dir, example["label"], "batch", key, summary.get("status"),
                           started, runtime_ms(t0), no_balance)
                emit("ok", f"batch finished: {summary['jobs_created']} created, {summary['jobs_rejected']} rejected "
                           f"({runtime_ms(t0) / 1000:.1f}s)")
                return render_template(
                    "_result.html",
                    ok=True,
                    title=f"Batch complete: {example['label']}",
                    job_id=job_id,
                    panel=panel_from_report(summary, job_id, out_dir, runtime_ms(t0),
                                            example["label"], kind="batch"),
                    report=summary,
                    created_at=now_str(),
                    message="All jobs were written into the batch folder. Open the JSON summary or browse the job directory.",
                )

            emit("run", "engine running (OpenCV, deterministic)…")
            report = run_job(example["config"], out_dir, no_balance=no_balance)
            write_meta(out_dir, example["label"], "job", key, report.get("status"),
                       started, runtime_ms(t0), no_balance)
            emit("ok", f"engine finished — {report.get('status')} · {report.get('quality_label', '')} "
                       f"({runtime_ms(t0) / 1000:.1f}s)")
            return render_template(
                "_result.html",
                ok=True,
                title=example["label"],
                job_id=job_id,
                panel=panel_from_report(report, job_id, out_dir, runtime_ms(t0),
                                        example["label"], kind="job"),
                report=report,
                created_at=now_str(),
                message="Processed using the packaged config and original sample files.",
            )
        except ProcessingError as exc:
            emit("error", f"engine rejected the job: {str(exc)[:160]}")
            write_meta(out_dir, example["label"], example["kind"], key, "rejected",
                       started, runtime_ms(t0), no_balance)
            return render_result_error(str(exc), job_id=job_id)

    def _custom_job(form: dict, files: dict, emit) -> str:
        config_file = (files.get("config") or [None])[0]
        config_text = (form.get("config_text") or "").strip()

        # Accept config from textarea OR uploaded file — textarea takes precedence
        # when the file input is hidden (no file chosen).
        has_file = config_file and config_file["filename"]
        if not has_file and not config_text:
            return render_result_error("Provide a JSON config — either type/paste it in the editor or upload a file.")

        mode = blank_to_none(form.get("mode"))
        direction = blank_to_none(form.get("direction"))
        no_balance = form.get("no_balance") == "on"
        run_as_batch = form.get("run_as_batch") == "on"
        label = "Custom batch" if run_as_batch else "Custom run"

        job_id = make_job_id("custom")
        out_dir = JOBS_DIR / job_id
        work_dir = UPLOADS_DIR / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        started = utc_now()
        t0 = time.time()
        try:
            if has_file:
                emit("run", f"reading uploaded config {config_file['filename']}…")
                config_path = work_dir / safe_upload_name(config_file["filename"])
                shutil.copyfile(config_file["path"], config_path)
                # Validate the uploaded file before any engine work so a bad
                # config fails with a readable message instead of a raw
                # JSONDecodeError from inside the first engine variant.
                try:
                    uploaded_text = config_path.read_text(encoding="utf-8")
                    json.loads(uploaded_text)
                except UnicodeDecodeError:
                    emit("error", "uploaded config is not UTF-8 text")
                    return render_result_error("The uploaded config is not readable text — export it as UTF-8 JSON and try again.")
                except json.JSONDecodeError as exc:
                    friendly = describe_config_error(uploaded_text, exc)
                    emit("error", friendly)
                    return render_result_error(friendly)
            else:
                emit("run", "validating config JSON from the editor…")
                try:
                    json.loads(config_text)
                except json.JSONDecodeError as exc:
                    friendly = describe_config_error(config_text, exc)
                    emit("error", friendly)
                    return render_result_error(friendly)
                config_path = work_dir / "config.json"
                config_path.write_text(config_text, encoding="utf-8")

            staged = 0
            for uploaded in files.get("sources", []):
                if uploaded and uploaded["filename"]:
                    shutil.copyfile(uploaded["path"], work_dir / safe_upload_name(uploaded["filename"]))
                    staged += 1
            if staged:
                emit("ok", f"staged {staged} uploaded image(s) next to the config")

            # Path jail: a custom config may only reference files staged inside
            # this run's upload folder — never arbitrary server paths.
            try:
                jailed = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                jailed = {}
            for name, info in (jailed.get("sources") or {}).items():
                rel = info.get("path") if isinstance(info, dict) else None
                if not isinstance(rel, str) or not rel:
                    continue
                resolved = (work_dir / rel).resolve()
                if work_dir.resolve() not in resolved.parents and resolved.parent != work_dir.resolve():
                    msg = (f"Source '{name}' path must stay inside the uploaded files folder "
                           f"(got '{rel}'). Upload the image alongside the config instead.")
                    emit("error", msg)
                    return render_result_error(msg)

            if run_as_batch and mode == "auto":
                raise ProcessingError("Auto mode (try single + combo) applies to single-job configs, not batch manifests.")

            if mode == "auto" and not run_as_batch:
                emit("run", "auto mode — running single AND combo variants, then picking…")
                return run_auto_custom(
                    config_path, out_dir, work_dir, job_id, label,
                    no_balance=no_balance, direction=direction,
                    started=started, t0=t0, emit=emit,
                )

            if run_as_batch:
                if mode and mode != "auto":
                    raise ProcessingError("Mode and direction assertions are only for single-job configs, not batch manifests.")
                emit("run", "engine running batch manifest…")
                summary = run_batch(config_path, out_dir, no_balance=no_balance)
                write_meta(out_dir, label, "batch", "custom", summary.get("status"),
                           started, runtime_ms(t0), no_balance)
                emit("ok", f"batch finished: {summary['jobs_created']} created, {summary['jobs_rejected']} rejected")
                return render_template(
                    "_result.html",
                    ok=True,
                    title=label,
                    job_id=job_id,
                    panel=panel_from_report(summary, job_id, out_dir, runtime_ms(t0), label, kind="batch"),
                    report=summary,
                    created_at=now_str(),
                    message="Batch completed. Make sure the uploaded config references the uploaded files by matching filenames.",
                )

            emit("run", "engine running (OpenCV, deterministic)…")
            report = run_job(config_path, out_dir, mode=mode, direction=direction, no_balance=no_balance)
            write_meta(out_dir, label, "job", "custom", report.get("status"),
                       started, runtime_ms(t0), no_balance)
            emit("ok", f"engine finished — {report.get('status')} · {report.get('quality_label', '')}")
            return render_template(
                "_result.html",
                ok=True,
                title=label,
                job_id=job_id,
                panel=panel_from_report(report, job_id, out_dir, runtime_ms(t0), label, kind="job"),
                report=report,
                created_at=now_str(),
                message="Uploaded config and files were staged in a temporary work folder, then processed by container_stitch.py.",
            )
        except ProcessingError as exc:
            emit("error", f"engine rejected the job: {str(exc)[:160]}")
            write_meta(out_dir, label, "batch" if run_as_batch else "job", "custom", "rejected",
                       started, runtime_ms(t0), no_balance)
            return render_result_error(str(exc), job_id=job_id)
        except Exception as exc:  # pragma: no cover - defensive path for manual UI use
            emit("error", f"unexpected server error: {exc}")
            write_meta(out_dir, label, "batch" if run_as_batch else "job", "custom", "rejected",
                       started, runtime_ms(t0), no_balance)
            return render_result_error(f"Unexpected error: {exc}", job_id=job_id)

    @app.post("/run-example")
    def run_example() -> str:
        return _example_job({k: v for k, v in request.form.items()}, {}, _noop_emit)

    @app.post("/run-custom")
    def run_custom() -> str:
        files, tmp_root = _materialize_files(request.files.items(multi=True))
        try:
            return _custom_job({k: v for k, v in request.form.items()}, files, _noop_emit)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    @app.post("/runs/stream")
    def runs_stream() -> Response:
        """SSE: runs any job kind while streaming live console events."""
        kind = request.form.get("kind", "custom")
        form = {k: v for k, v in request.form.items()}
        files, tmp_root = _materialize_files(request.files.items(multi=True))

        events: queue_mod.Queue = queue_mod.Queue()

        def emit(stage: str, message: str) -> None:
            events.put({"stage": stage, "message": message})

        def worker():
            with app.test_request_context():
                try:
                    if kind == "example":
                        html = _example_job(form, files, emit)
                    elif kind == "ai":
                        html = _ai_job(form, files, emit)
                    else:
                        html = _custom_job(form, files, emit)
                except Exception as exc:  # pragma: no cover - defensive
                    html = render_result_error(f"Unexpected server error: {exc}")
                events.put({"__done__": True, "html": html})

        threading.Thread(target=worker, daemon=True).start()

        def generate():
            try:
                yield ": stream open\n\n"
                while True:
                    item = events.get()
                    if item.get("__done__"):
                        yield "event: done\ndata: " + json.dumps({"html": item["html"]}) + "\n\n"
                        return
                    yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/run-paste")
    def run_paste() -> str:
        """Run a packaged recipe config but replace its source images with a pasted/uploaded image."""
        recipe_key = request.form.get("recipe", "")
        image_file = request.files.get("image")
        if not image_file or not image_file.filename:
            return render_result_error("No image received. Please paste or pick an image first.")

        example = EXAMPLE_BY_KEY.get(recipe_key)
        if example is None or example.get("kind") == "batch":
            return render_result_error("Unknown or unsupported recipe for paste mode.")

        job_id = make_job_id("paste")
        out_dir = JOBS_DIR / job_id
        work_dir = UPLOADS_DIR / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        label = f"Paste run — {example['label']}"
        started = utc_now()
        t0 = time.time()
        try:
            img_path = work_dir / "input.png"
            image_file.save(img_path)

            import cv2 as _cv2
            _img = _cv2.imread(str(img_path))
            if _img is None:
                raise ProcessingError("Could not decode the uploaded image. Please upload a valid PNG/JPEG.")
            img_h, img_w = _img.shape[:2]

            original_cfg = json.loads(example["config"].read_text(encoding="utf-8"))
            if "sources" in original_cfg and isinstance(original_cfg["sources"], dict):
                for key in original_cfg["sources"]:
                    original_cfg["sources"][key]["path"] = "input.png"
                    original_cfg["sources"][key]["expected_size_wh"] = [img_w, img_h]
                    original_cfg["sources"][key].pop("sha256", None)

            patched_path = work_dir / "config.json"
            patched_path.write_text(json.dumps(original_cfg, indent=2), encoding="utf-8")

            report = run_job(patched_path, out_dir)
            write_meta(out_dir, label, "job", recipe_key, report.get("status"),
                       started, runtime_ms(t0), False)
            return render_template(
                "_result.html",
                ok=True,
                title=label,
                job_id=job_id,
                panel=panel_from_report(report, job_id, out_dir, runtime_ms(t0), label, kind="job"),
                report=report,
                created_at=now_str(),
                message="Pasted image stitched using the selected packaged recipe's corner configuration.",
            )
        except ProcessingError as exc:
            write_meta(out_dir, label, "job", recipe_key, "rejected", started, runtime_ms(t0), False)
            return render_result_error(str(exc), job_id=job_id)
        except Exception as exc:  # pragma: no cover
            write_meta(out_dir, label, "job", recipe_key, "rejected", started, runtime_ms(t0), False)
            return render_result_error(f"Unexpected error: {exc}", job_id=job_id)

    def _ai_job(form: dict, files: dict, emit) -> str:
        """AI-assisted run: the vision planner proposes an ai_plan, this endpoint
        converts it to an engine config and runs the deterministic stitcher.
        On rejection the diagnostics go back to the (reviewer) model, capped at
        max_attempts. The engine remains the only component that touches pixels."""
        image = (files.get("image") or [None])[0]
        no_balance = form.get("no_balance") == "on"
        try:
            settings = json.loads(form.get("settings") or "{}")
        except json.JSONDecodeError:
            return render_result_error("AI settings are not valid JSON.")

        planner = settings.get("planner") or {}
        reviewer = settings.get("reviewer") or {}
        server_defaults = False
        if not (planner.get("base_url") and planner.get("model")):
            defaults = server_ai_defaults()
            if defaults:
                settings = {**settings, **defaults}
                planner = defaults["planner"]
                reviewer = defaults.get("reviewer", {})
                server_defaults = True
                emit("run", f"no custom AI settings — using operator-configured defaults "
                            f"(planner {planner.get('model', '')})")
        if not planner.get("base_url") or not planner.get("model"):
            return render_result_error("AI planning needs a planner endpoint and model — open ⚙ AI settings.")
        has_reviewer = bool(reviewer.get("base_url") and reviewer.get("model"))
        try:
            max_attempts = max(1, min(int(settings.get("max_attempts", 3)), 4))
        except (TypeError, ValueError):
            max_attempts = 3
        if server_defaults:
            try:
                max_attempts = max(1, min(int(os.environ.get("CST_AI_MAX_ATTEMPTS", max_attempts)), 4))
            except (TypeError, ValueError):
                pass
        if app.config.get("CST_PUBLIC"):
            guard_error = _assert_public_ai_endpoints(settings)
            if guard_error:
                emit("error", guard_error)
                return render_result_error(guard_error)

        if image is None or not image["filename"]:
            return render_result_error("Stage an image first — AI planning runs on the staged photo.")

        job_id = make_job_id("ai")
        work_dir = UPLOADS_DIR / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        img_path = work_dir / "input.png"
        shutil.copyfile(image["path"], img_path)

        import cv2
        probe = cv2.imread(str(img_path))
        if probe is None:
            return render_result_error("Could not decode the staged image.")
        height, width = probe.shape[:2]
        emit("run", f"AI planning on staged image ({width}×{height}) — planner {planner.get('model', '')}"
                    + (f", reviewer {reviewer.get('model', '')}" if has_reviewer else ", no reviewer")
                    + f", max {max_attempts} attempt(s)")

        started = utc_now()
        t0 = time.time()
        try:
            data_uri = ai_planner.build_image_data_uri(img_path)
        except Exception as exc:
            emit("error", f"could not prepare the image for the AI request: {exc}")
            return render_result_error(f"Could not prepare the image for the AI request: {exc}")

        attempt_log: list[dict] = []
        last_plan = None
        dead_providers: set[str] = set()
        transient_failures: dict[str, int] = {}
        try:
            request_timeout = max(15, min(int(settings.get("request_timeout", 90)), 240))
        except (TypeError, ValueError):
            request_timeout = 90
        planner_key = f"{planner.get('base_url')}|{planner.get('model')}"
        reviewer_key = f"{reviewer.get('base_url')}|{reviewer.get('model')}" if has_reviewer else None
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            # Attempt 1 always uses the planner. Retries prefer the reviewer;
            # if the reviewer is unreachable we fall back to the planner for
            # that attempt without consuming it, and skip the reviewer for the
            # rest of the run. A provider that fails twice in a row (timeout,
            # 5xx) is treated as down for this run instead of being retried
            # indefinitely.
            use_reviewer = (attempt > 1 and has_reviewer
                            and reviewer_key not in dead_providers)
            prov, role = ((reviewer, "reviewer") if use_reviewer else (planner, "planner"))
            prov_key = reviewer_key if use_reviewer else planner_key
            if role == "planner" and planner_key in dead_providers:
                emit("error", "no reachable AI provider remains — planner and reviewer are both unavailable, stopping early")
                break
            prov_name = prov.get("model", "model")
            emit("run", f"attempt {attempt}/{max_attempts}: asking {role} {prov_name}…")

            # 1) ask the model for a plan
            try:
                content = ai_planner.chat_completion(
                    prov["base_url"], prov.get("api_key", ""), prov["model"],
                    ai_planner.build_messages(data_uri, attempt_log),
                    timeout=request_timeout)
                plan = ai_planner.extract_json(content)
                ai_planner.validate_plan(plan)
            except ai_planner.AiProviderUnavailable as exc:
                raw = getattr(exc, "raw_detail", "")
                if role == "reviewer":
                    dead_providers.add(prov_key)
                    attempt_log.append({"attempt": attempt, "provider": prov_name, "stage": "fallback",
                                        "ok": False,
                                        "detail": f"reviewer unreachable ({str(exc)[:200]}) — falling back to planner {planner.get('model', '')}",
                                        "raw_detail": raw[:2000]})
                    emit("warn", f"{prov_name} unreachable — falling back to planner {planner.get('model', '')} (attempt not consumed)")
                    attempt -= 1  # availability failures do not consume an attempt
                    continue
                transient_failures[prov_key] = transient_failures.get(prov_key, 0) + 1
                if transient_failures[prov_key] >= 2:
                    dead_providers.add(prov_key)
                attempt_log.append({"attempt": attempt, "provider": prov_name, "stage": "planning",
                                    "ok": False, "detail": f"provider unreachable: {exc}",
                                    "raw_detail": raw[:2000]})
                emit("error", f"attempt {attempt}: provider unreachable — {str(exc)[:140]}")
                # Brief backoff before touching the same endpoint again; it
                # just failed, so an immediate retry almost always 504s too.
                if prov_key not in dead_providers:
                    emit("run", f"waiting 2s before retrying {prov_name}…")
                    time.sleep(2)
                continue
            except Exception as exc:
                attempt_log.append({"attempt": attempt, "provider": prov_name, "stage": "planning",
                                    "ok": False, "detail": str(exc)[:400]})
                emit("error", f"attempt {attempt}: unusable plan — {str(exc)[:140]}")
                continue
            last_plan = plan
            methods = ", ".join(c.get("method", "?") for c in plan.get("containers", []))
            conf = plan.get("confidence", {})
            emit("ok", f"plan received: {methods} · direction {plan.get('suggested_processing', {}).get('direction', 'n/a')}"
                       f" · confidence dir {conf.get('direction', '?')}/group {conf.get('container_grouping', '?')}/surface {conf.get('same_surface', '?')}")

            # 2) normalize plan -> engine config
            try:
                cfg = ai_planner.plan_to_config(plan, width, height)
                emit("run", f"normalized plan → engine config ({cfg['mode']}, {cfg['direction']})")
            except Exception as exc:
                attempt_log.append({"attempt": attempt, "provider": prov_name, "stage": "normalizing",
                                    "ok": False, "detail": str(exc)[:400], "plan": plan})
                emit("error", f"attempt {attempt}: could not normalize plan — {str(exc)[:140]}")
                continue

            # 3) deterministic engine — final authority
            cfg_path = work_dir / f"config_attempt{attempt}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            attempt_dir = JOBS_DIR / f"{job_id}-a{attempt}"
            emit("run", f"engine validating + stitching attempt {attempt}…")
            try:
                report = run_job(cfg_path, attempt_dir, no_balance=no_balance)
            except ProcessingError as exc:
                attempt_log.append({"attempt": attempt, "provider": prov_name, "stage": "engine",
                                    "ok": False, "detail": str(exc)[:400], "plan": plan,
                                    "validator_details": (exc.details or {}).get("details")})
                emit("error", f"engine rejected attempt {attempt}: {str(exc)[:150]}")
                write_meta(attempt_dir, f"AI plan attempt {attempt} — rejected", "job", "ai",
                           "rejected", started, runtime_ms(t0), no_balance)
                continue

            rt = runtime_ms(t0)
            write_meta(attempt_dir, f"AI plan run ({prov_name})", "job", "ai",
                       report.get("status"), started, rt, no_balance)
            attempt_log.append({"attempt": attempt, "provider": prov_name, "stage": "engine",
                                "ok": True, "detail": "accepted"})
            emit("ok", f"engine accepted attempt {attempt} — {report.get('quality_label', 'done')} ({rt / 1000:.1f}s total)")
            return render_template(
                "_result_ai.html",
                ok=True,
                job_id=attempt_dir.name,
                ai_meta={
                    "planner": planner.get("model", ""),
                    "reviewer": reviewer.get("model", "") if has_reviewer else "",
                    "attempts": attempt_log,
                    "attempt_count": attempt,
                    "max_attempts": max_attempts,
                    "confidence": plan.get("confidence", {}),
                    "reason": plan.get("reason", ""),
                    "exclude": plan.get("exclude", []),
                    "scene": plan.get("scene", {}),
                    "plan": plan,
                    "no_balance": no_balance,
                },
                picked_ctx={
                    "job_id": attempt_dir.name,
                    "panel": panel_from_report(report, attempt_dir.name, attempt_dir, rt,
                                               f"AI plan ({prov_name})", "job"),
                    "report": report,
                    "title": "AI-planned run",
                    "message": "AI proposed the configuration; the deterministic engine validated and stitched it.",
                },
                created_at=now_str(),
            )

        emit("error", "all attempts rejected — no composite was produced")
        # Keep the full raw provider responses (including unsanitized bodies)
        # in a server-side artifact; the UI only ever shows sanitized details.
        log_url = None
        try:
            log_dir = JOBS_DIR / job_id
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "ai_run_log.json").write_text(
                json.dumps(attempt_log, indent=2, ensure_ascii=False), encoding="utf-8")
            log_url = f"/jobs/{job_id}/ai_run_log.json"
        except OSError:
            pass
        write_meta(JOBS_DIR / job_id, "AI plan — all attempts rejected", "job", "ai",
                   "rejected", started, runtime_ms(t0), no_balance)
        return render_template(
            "_result_ai.html",
            ok=False,
            job_id=job_id,
            log_url=log_url,
            ai_meta={
                "planner": planner.get("model", ""),
                "reviewer": reviewer.get("model", "") if has_reviewer else "",
                "attempts": attempt_log,
                "attempt_count": max_attempts,
                "max_attempts": max_attempts,
                "confidence": (last_plan or {}).get("confidence", {}),
                "reason": (last_plan or {}).get("reason", ""),
                "exclude": (last_plan or {}).get("exclude", []),
                "scene": (last_plan or {}).get("scene", {}),
                "plan": last_plan,
                "no_balance": no_balance,
            },
            picked_ctx=None,
            created_at=now_str(),
        )

    @app.post("/run-ai")
    def run_ai() -> str:
        files, tmp_root = _materialize_files(request.files.items(multi=True))
        try:
            return _ai_job({k: v for k, v in request.form.items()}, files, _noop_emit)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    @app.get("/jobs")
    def jobs() -> str:
        return render_template("_jobs.html", jobs=list_recent_jobs())

    @app.get("/jobs/<job_id>/<path:filename>")
    def job_file(job_id: str, filename: str):
        job_path = (JOBS_DIR / job_id).resolve()
        if not job_path.exists() or job_path.parent != JOBS_DIR.resolve():
            abort(404)
        file_path = (job_path / filename).resolve()
        if not file_path.exists() or job_path not in file_path.parents and file_path != job_path:
            abort(404)
        return send_file(file_path)

    @app.get("/api/jobs/<job_id>/report")
    def job_report(job_id: str) -> Response:
        path = _job_summary_path(job_id)
        if path is None:
            abort(404)
        return Response(path.read_text(encoding="utf-8"), mimetype="application/json")

    @app.get("/api/jobs/<job_id>/meta")
    def job_meta(job_id: str) -> Response:
        path = _safe_job_path(job_id, "meta.json")
        if path is None:
            abort(404)
        return Response(path.read_text(encoding="utf-8"), mimetype="application/json")

    @app.get("/api/jobs/<job_id>/config")
    def job_config(job_id: str) -> Response:
        for candidate in (
            _safe_job_path(job_id, "resolved_config.json"),
            _safe_upload_path(job_id, "config.json"),
        ):
            if candidate is not None:
                return Response(candidate.read_text(encoding="utf-8"), mimetype="application/json")
        upload_dir = (UPLOADS_DIR / job_id).resolve()
        if upload_dir.parent == UPLOADS_DIR.resolve() and upload_dir.is_dir():
            for candidate in sorted(upload_dir.glob("*.json")):
                return Response(candidate.read_text(encoding="utf-8"), mimetype="application/json")
        abort(404)

    @app.get("/example-thumb/<key>")
    def example_thumb(key: str):
        example = EXAMPLE_BY_KEY.get(key)
        if example is None or example["kind"] != "job":
            abort(404)
        thumb = example["config"].parent.parent / "examples" / key / "result.png"
        if not thumb.is_file():
            # Seeded demo recipes ship a source image but no precomputed
            # examples/<key>/result.png; fall back to their main source.
            thumb = None
            try:
                cfg = json.loads(example["config"].read_text(encoding="utf-8"))
                src = str((cfg.get("sources") or {}).get("main", {}).get("path", ""))
                if src:
                    candidate = (example["config"].parent / src).resolve()
                    if candidate.is_file() and BASE_DIR in candidate.parents:
                        thumb = candidate
            except (OSError, ValueError):
                thumb = None
            if thumb is None:
                abort(404)
        return send_file(thumb, mimetype="image/png")

    @app.get("/api/ai-defaults")
    def api_ai_defaults() -> Response:
        """Whether operator-configured AI defaults exist — never the secrets."""
        defaults = server_ai_defaults()
        if not defaults:
            return Response(json.dumps({"configured": False}), mimetype="application/json")
        return Response(json.dumps({
            "configured": True,
            "planner_model": defaults["planner"].get("model", ""),
            "reviewer_model": defaults.get("reviewer", {}).get("model", ""),
        }), mimetype="application/json")

    @app.get("/api/profiles")
    def api_profiles() -> Response:
        profiles_dir = BASE_DIR / "profiles"
        items = []
        if profiles_dir.is_dir():
            for path in sorted(profiles_dir.glob("*.json")):
                try:
                    data = camera_profiles.load_profile(path)
                    items.append({
                        "key": data["profile_key"],
                        "file": path.stem,
                        "label": data.get("camera", {}).get("label", data["profile_key"]),
                        "expected_size_wh": data.get("camera", {}).get("expected_size_wh"),
                        "pinned": bool(data.get("pin", {}).get("enforce")),
                    })
                except Exception:
                    continue
        return Response(json.dumps(items), mimetype="application/json")

    @app.post("/profile-config")
    def profile_config() -> Response:
        """Generate an engine config from a camera profile + the staged capture."""
        key = request.form.get("profile", "")
        image = request.files.get("image")
        profiles_dir = BASE_DIR / "profiles"
        path = profiles_dir / f"{key}.json"
        if not path.is_file():
            return Response(json.dumps({"error": f"Unknown profile: {key}"}), status=404,
                            mimetype="application/json")
        if image is None or not image.filename:
            return Response(json.dumps({"error": "Stage a capture image first."}), status=400,
                            mimetype="application/json")
        work = UPLOADS_DIR / make_job_id("profile")
        work.mkdir(parents=True, exist_ok=True)
        capture = work / "capture.png"
        image.save(capture)
        try:
            profile = camera_profiles.load_profile(path)
            config = camera_profiles.profile_to_config(profile, capture)
        except (ProcessingError, Exception) as exc:
            return Response(json.dumps({"error": str(exc)}), status=422,
                            mimetype="application/json")
        # keep the staged capture reachable for the subsequent run
        target = work / "input.png"
        shutil.copyfile(capture, target)
        config["sources"][next(iter(config["sources"]))]["path"] = str(target)
        return Response(json.dumps({"config": json.dumps(config, indent=2)}, indent=None),
                        mimetype="application/json")

    @app.get("/api/docs/<name>")
    def api_docs(name: str) -> Response:
        path = DOCS.get(name)
        if path is None or not path.is_file():
            abort(404)
        return Response(path.read_text(encoding="utf-8"), mimetype="text/plain")

    @app.post("/delete-job/<job_id>")
    def delete_job(job_id: str) -> str:
        path = (JOBS_DIR / job_id).resolve()
        if path.exists() and path.parent == JOBS_DIR.resolve():
            shutil.rmtree(path, ignore_errors=True)
        upload = (UPLOADS_DIR / job_id).resolve()
        if upload.exists() and upload.parent == UPLOADS_DIR.resolve():
            shutil.rmtree(upload, ignore_errors=True)
        return render_template("_jobs.html", jobs=list_recent_jobs())

    return app


def _noop_emit(stage, message):
    pass


def _assert_public_ai_endpoints(settings: dict) -> str | None:
    """Public deployments refuse AI endpoints on private networks (SSRF guard).

    Returns a user-facing error message, or None when every configured
    endpoint resolves to a public address. Disabled locally (CST_PUBLIC)."""
    import urllib.parse

    def check(provider: str, role: str) -> str | None:
        base = (provider or {}).get("base_url") or ""
        if not base:
            return None
        parsed = urllib.parse.urlparse(base if "//" in base else f"https://{base}")
        host = parsed.hostname or ""
        if host in ("localhost",) or host.endswith(".localhost"):
            return f"{role} endpoint must be a public https URL, not localhost."
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            return f"{role} endpoint host '{host}' does not resolve."
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                return (f"{role} endpoint '{host}' resolves to a private address ({ip}); "
                        "public deployments only call public AI endpoints.")
        return None

    for provider, role in ((settings.get("planner"), "Planner"),
                           (settings.get("reviewer"), "Reviewer")):
        if provider and provider.get("base_url"):
            err = check(provider, role)
            if err:
                return err
    return None


def _materialize_files(multi_items) -> tuple[dict[str, list[dict]], Path]:
    """Copy uploaded files to a temp dir so worker threads can read them after
    the request stream is closed. Returns ({name: [{filename, path}]}, tmp_dir).

    Browsers submit a filename-less placeholder part for every unselected
    <input type="file">; those are dropped here so handlers never mistake
    them for real uploads."""
    tmp_root = Path(tempfile.mkdtemp(prefix="run-upload-"))
    files: dict[str, list[dict]] = {}
    for key, fs in multi_items:
        if not fs.filename:
            continue
        idx = len(files.get(key, []))
        dest = tmp_root / f"{idx:02d}_{Path(fs.filename).name}"
        fs.save(dest)
        files.setdefault(key, []).append({"filename": fs.filename, "path": str(dest)})
    return files, tmp_root


# ------------------------------------------------------------------ auto mode

def _split_combo_config(cfg: dict) -> dict | None:
    """Synthesize a combo variant (one rectified container per region) from a
    single-group, two-region edge-join config. Returns None when the config
    does not have that shape."""
    try:
        if cfg.get("mode") != "single":
            return None
        containers = cfg.get("containers")
        if not isinstance(containers, list) or len(containers) != 1:
            return None
        regions = containers[0].get("regions")
        if not isinstance(regions, list) or len(regions) != 2:
            return None
        combo = copy.deepcopy(cfg)
        combo["mode"] = "combo"
        combo["containers"] = [
            {
                "key": f"container_{i}",
                "label": f"Container {i}",
                "method": "rectify",
                "regions": [copy.deepcopy(region)],
            }
            for i, region in enumerate(regions, 1)
        ]
        return combo
    except Exception:
        return None


def detect_gap(image_path: Path, direction: str) -> bool | None:
    """UI-layer heuristic only: does a background-colored separator band cross
    the middle of the image along the stitch axis? True suggests two separate
    containers (combo), False suggests one continuous surface (single), None
    means undecidable. The engine itself never infers mode — this only ranks
    the two rendered variants."""
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
        bg = float(np.median(border))
        fg = np.abs(img.astype(np.float32) - bg) > 18.0
        # profile = fraction of container pixels per column (horizontal) / row (vertical)
        profile = fg.mean(axis=0 if direction == "horizontal" else 1)
        n = len(profile)
        lo, hi = int(n * 0.35), int(n * 0.65)
        edge = max(profile[: max(1, int(n * 0.05))].mean(),
                   profile[-max(1, int(n * 0.05)):].mean())
        if edge <= 0.05:
            return None
        return bool(profile[lo:hi].min() < 0.35 * edge)
    except Exception:
        return None


def run_auto_custom(config_path: Path, out_dir: Path, work_dir: Path, job_id: str,
                    label: str, *, no_balance: bool, direction: str | None,
                    started: str, t0: float, emit=None) -> str:
    """Run the config as-is (variant A) plus a synthesized combo variant
    (variant B), then render the heuristic-picked one full-size with the other
    as a compact alternate card."""
    import json as _json

    emit = emit or _noop_emit

    emit("run", "variant A: engine running as single (edge join)…")
    try:
        report_a = run_job(config_path, out_dir, no_balance=no_balance, direction=direction)
    except ProcessingError as exc:
        emit("error", f"variant A rejected: {str(exc)[:140]}")
        write_meta(out_dir, f"{label} — auto", "job", "custom", "rejected",
                   started, runtime_ms(t0), no_balance)
        return render_result_error(str(exc), job_id=job_id)
    emit("ok", f"variant A accepted — {report_a.get('quality_label', '')}")
    rt_a = runtime_ms(t0)
    write_meta(out_dir, f"{label} — auto:single", "job", "custom", report_a.get("status"),
               started, rt_a, no_balance)

    ctx_a = {
        "job_id": job_id,
        "panel": panel_from_report(report_a, job_id, out_dir, rt_a, "Single — edge join", "job"),
        "report": report_a,
        "title": "Variant A — Single (edge join)",
        "message": "Both halves joined as one container.",
    }

    # Variant B: combo (each region rectified as its own container)
    alt = None
    try:
        cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        combo_cfg = _split_combo_config(cfg)
    except Exception:
        combo_cfg = None
    if combo_cfg is not None:
        alt_job = f"{job_id}-combo"
        alt_dir = JOBS_DIR / alt_job
        combo_path = work_dir / "config_combo.json"
        combo_path.write_text(_json.dumps(combo_cfg, indent=2), encoding="utf-8")
        emit("run", "variant B: engine running as combo (separate containers)…")
        try:
            report_b = run_job(combo_path, alt_dir, no_balance=no_balance, direction=direction)
            write_meta(alt_dir, f"{label} — auto:combo", "job", "custom", report_b.get("status"),
                       started, runtime_ms(t0), no_balance)
            alt = {
                "job_id": alt_job,
                "title": "Variant B — Combo (separate containers)",
                "message": "Each half rectified as its own separate container.",
                "panel": panel_from_report(report_b, alt_job, alt_dir, runtime_ms(t0), "Combo — separate", "job"),
                "report": report_b,
            }
        except ProcessingError:
            write_meta(alt_dir, f"{label} — auto:combo", "job", "custom", "rejected",
                       started, runtime_ms(t0), no_balance)

    # Heuristic pick: background gap across the middle -> combo, else single.
    gap = None
    src_name = ""
    try:
        cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        first_source = next(iter(cfg.get("sources", {}).values()))
        src_name = first_source.get("path", "")
        gap = detect_gap(work_dir / src_name, report_a.get("direction", "horizontal"))
    except Exception:
        gap = None
    if gap is True and alt is not None:
        picked, reason = "combo", "background gap detected across the middle — likely separate containers"
    elif alt is not None:
        picked, reason = "single", "no background gap detected — appears to be one continuous surface"
    emit("ok", f"auto pick: {picked} — {reason}")

    picked_ctx = ctx_a if picked == "single" else alt
    other = alt if picked == "single" else {
        "job_id": ctx_a["job_id"],
        "title": "Variant A — Single (edge join)",
        "panel": ctx_a["panel"],
    }
    return render_template(
        "_result_auto.html",
        picked=picked,
        picked_label="Single — edge join" if picked == "single" else "Combo — separate containers",
        reason=reason,
        picked_ctx=picked_ctx,
        alt=other,
        created_at=now_str(),
        ok=True,
        job_id=job_id,
    )


# ------------------------------------------------------------------ helpers

def panel_from_report(report: dict, job_id: str, out_dir: Path, runtime_ms: int,
                      label: str, kind: str) -> dict:
    """Prepare everything the results panel template needs."""
    panel = {
        "kind": kind,
        "label": label,
        "status": report.get("status", "unknown"),
        "processing_state": report.get("processing_state", "created" if kind == "job" else ""),
        "quality_state": report.get("quality_state", ""),
        "quality_label": report.get("quality_label", ""),
        "runtime_ms": runtime_ms,
        "mode": report.get("mode", ""),
        "direction": report.get("direction", ""),
        "result_url": None,
        "sources": [],
        "output_size_wh": report.get("output_size_wh"),
        "file_size": None,
        "blend": None,
        "batch_rows": report.get("jobs") if kind == "batch" else None,
    }
    result_path = out_dir / "result.png"
    if result_path.is_file():
        panel["result_url"] = url_for("job_file", job_id=job_id, filename="result.png")
        size = result_path.stat().st_size
        panel["file_size"] = human_size(size)
    if (out_dir / "debug_overlay.png").is_file():
        panel["debug_url"] = url_for("job_file", job_id=job_id, filename="debug_overlay.png")
    if (out_dir / "diagnostics.json").is_file():
        panel["diag_url"] = url_for("job_file", job_id=job_id, filename="diagnostics.json")
    for record in report.get("source_records", []) or []:
        name = record.get("name", "")
        panel["sources"].append({
            "name": name,
            "file": record.get("file", ""),
            "size_wh": record.get("size_wh"),
            "url": url_for("job_file", job_id=job_id, filename=f"selected_regions/{name}.jpg"),
        })
    containers = report.get("containers") or []
    method = containers[0].get("method") if containers else None
    panel["blend"] = {
        "rectify": "Perspective rectify",
        "edge": "Edge join + exposure balance",
        "overlap": "OpenCV SIFT + RANSAC blend",
    }.get(method)
    seam_alignment = next((c.get("seam_alignment") for c in containers
                           if isinstance(c.get("seam_alignment"), dict)
                           and c.get("seam_alignment", {}).get("enabled")), None)
    if seam_alignment is None:
        seam_alignment = next((c.get("promoted_from_edge", {}).get("seam_alignment")
                               for c in containers
                               if isinstance(c.get("promoted_from_edge"), dict)), None)
    if seam_alignment:
        panel["seam_alignment"] = seam_alignment
        panel["promoted_from_edge"] = any(isinstance(c.get("promoted_from_edge"), dict)
                                          for c in containers)
    return panel


def write_meta(out_dir: Path, label: str, kind: str, recipe: str, status: str,
               started: str, runtime_ms: int, no_balance: bool) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "label": label,
            "kind": kind,
            "recipe": recipe,
            "status": status,
            "started_at": started,
            "completed_at": utc_now(),
            "runtime_ms": runtime_ms,
            "no_balance": bool(no_balance),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass


def load_meta(job_dir: Path) -> dict:
    try:
        return json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def runtime_ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} GB"


def time_ago(mtime: float) -> str:
    delta = max(0, int(time.time() - mtime))
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{delta // 60} minute{'s' if delta // 60 != 1 else ''} ago"
    if delta < 86400:
        return f"{delta // 3600} hour{'s' if delta // 3600 != 1 else ''} ago"
    return f"{delta // 86400} day{'s' if delta // 86400 != 1 else ''} ago"


def render_result_error(message: str, job_id: str | None = None) -> str:
    return render_template(
        "_result.html",
        ok=False,
        title="Run rejected",
        job_id=job_id,
        panel={"kind": "error", "label": "Run rejected", "status": "rejected",
               "debug_url": None, "diag_url": None,
               "processing_state": "rejected", "quality_state": "rejected", "quality_label": "",
               "runtime_ms": None, "mode": "", "direction": "", "result_url": None,
               "sources": [], "output_size_wh": None, "file_size": None,
               "blend": None, "batch_rows": None},
        report={"status": "rejected", "reason": message},
        created_at=now_str(),
        message=message,
    )


def blank_to_none(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def describe_config_error(text: str, exc: json.JSONDecodeError) -> str:
    """Turn a JSONDecodeError into a user-facing sentence with the offending line.

    The bare ``Expecting value: line 1 column 1`` message reads like an engine
    bug; this names the problem, the position and shows the line itself.
    """
    if not text.strip():
        return "The configuration is empty — paste or edit the JSON config (or upload a config file), then run again."
    lines = text.splitlines()
    line_no = exc.lineno or 1
    snippet = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
    msg = f"Config JSON is not valid: line {exc.lineno or '?'}, column {exc.colno or '?'} — {exc.msg}."
    if snippet:
        msg += f" Offending line: {snippet[:100]}"
    return msg


def safe_upload_name(name: str) -> str:
    name = Path(name).name
    if not name:
        raise ProcessingError("Uploaded filename is empty.")
    return name


def make_job_id(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid4().hex[:8]}"


def _safe_job_path(job_id: str, filename: str) -> Path | None:
    job_path = (JOBS_DIR / job_id).resolve()
    if job_path.parent != JOBS_DIR.resolve() or not job_path.is_dir():
        return None
    path = (job_path / filename).resolve()
    if job_path not in path.parents and path != job_path:
        return None
    return path if path.is_file() else None


def _safe_upload_path(job_id: str, filename: str) -> Path | None:
    up_path = (UPLOADS_DIR / job_id).resolve()
    if up_path.parent != UPLOADS_DIR.resolve() or not up_path.is_dir():
        return None
    path = (up_path / filename).resolve()
    if up_path not in path.parents and path != up_path:
        return None
    return path if path.is_file() else None


def _job_summary_path(job_id: str) -> Path | None:
    report_path = _safe_job_path(job_id, "report.json")
    if report_path is not None:
        return report_path
    return _safe_job_path(job_id, "batch_report.json")


def list_recent_jobs() -> list[dict[str, str]]:
    items = []
    for path in sorted(JOBS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        report_path = path / "report.json"
        batch_path = path / "batch_report.json"
        summary_path = report_path if report_path.exists() else batch_path if batch_path.exists() else None
        status = "unknown"
        mode = ""
        direction = ""
        quality_state = ""
        result_rel = None
        if summary_path:
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                status = data.get("status", status)
                mode = data.get("mode", "batch" if summary_path.name == "batch_report.json" else "")
                direction = data.get("direction", "")
                quality_state = data.get("quality_state", "")
                if (path / "result.png").exists():
                    result_rel = url_for("job_file", job_id=path.name, filename="result.png")
            except Exception:
                pass
        meta = load_meta(path)
        items.append(
            {
                "job_id": path.name,
                "label": meta.get("label") or path.name,
                "status": status,
                "quality_state": quality_state,
                "mode": mode,
                "direction": direction,
                "updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "ago": time_ago(path.stat().st_mtime),
                "runtime_ms": meta.get("runtime_ms"),
                "thumb_url": result_rel,
                "result_url": result_rel,
                "report_url": url_for("job_file", job_id=path.name, filename=summary_path.name) if summary_path else None,
            }
        )
    return items[:20]


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=8000)
