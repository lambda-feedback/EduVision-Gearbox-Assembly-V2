"""
EduVision — ocena montażu przekładni ze zdjęć studenta.

    response  zdjęcia (adresy URL / data URI / base64; formaty: inputs.py)
    answer    etap do sprawdzenia: nazwa etapu dołączonego do funkcji
              (np. "etap_pelny") albo definicja YAML/JSON (jak
              assemblies/*.yaml w aplikacji GearBox); lista = wszystkie
              etapy muszą być zaliczone
    params    opcjonalnie:
                stage        etap, gdy answer jest pusty
                images       ["3d", "3d_top", "top_view", "photos",
                             "detections"],
                             true (domyślne) lub false
                max_photos   ile zdjęć dla "photos"/"detections" (3)
                image_mode   "upload" (domyślnie) | "inline" | "none"
                image_folder katalog w magazynie obrazów ("eduvision")
                debug        true — log analizy, czas i pamięć w feedbacku

Wynik: is_correct = wszystkie etapy ZALICZONE; feedback (Markdown) — ocena,
powody, części, połączenia, problemy ze zdjęciami, obrazy.

Analiza działa w osobnym procesie (worker.py) z własnym katalogiem
roboczym: pamięć wraca do systemu po każdym zgłoszeniu, równoległe
zgłoszenia sobie nie przeszkadzają, a brak pamięci lub przekroczenie czasu
kończy się komunikatem dla studenta zamiast błędu serwera.

Zmienne środowiskowe:
    EDUVISION_WORK_DIR      katalog na zadania (domyślnie <temp>/eduvision)
    EDUVISION_TIME_LIMIT_S  limit czasu jednej analizy (domyślnie 600 s)
    EDUVISION_SUBPROCESS    "0" — analiza w tym procesie (diagnostyka)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from lf_toolkit.evaluation import Params, Result

from . import feedback, inputs

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Kody zakończenia procesu zabitego przez system (SIGKILL — zwykle brak
# pamięci): -9 (Python na Linuksie) lub 137 (128 + 9, przez powłokę).
KILLED_CODES = (-9, 137)


def _log(text):
    """Wpis w logach serwera (stderr — nie koliduje z kanałem RPC)."""
    print(f"[eduvision] {text}", file=sys.stderr, flush=True)


def _error(text):
    return Result(is_correct=False,
                  feedback_items=[("error", f"\n\n{text}\n\n")])


def _cannot_assess(text):
    return Result(is_correct=False, feedback_items=[(
        "error", f"\n\n### ⚠️ CANNOT ASSESS\n\n{text}\n\n")])


def _jobs_dir():
    folder = Path(os.environ.get("EDUVISION_WORK_DIR",
                                 Path(tempfile.gettempdir()) / "eduvision"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _debug_item(stats, log=""):
    lines = ["### Debug", ""]
    lines += [f"- {key}: `{value}`" for key, value in stats.items()]
    if log:
        lines += ["", "```", log[-4000:].replace("```", "'''"), "```"]
    return ("debug", "\n\n" + "\n".join(lines) + "\n\n")


def _run_in_process(images, stage_paths, problems, params):
    from . import pipeline
    from .attachments import Attacher
    return pipeline.run(
        images, stage_paths,
        finish=lambda run: feedback.build(run, problems, Attacher(params),
                                          params))


def _run_in_subprocess(job_dir, images, stage_paths, problems, params):
    (job_dir / "images").mkdir()
    for name, data in images:
        (job_dir / "images" / name).write_bytes(data)
    (job_dir / "job.json").write_text(json.dumps({
        "images": [name for name, _ in images],
        "stages": [str(p) for p in stage_paths],
        "problems": problems,
        "params": params,
    }, ensure_ascii=False), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE_ROOT)] + [p for p in sys.path if p])
    limit = float(os.environ.get("EDUVISION_TIME_LIMIT_S", "600"))
    started = time.time()
    try:
        # stdout procesu przechwycony (stdout serwera bywa kanałem RPC);
        # stderr — postęp etapów — idzie na bieżąco do logów serwera.
        proc = subprocess.run(
            [sys.executable, "-m", "evaluation_function.worker",
             str(job_dir)],
            cwd=str(PACKAGE_ROOT), env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, timeout=limit)
    except subprocess.TimeoutExpired:
        return None, {"status": "timeout", "limit_s": limit,
                      "process_seconds": round(time.time() - started, 1)}, ""

    log = proc.stdout.decode("utf-8", errors="replace")
    stats = {"return_code": proc.returncode,
             "process_seconds": round(time.time() - started, 1)}
    result_file = job_dir / "result.json"
    if proc.returncode != 0 or not result_file.is_file():
        stats["status"] = ("killed" if proc.returncode in KILLED_CODES
                           else "error")
        return None, stats, log
    result = json.loads(result_file.read_text(encoding="utf-8"))
    stats.update(result.get("stats", {}), status="ok")
    return result, stats, log


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:

    params = dict(params or {})
    request_start = time.time()
    _log("request received")

    try:
        stages = inputs.stage_definitions(answer, params)
    except ValueError as exc:
        return _error(f"Task configuration error: {feedback._e(exc)}")

    try:
        images, problems = inputs.load_images(response)
    except inputs.SubmissionError as exc:
        return _error(feedback._e(exc))
    download_seconds = round(time.time() - request_start, 1)
    _log(f"{len(images)} photo(s) downloaded in {download_seconds} s")

    job_dir = Path(tempfile.mkdtemp(prefix="job_", dir=_jobs_dir()))
    try:
        stage_paths = []
        (job_dir / "stages").mkdir()
        for index, (name, text) in enumerate(stages):
            path = job_dir / "stages" / f"{index:02d}_{name}"
            path.write_text(text, encoding="utf-8")
            stage_paths.append(path)

        if os.environ.get("EDUVISION_SUBPROCESS", "1") == "0":
            is_correct, items = _run_in_process(images, stage_paths,
                                                problems, params)
            return Result(is_correct=is_correct, feedback_items=items)

        result, stats, log = _run_in_subprocess(
            job_dir, images, stage_paths, problems, params)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    stats["photos"] = len(images)
    stats["download_seconds"] = download_seconds
    stats["total_seconds"] = round(time.time() - request_start, 1)
    _log(f"done: status {stats['status']}, total {stats['total_seconds']} s, "
         f"peak memory {stats.get('peak_memory_mb')} MB")
    debug = params.get("debug")
    if result is None:
        if stats["status"] == "timeout":
            text = (f"The analysis took too long (over "
                    f"{stats['limit_s']:.0f} s) and was stopped. Please try "
                    "again, with 4-10 photos.")
        elif stats["status"] == "killed":
            text = ("The analysis ran out of memory on the server and was "
                    "stopped. Please try again, with 4-10 photos.")
        else:
            text = "The analysis failed because of an internal error."
        res = _cannot_assess(text)
        if debug:
            res.add_feedback(*_debug_item(stats, log))
        return res

    items = [tuple(item) for item in result["items"]]
    if debug:
        items.append(_debug_item(stats))
    return Result(is_correct=result["is_correct"], feedback_items=items)
