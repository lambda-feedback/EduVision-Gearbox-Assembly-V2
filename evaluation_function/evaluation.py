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

Analiza działa w osobnym procesie (worker.py), nowym dla każdego
zgłoszenia: pamięć wraca do systemu po każdej analizie, a brak pamięci lub
przekroczenie czasu kończy się komunikatem dla studenta zamiast błędu
serwera. Opcjonalnie (EDUVISION_WORKER_REUSE=1) proces zostaje między
zgłoszeniami (worker.py --serve, biblioteki i model ładowane raz; ~10-15 s
szybciej) — wyłączone domyślnie, bo na AWS Lambda taki proces potrafił się
przerwać (SIGABRT); po awarii zadanie jest powtarzane w procesie
jednorazowym.

Zmienne środowiskowe:
    EDUVISION_WORK_DIR        katalog na zadania (domyślnie <temp>/eduvision)
    EDUVISION_TIME_LIMIT_S    limit czasu jednej analizy (domyślnie 600 s)
    EDUVISION_WORKER_REUSE    "1" — proces stały między zgłoszeniami
    EDUVISION_PRESTART        "1" — proces stały startuje razem z serwerem
    EDUVISION_WORKER_MAX_MB   próg pamięci restartu (domyślnie 3000)
    EDUVISION_WORKER_MAX_JOBS restart po tylu zadaniach (domyślnie 20)
    EDUVISION_SUBPROCESS      "0" — analiza w procesie serwera (diagnostyka)
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
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


def _write_job(job_dir, images, stage_paths, problems, params):
    (job_dir / "images").mkdir()
    for name, data in images:
        (job_dir / "images" / name).write_bytes(data)
    (job_dir / "job.json").write_text(json.dumps({
        "images": [name for name, _ in images],
        "stages": [str(p) for p in stage_paths],
        "problems": problems,
        "params": params,
    }, ensure_ascii=False), encoding="utf-8")


def _worker_command(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE_ROOT)] + [p for p in sys.path if p])
    return ([sys.executable, "-m", "evaluation_function.worker", *args],
            {"cwd": str(PACKAGE_ROOT), "env": env})


class _Worker:
    """Stały proces analizy (worker.py --serve). stdout procesu to kanał
    poleceń (stdout serwera bywa kanałem RPC, więc nie jest dziedziczony);
    stderr — postęp etapów — idzie na bieżąco do logów serwera."""

    def __init__(self):
        command, options = _worker_command("--serve")
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            **options)
        self.lines = queue.Queue()
        self.pid = None                 # PID właściwego procesu (@@READY)
        threading.Thread(target=self._read, daemon=True).start()
        _log(f"worker process started (pid {self.proc.pid})")

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)                     # proces się zakończył

    def alive(self):
        return self.proc.poll() is None

    def kill(self):
        """Zakończenie procesu i usunięcie jego katalogu roboczego."""
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        # PID z @@READY (na Windows python.exe z venv to pośrednik o innym
        # PID niż właściwy proces).
        shutil.rmtree(_jobs_dir() / f"worker_{self.pid or self.proc.pid}",
                      ignore_errors=True)

    def run(self, job_dir, limit):
        """-> (status "done"/"timeout"/"exited", dane, log)."""
        deadline = time.time() + limit
        log = []
        try:
            self.proc.stdin.write(f"{job_dir}\n")
            self.proc.stdin.flush()
        except OSError:
            return "exited", {"return_code": self.proc.poll()}, log
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self.kill()
                return "timeout", {}, log
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                self.proc.wait()
                return "exited", {"return_code": self.proc.returncode}, log
            if line.startswith("@@JOB_DONE "):
                return "done", json.loads(line[len("@@JOB_DONE "):]), log
            if line.startswith("@@READY"):
                self.pid = int(line.split()[1])
            else:
                log.append(line)


_WORKER = None
_WORKER_LOCK = threading.Lock()


def _get_worker():
    global _WORKER
    if _WORKER is not None and not _WORKER.alive():
        _WORKER.kill()                  # sprzątnięcie po martwym procesie
        _WORKER = None
    if _WORKER is None:
        _WORKER = _Worker()
    return _WORKER


def _run_in_worker(job_dir, limit):
    """Zadanie w stałym procesie analizy (uruchamianym w razie potrzeby)."""
    global _WORKER
    with _WORKER_LOCK:
        worker = _get_worker()
        status, data, log = worker.run(job_dir, limit)
        if status != "done" or data.get("restart"):
            if status == "done":
                try:
                    worker.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            worker.kill()
            _WORKER = None
            # Następny proces ładuje biblioteki już teraz, a nie przy
            # kolejnym zgłoszeniu.
            _WORKER = _Worker()
    stats = {"worker": status}
    if status == "timeout":
        stats["status"] = "timeout"
    elif status == "exited":
        code = data.get("return_code")
        stats.update(return_code=code, status="killed"
                     if code in KILLED_CODES else "error")
    else:
        stats.update(worker_rss_mb=data.get("rss_mb"),
                     worker_jobs=data.get("jobs"),
                     worker_restart=data.get("restart"),
                     status="ok" if data.get("status") == "ok" else "error")
    return stats, "\n".join(log)


def _run_once(job_dir, limit):
    """Zadanie w nowym procesie (EDUVISION_WORKER_REUSE=0)."""
    command, options = _worker_command(str(job_dir))
    try:
        proc = subprocess.run(command, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, timeout=limit,
                              **options)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}, ""
    code = proc.returncode
    status = "ok" if code == 0 else (
        "killed" if code in KILLED_CODES else "error")
    return ({"return_code": code, "status": status},
            proc.stdout.decode("utf-8", errors="replace"))


_REUSE_FAILED = False   # stały proces padł — dalej tylko procesy jednorazowe


def _reuse_enabled():
    """Proces stały tylko na życzenie (EDUVISION_WORKER_REUSE=1): na AWS
    Lambda proces czekający między wywołaniami potrafił się przerwać
    (SIGABRT) po odmrożeniu kontenera."""
    return os.environ.get("EDUVISION_WORKER_REUSE", "0") == "1"


def _run_in_subprocess(job_dir, images, stage_paths, problems, params):
    global _REUSE_FAILED, _WORKER
    _write_job(job_dir, images, stage_paths, problems, params)
    limit = float(os.environ.get("EDUVISION_TIME_LIMIT_S", "600"))
    started = time.time()
    if not _reuse_enabled() or _REUSE_FAILED:
        stats, log = _run_once(job_dir, limit)
    else:
        stats, log = _run_in_worker(job_dir, limit)
        if stats.get("worker") == "exited":
            # Awaria stałego procesu (np. SIGABRT): to samo zadanie jeszcze
            # raz w nowym procesie jednorazowym, a kolejne zgłoszenia już
            # bez procesu stałego.
            _log(f"worker process failed (return code "
                 f"{stats.get('return_code')}); retrying in a one-shot "
                 f"process and disabling worker reuse:\n{log[-3000:]}")
            _REUSE_FAILED = True
            with _WORKER_LOCK:
                if _WORKER is not None:
                    _WORKER.kill()
                    _WORKER = None
            remaining = max(limit - (time.time() - started), 1)
            failed = {f"worker_failure_{k}": v for k, v in stats.items()}
            stats, log = _run_once(job_dir, remaining)
            stats.update(failed)
    stats["process_seconds"] = round(time.time() - started, 1)
    stats["limit_s"] = limit

    result_file = job_dir / "result.json"
    if stats["status"] != "ok" or not result_file.is_file():
        if stats["status"] == "ok":
            stats["status"] = "error"
        return None, stats, log
    result = json.loads(result_file.read_text(encoding="utf-8"))
    stats.update(result.get("stats", {}))
    return result, stats, log


if os.environ.get("EDUVISION_PRESTART") == "1" and _reuse_enabled():
    # Proces analizy ładuje biblioteki od razu przy starcie serwera.
    _WORKER = _Worker()


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
