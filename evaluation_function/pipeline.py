"""
Uruchomienie analizy GearBox (kod skopiowany do gearbox/ skryptem
tools/sync_gearbox.py) dla jednego zestawu zdjęć.

Te same etapy co w aplikacji (run_pipeline.STEPS, bez podglądu 3D), w tym
samym procesie. Katalog roboczy jest stały (config.py czyta ścieżki przy
imporcie) i czyszczony przy każdym zgłoszeniu — zgłoszenia wykonywane są
po kolei (blokada).

Zmienne środowiskowe:
    EDUVISION_WORK_DIR   katalog roboczy (domyślnie <temp>/eduvision)
    EDUVISION_WORKERS    "auto" (domyślnie) — procesy, jeśli system na to
                         pozwala (AWS Lambda nie ma /dev/shm), "0" — wszystko
                         w jednym procesie, liczba — tyle procesów.
"""

import contextlib
import io
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from pathlib import Path

GEARBOX_DIR = Path(__file__).resolve().parent / "gearbox"
WORK_DIR = Path(os.environ.get("EDUVISION_WORK_DIR",
                               Path(tempfile.gettempdir()) / "eduvision"))
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output"

MODE_3D = "3d"
MODE_NO_TAG = "notag"

# Etapy, po których błędzie nadal warto sprawdzić etapy (z detekcji).
_SOFT_STEPS = {"camera", "cad", "fuse"}

_LOCK = threading.Lock()
_modules = None


class InlineExecutor:
    """Zamiennik ProcessPoolExecutor: zadania liczone od razu, w tym
    procesie (initializer wywoływany raz, jak w procesie roboczym)."""

    def __init__(self, max_workers=None, initializer=None, initargs=(),
                 **_):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def map(self, fn, *iterables, **_):
        return [fn(*args) for args in zip(*iterables)]

    def shutdown(self, wait=True, **_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def processes_available():
    """Czy można tworzyć procesy (AWS Lambda: brak semaforów /dev/shm)."""
    try:
        multiprocessing.get_context("spawn").Lock()
        return True
    except (OSError, ImportError, NotImplementedError):
        return False


def _load_modules():
    """Import kodu GearBox z katalogiem roboczym ustawionym w config.py."""
    global _modules
    if _modules is not None:
        return _modules

    os.environ["GEARBOX_INPUT_DIR"] = str(INPUT_DIR)
    os.environ["GEARBOX_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ.setdefault("YOLO_VERBOSE", "False")
    if str(GEARBOX_DIR) not in sys.path:
        sys.path.insert(0, str(GEARBOX_DIR))

    import config
    if Path(config.OUTPUT_DIR) != OUTPUT_DIR:
        raise RuntimeError(f"config.py was imported before the work "
                           f"directory was set ({config.OUTPUT_DIR})")
    import run_pipeline
    import estimate_part_poses
    import fuse_views

    setting = os.environ.get("EDUVISION_WORKERS", "auto").strip().lower()
    if setting in ("0", "inline", "off") or (
            setting == "auto" and not processes_available()):
        executor = InlineExecutor
    else:
        # "spawn" także na Linuksie: fork procesu z wątkami (serwer RPC,
        # torch) potrafi się zawiesić; spawn = jak w aplikacji na Windows.
        executor = partial(ProcessPoolExecutor,
                           mp_context=multiprocessing.get_context("spawn"))
        if setting.isdigit():
            estimate_part_poses.PARALLEL_WORKERS = int(setting)
            fuse_views.PARALLEL_WORKERS = int(setting)
    estimate_part_poses.ProcessPoolExecutor = executor
    fuse_views.ProcessPoolExecutor = executor

    _modules = {"config": config, "run_pipeline": run_pipeline,
                "executor": "inline" if executor is InlineExecutor
                else "processes"}
    return _modules


@contextlib.contextmanager
def _captured_stdout():
    """Wyjście etapów do bufora (stdout może być kanałem serwera RPC).
    Przekierowany jest też deskryptor 1 — biblioteki C i procesy robocze."""
    buffer = io.StringIO()
    sys.stdout.flush()
    log_file = tempfile.TemporaryFile(mode="w+b")
    saved_fd = None
    try:
        saved_fd = os.dup(1)
        os.dup2(log_file.fileno(), 1)
    except OSError:
        saved_fd = None
    try:
        with contextlib.redirect_stdout(buffer):
            yield buffer
    finally:
        sys.stdout.flush()
        if saved_fd is not None:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
        log_file.seek(0)
        extra = log_file.read().decode("utf-8", errors="replace")
        log_file.close()
        if extra.strip():
            buffer.write("\n--- native output ---\n" + extra)


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _reset_work_dir():
    for folder in (INPUT_DIR, OUTPUT_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)


def run(images, stage_paths, finish=None):
    """
    images: [(nazwa pliku, bajty)], stage_paths: pliki YAML etapów.
    Wynik: słownik z raportami etapów (None, gdy etap nie powstał):
        mode, steps [(klucz, ok, czas)], failed_step, log, executor,
        apriltag, image_info, camera, detections, part_poses, fused, check,
        output_dir.
    finish(wynik) — wywoływane przed zwolnieniem katalogu roboczego (pliki
    w output_dir są jeszcze aktualne); run() zwraca wtedy jego wartość.
    """
    with _LOCK:
        modules = _load_modules()
        config, rp = modules["config"], modules["run_pipeline"]
        _reset_work_dir()
        for name, data in images:
            (INPUT_DIR / name).write_bytes(data)

        stage_args = [str(Path(p).resolve()) for p in stage_paths]
        steps, failed, mode = [], None, MODE_3D
        saved_argv = sys.argv
        started = time.time()
        with _captured_stdout() as log:
            try:
                plan = [s for s in rp.STEPS if s[0] != "view"]
                while plan:
                    key, _, module, args = plan.pop(0)
                    if module == "check_assembly":
                        args = stage_args
                    step_start = time.time()
                    ok = rp.run_step(key, module, args, OUTPUT_DIR)
                    steps.append((key, ok, round(time.time() - step_start,
                                                 1)))
                    if not ok:
                        failed = failed or key
                        if key in _SOFT_STEPS:
                            # Bez 3D: kontrola z samych detekcji.
                            plan = [s for s in plan if s[0] == "check"]
                            continue
                        break
                    if key == "tag" and rp.accepted_count() == 0:
                        mode = MODE_NO_TAG
                        print("@@MODE notag", flush=True)
                        plan = list(rp.NO_TAG_STEPS)
            finally:
                sys.argv = saved_argv

        result = {
            "mode": mode,
            "steps": steps,
            "failed_step": failed,
            "seconds": round(time.time() - started, 1),
            "executor": modules["executor"],
            "log": log.getvalue(),
            "apriltag": _load_json(config.APRILTAG_REPORT_PATH),
            "image_info": _load_json(config.IMAGE_INFO_JSON_PATH),
            "camera": _load_json(config.CAMERA_POSES_PATH),
            "detections": _load_json(config.DETECTIONS_PATH),
            "part_poses": _load_json(config.PART_POSES_PATH),
            "fused": _load_json(config.FUSED_PARTS_PATH),
            "check": _load_json(config.ASSEMBLY_CHECK_PATH),
            "output_dir": OUTPUT_DIR,
        }
        return finish(result) if finish else result
