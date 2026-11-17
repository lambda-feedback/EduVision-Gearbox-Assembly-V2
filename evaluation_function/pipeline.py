"""
Uruchomienie analizy GearBox (kod skopiowany do gearbox/ skryptem
tools/sync_gearbox.py) dla jednego zestawu zdjęć.

Te same etapy co w aplikacji (run_pipeline.STEPS, bez podglądu 3D), w tym
samym procesie. Katalog roboczy jest stały (config.py czyta ścieżki przy
imporcie) i czyszczony przy każdym zgłoszeniu — zgłoszenia wykonywane są
po kolei (blokada).

Przyspieszenia (bez zmiany wyników analizy):
    accel.py      szybsze funkcje dopasowania sylwetki CAD (identyczne
                  wyniki),
    pula procesów zadania (forkpool.py; tworzy ją worker.py na początku
                  zadania) — etapy "cad" i "fuse" liczone równolegle, także
                  na AWS Lambda (bez /dev/shm); procesy są kopiami procesu
                  zadania, więc nie importują bibliotek od nowa,
    obrazy diagnostyczne GearBox tworzone tylko, gdy pokazuje je feedback
                  (run(..., outputs=...)) — etapy nie wczytują zdjęć tylko po
                  to, by na nich rysować, i nie zapisują JPEG 12 MPx,
    liczba wątków wg CPU rzeczywiście dostępnych (limit kontenera).

Zmienne środowiskowe:
    EDUVISION_WORK_DIR     katalog roboczy (domyślnie <temp>/eduvision)
    EDUVISION_WORKERS      "auto" (domyślnie) — pula procesów zadania (fork),
                           a bez niej procesy spawn, jeśli system na to
                           pozwala; "0" — wszystko w jednym procesie; liczba
                           — tyle procesów; "spawn" — zawsze nowe procesy
                           (jak w aplikacji na Windows)
    EDUVISION_DEBUG_IMAGES "1" — wszystkie obrazy diagnostyczne (jak
                           w aplikacji), także gdy feedback ich nie pokazuje
"""

import contextlib
import io
import itertools
import json
import math
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
            # Initializery puli ustawiają 1 wątek OpenCV (równoległość daje
            # pula procesów) — tu wszystko liczy się w jednym procesie,
            # więc liczba wątków wraca do poprzedniej.
            import cv2
            threads = cv2.getNumThreads()
            initializer(*initargs)
            cv2.setNumThreads(threads)

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


# --------------------------------------------------- pula procesów zadania

_POOL = {"pool": None, "fallback": InlineExecutor}
_POOL_STATE = {}                    # w procesie roboczym puli
_INLINE_LOCK = threading.Lock()


def set_shared_pool(pool):
    """Pula procesów (forkpool.ForkPool) dla etapów GearBox w tym
    zgłoszeniu; None — brak (etapy używają wtedy executora zastępczego)."""
    _POOL["pool"] = pool


def _pool_call(token, initializer, initargs, outputs, fn, args, kwargs):
    """Zadanie w procesie roboczym puli: initializer executora (raz na
    proces i executor), stan obrazów diagnostycznych, właściwe zadanie."""
    _OUTPUTS.update(outputs)
    if _POOL_STATE.get("token") != token:
        _POOL_STATE.clear()
        if initializer is not None:
            initializer(*initargs)
        _POOL_STATE["token"] = token
    return fn(*args, **kwargs)


class SharedPoolExecutor:
    """Zamiennik ProcessPoolExecutor dla etapów GearBox: zadania trafiają do
    puli procesów zadania (set_shared_pool), tworzonej raz na zgłoszenie —
    bez startu nowych procesów w każdym etapie. initializer wywoływany raz
    na proces roboczy (przed pierwszym zadaniem tego executora). Bez puli —
    executor zastępczy (spawn albo w tym procesie).

    Gdy proces puli padnie (np. zabity przy braku pamięci), zadania liczone
    są dalej w tym procesie — ten sam wynik (zadania nie zależą od procesu,
    w którym są liczone), tylko wolniej."""

    _ids = itertools.count()

    def __new__(cls, *args, **kwargs):
        if _POOL["pool"] is None:
            return _POOL["fallback"](*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, max_workers=None, initializer=None, initargs=(),
                 **_):
        self.pool = _POOL["pool"]
        self.token = f"{os.getpid()}:{next(self._ids)}"
        self.initializer = initializer
        self.initargs = tuple(initargs)
        self.inline_ready = False

    def _inline(self, fn, args, kwargs):
        """Zadanie w tym procesie (initializer raz, jak InlineExecutor)."""
        with _INLINE_LOCK:
            if not self.inline_ready:
                InlineExecutor(initializer=self.initializer,
                               initargs=self.initargs)
                self.inline_ready = True
            return fn(*args, **kwargs)

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures.process import BrokenProcessPool
        outer = Future()

        def run_inline():
            if outer.set_running_or_notify_cancel():
                try:
                    outer.set_result(self._inline(fn, args, kwargs))
                except BaseException as exc:
                    outer.set_exception(exc)

        def done(inner):
            error = inner.exception()
            if isinstance(error, BrokenProcessPool):
                run_inline()
            elif error is not None:
                outer.set_exception(error)
            else:
                outer.set_result(inner.result())

        try:
            inner = self.pool.submit(_pool_call, self.token,
                                     self.initializer, self.initargs,
                                     dict(_OUTPUTS), fn, args, kwargs)
        except BrokenProcessPool:
            run_inline()
            return outer
        inner.add_done_callback(done)
        return outer

    def map(self, fn, *iterables, **_):
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        return [future.result() for future in futures]

    def shutdown(self, wait=True, **_):
        pass                            # pula żyje do końca zadania

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def _setting():
    return os.environ.get("EDUVISION_WORKERS", "auto").strip().lower()


def fork_available():
    return hasattr(os, "fork") and sys.platform.startswith("linux")


def pool_size():
    """Ile procesów ma pula zadania (0 — bez puli): EDUVISION_WORKERS
    (liczba) albo liczba dostępnych CPU (od 2)."""
    setting = _setting()
    if not fork_available() or setting in ("0", "inline", "off", "spawn"):
        return 0
    if setting.isdigit():
        return int(setting) if int(setting) >= 2 else 0
    cpus = effective_cpus()
    return cpus if cpus >= 2 else 0


# ------------------------------------------------------ procesory (CPU)

def effective_cpus():
    """Procesory dostępne naprawdę: przypisanie procesu i limit cgroup
    (kontener z limitem CPU potrafi widzieć wszystkie rdzenie serwera)."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:          # Windows, macOS
        count = os.cpu_count() or 1
    quota = None
    try:                            # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as file:
            limit, period = file.read().split()[:2]
        if limit != "max":
            quota = int(limit) / int(period)
    except (OSError, ValueError):
        try:                        # cgroup v1
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as file:
                limit = int(file.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as file:
                period = int(file.read())
            if limit > 0 and period > 0:
                quota = limit / period
        except (OSError, ValueError):
            pass
    if quota:
        count = min(count, max(1, math.ceil(quota)))
    return max(1, count)


def _apply_cpu_limit(cpus):
    """Wątki AprilTaga i rozmiary pul wg rzeczywistej liczby CPU (wynik nie
    zależy od tych wartości — config.py)."""
    import config
    if cpus >= config.CPU_COUNT:
        return
    for name in ("config", "test_apriltag", "estimate_part_poses",
                 "fuse_views", "inspect_images"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "CPU_COUNT"):
            module.CPU_COUNT = cpus


def _keep_torch_threads():
    """ultralytics przy pierwszej predykcji (select_device) ustawia
    torch.set_num_threads(min(8, os.cpu_count() - 1)) — na 2 vCPU (AWS
    Lambda) to 1 wątek, a przy limicie CPU kontenera nadmiar wątków. Tu
    liczba wątków zostaje taka, jaką ustawił worker.py (torch_threads).
    Proces-matka trybu fork rozgrzewa model jednowątkowo: pula wątków
    OpenMP utworzona przed fork() zawiesza torch w kopii procesu."""
    try:
        import torch
        from ultralytics.engine import predictor
    except ImportError:
        return
    original = predictor.select_device
    if getattr(original, "_keeps_threads", False):
        return

    def select_device(*args, **kwargs):
        threads = torch.get_num_threads()
        try:
            return original(*args, **kwargs)
        finally:
            torch.set_num_threads(threads)

    select_device._keeps_threads = True
    predictor.select_device = select_device


def available_memory_mb():
    """Wolna pamięć: MemAvailable systemu albo limit cgroup minus zużycie,
    jeśli mniej (None — nieznana, np. Windows)."""
    available = None
    try:
        with open("/proc/meminfo") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 1024
                    break
    except (OSError, ValueError):
        return None
    for limit_file, usage_file in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:                        # cgroup v2 / v1: limit - zużycie
            with open(limit_file) as file:
                limit = file.read().strip()
            if limit == "max" or int(limit) >= 2**60:   # bez limitu
                continue
            with open(usage_file) as file:
                used = int(file.read())
            free = (int(limit) - used) / 2**20
            return free if available is None else min(available, free)
        except (OSError, ValueError):
            continue
    return available


# Pamięć potrzebna samemu zadaniu (zdjęcia, YOLO, maski) ponad procesy
# puli — zmierzone ~950 MB dla 8 zdjęć 12 MPx.
JOB_RESERVE_MB = float(os.environ.get("EDUVISION_JOB_RESERVE_MB", "1200"))


def processes_available():
    """Czy można tworzyć procesy (AWS Lambda: brak semaforów /dev/shm)."""
    try:
        multiprocessing.get_context("spawn").Lock()
        return True
    except (OSError, ImportError, NotImplementedError):
        return False


# ----------------------------------------------- obrazy diagnostyczne

# Obrazy diagnostyczne GearBox: czy tworzyć (run() ustawia wg feedbacku).
#   tag_debug, pose_debug, part_pose_debug — feedback ich nie pokazuje,
#   detections — detection_debug/*_det.jpg (obrazy "detections"),
#   photos     — fusion_debug/*_fused.jpg (obrazy "photos").
_OUTPUTS = {"tag_debug": True, "pose_debug": True, "part_pose_debug": True,
            "detections": True, "photos": True}


class _FilteredCv2:
    """cv2 dla modułu GearBox: pomija wczytanie zdjęcia, które służy tylko
    do rysowania obrazu diagnostycznego, i zapis takiego obrazu, gdy nie
    jest potrzebny (_OUTPUTS). Pozostałe funkcje — zwykłe cv2."""

    def __init__(self, cv2, reads=None, writes=None):
        self._cv2 = cv2
        self._reads = {str(k): v for k, v in (reads or {}).items()}
        self._writes = {str(k): v for k, v in (writes or {}).items()}

    def __getattr__(self, name):
        value = getattr(self._cv2, name)
        if name not in ("imread", "imwrite"):      # ultralytics je łata
            setattr(self, name, value)
        return value

    def imread(self, filename, *args, **kwargs):
        key = self._reads.get(os.path.dirname(str(filename)))
        if key is not None and not _OUTPUTS[key]:
            return None
        return self._cv2.imread(filename, *args, **kwargs)

    def imwrite(self, filename, img, *args, **kwargs):
        key = self._writes.get(os.path.dirname(str(filename)))
        if key is not None and not _OUTPUTS[key]:
            return True
        return self._cv2.imwrite(filename, img, *args, **kwargs)


def _install_output_filters(config):
    """Każdy z tych modułów wczytuje zdjęcie z accepted/ wyłącznie po to,
    by narysować obraz diagnostyczny (estimate_camera_poses, estimate_part_
    poses, fuse_views: rzuty części), albo rysuje go z gotowych danych
    (test_apriltag, detect_parts)."""
    import cv2

    import detect_parts
    import estimate_camera_poses
    import estimate_part_poses
    import fuse_views
    import test_apriltag

    test_apriltag.cv2 = _FilteredCv2(
        cv2, writes={config.TAG_DEBUG_DIR: "tag_debug"})
    estimate_camera_poses.cv2 = _FilteredCv2(
        cv2, reads={config.ACCEPTED_DIR: "pose_debug"},
        writes={config.POSE_DEBUG_DIR: "pose_debug"})
    detect_parts.cv2 = _FilteredCv2(
        cv2, writes={config.DETECTION_DEBUG_DIR: "detections"})
    draw_debug = detect_parts.draw_debug

    def draw_detections(*args, **kwargs):
        return draw_debug(*args, **kwargs) if _OUTPUTS["detections"] else None

    detect_parts.draw_debug = draw_detections
    estimate_part_poses.cv2 = _FilteredCv2(
        cv2, reads={config.ACCEPTED_DIR: "part_pose_debug"},
        writes={config.PART_POSE_DEBUG_DIR: "part_pose_debug"})
    fuse_views.cv2 = _FilteredCv2(cv2, reads={config.ACCEPTED_DIR: "photos"})


def _set_outputs(outputs, mode):
    """outputs: {tryb: rodzaje obrazów feedbacku} lub None (wszystkie)."""
    everything = (outputs is None
                  or os.environ.get("EDUVISION_DEBUG_IMAGES") == "1")
    wanted = set() if everything else set(outputs.get(mode, ()))
    _OUTPUTS.update(tag_debug=everything, pose_debug=everything,
                    part_pose_debug=everything,
                    detections=everything or "detections" in wanted,
                    photos=everything or "photos" in wanted)


# ---------------------------------------------------------------- moduły

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

    import detect_parts

    from . import accel
    accelerated = accel.apply(estimate_part_poses, fuse_views)
    accel.apply_detect(detect_parts)
    _install_output_filters(config)
    _apply_cpu_limit(effective_cpus())
    _keep_torch_threads()

    setting = _setting()
    if setting in ("0", "inline", "off") or (
            setting != "spawn" and not processes_available()):
        fallback = InlineExecutor
    else:
        # "spawn" także na Linuksie: fork procesu z wątkami (serwer IPC,
        # torch) potrafi się zawiesić; spawn = jak w aplikacji na Windows.
        fallback = partial(ProcessPoolExecutor,
                           mp_context=multiprocessing.get_context("spawn"))
        if setting.isdigit():
            estimate_part_poses.PARALLEL_WORKERS = int(setting)
            fuse_views.PARALLEL_WORKERS = int(setting)
    _POOL["fallback"] = fallback
    estimate_part_poses.ProcessPoolExecutor = SharedPoolExecutor
    fuse_views.ProcessPoolExecutor = SharedPoolExecutor

    _modules = {"config": config, "run_pipeline": run_pipeline,
                "accel": accelerated}
    return _modules


def executor_name():
    """Do sekcji debug: jak liczone są etapy cad/fuse."""
    if _POOL["pool"] is not None:
        return f"fork pool ({_POOL['pool'].size})"
    return "inline" if _POOL["fallback"] is InlineExecutor else "processes"


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


def run(images, stage_paths, finish=None, outputs=None):
    """
    images: [(nazwa pliku, bajty)], stage_paths: pliki YAML etapów.
    Wynik: słownik z raportami etapów (None, gdy etap nie powstał):
        mode, steps [(klucz, ok, czas)], failed_step, log, executor,
        apriltag, image_info, camera, detections, part_poses, fused, check,
        output_dir.
    finish(wynik) — wywoływane przed zwolnieniem katalogu roboczego (pliki
    w output_dir są jeszcze aktualne); run() zwraca wtedy jego wartość.
    outputs: {tryb ("3d"/"notag"): rodzaje obrazów feedbacku, np.
    ["3d", "top_view"]} — tworzone są tylko potrzebne obrazy diagnostyczne;
    None — wszystkie (jak w aplikacji).
    """
    with _LOCK:
        modules = _load_modules()
        config, rp = modules["config"], modules["run_pipeline"]
        _reset_work_dir()
        for name, data in images:
            (INPUT_DIR / name).write_bytes(data)

        stage_args = [str(Path(p).resolve()) for p in stage_paths]
        steps, failed, mode = [], None, MODE_3D
        _set_outputs(outputs, mode)
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
                        _set_outputs(outputs, mode)
                        print("@@MODE notag", flush=True)
                        plan = list(rp.NO_TAG_STEPS)
            finally:
                sys.argv = saved_argv

        result = {
            "mode": mode,
            "steps": steps,
            "failed_step": failed,
            "seconds": round(time.time() - started, 1),
            "executor": executor_name(),
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
