"""
Analiza zgłoszeń w osobnym procesie (uruchamiany przez evaluation.py).
Awaria procesu — np. zabicie przez brak pamięci — nie przerywa serwera.

    python -m evaluation_function.worker <katalog zadania>   # jedno zadanie
    python -m evaluation_function.worker --zygote            # proces-matka
    python -m evaluation_function.worker --serve             # proces stały

Tryb --zygote (Linux, serwer rpc — domyślny w kontenerze): biblioteki
(torch, ultralytics, open3d, scipy), kod GearBox i model YOLO (rozgrzany
jedną predykcją) ładowane są raz; każde zadanie liczy KOPIA tego procesu
(os.fork) — start w milisekundach zamiast ~10-15 s, a pamięć zadania wraca
do systemu po jego zakończeniu (jak w procesie jednorazowym). Proces-matka
nie uruchamia wątków obliczeniowych (torch i OpenCV: 1 wątek podczas
ładowania), więc kopia może bezpiecznie liczyć wielowątkowo; po starcie
sprawdza to krótki test w kopii (@@READY ... "fork": false, gdy się nie
powiedzie, albo gdy torch widzi GPU — CUDA nie działa w kopii procesu).
Protokół (stdin/stdout, linia tekstu):
    <- @@READY <pid> {"fork": true|false, "preload_s"}   raz
    -> <katalog zadania>
    <- @@STARTED <pid kopii>         kopia jest liderem grupy procesów
    <- @@JOB_DONE {"status", "return_code"}

Każde zadanie (kopia procesu-matki albo proces jednorazowy) tworzy na
początku pulę procesów (forkpool.py, EDUVISION_WORKERS) dla etapów "cad"
i "fuse" — bez /dev/shm, więc także na AWS Lambda.

Tryb --serve: biblioteki i model ładowane raz, proces czeka na kolejne
zadania. Protokół:
    -> <katalog zadania>
    <- @@READY <pid>                     po załadowaniu bibliotek (raz)
    <- @@JOB_DONE {"status", "rss_mb", "jobs", "restart"}
Po zadaniu, gdy pamięć procesu przekroczy EDUVISION_WORKER_MAX_MB albo
wykona on EDUVISION_WORKER_MAX_JOBS zadań, proces kończy się ("restart":
true), a evaluation.py uruchamia nowy.

Katalog zadania:
    job.json      {"images": [nazwy plików w images/], "stages": [ścieżki
                  YAML], "problems": [...], "params": {...}}
    images/       zdjęcia
    result.json   wynik: {"is_correct", "items", "stats"}
    crash.log     stos wywołań po awarii procesu

Zmienne środowiskowe (oprócz evaluation.py / pipeline.py):
    EDUVISION_TORCH_THREADS  wątki torch dla YOLO (domyślnie min(8, CPU))
    EDUVISION_POOL_WORKER_MB pamięć na proces puli przy doborze jej
                             rozmiaru (domyślnie 150; zmierzone ~40-70);
                             rezerwa dla samego zadania: pipeline.py
    EDUVISION_FORK_TEST_S    limit czasu testu fork (domyślnie 60 s)
"""

import faulthandler
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROCESS_START = time.time()
# Stos wywołań po awarii procesu (np. SIGSEGV) — evaluation.py dołącza go do
# sekcji debug, bo stderr trafia tylko do logów serwera.
CRASH_LOG = "crash.log"
MAX_MB = float(os.environ.get("EDUVISION_WORKER_MAX_MB", "3000"))
MAX_JOBS = int(os.environ.get("EDUVISION_WORKER_MAX_JOBS", "20"))
# Test fork trwa zwykle kilka sekund; dłużej = zawieszenie.
SELF_TEST_TIMEOUT_S = float(os.environ.get("EDUVISION_FORK_TEST_S", "60"))
POOL_WORKER_MB = float(os.environ.get("EDUVISION_POOL_WORKER_MB", "150"))


def peak_memory_mb():
    """Szczytowe zużycie pamięci procesu (od jego startu) [MB]."""
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                     / 1024)
    except ImportError:             # Windows
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 2**20)
        except (ImportError, AttributeError):
            return None


def current_memory_mb():
    try:
        with open("/proc/self/statm") as file:
            return round(int(file.read().split()[1])
                         * os.sysconf("SC_PAGE_SIZE") / 2**20)
    except (OSError, AttributeError, ValueError):
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / 2**20)
        except ImportError:
            return None


def cpu_info():
    """Do sekcji debug: ile CPU widzi proces i ile wątków używa torch."""
    info = {"cpu_count": os.cpu_count()}
    try:
        info["cpu_affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    from .pipeline import effective_cpus
    info["cpu_effective"] = effective_cpus()
    if "torch" in sys.modules:
        info["torch_threads"] = sys.modules["torch"].get_num_threads()
    return info


def gpu_info():
    """Do sekcji debug: czy torch widzi GPU i czy YOLO naprawdę z niego
    korzystał (szczyt pamięci karty > 0 MB w tym procesie)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    available = torch.cuda.is_available()
    info = {"torch_version": torch.__version__, "cuda_available": available}
    if available:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    detect_parts = sys.modules.get("detect_parts")
    if detect_parts is not None:
        info["yolo_timings"] = detect_parts.YOLO_TIMINGS
    return info


def _log(text):
    print(f"[eduvision] {text}", file=sys.stderr, flush=True)


def log_steps():
    """Postęp etapów na stderr (logi serwera, na bieżąco)."""
    from . import pipeline
    rp = pipeline._load_modules()["run_pipeline"]
    if getattr(rp.run_step, "_logged", False):
        return
    run_step = rp.run_step

    def logged_step(key, module, args, out):
        step_start = time.time()
        ok = run_step(key, module, args, out)
        print(f"[eduvision] step {key} {'ok' if ok else 'FAILED'} "
              f"{time.time() - step_start:.1f} s",
              file=sys.stderr, flush=True)
        return ok

    logged_step._logged = True
    rp.run_step = logged_step


def torch_threads(cpus):
    """Wątki torch (YOLO): EDUVISION_TORCH_THREADS, domyślnie min(8, CPU).
    ultralytics sam ustawiał min(8, os.cpu_count() - 1): na 2 vCPU (AWS
    Lambda) 1 wątek, a przy limicie CPU kontenera nadmiar wątków. Liczba
    wątków zmienia wyniki YOLO tylko w ostatnich bitach (ramki ±0.0003 px,
    maski identyczne) — tak jak dotąd różne liczby CPU instancji."""
    setting = os.environ.get("EDUVISION_TORCH_THREADS", "").strip()
    if setting.isdigit() and int(setting) > 0:
        return int(setting)
    return max(1, min(8, cpus))


def limit_threads():
    """Wątki na zadanie: OpenCV — wszystkie CPU (import ultralytics ustawia
    1; wyniki OpenCV nie zależą od liczby wątków), torch — torch_threads.
    CPU = rzeczywiście dostępne (limit kontenera)."""
    from .pipeline import effective_cpus
    cpus = effective_cpus()
    import cv2
    cv2.setNumThreads(cpus)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(torch_threads(cpus))


def preload(threads=None, everything=True):
    """Import bibliotek i kodu GearBox (bez modelu). threads=1: bez puli
    wątków (przed fork()). everything=False: tylko to, co importuje
    pipeline (proces jednorazowy — reszta i tak przy pierwszym użyciu)."""
    if threads is not None:
        import cv2
        cv2.setNumThreads(threads)
        import torch
        torch.set_num_threads(threads)
    from . import pipeline
    pipeline._load_modules()
    log_steps()
    if not everything:
        return
    from . import attachments, feedback, render  # noqa: F401
    import check_assembly  # noqa: F401
    import estimate_camera_poses  # noqa: F401
    import inspect_images  # noqa: F401
    import test_apriltag  # noqa: F401
    import view_assembly  # noqa: F401
    try:
        # Magazyn obrazów (klient chmury): ciężki import.
        from lf_toolkit.evaluation import image_upload  # noqa: F401
    except Exception:
        pass


def warm_up(threads=None):
    """Import bibliotek i model YOLO (z jedną predykcją), zanim przyjdzie
    pierwsze zadanie."""
    import numpy as np
    preload(threads)
    import detect_parts
    # Jednorazowa część pierwszej predykcji (fuzja warstw, konfiguracja
    # predyktora, leniwe importy; ~2.5 s) nie zależy od rozmiaru obrazu —
    # mały obraz wystarcza (pierwsza predykcja 1280 px po takiej
    # rozgrzewce jest tak szybka jak kolejne).
    detect_parts.load_model().predict(
        np.zeros((480, 640, 3), np.uint8), imgsz=160, retina_masks=True,
        verbose=False)


def job_outputs(params):
    """Obrazy feedbacku w każdym trybie -> potrzebne obrazy diagnostyczne
    (None — wszystkie, gdy params["images"] jest nietypowe)."""
    from . import feedback
    try:
        return {mode: sorted(feedback.wanted_images(params, mode))
                for mode in ("3d", "notag")}
    except Exception:
        return None


def native_threads():
    """Wątki systemowe procesu (Linux) — do logu procesu-matki."""
    try:
        with open("/proc/self/status") as file:
            for line in file:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


# ------------------------------------------------------ pula zadania

def _die_with_parent():
    """Proces kończy się razem z rodzicem (Linux: PR_SET_PDEATHSIG) —
    procesy puli nie zostają po zabitym zadaniu."""
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)     # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def job_pool_size():
    """Rozmiar puli: pipeline.pool_size() (CPU), ograniczony wolną pamięcią
    — przekroczenie pamięci na AWS Lambda kończy całe środowisko."""
    from . import pipeline
    size = pipeline.pool_size()
    if size < 2:
        return 0, None
    available = pipeline.available_memory_mb()
    if available is not None and not os.environ.get(
            "EDUVISION_WORKERS", "").strip().isdigit():
        size = min(size, int((available - pipeline.JOB_RESERVE_MB)
                             // POOL_WORKER_MB))
    return (size if size >= 2 else 0), available


def start_job_pool():
    """Pula procesów dla etapów cad/fuse, tworzona na samym początku
    zadania — kopie procesu jeszcze bez wątków obliczeniowych."""
    from . import pipeline
    size, available = job_pool_size()
    info = {"pool_workers": size, "memory_available_mb":
            None if available is None else round(available)}
    if size < 2:
        return None, info
    from .forkpool import ForkPool
    started = time.time()
    try:
        pool = ForkPool(size, on_start=_die_with_parent)
    except OSError as exc:
        info["pool_error"] = str(exc)
        return None, info
    pipeline.set_shared_pool(pool)
    info["pool_start_seconds"] = round(time.time() - started, 3)
    return pool, info


def _enable_crash_log(job_dir):
    """faulthandler -> crash.log zadania. Plik zostaje otwarty do końca
    procesu (faulthandler z niego korzysta)."""
    crash_log = open(job_dir / CRASH_LOG, "w", encoding="utf-8")
    faulthandler.enable(file=crash_log, all_threads=True)
    return crash_log


def run_job(job_dir, jobs_done=0, extra_stats=None):
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    from . import feedback, pipeline
    from .attachments import Attacher

    log_steps()
    images = [(name, (job_dir / "images" / name).read_bytes())
              for name in job["images"]]
    params = job["params"]
    started = time.time()
    info = {}

    def finish(run):
        info.update(executor=run["executor"], steps=run["steps"])
        return feedback.build(run, job["problems"], Attacher(params), params)

    is_correct, items = pipeline.run(images, job["stages"], finish=finish,
                                     outputs=job_outputs(params))
    stats = {
        "analysis_seconds": round(time.time() - started, 1),
        "worker_jobs_before": jobs_done,
        "peak_memory_mb": peak_memory_mb(),
        "memory_after_mb": current_memory_mb(),
        "accel": pipeline._load_modules()["accel"],
        **(extra_stats or {}),
        **info,
        **cpu_info(),
        **gpu_info(),
    }
    (job_dir / "result.json").write_text(
        json.dumps({"is_correct": is_correct, "items": items,
                    "stats": stats}, ensure_ascii=False),
        encoding="utf-8")


def serve():
    import shutil
    import tempfile
    base = Path(os.environ.get("EDUVISION_WORK_DIR",
                               Path(tempfile.gettempdir()) / "eduvision"))
    work_dir = base / f"worker_{os.getpid()}"
    os.environ["EDUVISION_WORK_DIR"] = str(work_dir)
    try:
        warm_up()
        limit_threads()
        print(f"[eduvision] worker ready in "
              f"{time.time() - PROCESS_START:.1f} s",
              file=sys.stderr, flush=True)
        print(f"@@READY {os.getpid()}", flush=True)
        jobs = 0
        for line in sys.stdin:          # koniec stdin = serwer zakończony
            job_dir = Path(line.strip())
            if not line.strip():
                continue
            try:
                run_job(job_dir, jobs, extra_stats={"worker_mode": "reuse"})
                status = "ok"
            except Exception:
                traceback.print_exc(file=sys.stderr)
                status = "error"
            jobs += 1
            rss = current_memory_mb()
            restart = jobs >= MAX_JOBS or (rss is not None and rss > MAX_MB)
            print("@@JOB_DONE " + json.dumps({
                "status": status, "rss_mb": rss, "jobs": jobs,
                "restart": restart}), flush=True)
            if restart:
                break
    except (OSError, ValueError):       # serwer zamknął potok
        try:
            traceback.print_exc(file=sys.stderr)
        except OSError:
            pass
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------- tryb zygote

def _pool_probe(value):
    """Zadanie testowe w procesie puli: to, czego używają etapy cad/fuse."""
    import cv2
    import numpy as np

    import estimate_part_poses
    from scipy.optimize import minimize
    cv2.setNumThreads(1)
    cv2.resize(np.zeros((600, 800), np.uint8), (256, 192),
               interpolation=cv2.INTER_NEAREST)
    minimize(lambda p: float((p ** 2).sum()), np.ones(3), method="Powell")
    estimate_part_poses.load_models()
    return value


def _job_child(job_dir, preload_s):
    """Kopia procesu-matki: jedno zadanie, potem os._exit (bez sprzątania
    bibliotek — pamięć zwalnia koniec procesu)."""
    code = 1
    crash_log = None
    try:
        os.setpgid(0, 0)                # grupa: zadanie + jego pula
        _die_with_parent()
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)             # stdin/stdout: kanał procesu-matki
        os.dup2(devnull, 1)
        crash_log = _enable_crash_log(job_dir)
        pool, pool_info = start_job_pool()  # przed wątkami obliczeniowymi
        limit_threads()
        run_job(job_dir, extra_stats={
            "worker_mode": "fork", "preload_seconds": preload_s,
            **pool_info})
        if pool is not None:
            pool.shutdown(wait=True)
        code = 0
    except BaseException:
        try:
            traceback.print_exc(file=sys.stderr)
            if crash_log is not None:
                traceback.print_exc(file=crash_log)
                crash_log.flush()
        except Exception:
            pass
    finally:
        try:
            # Pliki zadania (kopie zdjęć, maski) nie czekają do następnego
            # zgłoszenia: /tmp na Cloud Run zajmuje pamięć, a na AWS Lambda
            # ma domyślnie 512 MB.
            import shutil
            from . import pipeline
            for folder in (pipeline.INPUT_DIR, pipeline.OUTPUT_DIR):
                shutil.rmtree(folder, ignore_errors=True)
            sys.stderr.flush()
        finally:
            os._exit(code)


def _fork_self_test():
    """Kopia procesu-matki liczy wielowątkowo to samo co zadanie (torch +
    YOLO, OpenCV, numpy/scipy, open3d, pula procesów). Zawieszenie albo
    błąd = tryb fork niebezpieczny w tym środowisku."""
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            import numpy as np
            os.setpgid(0, 0)
            _die_with_parent()
            pool, _ = start_job_pool()
            limit_threads()
            import cv2
            import detect_parts
            import estimate_part_poses
            from scipy.optimize import minimize
            if pool is not None:
                list(pool.map(_pool_probe, range(2)))
            detect_parts.load_model().predict(
                np.zeros((480, 640, 3), np.uint8), imgsz=640,
                retina_masks=True, verbose=False)
            cv2.resize(np.zeros((3000, 4000, 3), np.uint8), (2000, 1500))
            np.linalg.inv(np.eye(200) + 1)
            minimize(lambda p: float((p ** 2).sum()), np.ones(3),
                     method="Powell")
            estimate_part_poses.load_models()
            if pool is not None:
                pool.shutdown(wait=True)
            code = 0
        except BaseException:
            traceback.print_exc(file=sys.stderr)
        finally:
            os._exit(code)
    deadline = time.time() + SELF_TEST_TIMEOUT_S
    while time.time() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status) == 0
        time.sleep(0.05)
    try:
        os.killpg(pid, 9)
    except OSError:
        pass
    os.waitpid(pid, 0)
    return False


def _cuda_visible():
    """GPU widoczne dla torch: CUDA nie działa w kopii procesu (fork)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def zygote():
    import shutil
    import tempfile
    base = Path(os.environ.get("EDUVISION_WORK_DIR",
                               Path(tempfile.gettempdir()) / "eduvision"))
    work_dir = base / f"zygote_{os.getpid()}"
    # Przed importem pipeline (czyta katalog roboczy przy imporcie).
    os.environ["EDUVISION_WORK_DIR"] = str(work_dir)
    try:
        if _cuda_visible():
            _log("zygote: GPU visible — fork mode disabled")
            print(f"@@READY {os.getpid()} "
                  + json.dumps({"fork": False, "reason": "cuda"}),
                  flush=True)
            return
        # 1 wątek torch / OpenCV: bez pul wątków przed fork() (wątki
        # ustawia kopia — limit_threads).
        warm_up(threads=1)
        preload_s = round(time.time() - PROCESS_START, 1)
        import threading
        python_threads = threading.active_count()
        fork_ok = _fork_self_test()
        _log(f"zygote ready in {time.time() - PROCESS_START:.1f} s "
             f"(preload {preload_s} s, fork test "
             f"{'ok' if fork_ok else 'FAILED'}; threads: python "
             f"{python_threads}, native {native_threads()})")
        print(f"@@READY {os.getpid()} "
              + json.dumps({"fork": fork_ok, "preload_s": preload_s}),
              flush=True)
        if not fork_ok:
            return
        for line in sys.stdin:          # koniec stdin = serwer zakończony
            job_dir = line.strip()
            if not job_dir:
                continue
            sys.stdout.flush()
            sys.stderr.flush()
            pid = os.fork()
            if pid == 0:
                _job_child(Path(job_dir), preload_s)
            print(f"@@STARTED {pid}", flush=True)
            _, status = os.waitpid(pid, 0)
            code = os.waitstatus_to_exitcode(status)
            print("@@JOB_DONE " + json.dumps({
                "status": "ok" if code == 0 else "error",
                "return_code": code}), flush=True)
    except (OSError, ValueError):       # serwer zamknął potok
        try:
            traceback.print_exc(file=sys.stderr)
        except OSError:
            pass
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    if sys.argv[1] == "--serve":
        serve()
        return
    if sys.argv[1] == "--zygote":
        zygote()
        return
    job_dir = Path(sys.argv[1])
    # Przed importem bibliotek, żeby objąć też awarię przy ich ładowaniu.
    crash_log = _enable_crash_log(job_dir)
    # Przed importem pipeline (czyta katalog roboczy przy imporcie).
    os.environ["EDUVISION_WORK_DIR"] = str(job_dir / "work")
    try:
        from . import pipeline
        # Biblioteki przed utworzeniem puli (procesy puli je dziedziczą),
        # 1 wątek do tego czasu — jak w procesie-matce.
        preload(threads=1 if pipeline.pool_size() >= 2 else None,
                everything=False)
        pool, pool_info = start_job_pool()
        limit_threads()
        run_job(job_dir, extra_stats={"worker_mode": "once", **pool_info})
        if pool is not None:
            pool.shutdown(wait=True)
    except Exception:
        traceback.print_exc(file=crash_log)
        crash_log.flush()
        raise


if __name__ == "__main__":
    main()
