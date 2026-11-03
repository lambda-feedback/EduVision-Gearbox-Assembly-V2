"""
Analiza zgłoszeń w osobnym procesie (uruchamiany przez evaluation.py).
Awaria procesu — np. zabicie przez brak pamięci — nie przerywa serwera.

    python -m evaluation_function.worker <katalog zadania>   # jedno zadanie
    python -m evaluation_function.worker --serve             # proces stały

Tryb --serve: biblioteki (torch, ultralytics, open3d) i model YOLO są
ładowane raz, a proces czeka na kolejne zadania — oszczędza ~10-15 s na
zgłoszeniu. Protokół (stdin/stdout, linia tekstu):
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


def warm_up():
    """Import bibliotek i model YOLO (z jedną predykcją), zanim przyjdzie
    pierwsze zadanie."""
    import numpy as np

    from . import pipeline
    pipeline._load_modules()
    import check_assembly  # noqa: F401
    import detect_parts
    import estimate_camera_poses  # noqa: F401
    import inspect_images  # noqa: F401
    import test_apriltag  # noqa: F401
    import view_assembly  # noqa: F401
    from config import IMAGE_SIZE
    detect_parts.load_model().predict(
        np.zeros((480, 640, 3), np.uint8), imgsz=IMAGE_SIZE,
        retina_masks=True, verbose=False)


def run_job(job_dir, jobs_done=0):
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

    is_correct, items = pipeline.run(images, job["stages"], finish=finish)
    stats = {
        "analysis_seconds": round(time.time() - started, 1),
        "worker_jobs_before": jobs_done,
        "peak_memory_mb": peak_memory_mb(),
        "memory_after_mb": current_memory_mb(),
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
                run_job(job_dir, jobs)
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


def main():
    if sys.argv[1] == "--serve":
        serve()
        return
    job_dir = Path(sys.argv[1])
    # Przed importem bibliotek, żeby objąć też awarię przy ich ładowaniu.
    # Plik zostaje otwarty do końca procesu (faulthandler z niego korzysta).
    crash_log = open(job_dir / CRASH_LOG, "w", encoding="utf-8")
    faulthandler.enable(file=crash_log, all_threads=True)
    # Przed importem pipeline (czyta katalog roboczy przy imporcie).
    os.environ["EDUVISION_WORK_DIR"] = str(job_dir / "work")
    try:
        run_job(job_dir)
    except Exception:
        traceback.print_exc(file=crash_log)
        crash_log.flush()
        raise


if __name__ == "__main__":
    main()
