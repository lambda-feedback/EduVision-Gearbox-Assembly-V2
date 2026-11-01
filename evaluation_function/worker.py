"""
Analiza jednego zgłoszenia w osobnym procesie (uruchamiany przez
evaluation.py). Po zakończeniu proces znika i oddaje całą pamięć (torch,
YOLO, zdjęcia), a jego awaria — np. zabicie przez brak pamięci — nie
przerywa serwera.

    python -m evaluation_function.worker <katalog zadania>

Katalog zadania:
    job.json      {"images": [nazwy plików w images/], "stages": [ścieżki
                  YAML], "problems": [...], "params": {...}}
    images/       zdjęcia
    work/         katalog roboczy analizy (tylko tego zgłoszenia)
    result.json   wynik: {"is_correct", "items", "stats"}
"""

import json
import os
import sys
import time
from pathlib import Path


def peak_memory_mb():
    """Szczytowe zużycie pamięci tego procesu i jego dzieci [MB]."""
    try:
        import resource
        peak_kb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                   + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        return round(peak_kb / 1024)
    except ImportError:             # Windows
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 2**20)
        except (ImportError, AttributeError):
            return None


def main():
    job_dir = Path(sys.argv[1])
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    # Przed importem pipeline (czyta katalog roboczy przy imporcie).
    os.environ["EDUVISION_WORK_DIR"] = str(job_dir / "work")

    from . import feedback, pipeline
    from .attachments import Attacher

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
        "seconds": round(time.time() - started, 1),
        "peak_memory_mb": peak_memory_mb(),
        **info,
    }
    (job_dir / "result.json").write_text(
        json.dumps({"is_correct": is_correct, "items": items,
                    "stats": stats}, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()
