
"""
Cała analiza w jednym procesie: etapy wykonywane po kolei (te same
funkcje main() co przy uruchamianiu skryptów osobno), ale biblioteki
(torch, ultralytics, open3d, scipy) i model ładowane są tylko raz.

Katalogi sesji: zmienne GEARBOX_INPUT_DIR / GEARBOX_OUTPUT_DIR (config.py).
Brak AprilTaga na wszystkich zdjęciach: tylko detekcja na wszystkich
zdjęciach i kontrola etapów "tylko elementy".

Znaczniki postępu (dla aplikacji), każdy w osobnej linii:
    @@START <klucz>
    @@END <klucz> <kod wyjścia> <czas s>
    @@MODE notag

Uruchomienie:
    python run_pipeline.py
"""

import importlib
import json
import sys
import time
import traceback

# (klucz, opis, moduł, argumenty)
STEPS = [
    ("tag", "Image filter and AprilTag detection", "test_apriltag", []),
    ("exif", "Image metadata (EXIF)", "inspect_images", []),
    ("camera", "Camera calibration and pose", "estimate_camera_poses", []),
    ("yolo", "Part detection (YOLO)", "detect_parts", []),
    ("cad", "CAD fitting per image", "estimate_part_poses", []),
    ("fuse", "Multi-view fusion (3D)", "fuse_views", []),
    ("check", "Assembly check", "check_assembly", []),
    ("view", "3D model", "view_assembly",
     ["--no-window", "--screenshot", "{output}/assembly_3d.png"]),
]
NO_TAG_STEPS = [
    ("yolo_all", "Part detection on all images", "detect_parts",
     ["--all-images"]),
    ("check_all", "Stage check (parts only)", "check_assembly", []),
]


def run_step(key, module_name, args, output_dir):
    print(f"@@START {key}", flush=True)
    sys.argv = [f"{module_name}.py"] + [
        a.replace("{output}", str(output_dir)) for a in args]
    started = time.time()
    try:
        importlib.import_module(module_name).main()
        code = 0
    except SystemExit as exc:
        # sys.exit("BLAD: ...") — komunikat jak przy osobnym skrypcie.
        if isinstance(exc.code, str):
            print(exc.code, flush=True)
            code = 1
        else:
            code = exc.code or 0
    except Exception:
        traceback.print_exc(file=sys.stdout)
        code = 1
    sys.stdout.flush()
    print(f"@@END {key} {code} {time.time() - started:.1f}", flush=True)
    return code == 0


def accepted_count():
    from config import APRILTAG_REPORT_PATH
    if not APRILTAG_REPORT_PATH.is_file():
        return 0
    with open(APRILTAG_REPORT_PATH, encoding="utf-8") as file:
        return sum(e.get("status") == "ACCEPTED" for e in json.load(file))


def main():
    from config import OUTPUT_DIR

    started = time.time()
    times = []
    for key, text, module, args in STEPS:
        step_start = time.time()
        if not run_step(key, module, args, OUTPUT_DIR):
            sys.exit(1)
        times.append((text, time.time() - step_start))
        if key == "tag" and accepted_count() == 0:
            print("@@MODE notag", flush=True)
            for k, t, m, a in NO_TAG_STEPS:
                step_start = time.time()
                if not run_step(k, m, a, OUTPUT_DIR):
                    sys.exit(1)
                times.append((t, time.time() - step_start))
            break

    print("\n--- STAGE TIMES ---")
    for text, seconds in times:
        print(f"  {text:<42} {seconds:6.1f} s")
    print(f"  {'TOTAL':<42} {time.time() - started:6.1f} s")


if __name__ == "__main__":
    main()
