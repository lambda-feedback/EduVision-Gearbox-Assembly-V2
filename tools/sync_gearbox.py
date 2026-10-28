"""
Copy the GearBox analysis code and assets from the main application into
evaluation_function/gearbox/ so the evaluation function runs exactly the
same pipeline as the desktop app.

Run after every change in the main application:

    python tools/sync_gearbox.py                 # main app = ../../ (default)
    python tools/sync_gearbox.py C:/path/to/GearBoxNew
"""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "evaluation_function" / "gearbox"

# Pipeline modules (GUI, 3D viewer and session handling are not needed).
MODULES = [
    "config.py",
    "geometry.py",
    "image_quality.py",
    "run_pipeline.py",
    "test_apriltag.py",
    "inspect_images.py",
    "estimate_camera_poses.py",
    "detect_parts.py",
    "estimate_part_poses.py",
    "fuse_views.py",
    "check_assembly.py",
    "view_assembly.py",     # tylko scene_geometry() — render robi render.py
]
ASSET_GLOBS = {
    "models": ["best.pt"],
    "cad": ["*.stl"],
    "assemblies": ["*.yaml", "*.yml"],
}


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parents[1]
    source = source.resolve()
    if not (source / "run_pipeline.py").is_file():
        sys.exit(f"ERROR: {source} is not the GearBox application folder")

    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    for name in MODULES:
        shutil.copy2(source / name, TARGET / name)
    for folder, patterns in ASSET_GLOBS.items():
        (TARGET / folder).mkdir()
        for pattern in patterns:
            for path in sorted((source / folder).glob(pattern)):
                shutil.copy2(path, TARGET / folder / path.name)

    (TARGET / "SOURCE.txt").write_text(
        "Copied by tools/sync_gearbox.py from the GearBox application.\n"
        "Do not edit here — change the application and re-run the script.\n",
        encoding="utf-8")
    files = sorted(p.relative_to(TARGET) for p in TARGET.rglob("*")
                   if p.is_file())
    print(f"Copied {len(files)} files from {source} to {TARGET}:")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
