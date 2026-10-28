
"""
Etap 3 — analiza metadanych i właściwości zdjęć.

Skrypt NIE wykonuje rekonstrukcji 3D ani kalibracji. Zbiera informacje,
które pozwolą zdecydować, jak w etapie 4 wyznaczyć parametry kamer:
- wymiary obrazu (surowe i po uwzględnieniu orientacji EXIF),
- dane EXIF (producent, model, ogniskowa, ekspozycja...),
- wstępne, przybliżone oszacowanie ogniskowej w pikselach,
- proste miary jakości obrazu (ostrość, jasność, prześwietlenia),
- rozmiar znacznika AprilTag w pikselach (z raportu etapu 2),
- ostrzeżenia, np. o usuniętym EXIF lub przeskalowaniu zdjęcia.

Uruchomienie:
    python inspect_images.py              # analizuje output/accepted/
    python inspect_images.py --dir input  # dowolny inny katalog
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

from config import (
    PROJECT_DIR,
    OUTPUT_DIR,
    ACCEPTED_DIR,
    IMAGE_EXTENSIONS,
    APRILTAG_REPORT_PATH,
    IMAGE_INFO_JSON_PATH,
    IMAGE_INFO_CSV_PATH,
    SENSOR_WIDTH_MM,
)

# Identyfikatory podkatalogów EXIF.
EXIF_IFD_POINTER = 0x8769
GPS_IFD_POINTER = 0x8825

# Przekątna klatki 35 mm (36 x 24 mm) — podstawa definicji
# "ogniskowej ekwiwalentnej 35 mm".
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

# Progi ostrzeżeń jakości (heurystyczne, do dostrojenia).
MAX_OVEREXPOSED_FRACTION = 0.05
MIN_TAG_SIDE_PX = 40.0

# Orientacja EXIF -> opis.
ORIENTATION_NAMES = {
    1: "normal",
    2: "mirrored horizontally",
    3: "rotated 180°",
    4: "mirrored vertically",
    5: "transposed",
    6: "rotated 90° clockwise",
    7: "transversed",
    8: "rotated 90° counter-clockwise",
}

# Typowe wymiary dłuższego boku zdjęć po wysłaniu przez komunikatory.
MESSENGER_LONG_SIDES = {1280, 1600, 2000, 2048}


def get_image_files(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def to_jsonable(value):
    """Konwertuje wartości EXIF (IFDRational, bytes, krotki) na typy JSON."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="strict").strip("\x00 ")
            return text if text.isprintable() else value.hex()
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, str):
        return value.strip("\x00 ")
    if isinstance(value, (int, bool)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number) or number <= 0:
        return None
    return number


def read_exif(pil_image: Image.Image):
    """Zwraca (płaski słownik EXIF z nazwami tagów, czy jest GPS)."""
    exif = pil_image.getexif()
    tags = {}

    for tag_id, value in exif.items():
        if tag_id in (EXIF_IFD_POINTER, GPS_IFD_POINTER):
            continue
        tags[ExifTags.TAGS.get(tag_id, f"tag_{tag_id}")] = value

    for tag_id, value in exif.get_ifd(EXIF_IFD_POINTER).items():
        name = ExifTags.TAGS.get(tag_id, f"tag_{tag_id}")
        # Pomijamy duże bloki binarne bez znaczenia dla geometrii.
        if name in ("MakerNote", "UserComment"):
            continue
        tags[name] = value

    gps_present = len(exif.get_ifd(GPS_IFD_POINTER)) > 0
    return tags, gps_present


def focal_length_estimates(exif, width, height):
    """
    Przybliżone ogniskowe w pikselach (fx ≈ fy) z dostępnych metadanych.

    To NIE jest kalibracja: nie znamy punktu głównego ani dystorsji,
    a ekwiwalent 35 mm bywa zaokrąglany przez producentów.
    """
    estimates = []
    diagonal_px = math.hypot(width, height)

    # Metoda 1: ogniskowa ekwiwalentna 35 mm (definicja po przekątnej).
    f35 = as_float(exif.get("FocalLengthIn35mmFilm"))
    if f35:
        estimates.append({
            "method": "FocalLengthIn35mmFilm",
            "f_px": round(f35 * diagonal_px / FULL_FRAME_DIAGONAL_MM, 1),
            "note": "approximation; assumes no cropping of the sensor",
        })

    # Metoda 2: ogniskowa fizyczna + rozdzielczość płaszczyzny ogniskowej.
    focal_mm = as_float(exif.get("FocalLength"))
    plane_res_x = as_float(exif.get("FocalPlaneXResolution"))
    plane_unit = exif.get("FocalPlaneResolutionUnit")
    unit_to_mm = {2: 25.4, 3: 10.0, 4: 1.0}.get(
        int(plane_unit) if plane_unit else 2
    )
    exif_width = as_float(exif.get("ExifImageWidth"))

    if focal_mm and plane_res_x and unit_to_mm:
        px_per_mm = plane_res_x / unit_to_mm
        # Rozdzielczość płaszczyzny odnosi się do ExifImageWidth;
        # przeskaluj, jeśli plik ma inną szerokość.
        scale = 1.0
        if exif_width:
            scale = max(width, height) / max(
                exif_width, as_float(exif.get("ExifImageHeight")) or 0
            )
        estimates.append({
            "method": "FocalLength+FocalPlaneResolution",
            "f_px": round(focal_mm * px_per_mm * scale, 1),
            "note": "approximation; depends on the manufacturer's EXIF tags",
        })

    # Metoda 3: ogniskowa fizyczna + szerokość matrycy ze specyfikacji.
    camera_key = (
        str(to_jsonable(exif.get("Make")) or ""),
        str(to_jsonable(exif.get("Model")) or ""),
    )
    sensor_width_mm = SENSOR_WIDTH_MM.get(camera_key)
    if focal_mm and sensor_width_mm:
        estimates.append({
            "method": "FocalLength+SENSOR_WIDTH_MM",
            "f_px": round(focal_mm / sensor_width_mm * max(width, height), 1),
            "note": "approximation; sensor width from public "
                    "specifications, assumes no cropping",
        })

    return estimates


def image_quality(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "sharpness_laplacian_var": round(sharpness, 1),
        "mean_brightness": round(float(gray.mean()), 1),
        "overexposed_fraction": round(float((gray >= 250).mean()), 4),
        "underexposed_fraction": round(float((gray <= 5).mean()), 4),
    }


def load_tag_info():
    """Słownik: nazwa pliku -> dane znacznika z raportu etapu 2."""
    if not APRILTAG_REPORT_PATH.is_file():
        return {}

    with open(APRILTAG_REPORT_PATH, encoding="utf-8") as file:
        report = json.load(file)

    info = {}
    for entry in report:
        detections = entry.get("detections") or []
        if len(detections) != 1:
            continue
        corners = np.asarray(detections[0]["corners"], dtype=float)
        sides = np.linalg.norm(
            corners - np.roll(corners, -1, axis=0), axis=1
        )
        info[entry["image"]] = {
            "tag_status": entry.get("status"),
            "tag_side_px_mean": round(float(sides.mean()), 1),
            "tag_side_px_min": round(float(sides.min()), 1),
            "tag_side_px_max": round(float(sides.max()), 1),
            "tag_decision_margin": round(
                float(detections[0]["decision_margin"]), 1
            ),
            "tag_center_px": [
                round(v, 1) for v in detections[0]["center"]
            ],
            "tag_report_image_size": [
                entry.get("image_width"), entry.get("image_height")
            ],
        }
    return info


def inspect_image(image_path: Path, tag_info: dict):
    warnings = []
    record = {
        "image": image_path.name,
        "path": str(image_path.relative_to(PROJECT_DIR))
        if image_path.is_relative_to(PROJECT_DIR) else str(image_path),
        "file_size_bytes": image_path.stat().st_size,
        "sha1": hashlib.sha1(image_path.read_bytes()).hexdigest(),
    }

    # --- Pillow: format, surowe wymiary, EXIF ---------------------------
    with Image.open(image_path) as pil_image:
        record["format"] = pil_image.format
        record["color_mode"] = pil_image.mode
        record["progressive_jpeg"] = bool(
            pil_image.info.get("progressive")
            or pil_image.info.get("progression")
        )
        record["jfif"] = "jfif" in pil_image.info
        record["icc_profile_present"] = "icc_profile" in pil_image.info
        raw_width, raw_height = pil_image.size
        exif, gps_present = read_exif(pil_image)

    orientation = exif.get("Orientation")
    orientation = int(orientation) if orientation else None
    rotated = orientation in (5, 6, 7, 8)
    width, height = (
        (raw_height, raw_width) if rotated else (raw_width, raw_height)
    )

    record.update({
        "raw_width": raw_width,
        "raw_height": raw_height,
        "exif_orientation": orientation,
        "exif_orientation_name": ORIENTATION_NAMES.get(orientation),
        # Wymiary w układzie, w którym pracują OpenCV/YOLO/AprilTag.
        "width": width,
        "height": height,
        "megapixels": round(width * height / 1e6, 2),
    })

    # --- EXIF: pola istotne dla geometrii kamery ------------------------
    record.update({
        "exif_present": len(exif) > 0,
        "exif_tag_count": len(exif),
        "camera_make": to_jsonable(exif.get("Make")),
        "camera_model": to_jsonable(exif.get("Model")),
        "lens_make": to_jsonable(exif.get("LensMake")),
        "lens_model": to_jsonable(exif.get("LensModel")),
        "software": to_jsonable(exif.get("Software")),
        "datetime_original": to_jsonable(exif.get("DateTimeOriginal")),
        "focal_length_mm": to_jsonable(exif.get("FocalLength")),
        "focal_length_35mm": to_jsonable(exif.get("FocalLengthIn35mmFilm")),
        "f_number": to_jsonable(exif.get("FNumber")),
        "exposure_time_s": to_jsonable(exif.get("ExposureTime")),
        "iso": to_jsonable(exif.get("ISOSpeedRatings")),
        "digital_zoom_ratio": to_jsonable(exif.get("DigitalZoomRatio")),
        "exif_image_width": to_jsonable(exif.get("ExifImageWidth")),
        "exif_image_height": to_jsonable(exif.get("ExifImageHeight")),
        "gps_present": gps_present,  # tylko informacja, bez współrzędnych
    })

    record["focal_px_estimates"] = focal_length_estimates(
        exif, width, height
    )

    # --- OpenCV: kontrola wymiarów i jakość obrazu ----------------------
    # cv2.imread domyślnie stosuje orientację EXIF, więc jego wymiary
    # powinny odpowiadać polom width/height.
    image = cv2.imread(str(image_path))
    if image is None:
        warnings.append("OpenCV cannot read the image")
        record["opencv_width"] = record["opencv_height"] = None
    else:
        record["opencv_width"] = int(image.shape[1])
        record["opencv_height"] = int(image.shape[0])
        if (image.shape[1], image.shape[0]) != (width, height):
            warnings.append(
                "OpenCV image size differs from the EXIF-oriented size"
            )
        record.update(image_quality(image))

        # Ostrość oceniana jest na znaczniku (image_quality.py) — wariancja
        # Laplasjanu całego zdjęcia zależy od treści i zostaje tylko jako
        # informacja.
        if record["overexposed_fraction"] > MAX_OVEREXPOSED_FRACTION:
            warnings.append("Large share of overexposed pixels")

    # --- AprilTag z etapu 2 ---------------------------------------------
    tag = tag_info.get(image_path.name)
    record.update(tag or {"tag_status": None})
    if tag:
        if tag["tag_report_image_size"] != [width, height]:
            warnings.append(
                "Image size differs from the one in the AprilTag report"
            )
        if tag["tag_side_px_mean"] < MIN_TAG_SIDE_PX:
            warnings.append(
                "AprilTag is small in the image — "
                "pose estimation will be less accurate"
            )
    else:
        warnings.append("No AprilTag data for this image")

    # --- Ostrzeżenia dot. pochodzenia zdjęcia ----------------------------
    if not record["exif_present"]:
        warnings.append(
            "No EXIF — phone model and focal length cannot be determined"
        )
        if record["jfif"] and record["progressive_jpeg"]:
            warnings.append(
                "Progressive JPEG without EXIF — the image probably "
                "went through a messenger or web service "
                "(rescaled and recompressed)"
            )
    else:
        if not record["focal_px_estimates"]:
            if as_float(record["focal_length_mm"]):
                warnings.append(
                    f"EXIF has focal length {record['focal_length_mm']} mm, "
                    "but no sensor size or 35 mm equivalent — cannot "
                    "convert to pixels (add the camera to SENSOR_WIDTH_MM)"
                )
            else:
                warnings.append("EXIF contains no focal length data")
        exif_w = as_float(record["exif_image_width"])
        exif_h = as_float(record["exif_image_height"])
        if exif_w and exif_h and {int(exif_w), int(exif_h)} != {
            raw_width, raw_height
        }:
            warnings.append(
                "EXIF dimensions differ from the actual ones — "
                "the image was rescaled or cropped"
            )
        zoom = as_float(record["digital_zoom_ratio"])
        if zoom and zoom > 1.0:
            warnings.append(
                f"Digital zoom used ({zoom:g}x) — it changes the "
                "effective focal length"
            )

    if max(width, height) in MESSENGER_LONG_SIDES and not record[
        "exif_present"
    ]:
        warnings.append(
            f"Long side of {max(width, height)} px is typical for images "
            "rescaled by messengers"
        )

    record["warnings"] = warnings
    return record


def build_summary(records):
    def unique(key):
        return sorted({
            str(r.get(key)) for r in records if r.get(key) is not None
        })

    with_exif = [r for r in records if r["exif_present"]]
    with_focal = [r for r in records if r["focal_px_estimates"]]

    by_hash = {}
    for r in records:
        by_hash.setdefault(r["sha1"], []).append(r["image"])
    duplicates = [names for names in by_hash.values() if len(names) > 1]

    cameras = sorted({
        f"{r.get('camera_make') or '?'} {r.get('camera_model') or '?'}"
        for r in with_exif
    })

    if not with_exif:
        calibration_hint = (
            "No image contains EXIF. Camera parameters cannot be read "
            "from metadata. Options: (a) get the original files from the "
            "phones (sent as a file/document or by cable), "
            "(b) calibrate the phones with a checkerboard/ChArUco, or "
            "(c) estimate the focal length from the AprilTag/multiple views "
            "— with higher uncertainty."
        )
    elif not with_focal:
        calibration_hint = (
            "Images have EXIF, but the focal length cannot be converted to "
            "pixels. Calibration or a SENSOR_WIDTH_MM entry is needed."
        )
    elif len(with_focal) < len(records):
        calibration_hint = (
            "Some images have focal length data, some do not. Images "
            "without it will need calibration or estimation."
        )
    else:
        calibration_hint = (
            "All images have focal length data. It can be used as a "
            "starting value, but distortion and the principal point still "
            "need calibration or estimation."
        )

    return {
        "images_count": len(records),
        "with_exif": len(with_exif),
        "with_focal_estimate": len(with_focal),
        "cameras": cameras,
        "resolutions": unique_resolutions(records),
        "orientations": unique("exif_orientation"),
        "duplicate_files": duplicates,
        "images_with_warnings": sum(1 for r in records if r["warnings"]),
        "calibration_hint": calibration_hint,
    }


def unique_resolutions(records):
    counts = {}
    for r in records:
        key = f"{r['width']}x{r['height']}"
        counts[key] = counts.get(key, 0) + 1
    return counts


CSV_FIELDS = [
    "image", "width", "height", "raw_width", "raw_height",
    "exif_orientation", "exif_present", "camera_make", "camera_model",
    "lens_model", "focal_length_mm", "focal_length_35mm", "f_px_estimate",
    "digital_zoom_ratio", "datetime_original", "file_size_bytes",
    "progressive_jpeg", "sharpness_laplacian_var", "mean_brightness",
    "overexposed_fraction", "tag_side_px_mean", "tag_decision_margin",
    "warnings",
]


def write_csv(records, path: Path):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=CSV_FIELDS, delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["f_px_estimate"] = (
                r["focal_px_estimates"][0]["f_px"]
                if r["focal_px_estimates"] else ""
            )
            row["warnings"] = " | ".join(r["warnings"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: image metadata and properties."
    )
    parser.add_argument(
        "--dir", type=Path, default=ACCEPTED_DIR,
        help=f"image folder (default {ACCEPTED_DIR})",
    )
    args = parser.parse_args()

    image_dir = args.dir
    if not image_dir.is_absolute():
        image_dir = (PROJECT_DIR / image_dir).resolve()

    if not image_dir.is_dir():
        sys.exit(
            f"ERROR: Folder {image_dir} does not exist. "
            "Run test_apriltag.py first."
        )

    image_files = get_image_files(image_dir)
    if not image_files:
        sys.exit(f"ERROR: No images in folder: {image_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag_info = load_tag_info()
    if not tag_info:
        print(f"WARNING: no AprilTag report ({APRILTAG_REPORT_PATH}).")

    records = []
    def safe_inspect(image_path):
        try:
            return inspect_image(image_path, tag_info), None
        except Exception as exc:  # jedno uszkodzone zdjęcie nie przerywa
            return None, exc

    # Zdjęcia analizowane równolegle (dekodowanie JPEG w OpenCV zwalnia
    # blokadę Pythona); map() zachowuje kolejność zdjęć w raporcie.
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        outcomes = list(pool.map(safe_inspect, image_files))

    for image_path, (record, error) in zip(image_files, outcomes):
        if error is not None:
            print(f"[ERROR] {image_path.name}: {error}")
            continue
        records.append(record)

        camera = (
            f"{record['camera_make'] or ''} {record['camera_model'] or ''}"
        ).strip() or "no EXIF"
        focal = (
            f"f≈{record['focal_px_estimates'][0]['f_px']} px"
            if record["focal_px_estimates"] else "f=?"
        )
        tag = (
            f"tag≈{record['tag_side_px_mean']} px"
            if record.get("tag_side_px_mean") else "tag=?"
        )
        print(
            f"{record['image']:<12} {record['width']}x{record['height']}  "
            f"{camera:<25} {focal:<14} {tag:<14} "
            f"sharpness={record.get('sharpness_laplacian_var')}"
        )

    if not records:
        sys.exit("ERROR: No image could be analysed.")

    summary = build_summary(records)

    with open(IMAGE_INFO_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {"source_dir": str(image_dir), "summary": summary,
             "images": records},
            file, indent=2, ensure_ascii=False,
        )
    write_csv(records, IMAGE_INFO_CSV_PATH)

    print("\n--- SUMMARY ---")
    print(f"Images: {summary['images_count']}")
    print(f"With EXIF: {summary['with_exif']}")
    print(f"With focal length estimate: {summary['with_focal_estimate']}")
    print(f"Cameras: {', '.join(summary['cameras']) or 'unknown'}")
    print(f"Resolutions: {summary['resolutions']}")
    if summary["duplicate_files"]:
        print(f"Identical files: {summary['duplicate_files']}")
    print(f"Images with warnings: {summary['images_with_warnings']}")

    # Unikalne ostrzeżenia z liczbą wystąpień.
    warning_counts = {}
    for r in records:
        for w in r["warnings"]:
            warning_counts[w] = warning_counts.get(w, 0) + 1
    for w, n in sorted(warning_counts.items(), key=lambda x: -x[1]):
        print(f"  [{n:>2}x] {w}")

    print(f"\nConclusion: {summary['calibration_hint']}")
    print(f"\nJSON report: {IMAGE_INFO_JSON_PATH}")
    print(f"CSV report:  {IMAGE_INFO_CSV_PATH}")


if __name__ == "__main__":
    main()
