
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

import image_quality
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    IMAGE_EXTENSIONS,
    APRILTAG_FAMILY,
    APRILTAG_ID,
    APRILTAG_SIZE_M,
    APRILTAG_QUAD_DECIMATE_STEPS,
    ACCEPTED_DIR,
    REJECTED_DIR,
    APRILTAG_REPORT_PATH,
    TAG_DEBUG_DIR,
)

# Minimalna jakość detekcji.
MIN_DECISION_MARGIN = 20.0

# Wymagamy, aby wszystkie narożniki znacznika były
# wewnątrz zdjęcia, z niewielkim zapasem.
TAG_EDGE_MARGIN_PX = 3


def get_image_files(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def detect_with_fallback(detectors, gray):
    """
    Próbuje kolejnych wartości quad_decimate, aż znajdzie znacznik
    o oczekiwanym ID. Zwraca (wszystkie detekcje, użyte quad_decimate).
    """
    detections = []
    for quad_decimate, detector in detectors:
        detections = detector.detect(gray, estimate_tag_pose=False)
        if any(tag.tag_id == APRILTAG_ID for tag in detections):
            return detections, quad_decimate
    return detections, None


def main():
    # Utwórz katalogi wynikowe.
    for folder in (
        OUTPUT_DIR,
        ACCEPTED_DIR,
        REJECTED_DIR,
        TAG_DEBUG_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    # Usuń kopie z poprzedniego uruchomienia, aby katalogi
    # odpowiadały wyłącznie bieżącej zawartości input/.
    for folder in (ACCEPTED_DIR, REJECTED_DIR, TAG_DEBUG_DIR):
        for old_file in get_image_files(folder):
            old_file.unlink()

    detectors = [
        (
            quad_decimate,
            Detector(
                families=APRILTAG_FAMILY,
                # Wątki dzielą pracę nad jednym zdjęciem; wynik detekcji
                # nie zależy od ich liczby.
                nthreads=os.cpu_count() or 2,
                quad_decimate=quad_decimate,
                refine_edges=1,
            ),
        )
        for quad_decimate in APRILTAG_QUAD_DECIMATE_STEPS
    ]

    image_files = get_image_files(INPUT_DIR)

    if not image_files:
        print(f"No images in folder: {INPUT_DIR}")
        return

    report = []

    accepted_count = 0
    rejected_count = 0

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            rejected_count += 1
            shutil.copy2(
                image_path,
                REJECTED_DIR / image_path.name,
            )
            report.append({
                "image": image_path.name,
                "status": "REJECTED",
                "reason": ["Cannot read the image"],
                "detections": [],
            })
            print(f"[REJECTED] {image_path.name}: read error")
            continue

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detections, used_quad_decimate = detect_with_fallback(
            detectors, gray
        )

        matching_tags = [
            tag for tag in detections
            if tag.tag_id == APRILTAG_ID
        ]

        reasons = []
        tag_details = []

        # Na potrzeby tego etapu wymagamy dokładnie jednego
        # znacznika o oczekiwanym ID.
        if len(matching_tags) == 0:
            reasons.append(
                f"AprilTag ID {APRILTAG_ID} not detected — make sure the "
                "whole tag is in the frame, sharp and well lit"
            )
        elif len(matching_tags) > 1:
            reasons.append(
                f"Multiple tags with ID {APRILTAG_ID} detected"
            )

        for tag in matching_tags:
            corners = np.asarray(tag.corners, dtype=float)

            corners_inside = bool(
                np.all(corners[:, 0] >= TAG_EDGE_MARGIN_PX)
                and np.all(
                    corners[:, 0] <= width - 1 - TAG_EDGE_MARGIN_PX
                )
                and np.all(corners[:, 1] >= TAG_EDGE_MARGIN_PX)
                and np.all(
                    corners[:, 1] <= height - 1 - TAG_EDGE_MARGIN_PX
                )
            )

            hamming_ok = int(tag.hamming) == 0
            margin_ok = (
                float(tag.decision_margin) >= MIN_DECISION_MARGIN
            )

            if not corners_inside:
                reasons.append(
                    "Tag is cut off or too close to the image edge"
                )

            if not hamming_ok:
                reasons.append(
                    f"Low tag code confidence: hamming={tag.hamming}"
                )

            if not margin_ok:
                reasons.append(
                    "decision_margin too low: "
                    f"{tag.decision_margin:.2f}"
                )

            tag_details.append({
                "tag_id": int(tag.tag_id),
                "hamming": int(tag.hamming),
                "decision_margin": float(tag.decision_margin),
                "corners_inside_image": corners_inside,
                "corners": corners.tolist(),
                "center": np.asarray(tag.center).tolist(),
            })

        # Jakość zdjęcia (skala, ostrość, jasność) — mierzona na znaczniku.
        # Także gdy znacznik ma problem (np. słaby odczyt): wtedy powód
        # jakościowy ("za ciemno") podpowiada, co poprawić.
        quality, quality_status, quality_warnings = None, None, []
        if len(matching_tags) == 1:
            quality = image_quality.measure(gray, matching_tags[0].corners)
            quality_status, quality_warnings, quality_rejects = \
                image_quality.classify(quality)
            if reasons:
                # Znacznik odrzucony — ostrzeżenia jakości jako wskazówki.
                quality_rejects = quality_rejects + quality_warnings
                quality_warnings = []
                quality_status = "REJECTED"
            reasons += quality_rejects

        accepted = (
            len(matching_tags) == 1
            and len(reasons) == 0
        )

        # Narysuj wykryte znaczniki na obrazie diagnostycznym.
        debug_image = image.copy()

        for tag in detections:
            corners = np.asarray(tag.corners, dtype=np.int32)
            color = (0, 200, 0) if (
                tag.tag_id == APRILTAG_ID
                and accepted
            ) else (0, 0, 255)

            cv2.polylines(
                debug_image,
                [corners.reshape((-1, 1, 2))],
                isClosed=True,
                color=color,
                thickness=3,
            )

            center = tuple(
                np.asarray(tag.center, dtype=int)
            )

            cv2.putText(
                debug_image,
                f"ID={tag.tag_id}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        status = "ACCEPTED" if accepted else "REJECTED"

        cv2.putText(
            debug_image,
            status,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 200, 0) if accepted else (0, 0, 255),
            2,
        )
        if quality:
            cv2.putText(
                debug_image,
                f"quality {quality_status}: "
                f"{quality['scale_px_per_mm']:.1f} px/mm  "
                f"blur {quality['edge_blur_mm']:.2f} mm  "
                f"white {quality['tag_white']:.0f}  "
                f"contrast {quality['contrast']:.0f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                {"OK": (0, 200, 0), "WARNING": (0, 165, 255)}.get(
                    quality_status, (0, 0, 255)),
                2,
            )

        cv2.imwrite(
            str(TAG_DEBUG_DIR / f"{image_path.stem}_tag.jpg"),
            debug_image,
        )

        destination_dir = ACCEPTED_DIR if accepted else REJECTED_DIR

        # Kopiujemy oryginał; nie zmieniamy pliku wejściowego.
        shutil.copy2(
            image_path,
            destination_dir / image_path.name,
        )

        report.append({
            "image": image_path.name,
            "status": status,
            "reason": reasons if reasons else ["OK"],
            "image_width": width,
            "image_height": height,
            "tag_family": APRILTAG_FAMILY,
            "expected_tag_id": APRILTAG_ID,
            "configured_tag_size_m": APRILTAG_SIZE_M,
            "quad_decimate": used_quad_decimate,
            "detections": tag_details,
            # OK / WARNING / REJECTED — ocena jakości zdjęcia.
            "quality_status": quality_status,
            "quality": quality,
            "warnings": quality_warnings,
        })

        if accepted:
            accepted_count += 1
            print(
                f"[OK] {image_path.name}: "
                f"AprilTag ID {APRILTAG_ID} "
                f"(quad_decimate={used_quad_decimate}), "
                f"quality {quality_status}"
            )
            for warning in quality_warnings:
                print(f"     WARNING: {warning}")
        else:
            rejected_count += 1
            print(
                f"[REJECTED] {image_path.name}: "
                + "; ".join(reasons)
            )

    with open(
        APRILTAG_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n--- SUMMARY ---")
    print(f"Total images: {len(image_files)}")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Report: {APRILTAG_REPORT_PATH}")
    print(f"Accepted images: {ACCEPTED_DIR}")
    print(f"Rejected images: {REJECTED_DIR}")
    print(f"Debug images: {TAG_DEBUG_DIR}")


if __name__ == "__main__":
    main()
