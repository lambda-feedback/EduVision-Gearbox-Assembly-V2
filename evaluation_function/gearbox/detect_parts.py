
"""
Etap 5 — segmentacja YOLO na zaakceptowanych zdjęciach.

Wejście:
    output/accepted/          zdjęcia z widocznym AprilTagiem (etap 2)
    output/camera_poses.json  parametry i pozy kamer (etap 4, opcjonalnie)

Dla każdej detekcji zapisywane są:
    klasa, confidence, bbox, środek i pole maski, kontur, elipsa
    (dla kół), maska w pełnej rozdzielczości (PNG, przycięta do bbox),
    promień w układzie świata przez środek maski (z pozy kamery).

Role klas:
    part    — osobna fizyczna część (umieszczana w przestrzeni),
    feature — cecha innej części (FEATURE_CLASS_PARENTS), np. gear_small
              jako zębnik koła gear_big; przypisywana do rodzica.
Liczba części nie jest z góry zakładana — zapisywane jest wszystko,
co wykrył model (poza IGNORED_CLASS_IDS).

Uruchomienie:
    python detect_parts.py
"""

import argparse
import json
import math
import shutil
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    INPUT_DIR,
    ACCEPTED_DIR,
    IMAGE_EXTENSIONS,
    CLASS_NAMES,
    ACTIVE_CLASS_IDS,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    CAMERA_POSES_PATH,
    DETECTIONS_PATH,
    MASKS_DIR,
    DETECTION_DEBUG_DIR,
    FEATURE_CLASS_PARENTS,
    CIRCULAR_CLASS_IDS,
    FEATURE_MAX_CENTER_DIST,
    FEATURE_MIN_CONTACT,
    FEATURE_COLOR_WEIGHT,
    FEATURE_AMBIGUITY_MARGIN,
    APRILTAG_REPORT_PATH,
    TAG_EXCLUSION_SCALE,
    TAG_EXCLUSION_MIN_OVERLAP,
)
from scipy.optimize import linear_sum_assignment

# Kolory klas (BGR) na obrazach diagnostycznych.
CLASS_COLORS = {
    0: (255, 128, 0),
    1: (255, 255, 0),
    2: (255, 255, 255),
    4: (128, 0, 128),
    5: (255, 0, 255),
    6: (0, 0, 255),
}

BORDER_MARGIN_PX = 2

_MODEL_CACHE = {}


def _load_model(model_path):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(str(model_path))
    return _MODEL_CACHE[model_path]


# --------------------------------------------------------------------------
# Maska przycięta do ramki
# --------------------------------------------------------------------------

class MaskCrop:
    """
    Maska części przechowywana jako wycinek w ramce (bbox) + położenie
    w zdjęciu. Wszystkie operacje dają wynik identyczny jak na masce
    w pełnej rozdzielczości, ale liczą tylko obszar wokół części.
    """

    def __init__(self, crop, x0, y0, image_shape):
        self.crop = crop                    # bool, ramka maski
        self.x0, self.y0 = x0, y0
        self.height, self.width = image_shape
        self.area = int(np.count_nonzero(crop))

    @classmethod
    def from_tensor(cls, mask_tensor, image_shape):
        """Wycina ramkę maski na GPU — na CPU trafia tylko wycinek."""
        mask = mask_tensor > 0.5
        rows = mask.any(dim=1).nonzero()
        cols = mask.any(dim=0).nonzero()
        if len(rows) == 0:
            return None
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])
        crop = mask[y0:y1 + 1, x0:x1 + 1].cpu().numpy()
        return cls(crop, x0, y0, image_shape)

    @property
    def bbox(self):
        h, w = self.crop.shape
        return self.x0, self.y0, self.x0 + w - 1, self.y0 + h - 1

    def region(self, margin, other=None):
        """
        Fragment PEŁNEJ maski (tej lub `other`) w ramce tej maski
        powiększonej o margin px i przyciętej do zdjęcia — z offsetem.
        """
        x0, y0, x1, y1 = self.bbox
        rx0, ry0 = max(0, x0 - margin), max(0, y0 - margin)
        rx1 = min(self.width - 1, x1 + margin)
        ry1 = min(self.height - 1, y1 + margin)
        out = np.zeros((ry1 - ry0 + 1, rx1 - rx0 + 1), dtype=bool)
        source = other or self
        sx0, sy0, sx1, sy1 = source.bbox
        ix0, iy0 = max(rx0, sx0), max(ry0, sy0)
        ix1, iy1 = min(rx1, sx1), min(ry1, sy1)
        if ix1 >= ix0 and iy1 >= iy0:
            out[iy0 - ry0:iy1 - ry0 + 1, ix0 - rx0:ix1 - rx0 + 1] = \
                source.crop[iy0 - sy0:iy1 - sy0 + 1, ix0 - sx0:ix1 - sx0 + 1]
        return out, rx0, ry0


# --------------------------------------------------------------------------
# Geometria maski
# --------------------------------------------------------------------------

def mask_geometry(mask, class_id):
    """Środek, pole, kontur, prostokąt i (dla kół) elipsa z MaskCrop."""
    if mask.area == 0:
        return None
    ys, xs = np.nonzero(mask.crop)
    xs = xs + mask.x0
    ys = ys + mask.y0
    area = mask.area

    height, width = mask.height, mask.width
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # Margines 1 px: śledzenie konturu jest lokalne, więc wynik jest taki
    # sam jak na pełnej masce (również przy krawędzi zdjęcia).
    region, rx0, ry0 = mask.region(1)
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        offset=(rx0, ry0),
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    main = contours[0]

    polygon = cv2.approxPolyDP(main, 1.0, True).reshape(-1, 2)
    (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = cv2.minAreaRect(main)

    geometry = {
        "mask_area_px": area,
        "mask_centroid_px": [round(float(xs.mean()), 2),
                             round(float(ys.mean()), 2)],
        "mask_bbox_xyxy": [x0, y0, x1, y1],
        "mask_components": len(contours),
        "polygon_px": polygon.tolist(),
        "min_area_rect": {
            "center_px": [round(rect_cx, 2), round(rect_cy, 2)],
            "size_px": [round(rect_w, 2), round(rect_h, 2)],
            "angle_deg": round(rect_angle, 2),
        },
        "touches_image_border": bool(
            x0 <= BORDER_MARGIN_PX or y0 <= BORDER_MARGIN_PX
            or x1 >= width - 1 - BORDER_MARGIN_PX
            or y1 >= height - 1 - BORDER_MARGIN_PX
        ),
        "ellipse": None,
    }

    if class_id in CIRCULAR_CLASS_IDS and len(main) >= 5:
        # Rzut okręgu to elipsa: półoś wielka ≈ promień koła w px,
        # stosunek osi ≈ cos(kąta nachylenia płaszczyzny koła do obrazu).
        # Przy zasłonięciach elipsa jest tylko przybliżeniem.
        (ecx, ecy), (ea, eb), eangle = cv2.fitEllipse(main)
        major, minor = max(ea, eb) / 2, min(ea, eb) / 2
        # Kierunek osi małej w obrazie (rzut osi obrotu koła).
        theta = math.radians(eangle) + (0 if ea < eb else math.pi / 2)
        geometry["ellipse"] = {
            "center_px": [round(ecx, 2), round(ecy, 2)],
            "semi_major_px": round(major, 2),
            "semi_minor_px": round(minor, 2),
            "axis_ratio": round(minor / major, 4) if major > 0 else None,
            "angle_deg": round(eangle, 2),
            # Pełne osie w kolejności z cv2.fitEllipse (do rysowania).
            "axes_px": [round(ea, 2), round(eb, 2)],
            "minor_axis_dir_px": [round(math.cos(theta), 4),
                                  round(math.sin(theta), 4)],
        }
    return geometry


def save_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.crop * 255).astype(np.uint8))


# --------------------------------------------------------------------------
# Promienie w układzie świata
# --------------------------------------------------------------------------

def pixel_ray_world(pixel, K, T_world_camera):
    """Promień (początek w mm, kierunek jednostkowy) przez dany piksel."""
    direction_camera = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    rotation = T_world_camera[:3, :3]
    direction = rotation @ direction_camera
    direction /= np.linalg.norm(direction)
    origin_mm = T_world_camera[:3, 3] * 1000
    return origin_mm, direction


def ray_record(pixel, camera):
    if camera is None:
        return None
    origin, direction = pixel_ray_world(
        pixel, camera["K"], camera["T_world_camera"]
    )
    record = {
        "origin_mm": np.round(origin, 2).tolist(),
        "direction": np.round(direction, 6).tolist(),
        "base_plane_hit_mm": None,
    }
    # Przecięcie z płaszczyzną znacznika Z=0 (podstawa). Środek części
    # leży nad podstawą, więc to tylko punkt odniesienia, nie pozycja 3D.
    if direction[2] < -1e-6:
        s = -origin[2] / direction[2]
        record["base_plane_hit_mm"] = np.round(
            origin + s * direction, 2
        ).tolist()
    return record


def load_tag_zones():
    """Nazwa zdjęcia -> wielokąt strefy znacznika (px), z etapu 2."""
    if not APRILTAG_REPORT_PATH.is_file():
        return {}
    with open(APRILTAG_REPORT_PATH, encoding="utf-8") as file:
        report = json.load(file)
    zones = {}
    for entry in report:
        if entry.get("status") != "ACCEPTED":
            continue
        corners = np.asarray(entry["detections"][0]["corners"], dtype=float)
        center = corners.mean(axis=0)
        zones[entry["image"]] = (
            center + (corners - center) * TAG_EXCLUSION_SCALE
        )
    return zones


def tag_zone_overlap(mask, zone):
    if zone is None:
        return 0.0
    # Strefa rysowana tylko w ramce maski (przesunięcie o całe piksele
    # nie zmienia rasteryzacji wielokąta).
    zone_mask = np.zeros(mask.crop.shape, dtype=np.uint8)
    polygon = np.round(zone).astype(np.int32) - [mask.x0, mask.y0]
    cv2.fillPoly(zone_mask, [polygon.astype(np.int32)], 1)
    return (float(np.count_nonzero(mask.crop & zone_mask.astype(bool)))
            / max(1, mask.area))


def load_cameras():
    if not CAMERA_POSES_PATH.is_file():
        print(f"WARNING: missing {CAMERA_POSES_PATH} — detections without "
              "3D rays. Run estimate_camera_poses.py.")
        return {}
    with open(CAMERA_POSES_PATH, encoding="utf-8") as file:
        report = json.load(file)
    cameras = {}
    for entry in report["images"]:
        if "T_world_camera" not in entry:
            continue
        cameras[entry["image"]] = {
            "status": entry["status"],
            "K": np.array(entry["K"], dtype=float),
            "T_world_camera": np.array(entry["T_world_camera"], dtype=float),
        }
    return cameras


# --------------------------------------------------------------------------
# Przypisanie cech (gear_small) do części (gear_big)
# --------------------------------------------------------------------------

def parent_radius_px(parent):
    ellipse = parent["ellipse"]
    if ellipse:
        return ellipse["semi_major_px"]
    return math.sqrt(parent["mask_area_px"] / math.pi)


def parent_center_px(parent):
    ellipse = parent["ellipse"]
    return np.array(
        ellipse["center_px"] if ellipse else parent["mask_centroid_px"]
    )


def contact_ratio(feature_mask, parent_mask, feature_area):
    """Jaka część obrzeża cechy styka się z maską rodzica (0..1)."""
    k = max(3, int(round(0.1 * math.sqrt(feature_area))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)
    # Dylatacja sięga k px — obszar z marginesem k+1 daje wynik identyczny
    # jak na pełnym zdjęciu.
    feature, _, _ = feature_mask.region(k + 1)
    parent, _, _ = feature_mask.region(k + 1, other=parent_mask)
    ring = cv2.dilate(feature.astype(np.uint8), kernel).astype(bool)
    ring &= ~feature
    ring_size = int(np.count_nonzero(ring))
    if ring_size == 0:
        return 0.0
    return float(np.count_nonzero(ring & parent)) / ring_size


def mean_color_lab(image, mask):
    """Mediana koloru (L, a, b) wnętrza maski — odporna na krawędzie."""
    # Erozja 5x5 sięga 2 px — margines 2 px daje wynik jak na pełnym
    # zdjęciu; Lab liczony tylko w tym obszarze (konwersja per piksel).
    region, rx0, ry0 = mask.region(2)
    h, w = region.shape
    lab = cv2.cvtColor(image[ry0:ry0 + h, rx0:rx0 + w],
                       cv2.COLOR_BGR2LAB).astype(float)
    inner = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8))
    pixels = lab[inner.astype(bool)]
    if len(pixels) < 20:
        pixels = lab[region]
    return np.median(pixels, axis=0)


def associate_features(detections, masks, image):
    """
    Przypisuje cechy do rodziców jeden do jednego, globalnie
    (algorytm węgierski) — minimalizuje łączny koszt:
        koszt = odległość środków / promień rodzica
                + FEATURE_COLOR_WEIGHT * różnica koloru (ΔE_ab / 50)
    Pary niespełniające warunków odległości i styku są niedozwolone.
    """
    colors = {}

    def color(index):
        if index not in colors:
            colors[index] = mean_color_lab(image, masks[index])
            detections[index]["color_lab"] = np.round(
                colors[index], 1
            ).tolist()
        return colors[index]

    feature_ids = [i for i, d in enumerate(detections)
                   if d["class_id"] in FEATURE_CLASS_PARENTS]
    parent_ids = [i for i, d in enumerate(detections)
                  if d["class_id"] in FEATURE_CLASS_PARENTS.values()]

    invalid = 1e6
    cost = np.full((len(feature_ids), len(parent_ids)), invalid)
    details = {}
    for row, f_index in enumerate(feature_ids):
        feature = detections[f_index]
        for col, p_index in enumerate(parent_ids):
            parent = detections[p_index]
            if FEATURE_CLASS_PARENTS[feature["class_id"]] != parent["class_id"]:
                continue
            distance = float(np.linalg.norm(
                np.array(feature["mask_centroid_px"])
                - parent_center_px(parent)
            )) / parent_radius_px(parent)
            if distance > FEATURE_MAX_CENTER_DIST:
                continue
            contact = contact_ratio(
                masks[f_index], masks[p_index], feature["mask_area_px"]
            )
            if contact < FEATURE_MIN_CONTACT:
                continue
            # Różnica barwy w płaszczyźnie (a, b) — niezależna od jasności.
            delta_e = float(np.linalg.norm(
                color(f_index)[1:] - color(p_index)[1:]
            ))
            cost[row, col] = distance + FEATURE_COLOR_WEIGHT * delta_e / 50
            details[(f_index, p_index)] = (distance, contact, delta_e)

    used_features, used_parents = set(), set()
    if cost.size:
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if cost[row, col] >= invalid:
                continue
            f_index, p_index = feature_ids[row], parent_ids[col]
            used_features.add(f_index)
            used_parents.add(p_index)

            # Alternatywni rodzice o podobnym koszcie.
            alternatives = [
                {"parent_id": detections[parent_ids[c]]["id"],
                 "cost": round(float(cost[row, c]), 3)}
                for c in range(len(parent_ids))
                if c != col and cost[row, c] < invalid
                and cost[row, c] - cost[row, col] < FEATURE_AMBIGUITY_MARGIN
            ]

            feature, parent = detections[f_index], detections[p_index]
            distance, contact, delta_e = details[(f_index, p_index)]
            offset = (
                np.array(feature["mask_centroid_px"])
                - parent_center_px(parent)
            )
            feature["parent_id"] = parent["id"]
            feature["association"] = {
                "cost": round(float(cost[row, col]), 3),
                "center_distance_norm": round(distance, 3),
                "contact_ratio": round(contact, 3),
                "color_delta_ab": round(delta_e, 1),
                "ambiguous": bool(alternatives),
                "alternatives": alternatives,
            }
            if alternatives:
                feature["warnings"].append(
                    "Ambiguous parent assignment — to be resolved "
                    "using other images"
                )
            parent["features"].append({
                "id": feature["id"],
                "class_name": feature["class_name"],
                # Wektor od środka koła do zębnika — po której stronie
                # koła widać zębnik (wskazówka orientacji dla etapu 6).
                "offset_px": np.round(offset, 2).tolist(),
                "offset_norm": round(
                    float(np.linalg.norm(offset)) / parent_radius_px(parent),
                    3,
                ),
                "ambiguous": bool(alternatives),
            })

    for index, detection in enumerate(detections):
        if detection["role"] == "feature" and index not in used_features:
            detection["warnings"].append(
                "No parent part found for this feature"
            )
        if (detection["class_id"] in FEATURE_CLASS_PARENTS.values()
                and index not in used_parents):
            detection["warnings"].append(
                "Feature (e.g. pinion) not visible — may be on the hidden "
                "side or occluded"
            )


# --------------------------------------------------------------------------
# Obraz diagnostyczny
# --------------------------------------------------------------------------

def draw_debug(image, detections, masks):
    overlay = image.copy()
    for detection, mask in zip(detections, masks):
        x0, y0, x1, y1 = mask.bbox
        overlay[y0:y1 + 1, x0:x1 + 1][mask.crop] = CLASS_COLORS.get(
            detection["class_id"], (0, 255, 0))
    debug = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)

    scale = max(image.shape[:2]) / 2000.0
    thickness = max(2, int(round(2 * scale)))
    by_id = {d["id"]: d for d in detections}

    for detection in detections:
        color = CLASS_COLORS.get(detection["class_id"], (0, 255, 0))
        polygon = np.array(detection["polygon_px"], dtype=np.int32)
        cv2.polylines(debug, [polygon], True, color, thickness)

        ellipse = detection["ellipse"]
        if ellipse:
            cv2.ellipse(
                debug,
                (tuple(ellipse["center_px"]), tuple(ellipse["axes_px"]),
                 ellipse["angle_deg"]),
                (0, 255, 255), max(1, thickness // 2),
            )

        if detection.get("parent_id"):
            parent = by_id[detection["parent_id"]]
            start = tuple(np.round(parent_center_px(parent)).astype(int))
            end = tuple(np.round(detection["mask_centroid_px"]).astype(int))
            # Żółta strzałka: pewne przypisanie, pomarańczowa: niejednoznaczne.
            arrow_color = (
                (0, 140, 255) if detection["association"]["ambiguous"]
                else (0, 255, 255)
            )
            cv2.arrowedLine(debug, start, end, arrow_color,
                            thickness + 1, tipLength=0.2)

    # Etykiety na końcu, aby nie były zasłaniane.
    for detection in detections:
        x, y = np.round(detection["mask_centroid_px"]).astype(int)
        label = f"{detection['index']}:{detection['class_name']}"
        for col, extra in (((0, 0, 0), 3), ((255, 255, 255), 0)):
            cv2.putText(debug, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * scale, col, thickness + extra)

    long_side = max(debug.shape[:2])
    if long_side > 2000:
        factor = 2000 / long_side
        debug = cv2.resize(debug, None, fx=factor, fy=factor,
                           interpolation=cv2.INTER_AREA)
    return debug


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 5: part detection.")
    parser.add_argument(
        "--all-images", action="store_true",
        help="analyse all images in INPUT_DIR (no-AprilTag mode — "
             "parts inventory only)")
    args = parser.parse_args()
    source_dir = INPUT_DIR if args.all_images else ACCEPTED_DIR

    if not MODEL_PATH.is_file():
        sys.exit(f"ERROR: Model not found: {MODEL_PATH}")
    if not source_dir.is_dir():
        sys.exit(f"ERROR: Missing {source_dir}. Run test_apriltag.py.")

    images = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        sys.exit(f"ERROR: No images in {source_dir}")

    model = _load_model(MODEL_PATH)
    mismatches = [
        f"ID {cid}: oczekiwano {name!r}, model ma {model.names.get(cid)!r}"
        for cid, name in CLASS_NAMES.items()
        if model.names.get(cid) != name
    ]
    if mismatches:
        sys.exit("ERROR: model class map does not match config.py:\n  "
                 + "\n  ".join(mismatches))

    cameras = load_cameras()
    tag_zones = load_tag_zones()

    # Wyniki poprzedniego uruchomienia.
    if MASKS_DIR.exists():
        shutil.rmtree(MASKS_DIR)
    DETECTION_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in DETECTION_DEBUG_DIR.glob("*_det.jpg"):
        old_file.unlink()

    report_images = []
    class_totals = {}

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[SKIPPED] {image_path.name}: cannot read")
            continue
        stem = image_path.stem
        camera = cameras.get(image_path.name)

        # retina_masks=True: maski w pełnej rozdzielczości zdjęcia.
        result = model.predict(
            source=image,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            classes=ACTIVE_CLASS_IDS,
            retina_masks=True,
            verbose=False,
        )[0]

        detections, masks, rejected = [], [], []
        if result.boxes is not None and result.masks is not None:
            for i in range(len(result.boxes)):
                class_id = int(result.boxes.cls[i].item())
                confidence = round(float(result.boxes.conf[i]), 4)
                mask = MaskCrop.from_tensor(result.masks.data[i],
                                            image.shape[:2])
                if mask is None:  # pusta maska
                    continue

                overlap = tag_zone_overlap(mask, tag_zones.get(image_path.name))
                if overlap >= TAG_EXCLUSION_MIN_OVERLAP:
                    rejected.append({
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": confidence,
                        "bbox_xyxy": [round(float(v), 2)
                                      for v in result.boxes.xyxy[i].tolist()],
                        "reason": f"Lies on the AprilTag "
                                  f"({overlap:.0%} of the mask)",
                    })
                    continue

                geometry = mask_geometry(mask, class_id)
                if geometry is None:
                    continue

                index = len(detections)
                det_id = f"{stem}#{index}"
                mask_file = MASKS_DIR / stem / f"{index}_{CLASS_NAMES[class_id]}.png"
                save_mask(mask, mask_file)

                is_feature = class_id in FEATURE_CLASS_PARENTS
                detection = {
                    "id": det_id,
                    "index": index,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "role": "feature" if is_feature else "part",
                    "confidence": confidence,
                    "bbox_xyxy": [round(float(v), 2)
                                  for v in result.boxes.xyxy[i].tolist()],
                    **geometry,
                    "mask_file": str(mask_file.relative_to(MASKS_DIR.parent)),
                    "mask_offset_px": geometry["mask_bbox_xyxy"][:2],
                    "ray_centroid": ray_record(
                        geometry["mask_centroid_px"], camera
                    ),
                    "features": [],
                    "parent_id": None,
                    "warnings": [],
                }
                if geometry["touches_image_border"]:
                    detection["warnings"].append(
                        "Mask touches the image edge — the part may be cut off"
                    )
                if geometry["mask_components"] > 1:
                    detection["warnings"].append(
                        f"Mask consists of {geometry['mask_components']} "
                        "fragments (occlusion?)"
                    )
                detections.append(detection)
                masks.append(mask)

        associate_features(detections, masks, image)

        cv2.imwrite(str(DETECTION_DEBUG_DIR / f"{stem}_det.jpg"),
                    draw_debug(image, detections, masks))

        counts = {}
        for d in detections:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
            class_totals[d["class_name"]] = (
                class_totals.get(d["class_name"], 0) + 1
            )
        linked = sum(1 for d in detections if d.get("parent_id"))
        ambiguous = sum(
            1 for d in detections
            if d.get("parent_id") and d["association"]["ambiguous"]
        )
        features = sum(1 for d in detections if d["role"] == "feature")

        report_images.append({
            "image": image_path.name,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "camera_pose_status": camera["status"] if camera else None,
            "class_counts": counts,
            "detections": detections,
            "rejected_detections": rejected,
        })

        counts_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{image_path.name:<8} {len(detections):>2} detections  "
              f"features linked {linked}/{features} "
              f"(ambiguous {ambiguous})  "
              f"rejected {len(rejected)}  "
              f"pose={camera['status'] if camera else 'NONE'}  "
              f"[{counts_text}]")

    with open(DETECTIONS_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "model": str(MODEL_PATH.name),
            # True = tryb bez AprilTaga (wszystkie zdjęcia, bez 3D).
            "all_images_mode": args.all_images,
            "image_size": IMAGE_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "feature_class_parents": {
                CLASS_NAMES[f]: CLASS_NAMES[p]
                for f, p in FEATURE_CLASS_PARENTS.items()
            },
            "images": report_images,
        }, file, indent=2, ensure_ascii=False)

    print("\n--- SUMMARY ---")
    print(f"Images: {len(report_images)}")
    print("Total detections (sum over images, the same part counted "
          "separately in each image):")
    for name, total in sorted(class_totals.items()):
        print(f"  {name:<14} {total}")
    print(f"Report: {DETECTIONS_PATH}")
    print(f"Masks: {MASKS_DIR}")
    print(f"Debug images: {DETECTION_DEBUG_DIR}")


if __name__ == "__main__":
    main()
