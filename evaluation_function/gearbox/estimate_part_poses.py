
"""
Etap 6 — poza 3D części z pojedynczych zdjęć (dopasowanie CAD do maski).

Wejście:
    output/detections.json    detekcje i maski (etap 5)
    output/camera_poses.json  parametry i pozy kamer (etap 4)
    cad/*.stl                 modele części (mm, oś symetrii = Z)

Dla każdej detekcji części (role == "part"):
    1. kandydaci pozy startowej z geometrii maski:
       - "disc": elipsa + promień z CAD -> odległość i nachylenie koła,
         dwie możliwe strony nachylenia i dwa zwroty osi; dla gear_big
         zwrot osi (+Z = zębnik) wskazuje przypisany gear_small,
       - "cylinder": prostokąt maski + średnica z CAD -> odległość,
         oś wzdłuż dłuższego/krótszego boku, kilka nachyleń,
    2. optymalizacja pozy (5 stopni swobody — obrót wokół osi symetrii
       jest nieobserwowalny) minimalizująca niezgodność sylwetki CAD
       z maską; piksele innych wykrytych części są neutralne (zasłonięcia),
    3. ocena: IoU, niejednoznaczność (alternatywny kandydat o podobnym
       koszcie), status OK / LOW_CONFIDENCE / UNKNOWN.

Pojedyncze zdjęcie słabo określa odległość — łączenie widoków w etapie 7.
Konwencje: geometry.py. Jednostki w tym etapie: milimetry.

Uruchomienie:
    python estimate_part_poses.py
"""

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize

from config import (
    OUTPUT_DIR,
    ACCEPTED_DIR,
    CLASS_NAMES,
    DETECTIONS_PATH,
    CAMERA_POSES_PATH,
    CAD_DIR,
    PART_MODELS,
    PART_POSES_PATH,
    PART_POSE_DEBUG_DIR,
    PART_FIT_ROI_PX,
    PART_FIT_EXTRA_WEIGHT,
    PART_FIT_NEUTRAL_WEIGHT,
    PART_MOUNTS,
    PARALLEL_WORKERS,
    PART_FIT_MAX_EVALS,
    PART_FIT_MIN_IOU_OK,
    PART_FIT_MIN_IOU,
    PART_FIT_AMBIGUITY_RATIO,
)
from geometry import make_transform

ROI_MARGIN = 0.6      # powiększenie obszaru wokół maski (względem rozmiaru)
PROFILE_SIDES = 32    # liczba boków walców przybliżających profil


# --------------------------------------------------------------------------
# Modele CAD
# --------------------------------------------------------------------------

def radial_profile(vertices, triangles):
    """
    Profil bryły obrotowej: lista odcinków (z0, z1, promień) wzdłuż osi Z.
    Promień odcinka = największa odległość od osi wierzchołków ścian
    bocznych, które przecinają wnętrze odcinka. Sylwetka części osiowo
    symetrycznej jest w pełni określona przez ten profil.
    """
    tri = vertices[triangles]
    z_min, z_max = tri[:, :, 2].min(axis=1), tri[:, :, 2].max(axis=1)
    radial = np.linalg.norm(tri[:, :, :2], axis=2).max(axis=1)
    levels = np.unique(np.round(vertices[:, 2], 4))

    segments = []
    for z0, z1 in zip(levels[:-1], levels[1:]):
        mid = (z0 + z1) / 2
        side = (z_min < mid) & (z_max > mid)
        if side.any():
            segments.append((float(z0), float(z1), float(radial[side].max())))
    return segments


def load_models():
    models = {}
    for class_id, spec in PART_MODELS.items():
        path = CAD_DIR / spec["stl"]
        if not path.is_file():
            sys.exit(f"ERROR: Missing CAD model {path}")
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices, dtype=float)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)

        # Wierzchołki walców profilu: (odcinek, góra/dół, kąt) -> punkt 3D.
        segments = radial_profile(vertices, triangles)
        angles = np.linspace(0, 2 * math.pi, PROFILE_SIDES, endpoint=False)
        ring = np.column_stack([np.cos(angles), np.sin(angles)])
        profile_points = np.array([
            [np.column_stack([ring * r, np.full(PROFILE_SIDES, z)])
             for z in (z0, z1)]
            for z0, z1, r in segments
        ]).reshape(len(segments), 2 * PROFILE_SIDES, 3)

        models[class_id] = {
            **spec,
            "vertices": vertices,
            "triangles": triangles,
            "profile": segments,
            "profile_points": profile_points,
            "center": (lo + hi) / 2,              # środek bryły (układ CAD)
            "radius_mm": float(max(hi[0] - lo[0], hi[1] - lo[1]) / 2),
            "length_mm": float(hi[2] - lo[2]),
        }
    return models


# --------------------------------------------------------------------------
# Poza: parametry optymalizacji <-> transformacja
# --------------------------------------------------------------------------

def axis_frame(axis):
    """Macierz obrotu, której trzecia kolumna to oś (pozostałe dowolne)."""
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    x = helper - helper.dot(axis) * axis
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(axis, x), axis])


def params_to_pose(params, K_inv):
    """[u, v, log z, theta, psi] -> (środek bryły w kamerze, oś w kamerze)."""
    u, v, log_z, theta, psi = params
    ray = K_inv @ np.array([u, v, 1.0])
    center = ray / ray[2] * math.exp(log_z)
    axis = np.array([
        math.sin(theta) * math.cos(psi),
        math.sin(theta) * math.sin(psi),
        math.cos(theta),
    ])
    return center, axis


def pose_to_params(center, axis, K):
    projected = K @ center
    theta = math.acos(np.clip(axis[2], -1, 1))
    psi = math.atan2(axis[1], axis[0])
    return np.array([projected[0] / projected[2], projected[1] / projected[2],
                     math.log(center[2]), theta, psi])


def T_camera_part(center, axis, model):
    rotation = axis_frame(axis)
    return make_transform(rotation, center - rotation @ model["center"])


# --------------------------------------------------------------------------
# Renderowanie sylwetki
# --------------------------------------------------------------------------

def render_silhouette(model, transform, K, shape):
    """
    Sylwetka jako suma rzutów walców profilu. Rzut walca (bryły wypukłej)
    to otoczka wypukła rzutów jego wierzchołków.
    """
    points = model["profile_points"] @ transform[:3, :3].T + transform[:3, 3]
    mask = np.zeros(shape, dtype=np.uint8)
    if np.any(points[..., 2] <= 1.0):
        return mask.astype(bool)
    projected = points @ K.T
    uv = projected[..., :2] / projected[..., 2:3]
    for segment_uv in uv:
        hull = cv2.convexHull(np.round(segment_uv * 16).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1, lineType=cv2.LINE_8, shift=4)
    return mask.view(bool)  # wartości 0/1 — widok bez kopiowania


# --------------------------------------------------------------------------
# Obszar dopasowania (ROI) i maski
# --------------------------------------------------------------------------

class MaskCache:
    def __init__(self):
        self.cache = {}

    def get(self, detection):
        key = detection["id"]
        if key not in self.cache:
            crop = cv2.imread(str(OUTPUT_DIR / detection["mask_file"]),
                              cv2.IMREAD_GRAYSCALE)
            self.cache[key] = (crop > 127, detection["mask_offset_px"])
        return self.cache[key]


def paste(target, crop, offset, roi_origin):
    """Wkleja przyciętą maskę (pełna rozdzielczość) do maski ROI."""
    x0 = offset[0] - roi_origin[0]
    y0 = offset[1] - roi_origin[1]
    h, w = crop.shape
    tx0, ty0 = max(0, x0), max(0, y0)
    tx1, ty1 = min(target.shape[1], x0 + w), min(target.shape[0], y0 + h)
    if tx1 <= tx0 or ty1 <= ty0:
        return
    target[ty0:ty1, tx0:tx1] |= crop[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]


def build_roi(target_dets, other_dets, image_size, K, cache):
    """
    Zwraca maski w ROI: target (do wyjaśnienia), neutral (inne części
    i obszar poza zdjęciem), oraz K przeliczone na współrzędne ROI.
    """
    boxes = np.array([d["mask_bbox_xyxy"] for d in target_dets], dtype=float)
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    size = max(x1 - x0, y1 - y0, 20)
    margin = ROI_MARGIN * size
    rx0, ry0 = int(math.floor(x0 - margin)), int(math.floor(y0 - margin))
    rx1, ry1 = int(math.ceil(x1 + margin)), int(math.ceil(y1 + margin))
    full_w, full_h = rx1 - rx0, ry1 - ry0

    target = np.zeros((full_h, full_w), dtype=bool)
    neutral = np.zeros((full_h, full_w), dtype=bool)
    for d in target_dets:
        paste(target, *cache.get(d), (rx0, ry0))
    for d in other_dets:
        paste(neutral, *cache.get(d), (rx0, ry0))

    # Poza kadrem nic nie wiemy — bez kary.
    width, height = image_size
    xs = np.arange(rx0, rx1)
    ys = np.arange(ry0, ry1)
    outside = (xs[None, :] < 0) | (xs[None, :] >= width) \
        | (ys[:, None] < 0) | (ys[:, None] >= height)
    neutral &= ~target
    outside &= ~target

    scale = PART_FIT_ROI_PX / max(full_w, full_h)
    roi_w = max(8, int(round(full_w * scale)))
    roi_h = max(8, int(round(full_h * scale)))
    resize = lambda m: cv2.resize(m.astype(np.uint8), (roi_w, roi_h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)

    K_roi = np.array([
        [K[0, 0] * scale, 0, (K[0, 2] - rx0) * scale],
        [0, K[1, 1] * scale, (K[1, 2] - ry0) * scale],
        [0, 0, 1],
    ])
    masks = {"target": resize(target), "occluders": resize(neutral),
             "outside": resize(outside)}
    # Składniki stałe dla ROI — liczone raz, nie przy każdej ocenie kosztu.
    masks["valid"] = ~masks["occluders"] & ~masks["outside"]
    masks["background"] = ~masks["target"] & masks["valid"]
    masks["target_count"] = int(np.count_nonzero(masks["target"]))
    return masks, K_roi, (rx0, ry0, scale)


def silhouette_cost(rendered, masks):
    """
    miss  — część maski niewyjaśniona przez sylwetkę CAD,
    extra — sylwetka CAD na tle (gdzie widać, że części nie ma),
    on_neutral — sylwetka na innych częściach (możliwe zasłonięcie).
    Obszar poza kadrem jest pomijany.
    """
    target = masks["target"]
    target_count = masks["target_count"]
    target_area = max(1, target_count)
    rendered_area = max(1, int(np.count_nonzero(rendered)))
    intersection = int(np.count_nonzero(target & rendered))
    # |target \ rendered| = |target| - |target ∩ rendered|
    miss = float(target_count - intersection) / target_area
    extra = float(np.count_nonzero(rendered & masks["background"])) \
        / rendered_area
    on_neutral = float(np.count_nonzero(rendered & masks["occluders"])) \
        / rendered_area
    union = float(np.count_nonzero((target | rendered) & masks["valid"]))
    iou = float(intersection) / union if union else 0.0
    total = (miss + PART_FIT_EXTRA_WEIGHT * extra
             + PART_FIT_NEUTRAL_WEIGHT * on_neutral)
    return total, iou, miss, extra


# --------------------------------------------------------------------------
# Kandydaci pozy startowej
# --------------------------------------------------------------------------

def lift_image_direction(direction_2d, view):
    """Kierunek w obrazie -> wektor 3D prostopadły do kierunku patrzenia."""
    d = np.array([direction_2d[0], direction_2d[1], 0.0])
    d -= d.dot(view) * view
    return d / np.linalg.norm(d)


def disc_candidates(detection, model, K, feature_offset):
    """
    Okrąg o promieniu R widziany jako elipsa (a, b): odległość z ≈ f R / a,
    kąt między osią koła a kierunkiem patrzenia φ = arccos(b / a),
    a rzut osi koła leży wzdłuż małej osi elipsy.
    """
    ellipse = detection["ellipse"]
    if ellipse is None:
        return []
    a, b = ellipse["semi_major_px"], ellipse["semi_minor_px"]
    u, v = ellipse["center_px"]
    view = np.linalg.inv(K) @ np.array([u, v, 1.0])
    view /= np.linalg.norm(view)
    z = K[0, 0] * model["disc_radius_mm"] / a
    disc_center = view / view[2] * z

    phi = math.acos(np.clip(b / a, 0.0, 1.0))
    minor = lift_image_direction(ellipse["minor_axis_dir_px"], view)

    # Znak składowej w płaszczyźnie obrazu: zębnik (+Z) wskazuje kierunek.
    if feature_offset is not None and model["directional"]:
        offset = np.asarray(feature_offset, dtype=float)
        side_signs = [1.0 if offset.dot(ellipse["minor_axis_dir_px"]) >= 0
                      else -1.0]
    else:
        side_signs = [1.0, -1.0]
    # Zwrot względem kamery (do kamery / od kamery). Dla części
    # niekierunkowych n i -n są równoważne, więc wystarczy jeden zwrot
    # przy obu znakach bocznych.
    toward_signs = [1.0, -1.0] if model["directional"] else [1.0]

    candidates = []
    for side in side_signs:
        for toward in toward_signs:
            axis = toward * math.cos(phi) * (-view) + side * math.sin(phi) * minor
            axis /= np.linalg.norm(axis)
            # Środek bryły leży na osi w odległości (środek - tarcza).
            center = disc_center + axis * (
                model["center"][2] - model["disc_z_mm"]
            )
            candidates.append((center, axis))
    return candidates


def cylinder_candidates(detection, model, K):
    rect = detection["min_area_rect"]
    (cx, cy), (w, h), angle = rect["center_px"], rect["size_px"], rect["angle_deg"]
    view = np.linalg.inv(K) @ np.array([cx, cy, 1.0])
    view /= np.linalg.norm(view)
    diameter = 2 * model["radius_mm"]

    candidates = []
    theta = math.radians(angle)
    sides = [((math.cos(theta), math.sin(theta)), w, h),
             ((-math.sin(theta), math.cos(theta)), h, w)]
    for direction_2d, along_px, across_px in sides:
        if across_px < 2:
            continue
        z = K[0, 0] * diameter / across_px
        center = view / view[2] * z
        in_plane = lift_image_direction(direction_2d, view)
        # Nachylenie osi poza płaszczyznę obrazu.
        for tilt_deg in (0.0, 40.0, -40.0):
            tilt = math.radians(tilt_deg)
            axis = math.cos(tilt) * in_plane + math.sin(tilt) * (-view)
            candidates.append((center, axis / np.linalg.norm(axis)))
    return candidates


# --------------------------------------------------------------------------
# Dopasowanie jednej części
# --------------------------------------------------------------------------

def fit_candidate(model, center, axis, K_roi, masks):
    K_inv = np.linalg.inv(K_roi)
    shape = masks["target"].shape

    def cost(params):
        c, n = params_to_pose(params, K_inv)
        rendered = render_silhouette(model, T_camera_part(c, n, model),
                                     K_roi, shape)
        return silhouette_cost(rendered, masks)[0]

    x0 = pose_to_params(center, axis, K_roi)
    result = minimize(
        cost, x0, method="Powell",
        options={
            "maxfev": PART_FIT_MAX_EVALS, "xtol": 1e-3, "ftol": 1e-4,
            # Początkowe kroki: piksele ROI, log-odległość, radiany.
            "direc": np.diag([3.0, 3.0, 0.05, 0.15, 0.15]),
        },
    )
    c, n = params_to_pose(result.x, K_inv)
    return evaluate(model, c, n, K_roi, masks, int(result.nfev),
                    float(center[2]))


def evaluate(model, center, axis, K_roi, masks, evals, init_depth):
    rendered = render_silhouette(model, T_camera_part(center, axis, model),
                                 K_roi, masks["target"].shape)
    total, iou, miss, extra = silhouette_cost(rendered, masks)
    return {
        "center": center, "axis": axis, "cost": total, "iou": iou,
        "miss": miss, "extra": extra, "evals": evals,
        "init_depth": init_depth,
    }


def fit_with_axis(model, axis, center_init, K_roi, masks):
    """
    Część osadzona na rodzicu: kierunek osi przejęty od rodzica (z jednego
    zdjęcia wyznaczony dobrze), położenie dopasowywane swobodnie — samo
    położenie osi rodzica ma z jednego zdjęcia kilka mm błędu.
    Pełna współosiowość: etap 7 (wiele zdjęć).
    """
    K_inv = np.linalg.inv(K_roi)
    shape = masks["target"].shape

    def pose(params):
        u, v, log_z = params
        ray = K_inv @ np.array([u, v, 1.0])
        return ray / ray[2] * math.exp(log_z)

    def cost(params):
        rendered = render_silhouette(
            model, T_camera_part(pose(params), axis, model), K_roi, shape)
        return silhouette_cost(rendered, masks)[0]

    x0 = pose_to_params(center_init, axis, K_roi)[:3]
    result = minimize(cost, x0, method="Powell", options={
        "maxfev": PART_FIT_MAX_EVALS, "xtol": 1e-3, "ftol": 1e-4,
        "direc": np.diag([3.0, 3.0, 0.05]),
    })
    return evaluate(model, pose(result.x), axis, K_roi, masks,
                    int(result.nfev), float(center_init[2]))


def closest_point_on_axis(ray, axis_point, axis_dir):
    """
    Punkt osi (rodzica) najbliższy promieniowi z kamery (początek 0).
    Zwraca (s wzdłuż osi, odległość promienia od osi w mm).
    """
    ray = ray / np.linalg.norm(ray)
    w0 = axis_point  # punkt osi minus początek promienia (0)
    a, b, c = axis_dir @ axis_dir, axis_dir @ ray, ray @ ray
    d, e = axis_dir @ w0, ray @ w0
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return 0.0, float("inf")
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    gap = np.linalg.norm((axis_point + s * axis_dir) - t * ray)
    return float(s), float(gap)


def axes_differ(a, b, directional):
    dot = float(np.dot(a, b))
    if not directional:
        dot = abs(dot)
    return dot < math.cos(math.radians(30))


def find_mount(detection, model, K, fitted):
    """
    Szuka części-rodzica (PART_MOUNTS), na której osi leży detekcja.
    fitted: id detekcji -> (klasa, środek, oś) części już dopasowanych.
    """
    rule = PART_MOUNTS.get(detection["class_id"])
    if rule is None:
        return None
    u, v = detection["mask_centroid_px"]
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    best = None
    for parent_id, (class_id, center, axis) in fitted.items():
        if class_id != rule["parent_class"]:
            continue
        s, gap = closest_point_on_axis(ray, center, axis)
        within = rule.get("rigid") or gap <= rule["max_axis_distance_mm"]
        if within and (best is None or gap < best["gap_mm"]):
            best = {"parent_id": parent_id, "center": center, "axis": axis,
                    "s_init": s, "gap_mm": gap}
    return best


def fit_part(detection, features, other_dets, image_size, camera, model,
             cache, fitted):
    K = camera["K"]
    target_dets = [detection] + features
    masks, K_roi, roi = build_roi(target_dets, other_dets, image_size, K,
                                  cache)

    mount = find_mount(detection, model, K, fitted)
    if mount is not None:
        center_init = mount["center"] + mount["axis"] * mount["s_init"]
        best = fit_with_axis(model, mount["axis"], center_init, K_roi, masks)
        # Odległość dopasowanego środka od osi rodzica (diagnostyka).
        offset = best["center"] - mount["center"]
        off_axis = offset - offset.dot(mount["axis"]) * mount["axis"]
        return {
            "best": best,
            "alternative": None,
            "ambiguity_ratio": None,
            "candidates": [],
            "mount": {"parent_id": mount["parent_id"],
                      "ray_axis_gap_mm": round(mount["gap_mm"], 2),
                      "fitted_off_axis_mm": round(
                          float(np.linalg.norm(off_axis)), 2)},
            "K_roi": K_roi,
            "roi": roi,
        }

    feature_offset = features[0]["offset_px"] if features else None
    if model["shape"] == "disc":
        candidates = disc_candidates(detection, model, K, feature_offset)
    else:
        candidates = cylinder_candidates(detection, model, K)
    if not candidates:
        return None

    fits = [fit_candidate(model, c, n, K_roi, masks) for c, n in candidates]
    fits.sort(key=lambda f: f["cost"])
    best = fits[0]

    alternative = next(
        (f for f in fits[1:]
         if axes_differ(f["axis"], best["axis"], model["directional"])),
        None,
    )
    ambiguity_ratio = (
        alternative["cost"] / max(best["cost"], 1e-6) if alternative else None
    )
    return {
        "best": best,
        "alternative": alternative,
        "ambiguity_ratio": ambiguity_ratio,
        "candidates": [
            {"cost": round(f["cost"], 4), "iou": round(f["iou"], 3),
             "axis_camera": np.round(f["axis"], 4).tolist()}
            for f in fits
        ],
        "mount": None,
        "K_roi": K_roi,
        "roi": roi,
    }


def fit_status(fit, detection):
    reasons = []
    best = fit["best"]
    if best["iou"] < PART_FIT_MIN_IOU:
        return "UNKNOWN", [f"Poor CAD fit (IoU {best['iou']:.2f})"]
    if best["iou"] < PART_FIT_MIN_IOU_OK:
        reasons.append(f"Mediocre CAD fit (IoU {best['iou']:.2f})")
    ratio = fit["ambiguity_ratio"]
    if ratio is not None and ratio < PART_FIT_AMBIGUITY_RATIO:
        reasons.append(
            f"Ambiguous orientation (alternative at {ratio:.2f}x cost)"
        )
    if detection["touches_image_border"]:
        reasons.append("Part cut off by the image edge")
    return ("LOW_CONFIDENCE" if reasons else "OK"), reasons or ["OK"]


# --------------------------------------------------------------------------
# Obraz diagnostyczny
# --------------------------------------------------------------------------

def draw_fit(debug, fit, model, camera, status):
    best = fit["best"]
    transform = T_camera_part(best["center"], best["axis"], model)
    rendered = render_silhouette(model, transform, camera["K"],
                                 debug.shape[:2])
    contours, _ = cv2.findContours(rendered.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = {"OK": (0, 255, 0), "LOW_CONFIDENCE": (0, 165, 255)}.get(
        status, (0, 0, 255))
    cv2.drawContours(debug, contours, -1, color, 3)

    # Oś części (+Z modelu) od środka bryły.
    length = max(model["length_mm"], 2 * model["radius_mm"]) * 0.8
    points = np.array([best["center"], best["center"] + best["axis"] * length])
    projected = points @ camera["K"].T
    uv = (projected[:, :2] / projected[:, 2:3]).astype(int)
    cv2.arrowedLine(debug, tuple(uv[0]), tuple(uv[1]), (255, 0, 255), 4,
                    tipLength=0.2)


# --------------------------------------------------------------------------

def load_cameras():
    with open(CAMERA_POSES_PATH, encoding="utf-8") as file:
        report = json.load(file)
    cameras = {}
    for entry in report["images"]:
        if "T_world_camera" not in entry or entry["status"] == "UNKNOWN":
            continue
        T_world_camera = np.array(entry["T_world_camera"], dtype=float)
        T_world_camera[:3, 3] *= 1000  # metry -> milimetry
        cameras[entry["image"]] = {
            "status": entry["status"],
            "K": np.array(entry["K"], dtype=float),
            "T_world_camera": T_world_camera,
        }
    return cameras


# Dane współdzielone przez procesy robocze (ładowane raz na proces).
_MODELS = None
_CAMERAS = None


def _init_worker():
    global _MODELS, _CAMERAS
    # Jeden wątek OpenCV na proces — równoległość daje pula procesów.
    cv2.setNumThreads(1)
    _MODELS = load_models()
    _CAMERAS = load_cameras()


def process_image(image_entry):
    """
    Dopasowanie wszystkich części jednego zdjęcia (zdjęcia są niezależne,
    więc mogą być liczone równolegle). Zwraca (wpisy, liczniki, tekst).
    """
    models, cameras = _MODELS, _CAMERAS
    results = []
    counts = {"OK": 0, "LOW_CONFIDENCE": 0, "UNKNOWN": 0, "SKIPPED": 0}

    name = image_entry["image"]
    camera = cameras.get(name)
    detections = image_entry["detections"]
    by_id = {d["id"]: d for d in detections}
    image_size = (image_entry["image_width"], image_entry["image_height"])

    if camera is None:
        counts["SKIPPED"] += sum(1 for d in detections if d["role"] == "part")
        return results, counts, f"[SKIPPED] {name}: no reliable " \
                                "camera pose"

    image = cv2.imread(str(ACCEPTED_DIR / name))
    debug = image.copy() if image is not None else None
    cache = MaskCache()
    line = []
    fitted = {}  # id -> (klasa, środek, oś) w układzie kamery

    # Najpierw części-rodzice z PART_MOUNTS (np. silnik), potem reszta.
    parent_classes = {r["parent_class"] for r in PART_MOUNTS.values()}
    ordered = sorted(detections,
                     key=lambda d: d["class_id"] not in parent_classes)

    for detection in ordered:
        if detection["role"] != "part":
            continue
        model = models.get(detection["class_id"])
        entry = {
            "image": name,
            "detection_id": detection["id"],
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "camera_pose_status": camera["status"],
        }
        if model is None:
            entry.update(status="SKIPPED",
                         reasons=["No CAD model for this class"])
            counts["SKIPPED"] += 1
            results.append(entry)
            continue

        features = [by_id[f["id"]] for f in detection["features"]
                    if f["id"] in by_id]
        # Cechy z offsetem (wektor środek koła -> zębnik).
        for feature, link in zip(features, detection["features"]):
            feature["offset_px"] = link["offset_px"]
        others = [d for d in detections
                  if d["id"] != detection["id"]
                  and d["id"] not in {f["id"] for f in features}]

        fit = fit_part(detection, features, others, image_size,
                       camera, model, cache, fitted)
        if fit is None:
            entry.update(status="UNKNOWN",
                         reasons=["Initial pose cannot be determined"])
            counts["UNKNOWN"] += 1
            results.append(entry)
            continue

        status, reasons = fit_status(fit, detection)
        if status != "UNKNOWN":
            fitted[detection["id"]] = (detection["class_id"],
                                       fit["best"]["center"],
                                       fit["best"]["axis"])
        if detection["class_id"] in PART_MOUNTS and fit["mount"] is None:
            reasons = [r for r in reasons if r != "OK"] + [
                "Not on the parent part's axis — axis fitted freely "
                "(weakly determined)"]
            status = "LOW_CONFIDENCE" if status == "OK" else status
        if camera["status"] != "OK":
            reasons = [r for r in reasons if r != "OK"] + [
                f"Camera pose: {camera['status']}"]
            status = "LOW_CONFIDENCE" if status == "OK" else status
        counts[status] += 1

        best = fit["best"]
        T_cp = T_camera_part(best["center"], best["axis"], model)
        T_wp = camera["T_world_camera"] @ T_cp
        center_world = (camera["T_world_camera"]
                        @ np.append(best["center"], 1.0))[:3]
        axis_world = camera["T_world_camera"][:3, :3] @ best["axis"]

        entry.update({
            "status": status,
            "reasons": reasons,
            "symmetry": "axial" + ("" if model["directional"]
                                   else "+flip"),
            "T_world_part_mm": np.round(T_wp, 4).tolist(),
            "center_world_mm": np.round(center_world, 2).tolist(),
            "axis_world": np.round(axis_world, 4).tolist(),
            "distance_from_camera_mm": round(
                float(np.linalg.norm(best["center"])), 1),
            "fit": {
                "iou": round(best["iou"], 3),
                "miss": round(best["miss"], 3),
                "extra": round(best["extra"], 3),
                "cost": round(best["cost"], 4),
                "evals": best["evals"],
                # >1: CAD trzeba było odsunąć względem pozycji startowej.
                "depth_vs_init": round(
                    float(best["center"][2] / best["init_depth"]), 3),
                "ambiguity_ratio": (
                    round(fit["ambiguity_ratio"], 3)
                    if fit["ambiguity_ratio"] is not None else None),
                "features_used": [f["id"] for f in features],
                "mounted_on": fit["mount"],
            },
            "candidates": fit["candidates"],
        })
        results.append(entry)
        line.append(f"{detection['index']}:{detection['class_name']}="
                    f"{best['iou']:.2f}")

        if debug is not None:
            draw_fit(debug, fit, model, camera, status)

    if debug is not None:
        long_side = max(debug.shape[:2])
        if long_side > 2000:
            debug = cv2.resize(debug, None, fx=2000 / long_side,
                               fy=2000 / long_side,
                               interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(PART_POSE_DEBUG_DIR
                        / f"{name.rsplit('.', 1)[0]}_parts.jpg"), debug)
    line.sort(key=lambda text: int(text.split(":")[0]))
    return results, counts, f"{name:<8} IoU: " + "  ".join(line)


def main():
    for path, stage in ((DETECTIONS_PATH, "detect_parts.py"),
                        (CAMERA_POSES_PATH, "estimate_camera_poses.py")):
        if not path.is_file():
            sys.exit(f"ERROR: Missing {path}. Run {stage} first.")

    with open(DETECTIONS_PATH, encoding="utf-8") as file:
        detections_report = json.load(file)

    PART_POSE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in PART_POSE_DEBUG_DIR.glob("*_parts.jpg"):
        old_file.unlink()

    results = []
    counts = {"OK": 0, "LOW_CONFIDENCE": 0, "UNKNOWN": 0, "SKIPPED": 0}
    started = time.time()

    images = detections_report["images"]
    workers = max(1, min(len(images), PARALLEL_WORKERS or os.cpu_count()))
    # map() zwraca wyniki w kolejności zdjęć — raport identyczny jak
    # przy liczeniu po kolei.
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as pool:
        for image_results, image_counts, text in pool.map(process_image,
                                                          images):
            results.extend(image_results)
            for key, value in image_counts.items():
                counts[key] += value
            print(text, flush=True)

    with open(PART_POSES_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "conventions": {
                "units": "mm",
                "T_world_part_mm": "p_world = T @ p_part (p_part in the CAD "
                                   "model frame, symmetry axis = Z)",
                "symmetry": "axial: rotation about Z undetermined; "
                            "+flip: axis direction also undetermined",
            },
            "parts": results,
            "summary": counts,
        }, file, indent=2, ensure_ascii=False)

    by_class = {}
    for r in results:
        if "fit" in r:
            by_class.setdefault(r["class_name"], []).append(r["fit"]["iou"])

    print("\n--- SUMMARY ---")
    print(f"Time: {time.time() - started:.1f} s")
    print("  ".join(f"{k}: {v}" for k, v in counts.items()))
    for name, ious in sorted(by_class.items()):
        print(f"  {name:<14} n={len(ious):>2}  IoU median "
              f"{np.median(ious):.2f}  min {min(ious):.2f}")
    print(f"Report: {PART_POSES_PATH}")
    print(f"Debug images: {PART_POSE_DEBUG_DIR}")


if __name__ == "__main__":
    main()
