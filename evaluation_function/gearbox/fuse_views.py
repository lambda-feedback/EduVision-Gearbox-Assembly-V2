
"""
Etap 7 — łączenie zdjęć: obserwacje -> fizyczne części w 3D.

Wejście:
    output/detections.json    detekcje, maski, promienie (etap 5)
    output/part_poses.json    pozy z pojedynczych zdjęć (etap 6)
    output/camera_poses.json  kamery (etap 4)

Kroki:
    1. Łączenie obserwacji tej samej części między zdjęciami.
       Para obserwacji z dwóch zdjęć pasuje, gdy promienie przez środki
       masek prawie się przecinają, klasy są zgodne (lub z jednej grupy
       CLASS_CONFUSION_GROUPS) i zgadza się kolor. Grupy łączone są
       zachłannie od najlepszych par, z kontrolą: najwyżej jedna obserwacja
       na zdjęcie, wszystkie promienie blisko wspólnego punktu.
    2. Klasa części = głosowanie obserwacji.
    3. Poza = wspólne dopasowanie sylwetki CAD (stały rozmiar) do masek
       ze wszystkich zdjęć; start z punktu triangulacji i osi z etapu 6.
       Kara za bryłę poniżej płaszczyzny znacznika.
    4. Reguły montażu (PART_MOUNTS): np. driving_gear na osi silnika.
    5. Kontrola skali CAD: odległość z rozmiaru na zdjęciu (etap 6)
       / odległość z triangulacji ≈ rozmiar CAD / rozmiar rzeczywisty.

Uruchomienie:
    python fuse_views.py
"""

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.optimize import minimize

from config import (
    ACCEPTED_DIR,
    CLASS_NAMES,
    DETECTIONS_PATH,
    PART_POSES_PATH,
    CAMERA_POSES_PATH,
    APRILTAG_SIZE_M,
    PART_MOUNTS,
    PART_FIT_MAX_EVALS,
    FUSED_PARTS_PATH,
    FUSION_DEBUG_DIR,
    CLASS_CONFUSION_GROUPS,
    FUSION_MAX_RAY_GAP_MM,
    FUSION_MAX_COLOR_DELTA,
    FUSION_MIN_VIEWS,
    FUSION_MIN_TRIANGULATION_DEG,
    FUSION_MAX_STARTS,
    FUSION_MIN_IOU_OK,
    FUSION_MIN_Z_MM,
    FUSION_BELOW_PLANE_WEIGHT,
    FUSION_MOUNT_MAX_COST_RATIO,
    FUSION_STRONG_VIEWS,
    FUSION_STRONG_VIEWS_FRACTION,
    FUSION_ABSORB_MIN_IOU,
    FUSION_EVICT_MAX_IOU,
    SHAFT_MAX_ANGLE_DEG,
    SHAFT_MAX_OFFSET_MM,
    SHAFT_MAX_COST_RATIO,
    SHAFT_ATTACH_MAX_OFFSET_MM,
    PARALLEL_WORKERS,
)
from estimate_part_poses import (
    MaskCache,
    T_camera_part,
    build_roi,
    load_cameras,
    load_models,
    render_silhouette,
    silhouette_cost,
)
from geometry import invert_transform


# --------------------------------------------------------------------------
# Obserwacje
# --------------------------------------------------------------------------

def class_group(class_id):
    for group in CLASS_CONFUSION_GROUPS:
        if class_id in group:
            return frozenset(group)
    return frozenset({class_id})


def load_observations(cameras):
    with open(DETECTIONS_PATH, encoding="utf-8") as file:
        detections = json.load(file)
    with open(PART_POSES_PATH, encoding="utf-8") as file:
        poses = {p["detection_id"]: p for p in json.load(file)["parts"]}

    images = {}
    observations = []
    for image_entry in detections["images"]:
        name = image_entry["image"]
        if name not in cameras:
            continue
        images[name] = image_entry
        by_id = {d["id"]: d for d in image_entry["detections"]}
        for det in image_entry["detections"]:
            pose = poses.get(det["id"])
            if det["role"] != "part" or det.get("ray_centroid") is None:
                continue
            if pose is None or pose.get("status") in (None, "UNKNOWN",
                                                       "SKIPPED"):
                continue
            ray = det["ray_centroid"]
            observations.append({
                "index": len(observations),
                "image": name,
                "detection": det,
                "features": [by_id[f["id"]] for f in det["features"]
                             if f["id"] in by_id],
                "class_id": det["class_id"],
                "group": class_group(det["class_id"]),
                "confidence": det["confidence"],
                "color": (np.array(det["color_lab"]) if det.get("color_lab")
                          else None),
                "origin": np.array(ray["origin_mm"], dtype=float),
                "direction": np.array(ray["direction"], dtype=float),
                "single_view": pose,
            })
    return observations, images


# --------------------------------------------------------------------------
# Triangulacja
# --------------------------------------------------------------------------

def triangulate(obs_list):
    """Punkt najbliższy wszystkim promieniom (najmniejsze kwadraty)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for obs in obs_list:
        d = obs["direction"]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ obs["origin"]
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def ray_distance(obs, point):
    v = point - obs["origin"]
    along = v @ obs["direction"]
    return float(np.linalg.norm(v - along * obs["direction"])), float(along)


def max_ray_angle_deg(obs_list):
    best = 0.0
    for i, a in enumerate(obs_list):
        for b in obs_list[i + 1:]:
            cos = np.clip(a["direction"] @ b["direction"], -1, 1)
            best = max(best, math.degrees(math.acos(cos)))
    return best


def colors_compatible(obs_list):
    colors = [o["color"] for o in obs_list if o["color"] is not None]
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            if np.linalg.norm(a[1:] - b[1:]) > FUSION_MAX_COLOR_DELTA:
                return False
    return True


def cluster_is_valid(obs_list):
    images = [o["image"] for o in obs_list]
    if len(set(images)) != len(images):
        return None
    if len({o["group"] for o in obs_list}) != 1:
        return None
    if not colors_compatible(obs_list):
        return None
    point = triangulate(obs_list)
    if point is None:
        return None
    for obs in obs_list:
        gap, along = ray_distance(obs, point)
        if gap > FUSION_MAX_RAY_GAP_MM or along <= 0:
            return None
    return point


def associate(observations):
    """Zachłanne łączenie par w grupy (fizyczne części)."""
    pairs = []
    for i, a in enumerate(observations):
        for b in observations[i + 1:]:
            if a["image"] == b["image"] or a["group"] != b["group"]:
                continue
            point = cluster_is_valid([a, b])
            if point is None:
                continue
            gap = max(ray_distance(a, point)[0], ray_distance(b, point)[0])
            pairs.append((gap, a["index"], b["index"]))
    pairs.sort()

    cluster_of = {o["index"]: {o["index"]} for o in observations}
    for _, i, j in pairs:
        if cluster_of[i] is cluster_of[j]:
            continue
        merged = cluster_of[i] | cluster_of[j]
        if cluster_is_valid([observations[k] for k in merged]) is None:
            continue
        for k in merged:
            cluster_of[k] = merged

    unique = {id(c): c for c in cluster_of.values()}
    return [sorted(c) for c in unique.values()]


def vote_class(obs_list):
    votes = Counter(o["class_id"] for o in obs_list)
    confidence = defaultdict(float)
    for o in obs_list:
        confidence[o["class_id"]] += o["confidence"]
    winner = max(votes, key=lambda c: (votes[c], confidence[c]))
    return winner, {CLASS_NAMES[c]: n for c, n in votes.items()}


# --------------------------------------------------------------------------
# Wspólne dopasowanie wielu widoków
# --------------------------------------------------------------------------

def axis_from_angles(theta, psi):
    return np.array([math.sin(theta) * math.cos(psi),
                     math.sin(theta) * math.sin(psi), math.cos(theta)])


def angles_from_axis(axis):
    return math.acos(np.clip(axis[2], -1, 1)), math.atan2(axis[1], axis[0])


def lowest_z(model, T_world_part):
    points = model["profile_points"].reshape(-1, 3)
    world = points @ T_world_part[:3, :3].T + T_world_part[:3, 3]
    return float(world[:, 2].min())


class MultiViewProblem:
    """Maski i kamery wszystkich zdjęć jednej części."""

    def __init__(self, obs_list, images, cameras, model, cache):
        self.model = model
        self.views = []
        for obs in obs_list:
            image_entry = images[obs["image"]]
            camera = cameras[obs["image"]]
            det = obs["detection"]
            own = {det["id"]} | {f["id"] for f in obs["features"]}
            others = [d for d in image_entry["detections"]
                      if d["id"] not in own]
            masks, K_roi, _ = build_roi(
                [det] + obs["features"], others,
                (image_entry["image_width"], image_entry["image_height"]),
                camera["K"], cache,
            )
            self.views.append({
                "image": obs["image"],
                "masks": masks,
                "K_roi": K_roi,
                "T_camera_world": invert_transform(camera["T_world_camera"]),
            })

    def evaluate(self, center, axis):
        T_world_part = T_camera_part(center, axis, self.model)
        costs, ious = [], []
        for view in self.views:
            T_cp = view["T_camera_world"] @ T_world_part
            rendered = render_silhouette(self.model, T_cp, view["K_roi"],
                                         view["masks"]["target"].shape)
            cost, iou, _, _ = silhouette_cost(rendered, view["masks"])
            costs.append(cost)
            ious.append(iou)
        below = max(0.0, FUSION_MIN_Z_MM - lowest_z(self.model, T_world_part))
        total = float(np.mean(costs)) + FUSION_BELOW_PLANE_WEIGHT * below
        return total, ious, below

    def fit(self, center, axis, fixed_axis=False):
        if fixed_axis:
            def unpack(p):
                return np.asarray(p[:3]), axis
            x0 = np.asarray(center, dtype=float)
            steps = np.diag([1.0, 1.0, 1.0])
        else:
            def unpack(p):
                return np.asarray(p[:3]), axis_from_angles(p[3], p[4])
            x0 = np.concatenate([center, angles_from_axis(axis)])
            steps = np.diag([1.0, 1.0, 1.0, 0.1, 0.1])

        result = minimize(
            lambda p: self.evaluate(*unpack(p))[0], x0, method="Powell",
            options={"maxfev": PART_FIT_MAX_EVALS, "xtol": 1e-3,
                     "ftol": 1e-4, "direc": steps},
        )
        c, n = unpack(result.x)
        total, ious, below = self.evaluate(c, n)
        return {"center": c, "axis": n, "cost": total, "ious": ious,
                "below_plane_mm": below, "evals": int(result.nfev)}

    def fit_on_axis(self, parent_center, parent_axis, s_init):
        """Środek na osi rodzica, oś = oś rodzica; szukane s (mm)."""
        def at(s):
            return parent_center + parent_axis * s

        grid = np.arange(s_init - 40, s_init + 40.01, 0.5)
        costs = [self.evaluate(at(s), parent_axis)[0] for s in grid]
        s_best = float(grid[int(np.argmin(costs))])
        fine = np.arange(s_best - 0.5, s_best + 0.51, 0.1)
        costs = [self.evaluate(at(s), parent_axis)[0] for s in fine]
        s_best = float(fine[int(np.argmin(costs))])
        total, ious, below = self.evaluate(at(s_best), parent_axis)
        return {"center": at(s_best), "axis": parent_axis, "cost": total,
                "ious": ious, "below_plane_mm": below, "evals": 0}


def start_axes(obs_list, directional):
    """Różne osie z etapu 6 (świat) jako punkty startowe."""
    axes = []
    for obs in sorted(obs_list, key=lambda o: -o["single_view"]["fit"]["iou"]):
        axis = np.array(obs["single_view"]["axis_world"], dtype=float)
        duplicate = any(
            (axis @ a if directional else abs(axis @ a))
            > math.cos(math.radians(15)) for a in axes
        )
        if not duplicate:
            axes.append(axis)
        if len(axes) >= FUSION_MAX_STARTS:
            break
    return axes


def axes_differ(a, b, directional):
    dot = float(a @ b) if directional else abs(float(a @ b))
    return dot < math.cos(math.radians(30))


# --------------------------------------------------------------------------
# Obrazy diagnostyczne
# --------------------------------------------------------------------------

PART_COLORS = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255),
               (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 255)]


def draw_views(parts, images, pool):
    """Rzuty części 3D na każde zdjęcie (zdjęcia rysowane równolegle)."""
    FUSION_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in FUSION_DEBUG_DIR.glob("*.jpg"):
        old_file.unlink()
    light = [{k: p[k] for k in ("class_id", "T_world_part_mm", "part_index",
                                "images")} for p in parts]
    list(pool.map(_draw_view_task, [(name, light) for name in images]))


def _draw_view_task(args):
    name, parts = args
    models, _, cameras, _ = _CTX
    image = cv2.imread(str(ACCEPTED_DIR / name))
    if image is None:
        return
    camera = cameras[name]
    T_cw = invert_transform(camera["T_world_camera"])
    scale = max(image.shape[:2]) / 2000
    for part in parts:
        model = models[part["class_id"]]
        T_cp = T_cw @ np.array(part["T_world_part_mm"])
        rendered = render_silhouette(model, T_cp, camera["K"],
                                     image.shape[:2])
        if not rendered.any():
            continue
        color = PART_COLORS[part["part_index"] % len(PART_COLORS)]
        seen = name in part["images"]
        contours, _ = cv2.findContours(rendered.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # Linia ciągła: część obserwowana na tym zdjęciu,
        # cienka: tylko rzut (np. zasłonięta lub niewykryta).
        cv2.drawContours(image, contours, -1, color,
                         int(4 * scale) if seen else max(1, int(scale)))
        x, y, w, h = cv2.boundingRect(contours[0])
        label = f"P{part['part_index']}"
        for col, extra in (((0, 0, 0), 4), (color, 0)):
            cv2.putText(image, label, (x + w + 5, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, col,
                        int(2 * scale) + extra)
    factor = 2000 / max(image.shape[:2])
    image = cv2.resize(image, None, fx=factor, fy=factor,
                       interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(FUSION_DEBUG_DIR / f"{name.rsplit('.', 1)[0]}_fused.jpg"),
                image)


def draw_top_view(parts, models):
    """Widok z góry (płaszczyzna XY świata), 10 px na mm."""
    if not parts:
        return
    px_per_mm = 10
    points = []
    for part in parts:
        model = models[part["class_id"]]
        T = np.array(part["T_world_part_mm"])
        world = model["profile_points"].reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
        points.append(world[:, :2])
    tag = APRILTAG_SIZE_M * 1000 / 2
    all_points = np.vstack(points + [np.array([[-tag, -tag], [tag, tag]])])
    lo, hi = all_points.min(axis=0) - 15, all_points.max(axis=0) + 15
    size = np.ceil((hi - lo) * px_per_mm).astype(int)
    canvas = np.full((size[1], size[0], 3), 255, np.uint8)

    def to_px(xy):
        # Y świata w górę obrazu.
        return np.column_stack([(xy[:, 0] - lo[0]) * px_per_mm,
                                (hi[1] - xy[:, 1]) * px_per_mm]).astype(np.int32)

    square = np.array([[-tag, -tag], [tag, -tag], [tag, tag], [-tag, tag]])
    cv2.fillPoly(canvas, [to_px(square)], (60, 60, 60))
    cv2.putText(canvas, "AprilTag", tuple(to_px(square)[3] + [5, 25]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for part, xy in zip(parts, points):
        color = PART_COLORS[part["part_index"] % len(PART_COLORS)]
        segments = xy.reshape(-1, models[part["class_id"]]["profile_points"]
                              .shape[1], 2)
        for segment in segments:
            hull = cv2.convexHull(to_px(segment))
            cv2.fillConvexPoly(canvas, hull, color)
            cv2.polylines(canvas, [hull], True, (0, 0, 0), 1)
        center = to_px(np.array([part["center_world_mm"][:2]]))[0]
        label = f"P{part['part_index']} {part['class_name']}"
        cv2.putText(canvas, label, tuple(center + [8, -8]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Podziałka 10 mm.
    cv2.line(canvas, (20, size[1] - 20), (20 + 10 * px_per_mm, size[1] - 20),
             (0, 0, 0), 3)
    cv2.putText(canvas, "10 mm", (20, size[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(FUSION_DEBUG_DIR / "top_view.png"), canvas)


# --------------------------------------------------------------------------
# Dopasowanie jednej fizycznej części
# --------------------------------------------------------------------------

def apply_mount(class_id, best, problem, parts):
    """Reguła PART_MOUNTS: zwraca (wynik, info, ostrzeżenia)."""
    rule = PART_MOUNTS.get(class_id)
    if rule is None:
        return best, None, []
    candidates = []
    for parent in parts:
        if parent["class_id"] != rule["parent_class"]:
            continue
        p_center = np.array(parent["center_world_mm"])
        p_axis = np.array(parent["axis_world"])
        offset = best["center"] - p_center
        s = float(offset @ p_axis)
        off_axis = float(np.linalg.norm(offset - s * p_axis))
        if rule.get("rigid") or off_axis <= rule["max_axis_distance_mm"]:
            candidates.append((off_axis, parent, p_center, p_axis, s))
    if not candidates:
        return best, None, [
            f"Not on the parent part's axis "
            f"({CLASS_NAMES[rule['parent_class']]})"]

    off_axis, parent, p_center, p_axis, s = min(candidates,
                                                key=lambda c: c[0])
    mounted = problem.fit_on_axis(p_center, p_axis, s)
    ratio = mounted["cost"] / max(best["cost"], 1e-6)
    rigid = bool(rule.get("rigid"))
    accepted = rigid or ratio <= FUSION_MOUNT_MAX_COST_RATIO
    info = {
        "parent": parent["id"],
        "rigid": rigid,
        "free_fit_off_axis_mm": round(off_axis, 2),
        "accepted": bool(accepted),
        "cost_ratio": round(ratio, 3),
    }
    if accepted:
        notes = []
        if rigid and off_axis > rule["max_axis_distance_mm"]:
            notes.append(
                f"Free fit was {off_axis:.0f} mm from the axis of "
                f"{parent['id']} — part placed on the axis by the mounting "
                "rule, check the images")
        return mounted, info, notes
    return best, info, ["Mounting rule rejected — fit on the parent's axis "
                        "is much worse"]


def geometry_class_check(class_id, best, obs_list, context):
    """
    Dla klas mylonych przez model (CLASS_CONFUSION_GROUPS): koszt
    dopasowania każdego modelu CAD z grupy, startując z najlepszej pozy.
    """
    models, images, cameras, cache = context
    group = class_group(class_id)
    if len(group) < 2:
        return None
    costs = {}
    for other in sorted(group):
        if other == class_id:
            costs[CLASS_NAMES[other]] = round(best["cost"], 4)
            continue
        problem = MultiViewProblem(obs_list, images, cameras, models[other],
                                   cache)
        costs[CLASS_NAMES[other]] = round(
            problem.fit(best["center"], best["axis"])["cost"], 4)
    return {"cad_fit_cost": costs,
            "best_cad_match": min(costs, key=costs.get)}


def fit_cluster(obs_list, parts, context, index=None):
    models, images, cameras, cache = context
    class_id, votes = vote_class(obs_list)
    model = models[class_id]
    point = triangulate(obs_list)
    problem = MultiViewProblem(obs_list, images, cameras, model, cache)

    fits = [problem.fit(point, axis)
            for axis in start_axes(obs_list, model["directional"])]
    fits.sort(key=lambda f: f["cost"])
    best = fits[0]
    alternative = next((f for f in fits[1:] if axes_differ(
        f["axis"], best["axis"], model["directional"])), None)

    best, mount_info, warnings = apply_mount(class_id, best, problem, parts)
    if mount_info and mount_info["accepted"]:
        alternative = None

    mean_iou = float(np.mean(best["ious"]))
    angle = max_ray_angle_deg(obs_list)
    ambiguity = (alternative["cost"] / max(best["cost"], 1e-6)
                 if alternative else None)
    if mean_iou < FUSION_MIN_IOU_OK:
        warnings.append(f"Poor fit (mean IoU {mean_iou:.2f})")
    if angle < FUSION_MIN_TRIANGULATION_DEG:
        warnings.append(f"Small angle between images ({angle:.1f}°) — "
                        "distance weakly determined")
    if ambiguity is not None and ambiguity < 1.1:
        warnings.append("Ambiguous orientation even after combining images")
    if best["below_plane_mm"] > 0:
        warnings.append(f"Body {best['below_plane_mm']:.1f} mm below "
                        "the tag plane")

    class_check = geometry_class_check(class_id, best, obs_list, context)
    if len(votes) > 1:
        note = f"Ambiguous class across observations: {votes}"
        if class_check:
            note += f"; CAD geometry fits: {class_check['best_cad_match']}"
        warnings.append(note)

    if index is None:
        index = len(parts)
    T_world_part = T_camera_part(best["center"], best["axis"], model)
    rays = [ray_distance(o, point)[0] for o in obs_list]
    return {
        "part_index": index,
        "id": f"P{index}",
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "status": "LOW_CONFIDENCE" if warnings else "CONFIRMED",
        "warnings": warnings,
        "class_votes": votes,
        "class_geometry_check": class_check,
        "n_views": len(obs_list),
        "images": [o["image"] for o in obs_list],
        "observations": [o["detection"]["id"] for o in obs_list],
        "T_world_part_mm": np.round(T_world_part, 4).tolist(),
        "center_world_mm": np.round(best["center"], 2).tolist(),
        "axis_world": np.round(best["axis"], 4).tolist(),
        "symmetry": "axial" + ("" if model["directional"] else "+flip"),
        "fit": {
            "mean_iou": round(mean_iou, 3),
            "per_view_iou": {o["image"]: round(i, 3)
                             for o, i in zip(obs_list, best["ious"])},
            "cost": round(best["cost"], 4),
            "ambiguity_ratio": (round(ambiguity, 3)
                                if ambiguity is not None else None),
            "below_plane_mm": round(best["below_plane_mm"], 2),
        },
        "triangulation": {
            "point_mm": np.round(point, 2).tolist(),
            "ray_gap_rms_mm": round(float(np.sqrt(np.mean(
                np.square(rays)))), 2),
            "max_ray_angle_deg": round(angle, 1),
            "fit_vs_triangulation_mm": round(float(np.linalg.norm(
                best["center"] - point)), 2),
        },
        "mount": mount_info,
        "shaft": None,
        "_obs": obs_list,
        "_problem": problem,
    }


def absorb(leftovers, parts, context):
    """
    Przypisuje obserwacje z małych grup do pewnych części: rzut części
    (pozy wielowidokowej) na zdjęcie obserwacji musi pokrywać jej maskę.
    Jedna obserwacja na zdjęcie na część; konflikty — wyższe IoU wygrywa.
    """
    models, images, cameras, cache = context
    candidates = []
    for obs in leftovers:
        for part in parts:
            if class_group(part["class_id"]) != obs["group"]:
                continue
            if obs["image"] in part["images"]:
                continue
            problem = MultiViewProblem([obs], images, cameras,
                                       models[part["class_id"]], cache)
            _, ious, _ = problem.evaluate(np.array(part["center_world_mm"]),
                                          np.array(part["axis_world"]))
            if ious[0] >= FUSION_ABSORB_MIN_IOU:
                candidates.append((ious[0], id(obs), obs, part))

    taken_obs, taken_slots, result = set(), set(), []
    for iou, obs_id, obs, part in sorted(candidates, key=lambda c: -c[0]):
        slot = (id(part), obs["image"])
        if obs_id in taken_obs or slot in taken_slots:
            continue
        taken_obs.add(obs_id)
        taken_slots.add(slot)
        result.append((part, obs))
    return result


# --------------------------------------------------------------------------
# Grupy współosiowe (wałki)
# --------------------------------------------------------------------------

def point_line_distance(point, line_point, line_dir):
    v = point - line_point
    return float(np.linalg.norm(v - (v @ line_dir) * line_dir))


def find_shaft_groups(parts):
    """Łączy części o prawie wspólnej osi (union-find)."""
    parent = list(range(len(parts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cos_limit = math.cos(math.radians(SHAFT_MAX_ANGLE_DEG))
    for i, a in enumerate(parts):
        ca, da = np.array(a["center_world_mm"]), np.array(a["axis_world"])
        for j in range(i + 1, len(parts)):
            b = parts[j]
            cb, db = np.array(b["center_world_mm"]), np.array(b["axis_world"])
            if abs(da @ db) < cos_limit:
                continue
            # Odległość każdego środka od osi drugiej części.
            if max(point_line_distance(ca, cb, db),
                   point_line_distance(cb, ca, da)) <= SHAFT_MAX_OFFSET_MM:
                parent[root(i)] = root(j)

    groups = defaultdict(list)
    for i in range(len(parts)):
        groups[root(i)].append(parts[i])
    return [g for g in groups.values() if len(g) >= 2]


def orthonormal_pair(direction):
    helper = np.array([1.0, 0, 0]) if abs(direction[0]) < 0.9 \
        else np.array([0, 1.0, 0])
    e1 = helper - (helper @ direction) * direction
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(direction, e1)


def group_line(core):
    """Prosta startowa grupy z osi i środków jej głównych członków."""
    axes = [np.array(p["axis_world"]) for p in core]
    d0 = sum((1.0 if a @ axes[0] >= 0 else -1.0) * a for a in axes)
    d0 /= np.linalg.norm(d0)
    p0 = np.mean([np.array(p["center_world_mm"]) for p in core], axis=0)
    return p0, d0


def attach_to_groups(parts, groups):
    """
    Dołącza części spoza grup, których środek leży blisko osi grupy
    (bez warunku na własną oś). Zwraca listę (grupa, główni członkowie).
    """
    grouped = {p["id"] for g in groups for p in g}
    lines = [group_line(g) for g in groups]
    result = [(list(g), list(g)) for g in groups]
    for part in parts:
        if part["id"] in grouped:
            continue
        center = np.array(part["center_world_mm"])
        distances = [point_line_distance(center, p0, d0) for p0, d0 in lines]
        if not distances:
            continue
        best = int(np.argmin(distances))
        if distances[best] <= SHAFT_ATTACH_MAX_OFFSET_MM:
            result[best][0].append(part)
    return result


def align_shaft(group, core):
    """
    Wspólna oś grupy: kierunek (2 kąty) + przesunięcie prostej w poprzek
    (2) + położenie każdej części wzdłuż osi. Kierunek startowy tylko
    z głównych członków (core). Zwrot osi części kierunkowych (np. zębnik
    gear_big) zachowany z dopasowania indywidualnego.
    """
    p0, d0 = group_line(core)
    signs = [1.0 if np.array(p["axis_world"]) @ d0 >= 0 else -1.0
             for p in group]
    centers = [np.array(p["center_world_mm"]) for p in group]
    e1, e2 = orthonormal_pair(d0)
    theta0, psi0 = angles_from_axis(d0)
    s0 = [float((c - p0) @ d0) for c in centers]

    def unpack(x):
        d = axis_from_angles(x[0], x[1])
        line_point = p0 + x[2] * e1 + x[3] * e2
        return d, line_point, x[4:]

    def poses(x):
        d, line_point, s = unpack(x)
        return [(line_point + s_i * d, sign * d)
                for s_i, sign in zip(s, signs)]

    def cost(x):
        return sum(p["_problem"].evaluate(c, n)[0]
                   for p, (c, n) in zip(group, poses(x)))

    x0 = np.concatenate([[theta0, psi0, 0.0, 0.0], s0])
    result = minimize(
        cost, x0, method="Powell",
        options={"maxfev": PART_FIT_MAX_EVALS * len(group), "xtol": 1e-3,
                 "ftol": 1e-4,
                 "direc": np.diag([0.05, 0.05, 0.5, 0.5]
                                  + [0.5] * len(group))},
    )
    d, line_point, _ = unpack(result.x)
    return {
        "poses": poses(result.x),
        "cost": float(result.fun),
        "axis": d,
        "point": line_point,
    }


def problem_of(part, context):
    """Problem wielowidokowy części (odtwarzany, jeśli liczony w innym
    procesie — budowa jest deterministyczna)."""
    if part.get("_problem") is None:
        models, images, cameras, cache = context
        part["_problem"] = MultiViewProblem(part["_obs"], images, cameras,
                                            models[part["class_id"]], cache)
    return part["_problem"]


def apply_shaft_alignment(parts, context, pool):
    models = context[0]
    groups = attach_to_groups(parts, find_shaft_groups(parts))
    # Wałki są od siebie niezależne — wyrównywane równolegle.
    tasks = [([strip_for_worker(p) for p in group],
              [p["id"] for p in core]) for group, core in groups]
    alignments = list(pool.map(_align_task, tasks))

    shafts = []
    for (group, core), aligned in zip(groups, alignments):
        before = sum(p["fit"]["cost"] for p in group)
        ratio = aligned["cost"] / max(before, 1e-6)
        shaft_id = f"S{len(shafts)}"
        accepted = ratio <= SHAFT_MAX_COST_RATIO
        shafts.append({
            "id": shaft_id,
            "members": [p["id"] for p in group],
            "accepted": bool(accepted),
            "cost_ratio": round(ratio, 3),
            "axis_world": np.round(aligned["axis"], 4).tolist(),
            "point_world_mm": np.round(aligned["point"], 2).tolist(),
        })
        for part, (center, axis) in zip(group, aligned["poses"]):
            if not accepted:
                part["warnings"].append(
                    f"Alignment to common axis {shaft_id} rejected "
                    f"(cost x{ratio:.2f})")
                part["status"] = "LOW_CONFIDENCE"
                continue
            model = models[part["class_id"]]
            total, ious, below = problem_of(part, context).evaluate(center,
                                                                     axis)
            part["center_world_mm"] = np.round(center, 2).tolist()
            part["axis_world"] = np.round(axis, 4).tolist()
            part["T_world_part_mm"] = np.round(
                T_camera_part(center, axis, model), 4).tolist()
            part["fit"].update({
                "mean_iou": round(float(np.mean(ious)), 3),
                "per_view_iou": {img: round(i, 3) for img, i in
                                 zip(part["images"], ious)},
                "cost": round(total, 4),
                "ambiguity_ratio": None,
                "below_plane_mm": round(below, 2),
            })
            part["shaft"] = shaft_id
            # Orientację wyznacza teraz cała grupa.
            part["warnings"] = [w for w in part["warnings"]
                                if not w.startswith("Ambiguous orientation")]
            part["status"] = ("LOW_CONFIDENCE" if part["warnings"]
                              else "CONFIRMED")
    return shafts


# --------------------------------------------------------------------------
# Obliczenia równoległe (procesy robocze)
# --------------------------------------------------------------------------

_CTX = None   # (modele, zdjęcia, kamery, cache masek) w procesie roboczym


def _init_worker(images, cameras):
    global _CTX
    cv2.setNumThreads(1)  # równoległość daje pula procesów
    _CTX = (load_models(), images, cameras, MaskCache())


def strip_for_worker(part):
    """Kopia części bez problemu wielowidokowego (odtwarzany w procesie)."""
    return {k: v for k, v in part.items() if k != "_problem"}


def mount_parents(parts):
    """Lekkie kopie części-rodziców reguł montażu (tyle wystarcza
    apply_mount)."""
    parent_classes = {r["parent_class"] for r in PART_MOUNTS.values()}
    return [{k: p[k] for k in ("id", "class_id", "center_world_mm",
                               "axis_world")}
            for p in parts if p["class_id"] in parent_classes]


def _fit_task(args):
    obs_list, parents, index = args
    part = fit_cluster(obs_list, parents, _CTX, index=index)
    part["_problem"] = None
    return part


def _align_task(args):
    group, core_ids = args
    for part in group:
        problem_of(part, _CTX)
    core = [p for p in group if p["id"] in core_ids]
    return align_shaft(group, core)


def fit_in_order(pool, tasks, parents_before):
    """
    Dopasowuje grupy obserwacji równolegle, zachowując zależności wersji
    sekwencyjnej: tasks = [(obs_list, index)] w kolejności numerów;
    parents_before(index, gotowe) -> części-rodzice widoczne dla tej
    części. Na rodziców czekają tylko klasy z regułą montażu (np.
    driving_gear na silniku) — reszta liczona od razu, równolegle.
    """
    parent_classes = {r["parent_class"] for r in PART_MOUNTS.values()}
    classes = [vote_class(obs)[0] for obs, _ in tasks]
    results, futures = {}, {}

    # Niezależne (w tym rodzice): od razu. Ich rodzice nie wpływają na
    # wynik, bo klasa nie ma reguły montażu.
    for (obs, idx), cls in zip(tasks, classes):
        if cls not in PART_MOUNTS:
            futures[idx] = pool.submit(_fit_task, (obs, [], idx))

    # Zależne: po zakończeniu rodziców, które widziałyby sekwencyjnie.
    for (obs, idx), cls in zip(tasks, classes):
        if cls in PART_MOUNTS:
            for (_, j), parent_cls in zip(tasks, classes):
                if parent_cls in parent_classes and j < idx:
                    results[j] = futures[j].result()
            futures[idx] = pool.submit(
                _fit_task, (obs, parents_before(idx, results), idx))

    for _, idx in tasks:
        results[idx] = futures[idx].result()
    return [results[idx] for _, idx in tasks]


# --------------------------------------------------------------------------

def main():
    for path, stage in ((PART_POSES_PATH, "estimate_part_poses.py"),
                        (DETECTIONS_PATH, "detect_parts.py"),
                        (CAMERA_POSES_PATH, "estimate_camera_poses.py")):
        if not path.is_file():
            sys.exit(f"ERROR: Missing {path}. Run {stage} first.")

    started = time.time()
    models = load_models()
    cameras = load_cameras()
    observations, images = load_observations(cameras)
    clusters = associate(observations)
    cache = MaskCache()

    groups = [[observations[i] for i in cluster] for cluster in clusters]
    strong_views = max(FUSION_STRONG_VIEWS,
                       math.ceil(FUSION_STRONG_VIEWS_FRACTION * len(images)))
    strong = [g for g in groups if len(g) >= strong_views]
    weak = [g for g in groups if len(g) < strong_views]

    # Rodzice reguł montażu najpierw (np. silnik przed driving_gear).
    parent_classes = {r["parent_class"] for r in PART_MOUNTS.values()}

    def by_parent_first(group_list):
        return sorted(group_list,
                      key=lambda g: vote_class(g)[0] not in parent_classes)

    context = (models, images, cameras, cache)
    # Tyle procesów, ile jest równoległych zadań (części / zdjęcia).
    workers = max(1, min(PARALLEL_WORKERS or os.cpu_count(),
                         max(len(strong), len(images))))
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                               initargs=(images, cameras))

    # 1. Pewne części (wiele zdjęć). Numer części = pozycja na liście;
    #    rodzic reguły montażu ma zawsze niższy numer niż część osadzona.
    ordered = by_parent_first(strong)
    parts = fit_in_order(
        pool, [(obs, i) for i, obs in enumerate(ordered)],
        lambda idx, done: mount_parents(
            [done[j] for j in sorted(done) if j < idx]))

    # 2a. Odłączenie obserwacji, których model 3D części nie pokrywa
    #     (dołączone do niewłaściwej części) — trafiają do puli wolnych.
    evicted, changed = [], set()
    for part in parts:
        ious = part["fit"]["per_view_iou"]
        bad = [obs for obs in part["_obs"]
               if ious.get(obs["image"], 1.0) < FUSION_EVICT_MAX_IOU]
        if bad and len(part["_obs"]) - len(bad) >= FUSION_MIN_VIEWS:
            bad_ids = {id(obs) for obs in bad}
            part["_obs"] = [obs for obs in part["_obs"]
                            if id(obs) not in bad_ids]
            part["evicted_observations"] = [o["detection"]["id"] for o in bad]
            evicted += bad
            changed.add(id(part))

    # 2b. Dołączanie wolnych obserwacji (małe grupy + odłączone) do
    #     pewnych części.
    leftovers = [obs for g in weak for obs in g] + evicted
    absorbed = absorb(leftovers, parts, context)
    changed |= {id(part) for part, _ in absorbed}
    for part, obs in absorbed:
        part["_obs"].append(obs)
    absorbed_obs = {id(obs) for _, obs in absorbed}

    # 3. Ponowne dopasowanie zmienionych części (te same numery). Rodzice
    #    liczeni przed częściami osadzonymi — jak w kolejności listy.
    refit = [(i, part) for i, part in enumerate(parts)
             if id(part) in changed]
    if refit:
        current = {i: part for i, part in enumerate(parts)}

        def parents_for_refit(idx, done):
            state = dict(current)
            state.update(done)
            return mount_parents([state[j] for j in sorted(state)
                                  if j != idx])

        new_parts = fit_in_order(
            pool, [(part["_obs"], part["part_index"]) for _, part in refit],
            parents_for_refit)
        for (i, old), new in zip(refit, new_parts):
            new["absorbed_observations"] = [
                obs["detection"]["id"] for p, obs in absorbed if p is old]
            parts[i] = new

    # 4. Pozostałe małe grupy: >= FUSION_MIN_VIEWS -> część, reszta
    #    to obserwacje niepotwierdzone.
    unconfirmed, extra = [], []
    for g in weak:
        rest = [obs for obs in g if id(obs) not in absorbed_obs]
        if len(rest) >= FUSION_MIN_VIEWS and cluster_is_valid(rest) is not None:
            extra.append((rest, len(parts) + len(extra)))
        else:
            unconfirmed.extend([obs] for obs in rest)
    unconfirmed.extend([obs] for obs in evicted
                       if id(obs) not in absorbed_obs)
    if extra:
        base = list(parts)
        parts += fit_in_order(
            pool, extra,
            lambda idx, done: mount_parents(
                base + [done[j] for j in sorted(done) if j < idx]))

    # 5. Wspólne osie (wałki).
    shafts = apply_shaft_alignment(parts, context, pool)

    # Kontrola skali CAD (por. docstring).
    scale_ratios = defaultdict(list)
    for part in parts:
        center = np.array(part["center_world_mm"])
        for obs in part["_obs"]:
            single = obs["single_view"].get("distance_from_camera_mm")
            if single:
                scale_ratios[part["class_name"]].append(
                    single / float(np.linalg.norm(center - obs["origin"])))
    scale_check = {
        name: {"median_single_over_fused": round(float(np.median(v)), 3),
               "n": len(v)}
        for name, v in scale_ratios.items()
    }

    for part in parts:
        del part["_obs"]
        del part["_problem"]

    unconfirmed_report = [{
        "class_name": CLASS_NAMES[obs_list[0]["class_id"]],
        "image": obs_list[0]["image"],
        "detection_id": obs_list[0]["detection"]["id"],
        "confidence": obs_list[0]["confidence"],
        "reason": "Not matched to any part — possibly a false detection "
                  "or a part occluded in the other images",
    } for obs_list in unconfirmed]

    class_counts = Counter(p["class_name"] for p in parts)
    with open(FUSED_PARTS_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "images_used": sorted(images),
            "parts": parts,
            "shafts": shafts,
            "unconfirmed_observations": unconfirmed_report,
            "class_counts": dict(class_counts),
            "cad_scale_check": scale_check,
        }, file, indent=2, ensure_ascii=False)

    draw_views(parts, list(images), pool)
    pool.shutdown()
    draw_top_view(parts, models)

    print(f"Observations: {len(observations)}  from images: {len(images)}")
    print(f"Physical parts (≥{FUSION_MIN_VIEWS} images): {len(parts)}  "
          f"unconfirmed observations: {len(unconfirmed)}\n")
    for p in parts:
        c = p["center_world_mm"]
        mount = (f"  on axis of {p['mount']['parent']}"
                 if p["mount"] and p["mount"]["accepted"] else "")
        print(f"{p['id']:<4} {p['class_name']:<13} {p['status']:<15} "
              f"views={p['n_views']:>2}  IoU={p['fit']['mean_iou']:.2f}  "
              f"centre=({c[0]:6.1f}, {c[1]:6.1f}, {c[2]:5.1f}) mm  "
              f"votes={p['class_votes']}{mount}")
        for w in p["warnings"]:
            print(f"       - {w}")

    print("\nCommon axes:")
    for s in shafts:
        state = "aligned" if s["accepted"] else "REJECTED"
        print(f"  {s['id']}: {', '.join(s['members'])}  ({state}, cost "
              f"x{s['cost_ratio']})")
    print("\nPart counts:", dict(class_counts))
    print("\nCAD scale check (distance from size / from triangulation;"
          " ~1.0 = CAD dimensions consistent):")
    for name, s in sorted(scale_check.items()):
        print(f"  {name:<14} {s['median_single_over_fused']:.3f}  (n={s['n']})")
    print(f"\nTime: {time.time() - started:.1f} s")
    print(f"Report: {FUSED_PARTS_PATH}")
    print(f"Images: {FUSION_DEBUG_DIR}  (top_view.png = top view)")


if __name__ == "__main__":
    main()
