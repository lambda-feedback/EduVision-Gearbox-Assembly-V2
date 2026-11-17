"""
Szybsze wersje najczęściej wywoływanych funkcji GearBox — dają wyniki
IDENTYCZNE (bit w bit) z oryginałem, więc ocena się nie zmienia.

Dopasowanie sylwetki CAD (etapy "cad" i "fuse") to dziesiątki tysięcy ocen
kosztu w optymalizatorze Powella; każda ocena to rzut bryły i porównanie
z maską ROI. Większość czasu oceny to narzut małych operacji numpy i kilka
przebiegów po całym ROI. Tutaj:

    axis_frame,       ta sama arytmetyka bez np.linalg.norm (dla wektora
    T_camera_part     float to dokładnie sqrt(x.dot(x))) i bez pośrednich
                      kopii (identyczne wyniki IEEE),
    render_silhouette ten sam obraz; zapamiętuje ramkę rzutu,
    silhouette_cost   liczniki pikseli tylko w ramce rzutu, jednym
                      przebiegiem po etykietach ROI (tło / cel /
                      zasłaniające / poza kadrem) — te same liczby
                      całkowite, więc te same koszty,
    build_roi         oryginał + tablica etykiet ROI,
    MaskCrop.from_tensor  (etap "yolo") ramka maski bez progowania całej
                      maski 12 MPx (~30 ms -> ~4 ms na detekcję).

Kod GearBox (gearbox/, kopia aplikacji) nie jest zmieniany — funkcje są
podmieniane w modułach po imporcie (apply()). EDUVISION_ACCEL=0 wyłącza
podmianę (np. do porównania wyników z oryginałem). Po aktualizacji gearbox/
(tools/sync_gearbox.py) accel_test.py sprawdza, czy zamienniki nadal dają
te same wyniki co nowe oryginały.
"""

import contextlib
import math
import os

import cv2
import numpy as np

# Etykiety pikseli ROI (rozłączne: build_roi wyklucza cel z pozostałych
# masek, a skalowanie INTER_NEAREST zachowuje rozłączność).
BACKGROUND, TARGET, OCCLUDER, OUTSIDE = 0, 1, 2, 3

# Wypełnianie wielokąta (shift=4, LINE_8) nie wychodzi poza ±1 px od
# wierzchołków — ramka z zapasem 2 px na pewno zawiera całą sylwetkę.
_MARGIN = 2
_MAX_COORD = 1 << 28        # współrzędne x16 bezpiecznie w int32

_ORIGINAL = {}              # oryginalne funkcje (fallback, testy)
_LAST = {"mask": None}      # ostatnio narysowana sylwetka i jej ramka
_WEIGHTS = {}               # wagi kosztu z config.py

_PATCHED = ("axis_frame", "T_camera_part", "render_silhouette",
            "silhouette_cost", "build_roi")


def enabled():
    return os.environ.get("EDUVISION_ACCEL", "1") != "0"


# ------------------------------------------------------------------ poza

def _norm(vector):
    # np.linalg.norm(1-D float) liczy dokładnie sqrt(x.dot(x)).
    return math.sqrt(vector.dot(vector))


_HELPER_X = np.array([1.0, 0, 0])
_HELPER_Y = np.array([0, 1.0, 0])
_HELPER_X.flags.writeable = False
_HELPER_Y.flags.writeable = False


def axis_frame(axis):
    """= estimate_part_poses.axis_frame (te same operacje na liczbach)."""
    axis = axis / _norm(axis)
    helper = _HELPER_X if abs(axis[0]) < 0.9 else _HELPER_Y
    x = helper - helper.dot(axis) * axis
    x /= _norm(x)
    a0, a1, a2 = axis.tolist()
    x0, x1, x2 = x.tolist()
    return np.array([
        [x0, a1 * x2 - a2 * x1, a0],
        [x1, a2 * x0 - a0 * x2, a1],
        [x2, a0 * x1 - a1 * x0, a2],
    ])


def T_camera_part(center, axis, model):
    """= estimate_part_poses.T_camera_part (geometry.make_transform)."""
    rotation = axis_frame(axis)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(center - rotation @ model["center"],
                                  dtype=float).ravel()
    return transform


# ------------------------------------------------------------- sylwetka

def render_silhouette(model, transform, K, shape):
    """= estimate_part_poses.render_silhouette (ten sam obraz); dodatkowo
    zapamiętuje ramkę rzutu dla silhouette_cost."""
    points = model["profile_points"] @ transform[:3, :3].T + transform[:3, 3]
    mask = np.zeros(shape, dtype=np.uint8)
    if (points[..., 2] <= 1.0).any():
        mask = mask.astype(bool)
        _LAST["mask"], _LAST["box"] = mask, (0, 0, 0, 0)
        return mask
    projected = points @ K.T
    uv = projected[..., :2] / projected[..., 2:3]
    x = y = x_hi = y_hi = None
    for segment_uv in np.round(uv * 16).astype(np.int32):
        hull = cv2.convexHull(segment_uv)
        cv2.fillConvexPoly(mask, hull, 1, lineType=cv2.LINE_8, shift=4)
        # Ramka wierzchołków (współrzędne x16).
        hx, hy, hw, hh = cv2.boundingRect(hull)
        if x is None:
            x, y, x_hi, y_hi = hx, hy, hx + hw - 1, hy + hh - 1
        else:
            x, y = min(x, hx), min(y, hy)
            x_hi, y_hi = max(x_hi, hx + hw - 1), max(y_hi, hy + hh - 1)
    box = None
    if x is not None and max(-x, -y, x_hi, y_hi) < _MAX_COORD:
        height, width = shape
        box = (max(0, (x >> 4) - _MARGIN), max(0, (y >> 4) - _MARGIN),
               min(width, (x_hi >> 4) + _MARGIN + 2),
               min(height, (y_hi >> 4) + _MARGIN + 2))
    mask = mask.view(bool)
    _LAST["mask"], _LAST["box"] = mask, box
    return mask


def _cost_from_counts(target_count, background, target, occluder, outside):
    """Wzory silhouette_cost z liczników pikseli sylwetki wg etykiet."""
    target_area = max(1, target_count)
    rendered_area = max(1, background + target + occluder + outside)
    intersection = target
    miss = float(target_count - intersection) / target_area
    extra = float(background) / rendered_area
    on_neutral = float(occluder) / rendered_area
    # (cel ∪ sylwetka) ∩ ważne = cel + sylwetka na tle (cel ⊂ ważne).
    union = float(target_count + background)
    iou = float(intersection) / union if union else 0.0
    total = (miss + _WEIGHTS["extra"] * extra
             + _WEIGHTS["neutral"] * on_neutral)
    return total, iou, miss, extra


def silhouette_cost(rendered, masks):
    """= estimate_part_poses.silhouette_cost."""
    label = masks.get("_label")
    if (label is None or rendered is not _LAST["mask"]
            or _LAST["box"] is None or rendered.shape != label.shape):
        return _ORIGINAL["silhouette_cost"](rendered, masks)
    x0, y0, x1, y1 = _LAST["box"]
    if x1 <= x0 or y1 <= y0:
        return _cost_from_counts(masks["target_count"], 0, 0, 0, 0)
    # Histogram etykiet pod sylwetką (float32 — dokładny do 2^24 pikseli).
    hist = cv2.calcHist([label[y0:y1, x0:x1]], [0],
                        rendered[y0:y1, x0:x1].view(np.uint8), [4], [0, 4])
    background, target, occluder, outside = (int(v) for v in hist.ravel())
    return _cost_from_counts(masks["target_count"], background, target,
                             occluder, outside)


def roi_labels(masks):
    """Etykiety ROI (uint8) albo None, gdy maski nie są rozłączne."""
    target, occluders, outside = (masks["target"], masks["occluders"],
                                  masks["outside"])
    if (target & occluders).any() or (target & outside).any() \
            or (occluders & outside).any():
        return None
    label = np.zeros(target.shape, np.uint8)
    label[target] = TARGET
    label[occluders] = OCCLUDER
    label[outside] = OUTSIDE
    return label


def build_roi(*args, **kwargs):
    """= estimate_part_poses.build_roi + masks["_label"]."""
    masks, K_roi, roi = _ORIGINAL["build_roi"](*args, **kwargs)
    masks["_label"] = roi_labels(masks)
    return masks, K_roi, roi


# ------------------------------------------------- detekcje (etap yolo)

def mask_from_tensor(cls, mask_tensor, image_shape):
    """= detect_parts.MaskCrop.from_tensor. Maski YOLO w pełnej
    rozdzielczości (retina_masks) są uint8, a dla uint8 "v > 0.5" to
    "v != 0" — ramka szukana bez progowania całej maski 12 MPx, próg tylko
    na wycinku."""
    if str(mask_tensor.dtype) != "torch.uint8":
        return _ORIGINAL["from_tensor"](cls, mask_tensor, image_shape)
    rows = mask_tensor.any(dim=1).nonzero()
    cols = mask_tensor.any(dim=0).nonzero()
    if len(rows) == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    crop = (mask_tensor[y0:y1 + 1, x0:x1 + 1] > 0.5).cpu().numpy()
    return cls(crop, x0, y0, image_shape)


def apply_detect(detect_parts):
    """Podmiana w detect_parts (raz)."""
    if not enabled() or "from_tensor" in _ORIGINAL:
        return
    _ORIGINAL["from_tensor"] = \
        detect_parts.MaskCrop.__dict__["from_tensor"].__func__
    detect_parts.MaskCrop.from_tensor = classmethod(mask_from_tensor)


# ------------------------------------------------------------- podmiana

def apply(estimate_part_poses, fuse_views):
    """Podmienia funkcje w modułach GearBox (raz). Zwraca True, gdy
    przyspieszenie jest aktywne."""
    if not enabled():
        return False
    if "silhouette_cost" in _ORIGINAL:
        return True
    epp = estimate_part_poses
    for name in _PATCHED:
        _ORIGINAL[name] = getattr(epp, name)
    _WEIGHTS["extra"] = epp.PART_FIT_EXTRA_WEIGHT
    _WEIGHTS["neutral"] = epp.PART_FIT_NEUTRAL_WEIGHT
    replacements = {name: globals()[name] for name in _PATCHED}
    for module in (epp, fuse_views):
        for name, function in replacements.items():
            if hasattr(module, name):
                setattr(module, name, function)
    return True


@contextlib.contextmanager
def original(*modules):
    """Tymczasowo oryginalne funkcje w podanych modułach (testy)."""
    saved = []
    for module in modules:
        for name, function in _ORIGINAL.items():
            if name != "from_tensor" and hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, function)
    try:
        yield
    finally:
        for module, name, function in saved:
            setattr(module, name, function)
