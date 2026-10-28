
"""
Etap 4 — parametry kamer i poza kamery względem AprilTaga.

Wejście (wyniki poprzednich etapów):
    output/accepted/            zaakceptowane zdjęcia (etap 2)
    output/apriltag_report.json narożniki znacznika (etap 2)
    output/image_info.json      wymiary i EXIF (etap 3)

Ogniskowa — hierarchia źródeł, niezależna od modelu telefonu:
    1. EXIF (ekwiwalent 35 mm / płaszczyzna ogniskowej / SENSOR_WIDTH_MM)
       jako wartość a priori z niepewnością FOCAL_REL_STD_EXIF,
    2. pomiar z AprilTaga na zdjęciach z tego samego aparatu — służy do
       kontroli zgodności z EXIF i zawężenia niepewności,
    3. bez EXIF: pomiar z AprilTaga, a gdy jest niewiarygodny —
       wartość domyślna z dużą niepewnością (wynik UNKNOWN).
Zdjęcia grupowane są po (marka, model, wymiary, ogniskowa mm, zoom),
więc różne telefony i różne obiektywy jednego telefonu są rozdzielone.

Konwencje osi i transformacji: patrz geometry.py.

Uruchomienie:
    python estimate_camera_poses.py
"""

import itertools
import json
import math
import sys

import cv2
import numpy as np

from config import (
    ACCEPTED_DIR,
    IMAGE_EXTENSIONS,
    APRILTAG_REPORT_PATH,
    IMAGE_INFO_JSON_PATH,
    APRILTAG_SIZE_M,
    CAMERA_POSES_PATH,
    POSE_DEBUG_DIR,
    FOCAL_REL_STD_EXIF,
    DEFAULT_FOCAL_FACTOR,
    FOCAL_REL_STD_DEFAULT,
    APRILTAG_CORNER_SIGMA_PX,
    TAG_FOCAL_MAX_VIEW_RMS_PX,
    TAG_FOCAL_MAX_REL_STD,
    FOCAL_CONSISTENCY_SIGMA,
    POSE_MAX_REPROJ_RMS_PX,
    POSE_MIN_AMBIGUITY_RATIO,
    POSE_MAX_FOCAL_REL_STD,
    COVERAGE_MIN_IMAGES,
    COVERAGE_MIN_AZIMUTH_SPAN_DEG,
    COVERAGE_MIN_VIEW_ANGLE_DEG,
    COVERAGE_MIN_ELEVATION_RANGE_DEG,
)
from geometry import (
    T_IPPE_WORLD,
    camera_matrix,
    invert_transform,
    rotation_to_euler_zyx_deg,
    rvec_tvec_from_transform,
    tag_corners_ippe,
    tag_corners_world,
    transform_from_rvec_tvec,
)

NO_DISTORTION = np.zeros(5)

TAG_WORLD = tag_corners_world(APRILTAG_SIZE_M)
TAG_IPPE = tag_corners_ippe(APRILTAG_SIZE_M)


# --------------------------------------------------------------------------
# Wczytanie danych z poprzednich etapów
# --------------------------------------------------------------------------

def load_inputs():
    for path, stage in (
        (APRILTAG_REPORT_PATH, "test_apriltag.py"),
        (IMAGE_INFO_JSON_PATH, "inspect_images.py"),
    ):
        if not path.is_file():
            sys.exit(f"ERROR: Missing file {path}. Run {stage} first.")

    with open(APRILTAG_REPORT_PATH, encoding="utf-8") as file:
        tag_report = {
            entry["image"]: entry for entry in json.load(file)
            if entry.get("status") == "ACCEPTED"
        }
    with open(IMAGE_INFO_JSON_PATH, encoding="utf-8") as file:
        image_info = {r["image"]: r for r in json.load(file)["images"]}

    names = sorted(
        p.name for p in ACCEPTED_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return names, tag_report, image_info


def focal_prior(info):
    """Ogniskowa a priori (px), jej sigma (px) i opis źródła."""
    long_side = max(info["width"], info["height"])
    estimates = info.get("focal_px_estimates") or []
    if estimates:
        f = float(estimates[0]["f_px"])
        return f, f * FOCAL_REL_STD_EXIF, f"exif:{estimates[0]['method']}"
    f = DEFAULT_FOCAL_FACTOR * long_side
    return f, f * FOCAL_REL_STD_DEFAULT, "default"


def camera_group_key(info):
    """Zdjęcia o tym samym kluczu dzielą parametry wewnętrzne kamery."""
    if not info.get("exif_present") or not info.get("camera_model"):
        # Bez EXIF nie wiemy, czy dwa zdjęcia są z tego samego aparatu.
        return f"unknown_camera/{info['image']}"
    return "/".join(str(v) for v in (
        info.get("camera_make"),
        info.get("camera_model"),
        f"{info['width']}x{info['height']}",
        f"f{info.get('focal_length_mm')}mm",
        f"zoom{info.get('digital_zoom_ratio') or 1}",
    ))


# --------------------------------------------------------------------------
# Rachunek niepewności
# --------------------------------------------------------------------------

def reprojection(object_points, rvec, tvec, K):
    projected, jacobian = cv2.projectPoints(
        object_points, rvec, tvec, K, NO_DISTORTION
    )
    return projected.reshape(-1, 2), jacobian


def parameter_covariance(jacobian, sigma_px, focal_sigma=None):
    """
    Kowariancja parametrów [rvec(3), tvec(3), f] z linearyzacji.
    focal_sigma: niepewność a priori ogniskowej (None = f swobodne).
    Zwraca None, gdy problem jest źle uwarunkowany.
    """
    J = np.hstack([
        jacobian[:, 0:6],
        (jacobian[:, 6] + jacobian[:, 7])[:, None],  # fx = fy = f
    ])
    information = J.T @ J / sigma_px ** 2
    if focal_sigma is not None:
        information[6, 6] += 1.0 / focal_sigma ** 2
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(covariance)) or np.any(
        np.diag(covariance) <= 0
    ):
        return None
    return covariance


def camera_center_covariance(rvec, tvec, pose_covariance):
    """Propagacja kowariancji pozy na położenie kamery w świecie."""
    def center(params):
        R, _ = cv2.Rodrigues(params[:3])
        return -R.T @ params[3:]

    params = np.concatenate([rvec, tvec])
    base = center(params)
    J = np.zeros((3, 6))
    for i in range(6):
        step = 1e-6
        shifted = params.copy()
        shifted[i] += step
        J[:, i] = (center(shifted) - base) / step
    return J @ pose_covariance @ J.T


# --------------------------------------------------------------------------
# Ogniskowa z AprilTaga (pojedyncze zdjęcie)
# --------------------------------------------------------------------------

def focal_from_view(corners, width, height, f_init):
    """
    Szacuje f z jednego widoku znacznika (pozostałe parametry stałe).
    Przy małym nachyleniu znacznika względem kamery f jest słabo
    określone — wtedy sigma wychodzi duża i widok zostaje pominięty.
    """
    K = camera_matrix(f_init, width, height)
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    )
    try:
        rms, K_est, _, rvecs, tvecs = cv2.calibrateCamera(
            [TAG_WORLD.astype(np.float32)],
            [corners.astype(np.float32)],
            (width, height), K, NO_DISTORTION.copy(), flags=flags,
        )
    except cv2.error:
        return None

    f = float(K_est[0, 0])
    long_side = max(width, height)
    if not 0.3 * long_side < f < 3.0 * long_side:
        return {"f_px": f, "f_std_px": None, "rms_px": float(rms),
                "used": False, "reason": "f poza zakresem fizycznym"}

    rvec, tvec = rvecs[0].ravel(), tvecs[0].ravel()
    _, jacobian = reprojection(TAG_WORLD, rvec, tvec, K_est)
    sigma_px = max(APRILTAG_CORNER_SIGMA_PX, float(rms))
    covariance = parameter_covariance(jacobian, sigma_px)
    f_std = math.sqrt(covariance[6, 6]) if covariance is not None else None

    result = {"f_px": f, "f_std_px": f_std, "rms_px": float(rms)}
    if f_std is None:
        result.update(used=False, reason="f undetermined (poor geometry)")
    elif rms > TAG_FOCAL_MAX_VIEW_RMS_PX:
        result.update(used=False, reason="corner error too large")
    elif f_std / f > TAG_FOCAL_MAX_REL_STD:
        result.update(used=False, reason="tag tilt too small")
    else:
        result.update(used=True, reason="OK")
    return result


def fuse_estimates(values, sigmas):
    """Średnia ważona odwrotnością wariancji z kontrolą zgodności."""
    weights = 1.0 / np.square(sigmas)
    mean = float(np.sum(weights * values) / np.sum(weights))
    sigma = float(1.0 / math.sqrt(np.sum(weights)))
    if len(values) > 1:
        chi2_per_dof = float(
            np.sum(weights * np.square(values - mean)) / (len(values) - 1)
        )
        if chi2_per_dof > 1.0:
            # Rozrzut większy niż wynika z sigm — zwiększ niepewność.
            sigma *= math.sqrt(chi2_per_dof)
    return mean, sigma


def resolve_group_focal(members):
    """
    Ustala ogniskową dla grupy zdjęć z tego samego aparatu.
    members: lista (name, info, corners).
    """
    _, first_info, _ = members[0]
    width, height = first_info["width"], first_info["height"]
    f_prior, sigma_prior, prior_source = focal_prior(first_info)

    views = {}
    for name, _, corners in members:
        views[name] = focal_from_view(corners, width, height, f_prior)

    used = [v for v in views.values() if v and v["used"]]
    tag_estimate = None
    if used:
        f_tag, sigma_tag = fuse_estimates(
            np.array([v["f_px"] for v in used]),
            np.array([v["f_std_px"] for v in used]),
        )
        tag_estimate = {
            "f_px": round(f_tag, 1),
            "f_std_px": round(sigma_tag, 1),
            "views_used": len(used),
        }

    warnings = []
    has_exif_prior = prior_source.startswith("exif")

    if tag_estimate and has_exif_prior:
        f_tag, sigma_tag = tag_estimate["f_px"], tag_estimate["f_std_px"]
        z_score = abs(f_tag - f_prior) / math.hypot(sigma_prior, sigma_tag)
        tag_estimate["z_score_vs_prior"] = round(z_score, 2)
        if z_score > FOCAL_CONSISTENCY_SIGMA:
            # Niezgodność: nie wybieramy na ślepo — zostaje EXIF,
            # ale niepewność obejmuje różnicę.
            f, sigma = f_prior, max(sigma_prior, abs(f_tag - f_prior))
            source = f"{prior_source} (inconsistent with AprilTag)"
            warnings.append(
                f"EXIF focal length ({f_prior:.0f} px) inconsistent with the "
                f"AprilTag measurement ({f_tag:.0f} ± {sigma_tag:.0f} px)"
            )
        else:
            f, sigma = fuse_estimates(
                np.array([f_prior, f_tag]), np.array([sigma_prior, sigma_tag])
            )
            source = f"{prior_source}+apriltag"
    elif has_exif_prior:
        f, sigma, source = f_prior, sigma_prior, prior_source
    elif tag_estimate:
        f, sigma = tag_estimate["f_px"], tag_estimate["f_std_px"]
        source = "apriltag"
    else:
        f, sigma, source = f_prior, sigma_prior, prior_source
        warnings.append(
            "No EXIF and no reliable AprilTag measurement — "
            "default focal length assumed"
        )

    return {
        "focal_px": round(f, 1),
        "focal_std_px": round(sigma, 1),
        "focal_rel_std": round(sigma / f, 4),
        "focal_source": source,
        "prior": {
            "f_px": round(f_prior, 1),
            "f_std_px": round(sigma_prior, 1),
            "source": prior_source,
        },
        "apriltag_estimate": tag_estimate,
        "per_view": {
            name: {k: (round(v, 3) if isinstance(v, float) else v)
                   for k, v in view.items()} if view else None
            for name, view in views.items()
        },
        "width": width,
        "height": height,
        "images": [name for name, _, _ in members],
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# Poza kamery
# --------------------------------------------------------------------------

def estimate_pose(corners, K, focal_std):
    count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        TAG_IPPE, corners, K, NO_DISTORTION,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if count == 0:
        return None

    errors = np.ravel(errors)
    order = np.argsort(errors)
    best = order[0]
    ambiguity_ratio = (
        float(errors[order[1]] / max(errors[best], 1e-9))
        if count > 1 else None
    )

    # Przejście z pomocniczego układu IPPE do układu świata.
    T_camera_ippe = transform_from_rvec_tvec(rvecs[best], tvecs[best])
    T_camera_world = T_camera_ippe @ T_IPPE_WORLD
    rvec, tvec = rvec_tvec_from_transform(T_camera_world)

    rvec, tvec = cv2.solvePnPRefineLM(
        TAG_WORLD, corners, K, NO_DISTORTION,
        rvec.reshape(3, 1).copy(), tvec.reshape(3, 1).copy(),
    )
    rvec, tvec = rvec.ravel(), tvec.ravel()
    T_camera_world = transform_from_rvec_tvec(rvec, tvec)
    T_world_camera = invert_transform(T_camera_world)

    projected, jacobian = reprojection(TAG_WORLD, rvec, tvec, K)
    residuals = np.linalg.norm(projected - corners, axis=1)
    rms = float(np.sqrt(np.mean(np.square(residuals))))

    sigma_px = max(APRILTAG_CORNER_SIGMA_PX, rms)
    covariance = parameter_covariance(jacobian, sigma_px, focal_std)

    camera_position = T_world_camera[:3, 3]
    distance = float(np.linalg.norm(camera_position))
    # Kąt między normalną znacznika a kierunkiem do kamery.
    view_angle = math.degrees(
        math.acos(np.clip(camera_position[2] / distance, -1.0, 1.0))
    )

    uncertainty = None
    if covariance is not None:
        pose_cov = covariance[:6, :6]
        center_cov = camera_center_covariance(rvec, tvec, pose_cov)
        uncertainty = {
            "position_std_mm": np.round(
                np.sqrt(np.diag(center_cov)) * 1000, 2
            ).tolist(),
            "distance_std_mm": round(float(math.sqrt(
                camera_position @ center_cov @ camera_position
            ) / distance * 1000), 2),
            "rotation_std_deg": round(float(math.degrees(math.sqrt(
                np.linalg.eigvalsh(pose_cov[:3, :3]).max()
            ))), 3),
        }

    return {
        "rvec": rvec,
        "tvec": tvec,
        "T_camera_world": T_camera_world,
        "T_world_camera": T_world_camera,
        "reprojection_rms_px": rms,
        "reprojection_max_px": float(residuals.max()),
        "reprojected_corners": projected,
        "ambiguity_ratio": ambiguity_ratio,
        "camera_position_mm": camera_position * 1000,
        "distance_mm": distance * 1000,
        "view_angle_deg": view_angle,
        "uncertainty": uncertainty,
    }


def pose_status(pose, focal):
    reasons = []
    if pose is None:
        return "UNKNOWN", ["Pose could not be estimated"]
    if focal["focal_rel_std"] > POSE_MAX_FOCAL_REL_STD:
        return "UNKNOWN", [
            f"Focal length uncertainty {focal['focal_rel_std']:.0%} — "
            "scale and distances unreliable"
        ]
    if pose["reprojection_rms_px"] > POSE_MAX_REPROJ_RMS_PX:
        reasons.append(
            f"Large reprojection error {pose['reprojection_rms_px']:.2f} px"
        )
    ratio = pose["ambiguity_ratio"]
    if ratio is not None and ratio < POSE_MIN_AMBIGUITY_RATIO:
        reasons.append(f"Ambiguous pose (ratio {ratio:.1f})")
    if pose["uncertainty"] is None:
        reasons.append("Pose uncertainty cannot be estimated")
    reasons.extend(focal["warnings"])
    return ("LOW_CONFIDENCE" if reasons else "OK"), reasons or ["OK"]


# --------------------------------------------------------------------------
# Obraz diagnostyczny
# --------------------------------------------------------------------------

def draw_debug(image, corners, pose, K, status, focal):
    debug = image.copy()
    scale = max(image.shape[:2]) / 2000.0
    thickness = max(2, int(round(3 * scale)))

    if pose is not None:
        # Sześcian o boku znacznika stojący NA znaczniku (oś +Z w górę).
        h = APRILTAG_SIZE_M / 2
        cube = np.array([
            [-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0],
            [-h, -h, 2 * h], [h, -h, 2 * h], [h, h, 2 * h], [-h, h, 2 * h],
        ])
        points, _ = cv2.projectPoints(
            cube, pose["rvec"], pose["tvec"], K, NO_DISTORTION
        )
        points = points.reshape(-1, 2).astype(int)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                     (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(debug, tuple(points[a]), tuple(points[b]),
                     (255, 200, 0), thickness)

        # Osie świata: X czerwona, Y zielona, Z niebieska (BGR).
        axes = np.array([[0, 0, 0], [2 * h, 0, 0], [0, 2 * h, 0],
                         [0, 0, 2 * h]])
        axes_px, _ = cv2.projectPoints(
            axes, pose["rvec"], pose["tvec"], K, NO_DISTORTION
        )
        axes_px = axes_px.reshape(-1, 2).astype(int)
        for end, color in ((1, (0, 0, 255)), (2, (0, 255, 0)),
                           (3, (255, 0, 0))):
            cv2.arrowedLine(debug, tuple(axes_px[0]), tuple(axes_px[end]),
                            color, thickness + 1, tipLength=0.15)

        for point in pose["reprojected_corners"]:
            center = tuple(np.round(point).astype(int))
            cv2.drawMarker(debug, center, (0, 0, 255),
                           cv2.MARKER_CROSS, int(25 * scale), thickness)

    for index, point in enumerate(corners):
        center = tuple(np.round(point).astype(int))
        cv2.circle(debug, center, int(10 * scale), (0, 220, 0), thickness)
        cv2.putText(debug, f"c{index}",
                    (center[0] + int(12 * scale), center[1] - int(12 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (0, 220, 0),
                    thickness)

    lines = [
        status,
        f"f={focal['focal_px']:.0f}+-{focal['focal_std_px']:.0f}px "
        f"({focal['focal_source']})",
    ]
    if pose is not None:
        lines.append(
            f"dist={pose['distance_mm']:.0f}mm  "
            f"angle={pose['view_angle_deg']:.1f}deg  "
            f"rms={pose['reprojection_rms_px']:.2f}px"
        )
    color = {"OK": (0, 200, 0), "LOW_CONFIDENCE": (0, 165, 255)}.get(
        status, (0, 0, 255)
    )
    for i, text in enumerate(lines):
        y = int((40 + 40 * i) * scale)
        cv2.putText(debug, text, (int(15 * scale), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (0, 0, 0),
                    thickness + 3)
        cv2.putText(debug, text, (int(15 * scale), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, color, thickness)

    # Zmniejsz do rozsądnego rozmiaru podglądu.
    long_side = max(debug.shape[:2])
    if long_side > 2000:
        factor = 2000 / long_side
        debug = cv2.resize(debug, None, fx=factor, fy=factor,
                           interpolation=cv2.INTER_AREA)
    return debug


# --------------------------------------------------------------------------
# Pokrycie widokami (czy zdjęcia są z wystarczająco różnych stron)
# --------------------------------------------------------------------------

def view_coverage(results):
    """
    Rozkład aparatów wokół mechanizmu — tylko ostrzeżenia. Kąty liczone
    względem punktu, na który patrzą aparaty (przecięcie osi optycznych,
    w praktyce środek mechanizmu), w układzie znacznika (Z w górę).
    """
    poses = [np.array(r["T_world_camera"]) for r in results
             if r.get("T_world_camera") and r["status"] != "UNKNOWN"]
    coverage = {"n_images": len(poses), "warnings": []}
    if len(poses) < COVERAGE_MIN_IMAGES:
        coverage["warnings"].append(
            f"Only {len(poses)} usable images — at least "
            f"{COVERAGE_MIN_IMAGES} recommended for a reliable 3D result")
    if len(poses) < 2:
        return coverage

    origins = [T[:3, 3] * 1000 for T in poses]
    axes = [T[:3, 2] for T in poses]
    A, b = np.zeros((3, 3)), np.zeros(3)
    for origin, axis in zip(origins, axes):
        P = np.eye(3) - np.outer(axis, axis)
        A += P
        b += P @ origin
    target = np.linalg.solve(A, b)
    offsets = [o - target for o in origins]

    azimuths = sorted(math.degrees(math.atan2(v[1], v[0])) % 360
                      for v in offsets)
    gaps = [b2 - a for a, b2 in zip(azimuths, azimuths[1:])]
    gaps.append(azimuths[0] + 360 - azimuths[-1])
    elevations = [math.degrees(math.asin(np.clip(v[2] / np.linalg.norm(v),
                                                 -1, 1))) for v in offsets]
    max_angle = max(
        math.degrees(math.acos(np.clip(
            a @ b2 / (np.linalg.norm(a) * np.linalg.norm(b2)), -1, 1)))
        for a, b2 in itertools.combinations(offsets, 2))

    span = 360 - max(gaps)
    coverage.update({
        "target_world_mm": np.round(target, 1).tolist(),
        "azimuths_deg": [round(a, 1) for a in azimuths],
        "azimuth_span_deg": round(span, 1),
        "largest_gap_deg": round(max(gaps), 1),
        "elevation_min_deg": round(min(elevations), 1),
        "elevation_max_deg": round(max(elevations), 1),
        "max_view_angle_deg": round(max_angle, 1),
    })
    if span < COVERAGE_MIN_AZIMUTH_SPAN_DEG:
        coverage["warnings"].append(
            f"Images cover only {span:.0f}° around the mechanism (recommended "
            f">= {COVERAGE_MIN_AZIMUTH_SPAN_DEG:.0f}°) — take photos from "
            "more sides")
    if max_angle < COVERAGE_MIN_VIEW_ANGLE_DEG:
        coverage["warnings"].append(
            f"Views are too similar (max {max_angle:.0f}° between images, "
            f"recommended >= {COVERAGE_MIN_VIEW_ANGLE_DEG:.0f}°) — the 3D "
            "positions will be poorly determined")
    if max(elevations) - min(elevations) < COVERAGE_MIN_ELEVATION_RANGE_DEG:
        coverage["warnings"].append(
            f"All images taken at a similar height ({min(elevations):.0f}–"
            f"{max(elevations):.0f}° above the table) — add views from "
            "higher or lower")
    return coverage


# --------------------------------------------------------------------------

def rounded(array, digits=4):
    return np.round(np.asarray(array, dtype=float), digits).tolist()


def main():
    names, tag_report, image_info = load_inputs()
    if not names:
        sys.exit(f"ERROR: No images in {ACCEPTED_DIR}.")

    POSE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in POSE_DEBUG_DIR.glob("*_pose.jpg"):
        old_file.unlink()

    # 1. Grupowanie zdjęć według aparatu.
    groups = {}
    skipped = []
    for name in names:
        info, tag = image_info.get(name), tag_report.get(name)
        if info is None or tag is None:
            skipped.append(name)
            continue
        corners = np.asarray(tag["detections"][0]["corners"], dtype=float)
        if [tag["image_width"], tag["image_height"]] != [
            info["width"], info["height"]
        ]:
            skipped.append(name)
            continue
        groups.setdefault(camera_group_key(info), []).append(
            (name, info, corners)
        )

    for name in skipped:
        print(f"[SKIPPED] {name}: missing or inconsistent stage 2/3 data — "
              "rerun test_apriltag.py and inspect_images.py")

    # 2. Ogniskowa dla każdej grupy.
    cameras = {}
    for key, members in groups.items():
        cameras[key] = resolve_group_focal(members)

    # 3. Poza kamery dla każdego zdjęcia.
    results = []
    counts = {"OK": 0, "LOW_CONFIDENCE": 0, "UNKNOWN": 0}
    for key, members in groups.items():
        focal = cameras[key]
        K = camera_matrix(focal["focal_px"], focal["width"], focal["height"])

        for name, info, corners in members:
            pose = estimate_pose(corners, K, focal["focal_std_px"])
            status, reasons = pose_status(pose, focal)
            counts[status] += 1

            image = cv2.imread(str(ACCEPTED_DIR / name))
            if image is not None:
                cv2.imwrite(
                    str(POSE_DEBUG_DIR / f"{name.rsplit('.', 1)[0]}_pose.jpg"),
                    draw_debug(image, corners, pose, K, status, focal),
                )

            entry = {
                "image": name,
                "status": status,
                "reasons": reasons,
                "camera_group": key,
                "K": rounded(K, 2),
                "dist_coeffs": NO_DISTORTION.tolist(),
                "focal_px": focal["focal_px"],
                "focal_std_px": focal["focal_std_px"],
                "focal_source": focal["focal_source"],
            }
            if pose is not None:
                entry.update({
                    "T_camera_world": rounded(pose["T_camera_world"], 6),
                    "T_world_camera": rounded(pose["T_world_camera"], 6),
                    "camera_position_world_mm": rounded(
                        pose["camera_position_mm"], 1
                    ),
                    "camera_euler_zyx_deg": rounded(
                        rotation_to_euler_zyx_deg(
                            pose["T_world_camera"][:3, :3]
                        ), 2
                    ),
                    "distance_to_tag_mm": round(pose["distance_mm"], 1),
                    "view_angle_deg": round(pose["view_angle_deg"], 1),
                    "reprojection_rms_px": round(
                        pose["reprojection_rms_px"], 3
                    ),
                    "reprojection_max_px": round(
                        pose["reprojection_max_px"], 3
                    ),
                    "ambiguity_ratio": (
                        round(pose["ambiguity_ratio"], 2)
                        if pose["ambiguity_ratio"] is not None else None
                    ),
                    "uncertainty": pose["uncertainty"],
                })
            results.append(entry)

            if pose is None:
                print(f"[{status}] {name}: no pose")
                continue
            unc = pose["uncertainty"] or {}
            print(
                f"[{status:<14}] {name:<10} "
                f"dist={pose['distance_mm']:6.1f} "
                f"± {unc.get('distance_std_mm', float('nan')):4.1f} mm  "
                f"angle={pose['view_angle_deg']:5.1f}°  "
                f"rms={pose['reprojection_rms_px']:.2f} px  "
                f"ambig.={pose['ambiguity_ratio'] or 0:6.1f}"
            )

    results.sort(key=lambda r: r["image"])

    report = {
        "conventions": {
            "units": "metres (T_*), millimetres (*_mm), pixels (*_px)",
            "camera_frame": "OpenCV: X right, Y down, Z forward",
            "world_frame": "AprilTag: tag centre; Z from the tag "
                           "towards the camera; X: c0->c1; Y: c0->c3",
            "transform": "p_a = T_a_b @ p_b",
            "camera_model": "pinhole, fx=fy, principal point at the "
                            "centre, no distortion",
            "apriltag_size_m": APRILTAG_SIZE_M,
        },
        "cameras": cameras,
        "images": results,
        "summary": counts,
        "coverage": view_coverage(results),
    }
    with open(CAMERA_POSES_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n--- CAMERAS ---")
    for key, cam in cameras.items():
        tag = cam["apriltag_estimate"]
        tag_text = (
            f"AprilTag {tag['f_px']:.0f} ± {tag['f_std_px']:.0f} px "
            f"from {tag['views_used']} images"
            if tag else "AprilTag: no reliable measurement"
        )
        print(
            f"{key}\n"
            f"  prior: {cam['prior']['f_px']:.0f} ± "
            f"{cam['prior']['f_std_px']:.0f} px ({cam['prior']['source']})\n"
            f"  {tag_text}\n"
            f"  used: {cam['focal_px']:.0f} ± {cam['focal_std_px']:.0f} px "
            f"({cam['focal_rel_std']:.1%}, {cam['focal_source']})"
        )
        for warning in cam["warnings"]:
            print(f"  WARNING: {warning}")

    coverage = report["coverage"]
    print("\n--- VIEW COVERAGE ---")
    if coverage.get("azimuth_span_deg") is not None:
        print(f"Images: {coverage['n_images']}  around the mechanism: "
              f"{coverage['azimuth_span_deg']:.0f}°  elevation: "
              f"{coverage['elevation_min_deg']:.0f}–"
              f"{coverage['elevation_max_deg']:.0f}°  max angle between "
              f"views: {coverage['max_view_angle_deg']:.0f}°")
    for warning in coverage["warnings"]:
        print(f"  WARNING: {warning}")

    print("\n--- SUMMARY ---")
    print(f"OK: {counts['OK']}  LOW_CONFIDENCE: {counts['LOW_CONFIDENCE']}  "
          f"UNKNOWN: {counts['UNKNOWN']}")
    print(f"Report: {CAMERA_POSES_PATH}")
    print(f"Debug images: {POSE_DEBUG_DIR}")


if __name__ == "__main__":
    main()
