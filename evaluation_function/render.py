"""
Obraz 3D złożenia bez OpenGL (serwer bez GPU i ekranu): te same bryły co
w aplikacji (view_assembly.scene_geometry), rzutowane perspektywicznie
i malowane od najdalszych trójkątów (algorytm malarza), z cieniowaniem
Lamberta. Siatki upraszczane do kilku tysięcy trójkątów — wystarcza do
podglądu, a liczy się w ułamku sekundy.
"""

import cv2
import numpy as np

MAX_TRIANGLES = 6000          # na jedną bryłę
SUPERSAMPLE = 3               # wygładzanie krawędzi
LIGHT = np.array([0.35, -0.45, 0.82])
BACKGROUND = (255, 255, 255)
# Płaskie bryły pod częściami (Z <= 0): malowane najpierw, w tej kolejności
# — duże trójkąty podstawy zasłoniłyby części przy sortowaniu po głębokości.
FLAT = ("base (Z=0)", "tag border", "AprilTag")

# Widoki: (kierunek od środka części do kamery w układzie tagu, Z w górę;
# kąt widzenia [deg]; odległość kamery / rozmiar złożenia). Ten sam kadr
# (odległość * tan(kąt / 2) stałe). "top": prosto z góry, prawie bez
# perspektywy (daleko, wąski kąt), +X w prawo, +Y w górę obrazu.
VIEWS = {  # widok "top" ma kadr ciaśniejszy (z góry nie widać wysokości)
    "oblique": ((0.7, -0.6, 0.7), 40.0, 1.6),
    "top": ((0.0, 0.0, 1.0), 15.0, 3.7),
}


def _look_at(eye, center, up):
    forward = center - eye
    forward /= np.linalg.norm(forward)
    if np.linalg.norm(np.cross(forward, up)) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])      # patrzenie wzdłuż Z
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])      # wiersze: osie kamery


def _triangles(mesh):
    if len(mesh.triangles) > MAX_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(MAX_TRIANGLES)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0:
        return None
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)[faces].mean(axis=1)
    else:
        colors = np.full((len(faces), 3), 0.6)
    return vertices[faces], colors


def render(geometries, labels=(), width=1200, height=800, fov_deg=40.0,
           direction=(0.7, -0.6, 0.7), distance=1.6):
    """
    geometries: lista {"name", "geometry"} z view_assembly.scene_geometry,
    labels: [(punkt 3D, tekst)]. Zwraca obraz BGR (uint8).
    """
    import open3d as o3d

    # Linie zazębień (LineSet) pominięte — bez bufora głębokości
    # przebijałyby koła.
    tris, cols, flat = [], [], []
    part_points = []
    for item in geometries:
        geometry = item["geometry"]
        if not isinstance(geometry, o3d.geometry.TriangleMesh):
            continue
        if item["name"].startswith("world frame"):
            continue
        result = _triangles(geometry)
        if result is None:
            continue
        if item["name"] in FLAT:
            flat.append(result)
            continue
        tris.append(result[0])
        cols.append(result[1])
        if item["name"].startswith("P"):
            part_points.append(np.asarray(geometry.vertices))

    image = np.full((height * SUPERSAMPLE, width * SUPERSAMPLE, 3),
                    BACKGROUND, np.uint8)
    if not tris:
        return cv2.resize(image, (width, height))
    tris = np.concatenate(tris)
    cols = np.concatenate(cols)

    # Kadr na same części (jak zrzut ekranu w aplikacji).
    points = (np.vstack(part_points) if part_points
              else tris.reshape(-1, 3))
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    extent = max(np.linalg.norm(hi - lo), 1.0)
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    eye = center + direction * extent * distance
    R = _look_at(eye, center, np.array([0.0, 0.0, 1.0]))
    W, H = width * SUPERSAMPLE, height * SUPERSAMPLE
    f = 0.5 * H / np.tan(np.radians(fov_deg) / 2)

    def project(p):
        cam = (np.asarray(p) - eye) @ R.T
        z = np.maximum(cam[..., 2], 1e-3)
        return np.stack([cam[..., 0] / z * f + W / 2,
                         cam[..., 1] / z * f + H / 2], axis=-1), cam[..., 2]

    light = LIGHT / np.linalg.norm(LIGHT)

    def paint(tris, cols, sort):
        # Cieniowanie: normalna trójkąta (dwustronnie), światło + otoczenie.
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        keep = length[:, 0] > 1e-12
        tris, cols = tris[keep], cols[keep]
        normals = normals[keep] / length[keep]
        to_eye = eye - tris.mean(axis=1)
        normals[(normals * to_eye).sum(axis=1) < 0] *= -1
        shade = 0.35 + 0.65 * np.clip(normals @ light, 0, 1)
        bgr = (np.clip(cols * shade[:, None], 0, 1)[:, ::-1] * 255
               ).astype(np.uint8)
        xy, depth = project(tris)
        order = (np.argsort(-depth.mean(axis=1)) if sort
                 else range(len(tris)))
        xy = np.round(xy * 16).astype(np.int32)      # 4 bity podpiksela
        for i in order:
            if (depth[i] <= 1e-3).any():
                continue
            cv2.fillConvexPoly(image, xy[i], bgr[i].tolist(),
                               lineType=cv2.LINE_8, shift=4)

    for flat_tris, flat_cols in flat:
        paint(flat_tris, flat_cols, sort=False)
    paint(tris, cols, sort=True)

    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Etykiety: przesuwane w górę, dopóki nachodzą na już postawione.
    placed = []
    for point, text in sorted(labels, key=lambda item: -item[0][2]):
        (p,), _ = project(np.array([point]))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = int(np.clip(p[0] / SUPERSAMPLE - tw / 2, 4, width - tw - 4))
        y = int(p[1] / SUPERSAMPLE)
        for _ in range(30):
            box = (x - 3, y - th - 4, x + tw + 3, y + 4)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3]
                       and b[1] < box[3] for b in placed):
                break
            y -= th + 6
        y = int(np.clip(y, th + 6, height - 6))
        box = (x - 3, y - th - 4, x + tw + 3, y + 4)
        placed.append(box)
        cv2.rectangle(image, box[:2], box[2:], (255, 255, 255), -1)
        cv2.rectangle(image, box[:2], box[2:], (200, 200, 200), 1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (30, 30, 30), 1, cv2.LINE_AA)
    return image


def render_assembly(fused, check, detections, view="oblique"):
    """Obraz złożenia z wyników analizy (None, gdy nie ma części 3D).
    view: klucz VIEWS."""
    if not fused or not fused.get("parts"):
        return None
    import view_assembly
    parts = fused["parts"]
    geometries = view_assembly.scene_geometry(parts, fused, check,
                                              detections)
    direction, fov_deg, distance = VIEWS[view]
    return render(geometries, view_assembly.labels(parts, check),
                  fov_deg=fov_deg, direction=direction, distance=distance)
