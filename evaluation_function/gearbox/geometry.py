
"""
Wspólne konwencje geometryczne projektu.

JEDNOSTKI
    Wewnętrznie: metry i radiany. W raportach dodatkowo milimetry/stopnie.

UKŁAD KAMERY (konwencja OpenCV)
    X w prawo, Y w dół, Z do przodu (oś optyczna). Piksele (u, v)
    odnoszą się do obrazu PO uwzględnieniu orientacji EXIF, czyli
    tak, jak wczytuje go cv2.imread.

UKŁAD ŚWIATA (W) = układ znacznika AprilTag
    Początek: środek czarnego kwadratu znacznika.
    Z: prostopadle do znacznika, w stronę kamery (od stołu w górę).
    X: wzdłuż krawędzi od narożnika c0 do c1.
    Y: wzdłuż krawędzi od narożnika c0 do c3.
    Narożniki c0..c3 w kolejności zwracanej przez pupil_apriltags;
    o ich fizycznym położeniu decyduje kod znacznika, więc układ jest
    taki sam na każdym zdjęciu.

TRANSFORMACJE
    T_a_b (macierz 4x4) przekształca punkt z układu b do układu a:
        p_a = T_a_b @ p_b
    Przykłady:
        T_camera_world — punkt świata -> współrzędne kamery (wynik PnP),
        T_world_camera — poza kamery w świecie (jej odwrotność),
        T_world_part   — poza części w świecie (kolejne etapy).
"""

import numpy as np
import cv2


def make_transform(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float).ravel()
    return transform


def invert_transform(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


def transform_from_rvec_tvec(rvec, tvec):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    return make_transform(rotation, tvec)


def rvec_tvec_from_transform(transform):
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return rvec.ravel(), transform[:3, 3].copy()


def tag_corners_world(tag_size_m):
    """Narożniki c0..c3 w układzie świata (kolejność jak w detektorze)."""
    h = tag_size_m / 2.0
    return np.array([
        [-h, -h, 0.0],
        [h, -h, 0.0],
        [h, h, 0.0],
        [-h, h, 0.0],
    ])


def tag_corners_ippe(tag_size_m):
    """
    Narożniki w kolejności wymaganej przez cv2.SOLVEPNP_IPPE_SQUARE.
    Ten pomocniczy układ (F) ma oś Z skierowaną w głąb znacznika;
    świat W powstaje z F przez obrót o 180° wokół osi X.
    """
    h = tag_size_m / 2.0
    return np.array([
        [-h, h, 0.0],
        [h, h, 0.0],
        [h, -h, 0.0],
        [-h, -h, 0.0],
    ])


# T_F_W: obrót o 180° wokół X (macierz jest swoją odwrotnością).
T_IPPE_WORLD = make_transform(np.diag([1.0, -1.0, -1.0]), np.zeros(3))


def camera_matrix(focal_px, width, height):
    """
    Macierz K przy założeniach: fx = fy, punkt główny w środku obrazu.
    Środek piksela (0,0) ma współrzędne (0,0), stąd (W-1)/2.
    """
    return np.array([
        [focal_px, 0.0, (width - 1) / 2.0],
        [0.0, focal_px, (height - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_to_euler_zyx_deg(rotation):
    """Kąty (yaw Z, pitch Y, roll X) w stopniach — tylko do czytania raportu."""
    sy = np.hypot(rotation[0, 0], rotation[1, 0])
    if sy > 1e-9:
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pitch = np.arctan2(-rotation[2, 0], sy)
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    else:
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        roll = 0.0
    return np.degrees([yaw, pitch, roll])
