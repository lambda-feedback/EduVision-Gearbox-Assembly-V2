
"""
Ocena jakości zdjęcia na podstawie znacznika AprilTag.

Znacznik ma znany rozmiar (APRILTAG_SIZE_M) i wydrukowaną biel i czerń,
więc służy jako wzorzec niezależny od telefonu i od treści zdjęcia:

    scale_px_per_mm   — ile pikseli na 1 mm mechanizmu w obrazie, który
                        widzi model YOLO (zdjęcie zmniejszone do IMAGE_SIZE);
                        miara odległości niezależna od rozdzielczości,
    edge_blur_mm      — szerokość rozmycia krawędzi znacznika (10–90%)
                        w milimetrach na obiekcie (nieostrość / poruszenie),
    tag_white / black — jasność białego marginesu i czarnej ramki
                        znacznika (0–255), contrast = biel - czerń.

Funkcja classify() porównuje miary z progami z config.py:
lekkie przekroczenie = WARNING, duże = REJECTED.
"""

import math

import cv2
import numpy as np

from config import (
    APRILTAG_SIZE_M,
    IMAGE_SIZE,
    QUALITY_MIN_SCALE_WARN,
    QUALITY_MIN_SCALE_REJECT,
    QUALITY_MAX_BLUR_MM_WARN,
    QUALITY_MAX_BLUR_MM_REJECT,
    QUALITY_MIN_WHITE_WARN,
    QUALITY_MIN_WHITE_REJECT,
    QUALITY_MIN_CONTRAST_WARN,
    QUALITY_MIN_CONTRAST_REJECT,
    QUALITY_MAX_WHITE_CLIPPED,
)

# Profil w poprzek krawędzi: od białego marginesu (na zewnątrz) do czarnej
# ramki (do środka). Margines i ramka mają po 1/8 boku czarnego kwadratu,
# więc +-0.1 boku mieści się w jednolitych obszarach.
PROFILE_HALF_LENGTH = 0.1
PROFILE_SAMPLES = 41
EDGE_POSITIONS = np.linspace(0.2, 0.8, 15)  # wzdłuż krawędzi, bez narożników


def _sample(gray, points):
    """Interpolowana jasność w punktach (x, y)."""
    map_x = points[..., 0].astype(np.float32)
    map_y = points[..., 1].astype(np.float32)
    return cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _rise_width(profile, step):
    """Odległość (px) między 10% a 90% przejścia biel -> czerń."""
    white = np.median(profile[:8])
    black = np.median(profile[-8:])
    span = white - black
    if span <= 1:
        return None, white, black
    norm = (white - profile) / span  # 0 = biel, 1 = czerń
    norm = np.maximum.accumulate(np.clip(norm, 0, 1))  # monotonicznie
    t = np.arange(len(norm)) * step
    t10 = np.interp(0.1, norm, t)
    t90 = np.interp(0.9, norm, t)
    return t90 - t10, white, black


def measure(gray, corners):
    """Miary jakości z obrazu w skali szarości i narożników znacznika (px)."""
    corners = np.asarray(corners, dtype=float)
    center = corners.mean(axis=0)
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    side_px = float(sides.mean())
    px_per_mm_full = side_px / (APRILTAG_SIZE_M * 1000)
    model_scale = IMAGE_SIZE / max(gray.shape[:2])

    widths, whites, blacks = [], [], []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-direction[1], direction[0]])
        # Normalna skierowana na zewnątrz kwadratu.
        if normal @ (((a + b) / 2) - center) < 0:
            normal = -normal
        half = PROFILE_HALF_LENGTH * side_px
        offsets = np.linspace(half, -half, PROFILE_SAMPLES)  # zewn. -> wewn.
        step = 2 * half / (PROFILE_SAMPLES - 1)
        bases = a + np.outer(EDGE_POSITIONS, b - a)
        points = bases[:, None, :] + offsets[None, :, None] * normal
        profiles = _sample(gray, points)
        for profile in profiles:
            width, white, black = _rise_width(profile.astype(float), step)
            whites.append(white)
            blacks.append(black)
            if width is not None:
                widths.append(width)

    white = float(np.median(whites))
    black = float(np.median(blacks))
    blur_px = float(np.median(widths)) if widths else float("nan")
    # Udział prześwietlonej bieli marginesu (odblask / prześwietlenie).
    clipped = float(np.mean(np.asarray(whites) >= 250))
    return {
        "tag_side_px": round(side_px, 1),
        "scale_px_per_mm": round(px_per_mm_full * model_scale, 2),
        "edge_blur_px": round(blur_px, 2),
        "edge_blur_mm": round(blur_px / px_per_mm_full, 3),
        "tag_white": round(white, 1),
        "tag_black": round(black, 1),
        "contrast": round(white - black, 1),
        "white_clipped_fraction": round(clipped, 2),
        "image_mean_brightness": round(float(gray.mean()), 1),
    }


def classify(metrics):
    """(status, ostrzeżenia, powody odrzucenia) z progów w config.py."""
    warnings, rejects = [], []

    def check(value, reject_below, warn_below, text_reject, text_warn):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        if value < reject_below:
            rejects.append(text_reject)
        elif value < warn_below:
            warnings.append(text_warn)

    scale = metrics["scale_px_per_mm"]
    check(scale, QUALITY_MIN_SCALE_REJECT, QUALITY_MIN_SCALE_WARN,
          f"Too far — the mechanism is too small in the image "
          f"({scale:.1f} px/mm, required >= {QUALITY_MIN_SCALE_REJECT}); "
          "move closer",
          f"Rather far ({scale:.1f} px/mm, recommended >= "
          f"{QUALITY_MIN_SCALE_WARN}); moving closer gives better detection")

    blur = metrics["edge_blur_mm"]
    if not math.isnan(blur):
        if blur > QUALITY_MAX_BLUR_MM_REJECT:
            rejects.append(f"Too blurry (edge blur {blur:.2f} mm, max "
                           f"{QUALITY_MAX_BLUR_MM_REJECT}); hold the phone "
                           "steady and let it focus")
        elif blur > QUALITY_MAX_BLUR_MM_WARN:
            warnings.append(f"Slightly blurry (edge blur {blur:.2f} mm, "
                            f"recommended <= {QUALITY_MAX_BLUR_MM_WARN})")

    white = metrics["tag_white"]
    check(white, QUALITY_MIN_WHITE_REJECT, QUALITY_MIN_WHITE_WARN,
          f"Too dark (tag white level {white:.0f}/255, required >= "
          f"{QUALITY_MIN_WHITE_REJECT}); add light",
          f"Rather dark (tag white level {white:.0f}/255, recommended >= "
          f"{QUALITY_MIN_WHITE_WARN})")

    contrast = metrics["contrast"]
    check(contrast, QUALITY_MIN_CONTRAST_REJECT, QUALITY_MIN_CONTRAST_WARN,
          f"Contrast too low ({contrast:.0f}, required >= "
          f"{QUALITY_MIN_CONTRAST_REJECT}); check lighting and glare",
          f"Low contrast ({contrast:.0f}, recommended >= "
          f"{QUALITY_MIN_CONTRAST_WARN})")

    if metrics["white_clipped_fraction"] > QUALITY_MAX_WHITE_CLIPPED:
        warnings.append("Overexposed / glare on the tag — avoid direct "
                        "light reflections")

    status = "REJECTED" if rejects else ("WARNING" if warnings else "OK")
    return status, warnings, rejects
