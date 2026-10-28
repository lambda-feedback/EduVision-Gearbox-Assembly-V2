"""
Obrazy w feedbacku (render 3D, widok z góry, zdjęcia z naniesionymi
częściami) jako obrazy Markdown: ![podpis](url).

Tryby (params["image_mode"] lub zmienna EDUVISION_IMAGE_MODE):
    "upload"  (domyślnie) — lf_toolkit.upload_image -> adres URL
              (S3: S3_BUCKET_URI + AWS_*, GCS: GCS_BUCKET); bez
              skonfigurowanego magazynu obrazy są pomijane z informacją,
    "inline"  — obraz jako data URI (duży feedback; do testów),
    "none"    — bez obrazów.
"""

import base64
import os
from pathlib import Path

import cv2
import numpy as np

UPLOAD, INLINE, NONE = "upload", "inline", "none"

MAX_SIDE_UPLOAD = 1600
MAX_SIDE_INLINE = 900


def image_mode(params):
    mode = str((params or {}).get(
        "image_mode", os.environ.get("EDUVISION_IMAGE_MODE", UPLOAD))).lower()
    return mode if mode in (UPLOAD, INLINE, NONE) else UPLOAD


def _load(source):
    if isinstance(source, np.ndarray):
        return source
    if source is None or not Path(source).is_file():
        return None
    return cv2.imread(str(source), cv2.IMREAD_COLOR)


def _shrink(image, max_side):
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    return image


class Attacher:
    """Zamienia obrazy na ![podpis](url); pamięta, czy magazyn działa."""

    def __init__(self, params):
        self.mode = image_mode(params)
        self.folder = str((params or {}).get(
            "image_folder", os.environ.get("EDUVISION_IMAGE_FOLDER",
                                           "eduvision")))
        self.problem = None          # pierwszy błąd zapisu (raz)

    def markdown(self, source, caption):
        """Obraz z podpisem (Markdown) albo None (brak obrazu / tryb none /
        błąd zapisu)."""
        if self.mode == NONE or self.problem:
            return None
        image = _load(source)
        if image is None:
            return None
        try:
            src = (self._inline(image) if self.mode == INLINE
                   else self._upload(image))
        except Exception as exc:
            self.problem = str(exc)
            return None
        return f"![{caption}]({src})\n\n*{caption}*\n"

    def _inline(self, image):
        ok, data = cv2.imencode(".jpg", _shrink(image, MAX_SIDE_INLINE),
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
        return "data:image/jpeg;base64," + base64.b64encode(
            data.tobytes()).decode("ascii")

    def _upload(self, image):
        from PIL import Image
        from lf_toolkit.evaluation.image_upload import upload_image
        image = _shrink(image, MAX_SIDE_UPLOAD)
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil.format = "JPEG"
        return upload_image(pil, self.folder)
