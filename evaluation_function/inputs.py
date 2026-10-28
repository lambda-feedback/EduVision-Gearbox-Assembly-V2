"""
Odczyt zgłoszenia: zdjęcia studenta (response) i etap do sprawdzenia
(answer / params).

Zdjęcia (response) — przyjmowane formaty, także zagnieżdżone:
    "https://.../IMG_1.jpg"                      adres URL
    "data:image/jpeg;base64,...."                data URI
    {"url": "...", "name": "IMG_1.jpg"}          (także src / href / uri)
    {"data": "<base64>", "name": "IMG_1.jpg"}    (także base64 / content)
    ["...", {...}, ...]   lub   {"images": [...]} / {"files": [...]}
    tekst JSON z dowolną z powyższych postaci
    ścieżka do pliku — tylko gdy EDUVISION_ALLOW_LOCAL_FILES=1 (testy)

Etap (answer, a gdy pusty — params["stage"]):
    "etap_pelny" / "etap_pelny.yaml"   etap dołączony do funkcji
                                       (evaluation_function/gearbox/assemblies)
    tekst YAML/JSON z definicją etapu  (format jak w assemblies/*.yaml)
    słownik z definicją etapu          {"name": ..., "parts": {...}, ...}
    lista powyższych                   wszystkie etapy muszą być zaliczone
"""

import base64
import binascii
import json
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ASSEMBLIES_DIR = Path(__file__).resolve().parent / "gearbox" / "assemblies"

MAX_IMAGES = int(os.environ.get("EDUVISION_MAX_IMAGES", "30"))
MAX_IMAGE_BYTES = int(os.environ.get("EDUVISION_MAX_IMAGE_MB", "40")) << 20
DOWNLOAD_TIMEOUT_S = 30

URL_KEYS = ("url", "src", "href", "uri", "link")
DATA_KEYS = ("data", "base64", "content", "dataUrl", "data_url")
NAME_KEYS = ("name", "filename", "fileName", "file_name", "title")
PATH_KEYS = ("path", "file", "filePath", "file_path")

# Rozszerzenia obsługiwane przez analizę (config.IMAGE_EXTENSIONS).
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SubmissionError(ValueError):
    """Zgłoszenie bez poprawnych zdjęć (lf_toolkit: InvalidSubmission)."""


# ---------------------------------------------------------------- zdjęcia

def _allow_local():
    return os.environ.get("EDUVISION_ALLOW_LOCAL_FILES", "") == "1"


def _looks_like_url(text):
    return re.match(r"^(https?|data):", text.strip(), re.I) is not None


def collect_sources(value, name=None, out=None):
    """Lista (nazwa lub None, źródło) — źródło: URL, data URI, base64,
    ścieżka."""
    out = [] if out is None else out
    if value is None:
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            collect_sources(item, None, out)
    elif isinstance(value, dict):
        item_name = next((value[k] for k in NAME_KEYS
                          if isinstance(value.get(k), str)), None)
        for keys, kind in ((URL_KEYS, "url"), (DATA_KEYS, "data"),
                           (PATH_KEYS, "path")):
            source = next((value[k] for k in keys
                           if isinstance(value.get(k), str)), None)
            if source:
                if kind == "data" and not source.startswith("data:"):
                    source = "base64:" + source
                out.append((item_name, source))
                return out
        for item in value.values():
            if isinstance(item, (list, tuple, dict, str)):
                collect_sources(item, None, out)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return out
        if text[0] in "[{":
            try:
                return collect_sources(json.loads(text), name, out)
            except ValueError:
                pass
        # Kilka adresów w jednym tekście (po przecinku / w liniach);
        # data URI zawiera przecinek — zawsze jeden obraz.
        parts = ([text] if text.lower().startswith("data:")
                 else [p for p in re.split(r"[\s,;]+", text) if p])
        if len(parts) > 1 and all(_looks_like_url(p) for p in parts):
            for part in parts:
                out.append((None, part))
        elif _looks_like_url(text):
            out.append((name, text))
        elif _allow_local() and Path(text).is_file():
            out.append((name, text))
    return out


def _sniff_extension(data):
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"BM":
        return ".bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[4:8] == b"ftyp":
        return ".heic"
    return None


def _fetch(source):
    if source.startswith("data:"):
        try:
            header, payload = source.split(",", 1)
        except ValueError:
            raise SubmissionError("invalid data URI")
        if ";base64" in header:
            return base64.b64decode(payload), None
        return unquote(payload).encode("latin-1"), None
    if source.startswith("base64:"):
        try:
            return base64.b64decode(source[7:], validate=False), None
        except binascii.Error:
            raise SubmissionError("invalid base64 image data")
    if re.match(r"^https?:", source, re.I):
        import requests
        with requests.get(source, timeout=DOWNLOAD_TIMEOUT_S,
                          stream=True) as reply:
            reply.raise_for_status()
            data = b""
            for chunk in reply.iter_content(1 << 20):
                data += chunk
                if len(data) > MAX_IMAGE_BYTES:
                    raise SubmissionError("image is too large")
        return data, Path(unquote(urlparse(source).path)).name
    path = Path(source)
    return path.read_bytes(), path.name


def _to_supported(data):
    """Format spoza listy (np. HEIC, gdy jest pillow-heif) -> JPEG z EXIF."""
    from PIL import Image
    try:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        import io
        image = Image.open(io.BytesIO(data))
        exif = image.info.get("exif")
        out = io.BytesIO()
        image.convert("RGB").save(out, "JPEG", quality=95,
                                  **({"exif": exif} if exif else {}))
        return out.getvalue()
    except Exception:
        return None


def _safe_name(name, index, extension):
    stem = Path(name).stem if name else ""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")[:60]
    return f"{stem or f'image_{index + 1:02d}'}{extension}"


def load_images(response):
    """
    Pobiera zdjęcia. Zwraca (images, problems):
        images   [(unikalna nazwa pliku, bajty)]
        problems [tekst] — źródła, których nie udało się wczytać
    """
    sources = collect_sources(response)
    if not sources:
        raise SubmissionError(
            "No photos found in the submission. Upload 4-10 photos of the "
            "gearbox with the AprilTag visible.")
    images, problems, used = [], [], set()
    for index, (name, source) in enumerate(sources[:MAX_IMAGES]):
        label = name or (source if len(source) < 80 else source[:77] + "...")
        if source.startswith(("data:", "base64:")):
            label = name or f"image {index + 1}"
        try:
            data, url_name = _fetch(source)
        except SubmissionError as exc:
            problems.append(f"{label}: {exc}")
            continue
        except Exception as exc:
            problems.append(f"{label}: cannot download ({exc})")
            continue
        if len(data) > MAX_IMAGE_BYTES:
            problems.append(f"{label}: image is too large")
            continue
        extension = _sniff_extension(data)
        if extension not in SUPPORTED:
            converted = _to_supported(data)
            if converted is None:
                problems.append(f"{label}: not a supported image "
                                f"(JPEG, PNG, BMP, WEBP)")
                continue
            data, extension = converted, ".jpg"
        file_name = _safe_name(name or url_name, index, extension)
        while file_name.lower() in used:
            file_name = _safe_name(f"{Path(file_name).stem}_{index + 1}",
                                   index, extension)
        used.add(file_name.lower())
        images.append((file_name, data))
    if len(sources) > MAX_IMAGES:
        problems.append(f"only the first {MAX_IMAGES} of {len(sources)} "
                        f"photos were analysed")
    if not images:
        raise SubmissionError("None of the submitted photos could be read: "
                              + "; ".join(problems))
    return images, problems


# ------------------------------------------------------------------ etapy

def bundled_stages():
    return sorted(p.stem for p in ASSEMBLIES_DIR.glob("*.y*ml"))


def _stage_from_value(value):
    """-> (nazwa pliku, tekst YAML) lub ValueError."""
    if isinstance(value, dict):
        if "parts" not in value:
            raise ValueError("stage definition has no 'parts'")
        name = re.sub(r"[^A-Za-z0-9_-]+", "_",
                      str(value.get("name", "stage")))[:40]
        return f"{name or 'stage'}.yaml", yaml.safe_dump(
            value, allow_unicode=True, sort_keys=False)
    if isinstance(value, str):
        text = value.strip()
        stem = Path(text).stem if re.fullmatch(r"[\w.-]+", text) else None
        if stem:
            for path in ASSEMBLIES_DIR.glob("*.y*ml"):
                if path.stem == stem:
                    return path.name, path.read_text(encoding="utf-8")
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"stage definition is not valid YAML: {exc}")
        if isinstance(parsed, dict):
            return _stage_from_value(parsed)
        if isinstance(parsed, list):
            return None
        raise ValueError(f"unknown stage '{text}' (bundled stages: "
                         f"{', '.join(bundled_stages())})")
    raise ValueError(f"unsupported stage definition: {type(value).__name__}")


def stage_definitions(answer, params):
    """Lista (nazwa pliku, tekst YAML) etapów do sprawdzenia."""
    value = answer
    if value in (None, "", [], {}):
        value = (params or {}).get("stage")
    if value in (None, "", [], {}):
        raise ValueError("no stage to check: set the answer (bundled stage "
                         "name or YAML definition) — bundled stages: "
                         + ", ".join(bundled_stages()))
    if isinstance(value, str):
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, list):
            value = parsed
    values = value if isinstance(value, list) else [value]
    stages = []
    for item in values:
        stage = _stage_from_value(item)
        if stage is None:
            raise ValueError("a stage list cannot be nested")
        stages.append(stage)
    return stages
