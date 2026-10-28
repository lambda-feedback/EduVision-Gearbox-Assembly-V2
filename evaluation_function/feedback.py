"""
Feedback dla studenta (HTML) z wyników analizy GearBox.

Elementy (tag, html) w kolejności wyświetlania:
    verdict  wynik każdego etapu: ocena, powody, liczby części, relacje
    photos   zdjęcia: odrzucone (dlaczego), ostrzeżenia jakości, pokrycie
             kątów, problemy z pobraniem
    parts    znalezione części i napęd (co się obraca)
    images   obrazy (render 3D, widok z góry, zdjęcia) — wg params["images"]
    error    analiza przerwana (opcjonalnie log: params["debug"])
"""

from html import escape

PASSED, FAILED, UNKNOWN = "PASSED", "FAILED", "CANNOT ASSESS"
OK, ERROR = "OK", "ERROR"

VERDICT_STYLE = {
    PASSED: ("#1a7f37", "PASSED"),
    FAILED: ("#cf222e", "FAILED"),
    UNKNOWN: ("#9a6700", "CANNOT ASSESS"),
}
RESULT_STYLE = {OK: "#1a7f37", ERROR: "#cf222e"}

STEP_NAMES = {
    "tag": "image filter and AprilTag detection",
    "exif": "image metadata",
    "camera": "camera calibration",
    "yolo": "part detection",
    "yolo_all": "part detection",
    "cad": "CAD fitting",
    "fuse": "3D reconstruction",
    "check": "assembly check",
    "check_all": "stage check",
}

DEFAULT_IMAGES_3D = ("3d", "top_view")
DEFAULT_IMAGES_NO_TAG = ("detections",)


def _e(value):
    return escape(str(value))


def _ul(items):
    items = [i for i in items if i]
    if not items:
        return ""
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _badge(result):
    color = RESULT_STYLE.get(result, "#9a6700")
    return f'<b style="color:{color}">{_e(result)}</b>'


# ------------------------------------------------------------------ etapy

def _relation_line(rel):
    text = _e(rel.get("description") or f"{rel['type']}: "
              f"{', '.join(rel.get('roles', []))}")
    detail = rel.get("detail")
    measurement = rel.get("measurement") or {}
    extra = []
    if detail and rel.get("result") != OK:
        extra.append(_e(detail))
    if rel["type"] == "mesh" and "center_distance_mm" in measurement:
        extra.append(f"axis distance {measurement['center_distance_mm']:.1f}"
                     f" mm, required {measurement['required_mm']:.1f} mm")
    if rel.get("uncertain_parts"):
        extra.append("uncertain: " + _e(", ".join(rel["uncertain_parts"])))
    suffix = f" <small>({'; '.join(extra)})</small>" if extra else ""
    return f"{_badge(rel.get('result', UNKNOWN))} {text}{suffix}"


def _count_line(count):
    note = " <small>(uncertain)</small>" if count.get("uncertain") else ""
    return (f"{_badge(count['result'])} {_e(count['class'])}: found "
            f"{count['found']}, expected {count['expected']}{note}")


def stage_html(stage):
    color, label = VERDICT_STYLE.get(stage["verdict"],
                                     ("#9a6700", stage["verdict"]))
    html = [f'<p><b style="color:{color}">{label}</b> &mdash; '
            f'<b>{_e(stage["name"])}</b></p>']
    if stage.get("description"):
        html.append(f"<p><small>{_e(stage['description'])}</small></p>")
    if stage["verdict"] != PASSED and stage.get("reasons"):
        html.append("<p>What is wrong:</p>"
                    + _ul(_e(r) for r in stage["reasons"]))
    if stage.get("counts"):
        html.append("<p>Parts:</p>" + _ul(_count_line(c)
                                          for c in stage["counts"]))
    if stage.get("relations"):
        html.append("<p>Connections:</p>" + _ul(_relation_line(r)
                                                for r in stage["relations"]))
    if stage.get("uncertain"):
        html.append("<p><small>Uncertain: "
                    f"{_e(', '.join(stage['uncertain']))}</small></p>")
    return "".join(html)


# ---------------------------------------------------------------- zdjęcia

def photos_html(run, download_problems):
    tag_report = run.get("apriltag") or []
    accepted = [e for e in tag_report if e.get("status") == "ACCEPTED"]
    rejected = [e for e in tag_report if e.get("status") != "ACCEPTED"]
    html = [f"<p><b>Photos:</b> {len(tag_report)} analysed, "
            f"{len(accepted)} accepted, {len(rejected)} rejected.</p>"]

    if run["mode"] == "notag":
        html.append("<p><b>The AprilTag was not found on any photo.</b> "
                    "Only the list of parts could be checked; connections "
                    "between parts cannot be assessed without the tag. "
                    "Make sure the whole tag is visible and sharp.</p>")
    elif len(accepted) == 1:
        html.append("<p>Only one photo shows the AprilTag &mdash; at least "
                    "two (better 5 or more, from different sides) are "
                    "needed for the 3D analysis.</p>")

    if rejected:
        title = ("Photos without a usable AprilTag" if run["mode"] == "notag"
                 else "Rejected photos (not used for 3D)")
        html.append(f"<p>{title}:</p>" + _ul(
            f"{_e(e['image'])}: {_e('; '.join(_reasons(e)))}"
            for e in rejected))

    warned = [e for e in accepted if e.get("warnings")]
    if warned:
        html.append("<p>Photo quality warnings:</p>" + _ul(
            f"{_e(e['image'])}: {_e('; '.join(e['warnings']))}"
            for e in warned))

    camera = run.get("camera") or {}
    unknown = [c for c in camera.get("images", [])
               if c.get("status") == "UNKNOWN"]
    if unknown:
        html.append("<p>Camera position could not be determined:</p>" + _ul(
            f"{_e(c['image'])}: {_e('; '.join(c.get('reasons') or []))}"
            for c in unknown))

    coverage = camera.get("coverage") or {}
    if coverage.get("warnings"):
        html.append("<p>Viewpoints:</p>" + _ul(
            _e(w) for w in coverage["warnings"]))

    if download_problems:
        html.append("<p>Could not use:</p>" + _ul(
            _e(p) for p in download_problems))
    return "".join(html)


def _reasons(entry):
    reasons = entry.get("reason") or entry.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    return [r for r in reasons if r and r != "OK"] or ["rejected"]


# ------------------------------------------------------------------ części

def parts_html(run):
    check = run.get("check") or {}
    inventory = check.get("inventory") or {}
    parts = inventory.get("parts") or {}
    if not parts:
        return "<p><b>Parts found:</b> none.</p>"
    uncertain = set(inventory.get("uncertain_classes") or [])
    source = ("3D reconstruction" if inventory.get("source") == "3D"
              else "detections on the photos, no 3D")
    items = [f"{_e(name)}: {count}"
             + (" <small>(uncertain)</small>" if name in uncertain else "")
             for name, count in sorted(parts.items())]
    html = [f"<p><b>Parts found</b> <small>({source})</small>:</p>",
            _ul(items)]

    train = check.get("drive_train") or []
    rotating = sorted((d for d in train if d.get("driven")
                       and d["class_name"] not in ("engine", "driving_gear")),
                      key=lambda d: d.get("reduction") or 0)
    if train:
        if rotating:
            html.append("<p>Driven by the motor:</p>" + _ul(
                f"{_e(d['part'])} {_e(d['class_name'])}: ratio "
                f"1:{d['reduction']}, {_e(d.get('direction', ''))}"
                for d in rotating))
        elif any(d["class_name"] == "engine" for d in train):
            html.append("<p>No gear is driven by the motor.</p>")
    return "".join(html)


# ------------------------------------------------------------------ obrazy

def wanted_images(params, mode):
    value = (params or {}).get("images")
    if value is False:
        return []
    if value in (None, True):
        return list(DEFAULT_IMAGES_3D if mode == "3d"
                    else DEFAULT_IMAGES_NO_TAG)
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    return [str(v).lower() for v in value]


def image_sources(run, params):
    """[(źródło: ścieżka lub obraz, podpis)] wg params["images"]."""
    from .render import render_assembly

    output = run["output_dir"]
    max_photos = int((params or {}).get("max_photos", 3))
    sources = []
    for kind in wanted_images(params, run["mode"]):
        if kind == "3d":
            try:
                image = render_assembly(run.get("fused"), run.get("check"),
                                        run.get("detections"))
            except Exception:
                image = None
            if image is not None:
                sources.append((image, "3D model of the recognised "
                                       "assembly"))
        elif kind == "top_view":
            sources.append((output / "fusion_debug" / "top_view.png",
                            "Top view (parts and axes)"))
        elif kind == "photos":
            for path in sorted((output / "fusion_debug").glob(
                    "*_fused.jpg"))[:max_photos]:
                sources.append((path, f"3D model overlaid on "
                                      f"{path.stem[:-6]}"))
        elif kind == "detections":
            for path in sorted((output / "detection_debug").glob(
                    "*_det.jpg"))[:max_photos]:
                sources.append((path, f"Parts detected on {path.stem[:-4]}"))
    return sources


# ------------------------------------------------------------------- całość

def build(run, download_problems, attacher, params):
    """-> (is_correct, [(tag, html)])"""
    items = []
    check = run.get("check") or {}
    stages = check.get("assemblies") or []

    if stages:
        for stage in stages:
            items.append(("verdict", stage_html(stage)))
    else:
        items.append(("verdict", '<p><b style="color:#9a6700">CANNOT '
                                 'ASSESS</b> &mdash; the analysis did not '
                                 'finish.</p>'))

    items.append(("photos", photos_html(run, download_problems)))
    if check:
        items.append(("parts", parts_html(run)))

    figures = [attacher.html(src, caption)
               for src, caption in image_sources(run, params)]
    figures = [f for f in figures if f]
    if figures:
        items.append(("images", "".join(figures)))
    elif attacher.problem:
        items.append(("images", "<p><small>Images could not be attached: "
                                f"{_e(attacher.problem)}</small></p>"))

    if run.get("failed_step"):
        name = STEP_NAMES.get(run["failed_step"], run["failed_step"])
        html = (f"<p><small>The analysis stopped at: {_e(name)}. The result "
                "above may be incomplete.</small></p>")
        if (params or {}).get("debug"):
            html += f"<pre>{_e(run.get('log', '')[-6000:])}</pre>"
        items.append(("error", html))
    elif (params or {}).get("debug"):
        items.append(("error", f"<pre>{_e(run.get('log', '')[-6000:])}"
                               "</pre>"))

    is_correct = bool(stages) and all(s["verdict"] == PASSED for s in stages)
    return is_correct, items
