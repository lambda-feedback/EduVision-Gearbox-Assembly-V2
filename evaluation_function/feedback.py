"""
Student feedback (Markdown) from GearBox analysis results.

Elements (tag, markdown), in display order:
    verdict  result of each stage: status, reasons, part counts, relationships
    photos   photos: rejected (why), quality warnings, viewpoint coverage,
             download problems
    parts    detected parts and drivetrain (what rotates)
    images   images (3D render, top view, photos) - according to params["images"]
    error    analysis interrupted (optional log: params["debug"])

lf_toolkit joins elements using "<br>" and displays only the content (the tag
is not displayed). Each element starts and ends with a blank line so that
Markdown (headings, lists) works after joining.
"""

import re

PASSED, FAILED, UNKNOWN = "PASSED", "FAILED", "CANNOT ASSESS"
OK, ERROR = "OK", "ERROR"

VERDICT_LABEL = {
    PASSED: "[OK] PASSED",
    FAILED: "[ERROR] FAILED",
    UNKNOWN: "[WARNING] CANNOT ASSESS",
}
RESULT_MARK = {OK: "[OK]", ERROR: "[ERROR]"}

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

DEFAULT_IMAGES_3D = ("3d", "3d_top", "top_view")
DEFAULT_IMAGES_NO_TAG = ("detections",)
# 3D render image types: {name in params: (render.VIEWS view, caption)}
RENDER_VIEWS = {
    "3d": ("oblique", "3D model of the recognised assembly"),
    "3d_top": ("top", "3D model - top view"),
}

_SPECIAL = re.compile(r"([\\`*_\[\]<>|])")


def _e(value):
    """Text without Markdown formatting (e.g. "_" in names, "<" in descriptions)."""
    return _SPECIAL.sub(r"\\\1", str(value))


def _code(value):
    return f"`{str(value).replace('`', '')}`"


def _block(*lines):
    return "\n\n" + "\n".join(lines).strip("\n") + "\n\n"


def _ul(items):
    return "\n".join(f"- {i}" for i in items if i)


def _mark(result):
    return RESULT_MARK.get(result, "[WARNING]")


# ------------------------------------------------------------------ stages

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
    suffix = f" *({'; '.join(extra)})*" if extra else ""
    return f"{_mark(rel.get('result', UNKNOWN))} {text}{suffix}"


def _count_line(count):
    note = " *(uncertain)*" if count.get("uncertain") else ""
    return (f"{_mark(count['result'])} {_code(count['class'])}: found "
            f"**{count['found']}**, expected **{count['expected']}**{note}")


def stage_markdown(stage):
    label = VERDICT_LABEL.get(stage["verdict"], f"[WARNING] {stage['verdict']}")
    lines = [f"### {label} - {_e(stage['name'])}"]
    if stage.get("description"):
        lines += ["", f"*{_e(stage['description'])}*"]
    if stage["verdict"] != PASSED and stage.get("reasons"):
        lines += ["", "**What is wrong:**", "",
                  _ul(_e(r) for r in stage["reasons"])]
    if stage.get("counts"):
        lines += ["", "**Parts:**", "",
                  _ul(_count_line(c) for c in stage["counts"])]
    if stage.get("relations"):
        lines += ["", "**Connections:**", "",
                  _ul(_relation_line(r) for r in stage["relations"])]
    if stage.get("uncertain"):
        lines += ["", f"*Uncertain: {_e(', '.join(stage['uncertain']))}*"]
    return _block(*lines)


# ---------------------------------------------------------------- photos

def photos_markdown(run, download_problems):
    tag_report = run.get("apriltag") or []
    accepted = [e for e in tag_report if e.get("status") == "ACCEPTED"]
    rejected = [e for e in tag_report if e.get("status") != "ACCEPTED"]
    lines = ["### Photos", "",
             f"{len(tag_report)} analysed, **{len(accepted)} accepted**, "
             f"{len(rejected)} rejected."]

    if run["mode"] == "notag":
        lines += ["", "**The AprilTag was not found on any photo.** Only the "
                      "list of parts could be checked; connections between "
                      "parts cannot be assessed without the tag. Make sure "
                      "the whole tag is visible and sharp."]
    elif len(accepted) == 1:
        lines += ["", "Only one photo shows the AprilTag - at least two "
                      "(better 5 or more, from different sides) are needed "
                      "for the 3D analysis."]

    if rejected:
        title = ("Photos without a usable AprilTag" if run["mode"] == "notag"
                 else "Rejected photos (not used for 3D)")
        lines += ["", f"**{title}:**", "", _ul(
            f"{_code(e['image'])}: {_e('; '.join(_reasons(e)))}"
            for e in rejected)]

    warned = [e for e in accepted if e.get("warnings")]
    if warned:
        lines += ["", "**Photo quality warnings:**", "", _ul(
            f"{_code(e['image'])}: {_e('; '.join(e['warnings']))}"
            for e in warned)]

    camera = run.get("camera") or {}
    unknown = [c for c in camera.get("images", [])
               if c.get("status") == "UNKNOWN"]
    if unknown:
        lines += ["", "**Camera position could not be determined:**", "",
                  _ul(f"{_code(c['image'])}: "
                      f"{_e('; '.join(c.get('reasons') or []))}"
                      for c in unknown)]

    coverage = camera.get("coverage") or {}
    if coverage.get("warnings"):
        lines += ["", "**Viewpoints:**", "",
                  _ul(_e(w) for w in coverage["warnings"])]

    if download_problems:
        lines += ["", "**Could not use:**", "",
                  _ul(_e(p) for p in download_problems)]
    return _block(*lines)


def _reasons(entry):
    reasons = entry.get("reason") or entry.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    return [r for r in reasons if r and r != "OK"] or ["rejected"]


# ------------------------------------------------------------------ parts

def parts_markdown(run):
    check = run.get("check") or {}
    inventory = check.get("inventory") or {}
    parts = inventory.get("parts") or {}
    if not parts:
        return _block("### Parts found", "", "None.")
    uncertain = set(inventory.get("uncertain_classes") or [])
    source = ("3D reconstruction" if inventory.get("source") == "3D"
              else "detections on the photos, no 3D")
    lines = ["### Parts found", "", f"*Source: {source}*", "", _ul(
        f"{_code(name)}: **{count}**"
        + (" *(uncertain)*" if name in uncertain else "")
        for name, count in sorted(parts.items()))]

    train = check.get("drive_train") or []
    rotating = sorted((d for d in train if d.get("driven")
                       and d["class_name"] not in ("engine", "driving_gear")),
                      key=lambda d: d.get("reduction") or 0)
    if train:
        if rotating:
            lines += ["", "**Driven by the motor:**", "", _ul(
                f"{_e(d['part'])} {_code(d['class_name'])}: ratio "
                f"1:{d['reduction']}, {_e(d.get('direction', ''))}"
                for d in rotating)]
        elif any(d["class_name"] == "engine" for d in train):
            lines += ["", "No gear is driven by the motor."]
    return _block(*lines)


# ------------------------------------------------------------------ images

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
    """[(source: path or image, caption)] according to params["images"]."""
    from .render import render_assembly

    output = run["output_dir"]
    max_photos = int((params or {}).get("max_photos", 3))
    sources = []
    for kind in wanted_images(params, run["mode"]):
        if kind in RENDER_VIEWS:
            view, caption = RENDER_VIEWS[kind]
            try:
                image = render_assembly(run.get("fused"), run.get("check"),
                                        run.get("detections"), view=view)
            except Exception:
                image = None
            if image is not None:
                sources.append((image, caption))
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


# ------------------------------------------------------------------- full report

def _log_block(run):
    log = run.get("log", "")[-6000:].replace("```", "'''")
    return f"```\n{log}\n```"


def build(run, download_problems, attacher, params):
    """-> (is_correct, [(tag, markdown)])"""
    items = []
    check = run.get("check") or {}
    stages = check.get("assemblies") or []

    if stages:
        for stage in stages:
            items.append(("verdict", stage_markdown(stage)))
    else:
        items.append(("verdict", _block(
            "### [WARNING] CANNOT ASSESS", "", "The analysis did not finish.")))

    items.append(("photos", photos_markdown(run, download_problems)))
    if check:
        items.append(("parts", parts_markdown(run)))

    figures = [attacher.markdown(src, caption)
               for src, caption in image_sources(run, params)]
    figures = [f for f in figures if f]
    if figures:
        items.append(("images", _block("### Images", "", *figures)))
    elif attacher.problem:
        items.append(("images", _block(
            f"*Images could not be attached: {_e(attacher.problem)}*")))

    debug = (params or {}).get("debug")
    if run.get("failed_step"):
        name = STEP_NAMES.get(run["failed_step"], run["failed_step"])
        lines = [f"*The analysis stopped at: {_e(name)}. The result above "
                 "may be incomplete.*"]
        if debug:
            lines += ["", _log_block(run)]
        items.append(("error", _block(*lines)))
    elif debug:
        items.append(("error", _block(_log_block(run))))

    is_correct = bool(stages) and all(s["verdict"] == PASSED for s in stages)
    return is_correct, items
