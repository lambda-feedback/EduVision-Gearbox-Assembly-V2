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
    """Return (source path or image, caption) pairs according to params["images"]."""
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


# ------------------------------------------------------------------- complete output

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
            "### [CANNOT ASSESS]", "", "The analysis did not finish.")))

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