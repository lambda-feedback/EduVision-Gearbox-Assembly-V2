
"""
Etap 8 — kontrola montażu.

Wejście:
    output/fused_parts.json   fizyczne części w 3D (etap 7) — jeśli są,
    output/detections.json    detekcje (etap 5) — spis bez 3D,
    assemblies/*.yaml         definicje etapów (jeden plik = jeden etap).

Format etapu (YAML):
    name: "Etap 2 — silnik + dwa koła"
    parts:                      # DOKŁADNA liczba sztuk każdej klasy
      engine: 1
      driving_gear: 1
      gear_big: 2
    allow_extra: false          # opcjonalnie: dopuść dodatkowe części
    relations:                  # brak/puste = etap "tylko elementy"
      - mesh: [driving_gear, gear_big#1]
        sections: [main, big]   # big | pinion | main | any
      - mesh: [gear_big#1, gear_big#2]
        sections: [pinion, big]
      - same_axis: [gear_big#1, short_spacer]
      - adjacent: [gear_big#1, gear_big#2]   # opcjonalnie max_gap_mm
      - driven: gear_big#2
      - not_driven: gear_big#2
      - upright: gear_big#1         # stoi na krawędzi (opcj. tolerance_deg)
      - speed: gear_big#2           # prędkość obrotowa (silnik: motor_rpm)
        rpm: 2000                   #   wymagana, ± tolerance_pct (dom. 10)
        # albo min_rpm / max_rpm; opcjonalnie direction: same | opposite
      - speed: output               # j.w. dla wyjścia przekładni
        ratio: 1024                 #   (najwolniejsze napędzane koło);
                                    #   ratio = przełożenie zamiast rpm
    Każda relacja może mieć "group" — nagłówek sekcji w feedbacku.
    motor_rpm: 8000                 # opcjonalnie (domyślnie config.MOTOR_RPM)
    Każda relacja może mieć "description".
    Części tej samej klasy: klasa#1, klasa#2, ... (klasa = klasa#1).

Rodzaje etapów:
    "tylko elementy" (bez relacji) — liczba sztuk; działa także bez
        AprilTaga (spis z detekcji: najwięcej sztuk na jednym zdjęciu),
    "złożenie" (z relacjami) — wymaga analizy 3D (AprilTag na >= 2
        zdjęciach), inaczej NIE MOŻNA OCENIĆ.

Wynik (output/assembly_check.json): spis elementów, zazębienia, napęd
i ocena każdego etapu: ZALICZONE / NIEZALICZONE / NIE MOŻNA OCENIĆ.

Uruchomienie:
    python check_assembly.py                        # wszystkie etapy
    python check_assembly.py assemblies/etap_2.yaml
"""

import itertools
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import yaml

from config import (
    CLASS_NAMES,
    IGNORED_CLASS_IDS,
    FEATURE_CLASS_PARENTS,
    CLASS_CONFUSION_GROUPS,
    PART_MODELS,
    CAD_DIR,
    FUSED_PARTS_PATH,
    DETECTIONS_PATH,
    ASSEMBLIES_DIR,
    ASSEMBLY_CHECK_PATH,
    GEAR_SECTIONS,
    DRIVE_SOURCE_CLASS,
    CHECK_PARALLEL_TOL_DEG,
    CHECK_COAXIAL_TOL_MM,
    CHECK_MESH_CENTER_TOL_MM,
    CHECK_MESH_MIN_OVERLAP_MM,
    CHECK_ADJACENT_MAX_GAP_MM,
    CHECK_MAX_PENETRATION_MM,
    CHECK_UPRIGHT_TOL_DEG,
    CHECK_UNCERTAIN_VOTE_MARGIN,
    MOTOR_RPM,
    CHECK_SPEED_TOL_PCT,
)

OK, ERROR, UNKNOWN = "OK", "ERROR", "CANNOT ASSESS"
PASSED, FAILED = "PASSED", "FAILED"
KIND_ASSEMBLY, KIND_PARTS_ONLY = "assembly", "parts only"
SOURCE_3D, SOURCE_DETECTIONS = "3D", "detections (no 3D)"
OTHER_SECTION = "meshed with a different section"
CLASS_IDS = {name: cid for cid, name in CLASS_NAMES.items()}
PART_CLASSES = [name for cid, name in CLASS_NAMES.items()
                if cid not in IGNORED_CLASS_IDS
                and cid not in FEATURE_CLASS_PARENTS]
RELATION_TYPES = ("mesh", "same_axis", "adjacent", "driven", "not_driven",
                  "upright", "speed")
# "speed: output" — wyjście przekładni: najwolniej obracające się koło
# napędzane przez silnik (koniec łańcucha), bez wskazywania konkretnej części.
OUTPUT_ROLE = "output"
DRIVE_EXCLUDED_OUTPUT = ("engine", "driving_gear")


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------

_Z_RANGE = {}


def model_z_range(class_id):
    """Zakres bryły CAD wzdłuż osi Z modelu (mm)."""
    if class_id not in _Z_RANGE:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(
            str(CAD_DIR / PART_MODELS[class_id]["stl"]))
        z = np.asarray(mesh.vertices)[:, 2]
        _Z_RANGE[class_id] = (float(z.min()), float(z.max()))
    return _Z_RANGE[class_id]


def part_axis(part):
    T = np.array(part["T_world_part_mm"])
    return T[:3, 3], T[:3, 2]  # początek modelu CAD, oś Z modelu


def axis_angle_deg(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))


def line_distance(p1, d1, p2):
    v = p2 - p1
    return float(np.linalg.norm(v - (v @ d1) * d1))


def interval_on_axis(part, z_range, direction):
    origin, axis = part_axis(part)
    ends = [float((origin + axis * z) @ direction) for z in z_range]
    return min(ends), max(ends)


def check_same_axis(a, b):
    pa, da = part_axis(a)
    pb, db = part_axis(b)
    angle = axis_angle_deg(da, db)
    distance = max(line_distance(pa, da, pb), line_distance(pb, db, pa))
    ok = angle <= CHECK_PARALLEL_TOL_DEG and distance <= CHECK_COAXIAL_TOL_MM
    info = {"angle_deg": round(angle, 2),
            "axis_distance_mm": round(distance, 2),
            "tolerance_mm": CHECK_COAXIAL_TOL_MM}
    if angle > CHECK_PARALLEL_TOL_DEG:
        info["detail"] = f"axes not parallel ({angle:.1f}°)"
    elif distance > CHECK_COAXIAL_TOL_MM:
        info["detail"] = f"not on a common axis ({distance:.1f} mm apart)"
    return ok, info


def check_adjacent(a, b, max_gap):
    ok, info = check_same_axis(a, b)
    _, da = part_axis(a)
    ia = interval_on_axis(a, model_z_range(a["class_id"]), da)
    ib = interval_on_axis(b, model_z_range(b["class_id"]), da)
    gap = max(ib[0] - ia[1], ia[0] - ib[1])  # < 0: bryły nachodzą
    info.update({"gap_mm": round(gap, 2), "max_gap_mm": max_gap})
    if gap < -CHECK_MAX_PENETRATION_MM:
        info["detail"] = "parts overlap (collision)"
        return False, info
    if not ok:
        info["detail"] = "not on a common axis"
        return False, info
    if gap > max_gap:
        info["detail"] = f"gap too large ({gap:.1f} mm)"
        return False, info
    return True, info


def check_mesh_sections(a, sa, b, sb):
    pa, da = part_axis(a)
    pb, db = part_axis(b)
    angle = axis_angle_deg(da, db)
    distance = line_distance(pa, da, pb)
    if sa["module"] != sb["module"]:
        return False, {"reason": "different modules"}
    required = sa["module"] * (sa["teeth"] + sb["teeth"]) / 2
    ia = interval_on_axis(a, sa["z_mm"], da)
    ib = interval_on_axis(b, sb["z_mm"], da)
    overlap = min(ia[1], ib[1]) - max(ia[0], ib[0])
    ok = (angle <= CHECK_PARALLEL_TOL_DEG
          and abs(distance - required) <= CHECK_MESH_CENTER_TOL_MM
          and overlap >= CHECK_MESH_MIN_OVERLAP_MM)
    return ok, {
        "a_section": sa["name"], "b_section": sb["name"],
        "angle_deg": round(angle, 2),
        "center_distance_mm": round(distance, 2),
        "required_mm": round(required, 2),
        "deviation_mm": round(distance - required, 2),
        "axial_overlap_mm": round(overlap, 2),
        "tolerance_mm": CHECK_MESH_CENTER_TOL_MM,
    }


def check_upright(part, tolerance):
    """Kąt tarczy do podstawy = kąt osi części do pionu (Z świata)."""
    _, axis = part_axis(part)
    tilt = axis_angle_deg(axis, np.array([0.0, 0.0, 1.0]))
    ok = tilt >= 90.0 - tolerance
    info = {"tilt_deg": round(tilt, 1), "tolerance_deg": tolerance}
    if not ok:
        info["detail"] = (f"not standing upright ({tilt:.0f}° to the base, "
                          f"expected about 90°)")
    return ok, info


def best_mesh(a, b, a_section=None, b_section=None):
    """Najlepsza para sekcji uzębienia (lub wskazana) — (ok, pomiar)."""
    a_section = None if a_section in (None, "any") else a_section
    b_section = None if b_section in (None, "any") else b_section
    results = []
    for sa in GEAR_SECTIONS.get(a["class_id"], []):
        if a_section and sa["name"] != a_section:
            continue
        for sb in GEAR_SECTIONS.get(b["class_id"], []):
            if b_section and sb["name"] != b_section:
                continue
            ok, info = check_mesh_sections(a, sa, b, sb)
            if "deviation_mm" in info:
                results.append((ok, info))
    if not results:
        return False, {"reason": "no matching gear sections"}
    results.sort(key=lambda r: (not r[0], abs(r[1]["deviation_mm"])))
    ok, info = results[0]
    if not ok:
        # Czy zazębienie istnieje, ale inną sekcją niż wymagana?
        other, other_info = best_mesh(a, b) if (a_section or b_section) \
            else (False, None)
        if other:
            info["detail"] = (f"{OTHER_SECTION}: "
                              f"{other_info['a_section']}↔"
                              f"{other_info['b_section']}")
        elif abs(info["deviation_mm"]) > CHECK_MESH_CENTER_TOL_MM:
            info["detail"] = "axis distance does not match a mesh"
        elif info["axial_overlap_mm"] < CHECK_MESH_MIN_OVERLAP_MM:
            info["detail"] = "teeth offset along the axis (no overlap)"
        elif info["angle_deg"] > CHECK_PARALLEL_TOL_DEG:
            info["detail"] = f"axes not parallel ({info['angle_deg']:.1f}°)"
    return ok, info


# --------------------------------------------------------------------------
# Zazębienia i napęd (całe złożenie)
# --------------------------------------------------------------------------

def find_all_meshes(parts):
    gears = [p for p in parts if p["class_id"] in GEAR_SECTIONS]
    meshes = []
    for a, b in itertools.combinations(gears, 2):
        ok, info = best_mesh(a, b)
        if ok:
            meshes.append({"a": a["id"], "b": b["id"], **info})
    return meshes


def drive_train(parts, meshes):
    """
    Obrót od silnika: silnik i części na nim osadzone (PART_MOUNTS)
    obracają się razem, dalej przez zazębienia; część obraca się jako
    całość. Prędkość względna: w_b = -w_a * z_a / z_b.
    """
    teeth = {(p["id"], s["name"]): s["teeth"]
             for p in parts for s in GEAR_SECTIONS.get(p["class_id"], [])}
    speed, links = {}, {}
    queue = deque()
    for p in parts:
        if p["class_id"] == DRIVE_SOURCE_CLASS:
            speed[p["id"]] = 1.0
            queue.append(p["id"])
    for p in parts:
        mount = p.get("mount")
        if mount and mount.get("accepted") and mount["parent"] in speed:
            speed[p["id"]] = speed[mount["parent"]]
            queue.append(p["id"])

    while queue:
        current = queue.popleft()
        for m in meshes:
            if current not in (m["a"], m["b"]):
                continue
            if current == m["a"]:
                other, s_cur, s_other = m["b"], m["a_section"], m["b_section"]
            else:
                other, s_cur, s_other = m["a"], m["b_section"], m["a_section"]
            if other in speed:
                continue
            speed[other] = -speed[current] * (
                teeth[(current, s_cur)] / teeth[(other, s_other)])
            links[other] = {"from": current, "via": f"{s_cur}->{s_other}"}
            queue.append(other)

    result = []
    for p in parts:
        entry = {"part": p["id"], "class_name": p["class_name"],
                 "driven": p["id"] in speed}
        if p["id"] in speed:
            w = speed[p["id"]]
            entry.update({
                # Cyfry znaczące, nie miejsca po przecinku: 1/4096 = 0.000244.
                "speed_vs_motor": float(f"{abs(w):.6g}"),
                "reduction": round(1 / abs(w), 2),
                "rpm": float(f"{MOTOR_RPM * abs(w):.4g}"),
                "direction": "same as motor" if w > 0
                             else "opposite to motor",
                "driven_by": links.get(p["id"]),
            })
        result.append(entry)
    return result


# --------------------------------------------------------------------------
# Spis elementów
# --------------------------------------------------------------------------

def uncertain_class(part):
    """Czy klasa części wynika z wyrównanego głosowania (np. 5:5)."""
    votes = sorted(part.get("class_votes", {}).values(), reverse=True)
    return len(votes) > 1 and votes[0] - votes[1] <= CHECK_UNCERTAIN_VOTE_MARGIN


def inventory_3d(parts):
    counts = Counter(p["class_name"] for p in parts)
    uncertain = {p["class_name"] for p in parts if uncertain_class(p)}
    return dict(counts), uncertain, {}


def inventory_from_detections(detections):
    """
    Bez 3D: najwięcej sztuk na jednym zdjęciu. Klasy mylone przez model
    liczone razem (najwięcej sztuk grupy na jednym zdjęciu), a podział na
    klasy według głosów ze wszystkich zdjęć — oznaczony jako niepewny.
    """
    counts, uncertain, uncertain_groups = {}, set(), {}
    group_of = {CLASS_NAMES[c]: tuple(sorted(CLASS_NAMES[x] for x in g))
                for g in CLASS_CONFUSION_GROUPS for c in g}
    group_max, votes = Counter(), Counter()
    for image in detections["images"]:
        per_group = Counter()
        for name, n in image["class_counts"].items():
            if name not in PART_CLASSES:
                continue
            if name in group_of:
                per_group[group_of[name]] += n
                votes[name] += n
            else:
                counts[name] = max(counts.get(name, 0), n)
        for group, n in per_group.items():
            group_max[group] = max(group_max[group], n)

    for group, total in group_max.items():
        group_votes = {name: votes[name] for name in group}
        vote_sum = sum(group_votes.values())
        remaining = total
        ordered = sorted(group, key=lambda n: -group_votes[n])
        for i, name in enumerate(ordered):
            share = (remaining if i == len(ordered) - 1
                     else round(total * group_votes[name] / vote_sum))
            share = min(share, remaining)
            if share:
                counts[name] = share
            remaining -= share
        if len([n for n in group if group_votes[n]]) > 1:
            # Model różnie nazywał te części na różnych zdjęciach — bez 3D
            # pewna jest tylko łączna liczba sztuk grupy.
            uncertain.update(group)
            uncertain_groups[group] = total
    return counts, uncertain, uncertain_groups


# --------------------------------------------------------------------------
# Etapy montażu
# --------------------------------------------------------------------------

def load_spec(path):
    with open(path, encoding="utf-8") as file:
        spec = yaml.safe_load(file) or {}
    spec.setdefault("name", path.stem)
    spec.setdefault("parts", {})
    spec["relations"] = spec.get("relations") or []
    for name in spec["parts"]:
        if name not in CLASS_IDS:
            raise ValueError(f"unknown class '{name}' (available: "
                             f"{', '.join(PART_CLASSES)})")
    spec["motor_rpm"] = float(spec.get("motor_rpm", MOTOR_RPM))
    roles = set(role_names(spec))
    for rel in spec["relations"]:
        kind, rel_roles = parse_relation(rel)
        if kind == "speed":
            if not any(k in rel for k in ("rpm", "ratio", "min_rpm",
                                          "max_rpm")):
                raise ValueError("'speed' needs rpm, ratio or "
                                 "min_rpm / max_rpm")
            if rel.get("direction") not in (None, "same", "opposite"):
                raise ValueError("'speed' direction must be same or opposite")
        for role in rel_roles:
            if role not in roles:
                raise ValueError(
                    f"relation refers to '{role}', which is not in "
                    f"'parts' (available: {', '.join(sorted(roles))})")
    return spec


def role_names(spec):
    roles = []
    for name, count in spec["parts"].items():
        roles += [f"{name}#{i}" for i in range(1, int(count) + 1)]
    return roles


def normalize_role(role):
    return role if "#" in role else f"{role}#1"


def parse_relation(rel):
    kind = next((k for k in RELATION_TYPES if k in rel), None)
    if kind is None:
        raise ValueError(f"Unknown relation: {rel}")
    value = rel[kind]
    if kind == "speed" and value == OUTPUT_ROLE:
        return kind, []
    roles = [normalize_role(r) for r in
             (value if isinstance(value, list) else [value])]
    return kind, roles


def output_part(drive, by_id):
    """Wyjście przekładni: najwolniejsze napędzane koło (poza silnikiem)."""
    gears = [d for d in drive.values()
             if d["class_name"] not in DRIVE_EXCLUDED_OUTPUT]
    if not gears:
        return None
    return by_id[min(gears, key=lambda d: d["speed_vs_motor"])["part"]]


def fmt_rpm(rpm):
    """2000 -> "2000", 7.8125 -> "7.81", 1.953 -> "1.95"."""
    return f"{rpm:.0f}" if rpm >= 100 else f"{rpm:.3g}"


def check_speed(rel, part, drive, motor_rpm):
    info = {"motor_rpm": motor_rpm}
    if part is None:
        info["detail"] = "no gear is driven by the motor"
        return False, info
    entry = drive.get(part["id"])
    rpm = motor_rpm * entry["speed_vs_motor"] if entry else 0.0
    info.update({"part": part["id"], "rpm": float(f"{rpm:.4g}")})
    if "rpm" in rel or "ratio" in rel:
        tol = float(rel.get("tolerance_pct", CHECK_SPEED_TOL_PCT))
        target = (rel["rpm"] if "rpm" in rel
                  else motor_rpm / float(rel["ratio"]))
        low = target * (1 - tol / 100)
        high = target * (1 + tol / 100)
        expected = (f"about {rel['rpm']:g} rpm (±{tol:g}%)" if "rpm" in rel
                    else f"ratio about 1:{float(rel['ratio']):g} "
                         f"({fmt_rpm(target)} rpm, ±{tol:g}%)")
    else:
        low = float(rel.get("min_rpm", 0))
        high = float(rel.get("max_rpm", math.inf))
        expected = (f"{low:g}–{high:g} rpm" if "max_rpm" in rel
                    else f"at least {low:g} rpm")
    info.update({"min_rpm": float(f"{low:.4g}"),
                 "max_rpm": float(f"{high:.4g}")})
    if entry is None:
        info["detail"] = f"not driven by the motor, expected {expected}"
        return False, info
    info["direction"] = entry["direction"]
    if not low <= rpm <= high:
        info["detail"] = (f"rotates at {fmt_rpm(rpm)} rpm (ratio "
                          f"1:{entry['reduction']:g}), expected {expected}")
        return False, info
    wanted = rel.get("direction")
    if wanted and not entry["direction"].startswith(wanted):
        info["detail"] = (f"rotates {entry['direction']}, expected "
                          f"{wanted} {'as' if wanted == 'same' else 'to'} "
                          f"the motor")
        return False, info
    return True, info


def evaluate_relation(rel, binding, by_id, drive, motor_rpm=MOTOR_RPM):
    """drive: {id części napędzanej: wpis z drive_train()}."""
    kind, roles = parse_relation(rel)
    missing = [r for r in roles if binding.get(r) is None]
    base = {"type": kind, "roles": roles or [OUTPUT_ROLE],
            "parts": [binding.get(r) for r in roles],
            "description": rel.get("description", ""),
            "group": rel.get("group", "")}
    if missing:
        return {**base, "result": ERROR,
                "detail": f"missing part: {', '.join(missing)}"}
    parts = [by_id[binding[r]] for r in roles]

    if kind == "speed":
        if not roles:
            output = output_part(drive, by_id)
            parts = [output] if output else []
            base["parts"] = [p["id"] for p in parts]
        ok, info = check_speed(rel, parts[0] if parts else None, drive,
                               motor_rpm)
    elif kind == "mesh":
        sections = rel.get("sections") or [None, None]
        ok, info = best_mesh(parts[0], parts[1], sections[0], sections[1])
    elif kind == "same_axis":
        ok, info = check_same_axis(parts[0], parts[1])
    elif kind == "adjacent":
        ok, info = check_adjacent(parts[0], parts[1],
                                  rel.get("max_gap_mm",
                                          CHECK_ADJACENT_MAX_GAP_MM))
    elif kind == "upright":
        ok, info = check_upright(parts[0], rel.get("tolerance_deg",
                                                   CHECK_UPRIGHT_TOL_DEG))
    elif kind == "driven":
        ok, info = parts[0]["id"] in drive, {}
        if not ok:
            info["detail"] = "not driven by the motor"
    else:  # not_driven
        ok, info = parts[0]["id"] not in drive, {}
        if not ok:
            info["detail"] = "rotates although it should not"

    uncertain = [p["id"] for p in parts if p.get("status") != "CONFIRMED"]
    return {**base, "result": OK if ok else ERROR, "measurement": info,
            "detail": info.get("detail", ""), "uncertain_parts": uncertain}


def bind_roles(roles, relations, parts, drive, motor_rpm=MOTOR_RPM):
    """Przypisanie ról do części maksymalizujące liczbę spełnionych relacji."""
    by_id = {p["id"]: p for p in parts}
    # Przypisania w obrębie klasy: permutacje części (bez powtórzeń). Rola
    # bez części (None) tylko, gdy części brakuje — przypisanie części
    # zamiast None nigdy nie obniża wyniku (relacja z brakującą częścią
    # i tak jest błędem), a przy 6 kołach pełny iloczyn z None to setki
    # tysięcy kombinacji zamiast 720.
    by_class = defaultdict(list)
    for role in roles:
        by_class[role.split("#")[0]].append(role)
    class_roles, class_options = [], []
    for cls, cls_roles in by_class.items():
        ids = [p["id"] for p in parts if p["class_name"] == cls]
        pool = ids + [None] * max(0, len(cls_roles) - len(ids))
        class_roles.append(cls_roles)
        # Kolejność jak w spisie części, brak (None) na końcu — przy remisie
        # brakuje ostatniego koła łańcucha, nie pierwszego.
        order = {pid: i for i, pid in enumerate(ids)}
        class_options.append(sorted(
            set(itertools.permutations(pool, len(cls_roles))),
            key=lambda c: [order.get(x, len(ids)) for x in c]))
    # Wynik relacji zależy tylko od części przypisanych do jej ról — liczony
    # raz dla każdego układu tych części (kombinacji przypisań są dziesiątki
    # tysięcy, a geometria relacji jest kosztowna).
    relation_roles = [parse_relation(r)[1] for r in relations]
    cache = {}

    def relation_result(i, binding):
        key = (i, tuple(binding.get(r) for r in relation_roles[i]))
        if key not in cache:
            cache[key] = evaluate_relation(relations[i], binding, by_id,
                                           drive, motor_rpm)
        return cache[key]

    best = None
    for combo in itertools.product(*class_options):
        binding = {role: pid for cls_roles, assigned in zip(class_roles, combo)
                   for role, pid in zip(cls_roles, assigned)}
        used = [pid for pid in binding.values() if pid is not None]
        results = [relation_result(i, binding)
                   for i in range(len(relations))]
        # Pół punktu za zazębienie istniejące, ale inną sekcją — wtedy
        # raport wskaże właściwe koło i powód ("inną sekcją").
        score = (sum(r["result"] == OK for r in results)
                 + 0.5 * sum(OTHER_SECTION in (r.get("detail") or "")
                             for r in results),
                 len(used))
        if best is None or score > best[0]:
            best = (score, binding, results)
    return best[1], [dict(r) for r in best[2]]


def evaluate_assembly(spec, inventory, uncertain, uncertain_groups, parts,
                      has_3d, drive):
    kind = KIND_ASSEMBLY if spec["relations"] else KIND_PARTS_ONLY
    allow_extra = bool(spec.get("allow_extra", False))

    def compare(expected, found):
        if expected == found or (found > expected and allow_extra):
            return OK
        return ERROR

    group_of = {name: group for group in uncertain_groups for name in group}
    counts = []
    for name in sorted(set(spec["parts"]) | set(inventory)
                       | set(group_of)):
        expected = int(spec["parts"].get(name, 0))
        found = int(inventory.get(name, 0))
        entry = {"class": name, "expected": expected, "found": found,
                 "result": compare(expected, found),
                 "uncertain": name in uncertain}
        group = group_of.get(name)
        if group:
            # Podział między klasy niepewny: oceniana suma grupy.
            group_expected = sum(int(spec["parts"].get(n, 0)) for n in group)
            entry["group"] = "/".join(group)
            entry["group_expected"] = group_expected
            entry["group_found"] = uncertain_groups[group]
            entry["result"] = compare(group_expected, uncertain_groups[group])
        if expected or found or group:
            counts.append(entry)

    reasons = []
    reported_groups = set()
    for c in counts:
        if c["result"] != ERROR:
            continue
        if "group" in c:
            if c["group"] in reported_groups:
                continue
            reported_groups.add(c["group"])
            reasons.append(
                f"count of {c['group']} (combined): found {c['group_found']}, "
                f"expected {c['group_expected']}")
        elif c["found"] < c["expected"]:
            reasons.append(f"missing {c['class']}: found {c['found']}, "
                           f"expected {c['expected']}")
        else:
            reasons.append(f"too many {c['class']}: found {c['found']}, "
                           f"expected {c['expected']}")
    counts_ok = not reasons

    result = {"name": spec["name"], "kind": kind,
              "description": spec.get("description", ""),
              "source": SOURCE_3D if has_3d else SOURCE_DETECTIONS,
              "counts": counts, "allow_extra": allow_extra,
              "relations": [], "binding": {},
              "motor_rpm": spec["motor_rpm"]}

    if spec["relations"] and not has_3d:
        reasons.append("stage requires 3D analysis: AprilTag in at least "
                       "2 images")
        verdict = UNKNOWN
    else:
        if spec["relations"]:
            binding, relations = bind_roles(role_names(spec),
                                            spec["relations"], parts,
                                            drive, spec["motor_rpm"])
            result["binding"] = binding
            result["relations"] = relations
            for r in relations:
                if r["result"] == ERROR:
                    text = r["description"] or f"{r['type']} {r['roles']}"
                    reasons.append(f"{text}: {r.get('detail') or 'not met'}")
        failed = (not counts_ok
                  or any(r["result"] == ERROR for r in result["relations"]))
        verdict = FAILED if failed else PASSED

    result["verdict"] = verdict
    result["reasons"] = reasons
    result["uncertain"] = sorted(
        {c["class"] for c in counts if c["uncertain"]}
        | {pid for r in result["relations"]
           for pid in r.get("uncertain_parts", [])})
    return result


# --------------------------------------------------------------------------

def load_json(path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def main():
    fused = load_json(FUSED_PARTS_PATH)
    detections = load_json(DETECTIONS_PATH)
    if fused is None and detections is None:
        sys.exit("ERROR: No detection or 3D results. Run the earlier "
                 "stages.")

    parts = (fused or {}).get("parts", [])
    has_3d = bool(parts)
    if has_3d:
        inventory, uncertain, uncertain_groups = inventory_3d(parts)
    else:
        inventory, uncertain, uncertain_groups = \
            inventory_from_detections(detections)

    meshes = find_all_meshes(parts) if has_3d else []
    train = drive_train(parts, meshes) if has_3d else []
    drive = {d["part"]: d for d in train if d["driven"]}

    spec_paths = ([Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1
                  else sorted(list(ASSEMBLIES_DIR.glob("*.yaml"))
                              + list(ASSEMBLIES_DIR.glob("*.yml"))))
    assemblies = []
    for path in spec_paths:
        try:
            spec = load_spec(path)
            result = evaluate_assembly(spec, inventory, uncertain,
                                       uncertain_groups, parts, has_3d,
                                       drive)
        except (ValueError, yaml.YAMLError) as exc:
            result = {"name": path.stem, "verdict": UNKNOWN, "kind": "?",
                      "reasons": [f"error in the stage file: {exc}"],
                      "counts": [], "relations": [], "binding": {},
                      "uncertain": []}
        result["file"] = path.name
        assemblies.append(result)

    report = {
        "inventory": {
            "source": SOURCE_3D if has_3d else SOURCE_DETECTIONS,
            "parts": inventory,
            "uncertain_classes": sorted(uncertain),
            "uncertain_groups": {"/".join(g): n
                                 for g, n in uncertain_groups.items()},
            "unconfirmed_observations": dict(Counter(
                o["class_name"]
                for o in (fused or {}).get("unconfirmed_observations", []))),
            "images_used": (fused or {}).get("images_used", []),
        },
        "gear_meshes": meshes,
        "drive_train": train,
        "assemblies": assemblies,
    }
    with open(ASSEMBLY_CHECK_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"=== PARTS FOUND ({report['inventory']['source']}) ===")
    for name, count in sorted(inventory.items()):
        note = "  (uncertain)" if name in uncertain else ""
        print(f"  {name:<14} {count}{note}")
    if report["inventory"]["unconfirmed_observations"]:
        print("  unconfirmed (not counted): "
              f"{report['inventory']['unconfirmed_observations']}")

    if has_3d:
        print("\n=== GEAR MESHES ===")
        for m in meshes:
            print(f"  {m['a']}.{m['a_section']} <-> {m['b']}.{m['b_section']}"
                  f"  distance {m['center_distance_mm']:.2f} mm (req. "
                  f"{m['required_mm']:.2f}, deviation {m['deviation_mm']:+.2f})")
        if not meshes:
            print("  none")
        print("\n=== DRIVE ===")
        for d in train:
            state = (f"rotates 1:{d['reduction']} ({d['rpm']:g} rpm at "
                     f"{MOTOR_RPM} rpm motor)" if d["driven"]
                     else "does not rotate")
            print(f"  {d['part']:<4} {d['class_name']:<13} {state}")

    print("\n=== STAGES ===")
    for a in assemblies:
        print(f"  [{a['verdict']:<14}] {a['name']}  ({a['file']}, "
              f"{a['kind']})")
        for reason in a["reasons"]:
            print(f"        - {reason}")
        if a.get("uncertain"):
            print(f"        ~ uncertain: {', '.join(a['uncertain'])}")
    print(f"\nReport: {ASSEMBLY_CHECK_PATH}")


if __name__ == "__main__":
    main()
