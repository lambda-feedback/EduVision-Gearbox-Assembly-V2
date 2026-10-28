
"""
Podgląd 3D złożenia (Open3D).

Pokazuje modele CAD wszystkich znalezionych części w wyznaczonych pozach
(etap 7), w kolorach ze zdjęć, oraz:
    - AprilTag i płaszczyznę podstawy (Z = 0),
    - osie wspólnych wałków (szare linie),
    - zazębienia z etapu 8 (zielone linie między osiami kół),
    - etykiety P0, P1, ... z klasą i informacją o napędzie.
Całe złożenie zapisywane jest też do output/assembly_3d.ply (otwiera się
w każdej przeglądarce 3D, np. MeshLab, Blender, Windows 3D Viewer).

Uruchomienie:
    python view_assembly.py              # okno 3D + zapis .ply
    python view_assembly.py --no-window  # tylko zapis .ply
    python view_assembly.py --screenshot output/assembly_3d.png
"""

import argparse
import json
import sys

import cv2
import numpy as np
import open3d as o3d

from config import (
    CAD_DIR,
    PART_MODELS,
    APRILTAG_SIZE_M,
    DETECTIONS_PATH,
    FUSED_PARTS_PATH,
    ASSEMBLY_CHECK_PATH,
    ASSEMBLY_3D_PATH,
    DRIVE_SOURCE_CLASS,
    MOTOR_SHAFT_DIAMETER_MM,
    SHAFT_VISUAL_DIAMETER_MM,
    SHAFT_VISUAL_MARGIN_MM,
)

STEEL = (0.72, 0.74, 0.78)

# Kolory domyślne (RGB 0..1), gdy kolor ze zdjęć nie jest znany.
CLASS_COLORS = {
    "engine": (0.75, 0.75, 0.78),
    "driving_gear": (0.95, 0.95, 0.95),
    "gear_big": (0.9, 0.5, 0.1),
    "short_spacer": (0.15, 0.15, 0.15),
    "long_spacer": (0.3, 0.3, 0.3),
}


def load_json(path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def observed_colors(parts, detections):
    """Średni kolor części ze zdjęć (mediana Lab zapisana w etapie 5)."""
    if not detections:
        return {}
    lab_by_id = {d["id"]: d["color_lab"]
                 for image in detections["images"]
                 for d in image["detections"] if d.get("color_lab")}
    colors = {}
    for part in parts:
        labs = [lab_by_id[o] for o in part["observations"] if o in lab_by_id]
        if not labs:
            continue
        lab = np.median(np.array(labs), axis=0).astype(np.uint8)
        bgr = cv2.cvtColor(lab.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
        colors[part["id"]] = tuple(float(c) / 255 for c in bgr[::-1])
    return colors


def part_meshes(parts, detections=None):
    colors = observed_colors(parts, detections)
    cache = {}
    items = []
    for part in parts:
        spec = PART_MODELS[part["class_id"]]
        if spec["stl"] not in cache:
            mesh = o3d.io.read_triangle_mesh(str(CAD_DIR / spec["stl"]))
            mesh.compute_vertex_normals()
            cache[spec["stl"]] = mesh
        mesh = o3d.geometry.TriangleMesh(cache[spec["stl"]])
        mesh.transform(np.array(part["T_world_part_mm"]))
        mesh.paint_uniform_color(
            colors.get(part["id"], CLASS_COLORS.get(part["class_name"],
                                                    (0.6, 0.6, 0.6))))
        items.append((part, mesh))
    return items


def shaft_cylinder(point, direction, start, end, diameter):
    """Walec wzdłuż prostej (point + t * direction) od t=start do t=end."""
    direction = direction / np.linalg.norm(direction)
    length = max(end - start, 0.1)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=diameter / 2, height=length, resolution=32)
    # create_cylinder: oś Z, środek w początku układu -> obrót Z na kierunek.
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    angle = np.arccos(np.clip(z @ direction, -1, 1))
    if np.linalg.norm(axis) > 1e-9:
        cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis / np.linalg.norm(axis) * angle), center=(0, 0, 0))
    elif direction[2] < 0:
        cylinder.rotate(np.diag([1.0, -1.0, -1.0]), center=(0, 0, 0))
    cylinder.translate(point + direction * (start + end) / 2)
    cylinder.paint_uniform_color(STEEL)
    cylinder.compute_vertex_normals()
    return cylinder


def extent_on_axis(mesh, point, direction):
    t = (np.asarray(mesh.vertices) - point) @ direction
    return float(t.min()), float(t.max())


def shaft_geometry(meshes, fused):
    """
    Wałek silnika (od czoła silnika do końca osadzonego driving_gear)
    i wałki kół (wspólne osie z etapu 7) jako walce.
    """
    mesh_by_id = {part["id"]: (part, mesh) for part, mesh in meshes}
    items = []
    motor_members = set()

    for part, engine_mesh in meshes:
        if part["class_id"] != DRIVE_SOURCE_CLASS:
            continue
        T = np.array(part["T_world_part_mm"])
        point, direction = T[:3, 3], T[:3, 2]
        mounted = [m for p, m in meshes
                   if (p.get("mount") or {}).get("accepted")
                   and p["mount"]["parent"] == part["id"]]
        if not mounted:
            continue
        motor_members.add(part["id"])
        motor_members.update(p["id"] for p, _ in meshes
                             if (p.get("mount") or {}).get("parent")
                             == part["id"])
        e0, e1 = extent_on_axis(engine_mesh, point, direction)
        e_mid = (e0 + e1) / 2
        for gear_mesh in mounted:
            g0, g1 = extent_on_axis(gear_mesh, point, direction)
            if (g0 + g1) / 2 >= e_mid:   # koło po stronie +Z silnika
                start, end = e1 - 0.5, g1
            else:
                start, end = g0, e0 + 0.5
            items.append({"name": f"motor shaft {part['id']}",
                          "geometry": shaft_cylinder(
                              point, direction, start, end,
                              MOTOR_SHAFT_DIAMETER_MM)})

    for shaft in fused.get("shafts", []):
        if not shaft["accepted"] or motor_members & set(shaft["members"]):
            continue
        point = np.array(shaft["point_world_mm"])
        direction = np.array(shaft["axis_world"])
        direction /= np.linalg.norm(direction)
        ranges = [extent_on_axis(mesh_by_id[m][1], point, direction)
                  for m in shaft["members"] if m in mesh_by_id]
        if not ranges:
            continue
        start = min(r[0] for r in ranges) - SHAFT_VISUAL_MARGIN_MM
        end = max(r[1] for r in ranges) + SHAFT_VISUAL_MARGIN_MM
        items.append({"name": f"shaft {shaft['id']}",
                      "geometry": shaft_cylinder(point, direction, start,
                                                 end,
                                                 SHAFT_VISUAL_DIAMETER_MM)})
    return items


def scene_geometry(parts, fused, check, detections=None):
    """Lista {"name", "geometry"} (Open3D) — używana też przez aplikację."""
    geometries = []

    meshes = part_meshes(parts, detections)
    for part, mesh in meshes:
        geometries.append({"name": f"{part['id']} {part['class_name']}",
                           "geometry": mesh})

    # Podstawa (płaszczyzna Z=0) pod wszystkimi częściami.
    points = np.vstack([np.asarray(m.vertices)
                        for _, m in meshes] or [np.zeros((1, 3))])
    tag = APRILTAG_SIZE_M * 1000
    lo = np.minimum(points.min(axis=0)[:2], [-tag, -tag]) - 15
    hi = np.maximum(points.max(axis=0)[:2], [tag, tag]) + 15
    base = o3d.geometry.TriangleMesh.create_box(hi[0] - lo[0], hi[1] - lo[1],
                                                1.0)
    base.translate((lo[0], lo[1], -1.05))
    base.paint_uniform_color((0.87, 0.78, 0.62))
    base.compute_vertex_normals()
    geometries.append({"name": "base (Z=0)", "geometry": base})

    # AprilTag: biały margines 30 mm + czarny kwadrat 24 mm.
    for size, color, z, name in ((tag * 1.25, (1, 1, 1), -0.04, "tag border"),
                                 (tag, (0.05, 0.05, 0.05), 0.0, "AprilTag")):
        square = o3d.geometry.TriangleMesh.create_box(size, size, 0.05)
        square.translate((-size / 2, -size / 2, z))
        square.paint_uniform_color(color)
        square.compute_vertex_normals()
        geometries.append({"name": name, "geometry": square})

    geometries.append({
        "name": "world frame (X red, Y green, Z blue)",
        "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(10.0),
    })

    # Wałki (walce stalowe) — tylko wizualnie.
    geometries.extend(shaft_geometry(meshes, fused))

    # Zazębienia: odcinek między osiami w płaszczyźnie wspólnych zębów.
    if check:
        by_id = {p["id"]: p for p in parts}
        for m in check.get("gear_meshes", []):
            a, b = by_id.get(m["a"]), by_id.get(m["b"])
            if not a or not b:
                continue
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([a["center_world_mm"],
                                                   b["center_world_mm"]]),
                lines=o3d.utility.Vector2iVector([[0, 1]]))
            line.paint_uniform_color((0.0, 0.8, 0.0))
            geometries.append({
                "name": f"mesh {m['a']}.{m['a_section']}-"
                        f"{m['b']}.{m['b_section']}",
                "geometry": line})
    return geometries


def labels(parts, check):
    driven = {}
    if check:
        driven = {d["part"]: d for d in check.get("drive_train", [])}
    result = []
    for part in parts:
        text = f"{part['id']} {part['class_name']}"
        d = driven.get(part["id"])
        if d and d["driven"] and part["class_name"] != "engine":
            text += f"  1:{d['reduction']}"
        position = np.array(part["center_world_mm"]) + [0, 0, 14]
        result.append((position, text))
    return result


def export_ply(geometries, path):
    combined = o3d.geometry.TriangleMesh()
    for item in geometries:
        if isinstance(item["geometry"], o3d.geometry.TriangleMesh):
            combined += item["geometry"]
    o3d.io.write_triangle_mesh(str(path), combined)


def screenshot(geometries, path, width=1600, height=1000):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    line_material = o3d.visualization.rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = 3
    for i, item in enumerate(geometries):
        is_line = isinstance(item["geometry"], o3d.geometry.LineSet)
        renderer.scene.add_geometry(f"g{i}", item["geometry"],
                                    line_material if is_line else material)
    renderer.scene.set_background([1, 1, 1, 1])
    # Kadr na same części (nazwy "P<n> ...").
    points = np.vstack([np.asarray(item["geometry"].vertices)
                        for item in geometries
                        if item["name"].startswith("P")])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    eye = center + np.array([0.7, -0.6, 0.7]) * extent * 1.4
    renderer.setup_camera(45.0, center, eye, [0, 0, 1])
    image = renderer.render_to_image()
    o3d.io.write_image(str(path), image)


def main():
    parser = argparse.ArgumentParser(description="3D assembly viewer.")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--screenshot", type=str, default=None)
    args = parser.parse_args()

    fused = load_json(FUSED_PARTS_PATH)
    if not fused or not fused["parts"]:
        sys.exit(f"ERROR: No parts in {FUSED_PARTS_PATH}. "
                 "Run fuse_views.py.")
    check = load_json(ASSEMBLY_CHECK_PATH)
    parts = fused["parts"]

    geometries = scene_geometry(parts, fused, check,
                                load_json(DETECTIONS_PATH))
    export_ply(geometries, ASSEMBLY_3D_PATH)
    print(f"Saved assembly model: {ASSEMBLY_3D_PATH}")

    if args.screenshot:
        screenshot(geometries, args.screenshot)
        print(f"Saved screenshot: {args.screenshot}")

    if args.no_window:
        return

    def on_init(visualizer):
        for position, text in labels(parts, check):
            visualizer.add_3d_label(position, text)

    o3d.visualization.draw(
        geometries,
        title="GearBox — 3D assembly",
        bg_color=(1.0, 1.0, 1.0, 1.0),
        show_ui=True,
        line_width=4,
        on_init=on_init,
    )


if __name__ == "__main__":
    main()
