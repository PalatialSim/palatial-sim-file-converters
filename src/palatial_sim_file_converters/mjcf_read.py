"""Read an MJCF model into the shared scene.

MuJoCo's compiler default for ``angle`` is degrees. Joint ranges, Euler
angles, and spring references are converted to radians here so every writer
sees the same units as the USD reader. Geom ``contype="0"`` and
``conaffinity="0"`` are visuals. A nested body with no joint is a fixed weld.
A spatial tendon becomes a distance joint and does not parent the child.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from palatial_sim_file_converters.frame import align_axis, apply_pose, diagonalize_inertia, world_pose
from palatial_sim_file_converters.meshes import load_surface
from palatial_sim_file_converters.model import Asset, Body, Collider, Joint, Visual
from palatial_sim_file_converters.xform import transform_points


def load_mjcf(path: str | Path) -> Asset:
    path = Path(path)
    return parse_mjcf(path.read_text(encoding="utf-8"), path.name, path.parent)


def parse_mjcf(text: str, name: str = "model.xml", mesh_dir: Path | None = None) -> Asset:
    root = ET.fromstring(text)
    if root.tag != "mujoco":
        raise ValueError(f"{name} is not an MJCF model")
    if root.find(".//include") is not None:
        raise ValueError(f"{name} uses MJCF include. Expand includes before converting.")
    compiler = root.find("compiler")
    degrees = compiler is None or compiler.get("angle", "degree") != "radian"
    if mesh_dir is not None and compiler is not None and compiler.get("meshdir"):
        mesh_dir = mesh_dir / compiler.get("meshdir")
    meshes = _mesh_assets(root, mesh_dir)
    classes = _default_classes(root)
    option = root.find("option")
    gravity = _vector(option.get("gravity") if option is not None else None, (0.0, 0.0, -9.81))
    used: set[str] = set()
    bodies: list[Body] = []
    joints: list[Joint] = []
    sites: dict[str, tuple[str, tuple[float, float, float]]] = {}
    names: dict[str, str] = {}
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{name} has no worldbody")
    _consume_body_children(world, None, np.eye(4), "", classes, degrees, meshes, used, bodies, joints, sites, names)
    joints.extend(_tendons(root, sites, names, degrees))
    return Asset(
        name=root.get("model") or Path(name).stem,
        source=Path(name).name,
        up_axis="Z",
        meters_per_unit=1.0,
        bodies=bodies,
        joints=joints,
        gravity=gravity,
        filtered_pairs=_excludes(root, names),
    )


def _consume_body_children(node, parent, parent_rigid, active_class, classes, degrees, meshes, used, bodies, joints, sites, names):
    for element in list(node):
        if element.tag != "body":
            continue
        _read_body(element, parent, parent_rigid, active_class, classes, degrees, meshes, used, bodies, joints, sites, names)
    if parent is not None:
        return
    geoms = [element for element in list(node) if element.tag == "geom"]
    if not geoms:
        return
    body = _new_body("world", used, names, np.eye(4), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), kinematic=True)
    bodies.append(body)
    for geom in geoms:
        _read_geom(geom, body, active_class, classes, degrees, meshes)


def _read_body(element, parent, parent_rigid, active_class, classes, degrees, meshes, used, bodies, joints, sites, names):
    attrib = _merged("body", element, active_class, classes)
    local_pos = _vector(attrib.get("pos"), (0.0, 0.0, 0.0))
    local_quat = _orientation(attrib, degrees)
    rigid, pos, quat = world_pose(local_pos, local_quat, parent_rigid)
    raw_name = element.get("name") or attrib.get("name") or "body"
    body = _new_body(raw_name, used, names, rigid, pos, quat, kinematic=False)
    bodies.append(body)
    _read_inertial(element.find("inertial"), body, degrees)
    child_class = element.get("childclass") or active_class
    motion: list[Joint] = []
    has_free = False
    for child in list(element):
        if child.tag == "freejoint":
            has_free = True
        elif child.tag == "joint":
            spec = _merged("joint", child, child_class, classes)
            if spec.get("type", "hinge") == "free":
                has_free = True
                continue
            built = _read_joint(child, spec, body, parent, degrees, used)
            if built is not None:
                motion.append(built)
        elif child.tag == "geom":
            _read_geom(child, body, child_class, classes, degrees, meshes)
        elif child.tag == "site":
            spec = _merged("site", child, child_class, classes)
            site_name = child.get("name") or spec.get("name")
            if site_name:
                sites[site_name] = (body.path, _vector(spec.get("pos"), (0.0, 0.0, 0.0)))
    if parent is not None and not has_free and not motion:
        joints.append(_fixed(body, parent, used))
    elif len(motion) == 1:
        joints.append(motion[0])
    elif len(motion) > 1:
        joints.append(_compound(motion))
    for child in list(element):
        if child.tag == "body":
            _read_body(child, body, rigid, child_class, classes, degrees, meshes, used, bodies, joints, sites, names)


def _new_body(raw_name, used, names, rigid, pos, quat, kinematic: bool) -> Body:
    name = _unique(raw_name, used)
    path = f"/World/{_token(name)}"
    names[raw_name] = path
    names[name] = path
    return Body(
        path=path,
        name=name,
        pos=pos,
        quat=quat,
        rigid=rigid,
        mass=None,
        com=None,
        diaginertia=None,
        principal_quat=None,
        kinematic=kinematic,
    )


def _read_inertial(element, body: Body, degrees: bool) -> None:
    if element is None:
        return
    if element.get("mass") is not None:
        body.mass = float(element.get("mass"))
    origin = _orientation({"quat": element.get("quat"), "euler": element.get("euler")}, degrees) if element.get("quat") or element.get("euler") else (1.0, 0.0, 0.0, 0.0)
    body.com = _vector(element.get("pos"), (0.0, 0.0, 0.0))
    if element.get("diaginertia"):
        body.diaginertia = _vector(element.get("diaginertia"), (0.0, 0.0, 0.0))
        body.principal_quat = origin
    elif element.get("fullinertia"):
        values = _floats(element.get("fullinertia"))
        while len(values) < 6:
            values.append(0.0)
        diag, principal = diagonalize_inertia(*values[:6])
        body.diaginertia = diag
        body.principal_quat = principal


def _read_joint(element, spec, body: Body, parent, degrees: bool, used) -> Joint | None:
    mj_type = spec.get("type", "hinge")
    kind = {"hinge": "revolute", "slide": "prismatic", "ball": "ball"}.get(mj_type)
    if kind is None:
        return None
    axis = _vector(spec.get("axis"), (0.0, 0.0, 1.0))
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        axis = (0.0, 0.0, 1.0)
    else:
        axis = tuple(float(value / length) for value in axis)
    angular = kind in {"revolute", "ball"}
    return Joint(
        name=_unique(element.get("name") or spec.get("name") or body.name, used),
        path=f"/World/joints/{_token(element.get('name') or body.name)}",
        kind=kind,
        parent=None if parent is None else parent.path,
        child=body.path,
        axis=axis,
        anchor=_vector(spec.get("pos"), (0.0, 0.0, 0.0)),
        range=_span(spec.get("range"), angular, degrees),
        stiffness=_optional_float(spec.get("stiffness")),
        damping=_optional_float(spec.get("damping")),
        springref=_angle_value(spec.get("springref"), angular, degrees),
        disable_collision=False,
    )


def _fixed(body: Body, parent, used) -> Joint:
    return Joint(
        name=_unique(f"{body.name}_fixed", used),
        path=f"/World/joints/{_token(body.name)}_fixed",
        kind="fixed",
        parent=parent.path,
        child=body.path,
        axis=(0.0, 0.0, 1.0),
        anchor=(0.0, 0.0, 0.0),
        range=None,
        stiffness=None,
        damping=None,
        springref=None,
        disable_collision=False,
    )


def _compound(motion: list[Joint]) -> Joint:
    first = motion[0]
    first.kind = "compound"
    first.dofs = [
        {
            "type": "hinge" if joint.kind == "revolute" else "slide",
            "axis": joint.axis,
            "range": joint.range,
        }
        for joint in motion
    ]
    return first


def _read_geom(element, body: Body, active_class, classes, degrees: bool, meshes) -> None:
    spec = _merged("geom", element, active_class, classes)
    geom_type = spec.get("type", "sphere")
    pos = _vector(spec.get("pos"), (0.0, 0.0, 0.0))
    quat = _orientation(spec, degrees)
    size = _floats(spec.get("size")) if spec.get("size") else []
    if spec.get("fromto"):
        pos, quat, size, geom_type = _fromto(spec.get("fromto"), size, geom_type)
    name = element.get("name") or spec.get("name") or f"{body.name}_{len(body.colliders) + len(body.visuals)}"
    path = f"{body.path}/{_token(name)}"
    friction = None
    if spec.get("friction"):
        slide = _floats(spec.get("friction"))[0]
        friction = (slide, slide, 0.0)
    contype = int(float(spec.get("contype", "1")))
    conaffinity = int(float(spec.get("conaffinity", "1")))
    visual = contype == 0 and conaffinity == 0
    points = faces = None
    primitive_size = None
    if geom_type in {"mesh", "sdf"}:
        mesh_name = spec.get("mesh")
        if mesh_name is None or mesh_name not in meshes:
            raise ValueError(f"MJCF geom {name} references missing mesh {mesh_name}")
        points, faces, scale = meshes[mesh_name]
        points = points * scale
        points = apply_pose(points, pos, quat)
        geom_type = "sdf" if geom_type == "sdf" else "mesh"
        pos = (0.0, 0.0, 0.0)
        quat = (1.0, 0.0, 0.0, 0.0)
    else:
        primitive_size = _mujoco_size(geom_type, size)
    world_min, world_max = _bounds(body.rigid, geom_type, primitive_size, pos, points)
    if visual:
        body.visuals.append(
            Visual(
                path=path,
                local_points=points,
                faces=faces,
                world_min=world_min,
                world_max=world_max,
                geom_type=geom_type,
                primitive_pos=pos,
                primitive_quat=quat,
                primitive_size=primitive_size,
            )
        )
        return
    body.colliders.append(
        Collider(
            path=path,
            kind=geom_type,
            geom_type=geom_type,
            approximation="sdf" if geom_type == "sdf" else ("convexHull" if geom_type == "mesh" else None),
            dedicated=True,
            unit_cube=False,
            scales=[],
            world_min=world_min,
            world_max=world_max,
            friction=friction,
            local_points=points,
            faces=faces,
            primitive_pos=pos,
            primitive_quat=quat,
            primitive_size=primitive_size,
        )
    )


def _fromto(text, size, geom_type):
    values = _floats(text)
    start = np.array(values[:3], dtype=np.float64)
    end = np.array(values[3:6], dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    radius = float(size[0]) if size else 0.01
    kind = geom_type if geom_type in {"capsule", "cylinder"} else "capsule"
    return (
        tuple(float(value) for value in (start + end) * 0.5),
        align_axis(np.array([0.0, 0.0, 1.0]), delta),
        [radius, length * 0.5],
        kind,
    )


def _mujoco_size(geom_type: str, size: list[float]) -> tuple[float, ...]:
    if geom_type == "sphere":
        return (float(size[0]) if size else 0.01,)
    if geom_type == "ellipsoid":
        values = (size + [0.01, 0.01, 0.01])[:3]
        return tuple(float(value) for value in values)
    if geom_type == "box":
        values = (size + [0.01, 0.01, 0.01])[:3]
        return tuple(float(value) for value in values)
    if geom_type in {"capsule", "cylinder"}:
        radius = float(size[0]) if size else 0.01
        half = float(size[1]) if len(size) > 1 else radius
        return (radius, half)
    if geom_type == "plane":
        values = (size + [1.0, 1.0, 0.01])[:3]
        return tuple(float(value) for value in values)
    raise ValueError(f"Unsupported MJCF geom type {geom_type}")


def _bounds(rigid, geom_type, size, pos, points):
    if points is not None:
        world = transform_points(rigid, points)
        return world.min(axis=0), world.max(axis=0)
    half = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    if geom_type == "sphere":
        half[:] = size[0]
    elif geom_type in {"box", "ellipsoid", "plane"}:
        half[:] = size[:3]
    elif geom_type in {"capsule", "cylinder"}:
        extra = size[0] if geom_type == "capsule" else 0.0
        half[:] = (size[0], size[0], size[1] + extra)
    center = np.asarray(pos, dtype=np.float64)
    corners = np.array([[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    corners += center
    world = transform_points(rigid, corners)
    return world.min(axis=0), world.max(axis=0)


def _tendons(root, sites, names, degrees: bool) -> list[Joint]:
    joints = []
    for spatial in root.findall("./tendon/spatial"):
        endpoints = spatial.findall("site")
        if len(endpoints) < 2:
            continue
        start = sites.get(endpoints[0].get("site"))
        end = sites.get(endpoints[-1].get("site"))
        if start is None or end is None:
            continue
        name = spatial.get("name") or "distance"
        joints.append(
            Joint(
                name=name,
                path=f"/World/joints/{_token(name)}",
                kind="distance",
                parent=start[0],
                child=end[0],
                axis=(0.0, 0.0, 1.0),
                anchor=end[1],
                range=_span(spatial.get("range"), False, degrees),
                stiffness=_optional_float(spatial.get("stiffness")),
                damping=_optional_float(spatial.get("damping")),
                springref=None,
                disable_collision=False,
                parent_anchor=start[1],
            )
        )
    return joints


def _excludes(root, names) -> list[tuple[str, str]]:
    pairs = []
    for exclude in root.findall("./contact/exclude"):
        first = names.get(exclude.get("body1"))
        second = names.get(exclude.get("body2"))
        if first and second:
            pairs.append((first, second))
    return pairs


def _mesh_assets(root, mesh_dir: Path | None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    assets = {}
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name") or Path(mesh.get("file", "mesh")).stem
        scale = np.array(_vector(mesh.get("scale"), (1.0, 1.0, 1.0)), dtype=np.float64)
        if mesh.get("file"):
            if mesh_dir is None:
                raise ValueError(f"MJCF mesh {mesh.get('file')} needs the model directory")
            points, faces = load_surface(mesh_dir / mesh.get("file"))
        elif mesh.get("vertex"):
            values = _floats(mesh.get("vertex"))
            points = np.array(values, dtype=np.float64).reshape(-1, 3)
            faces = np.arange(len(points), dtype=np.int32).reshape(-1, 3) if len(points) % 3 == 0 else np.zeros((0, 3), dtype=np.int32)
        else:
            continue
        assets[name] = (points, faces, scale)
    return assets


def _default_classes(root) -> dict[str, dict[str, dict[str, str]]]:
    classes: dict[str, dict[str, dict[str, str]]] = {}

    def walk(node, inherited):
        current = {key: dict(value) for key, value in inherited.items()}
        for child in list(node):
            if child.tag in {"geom", "joint", "body", "site"}:
                current.setdefault(child.tag, {}).update(child.attrib)
        classes[node.get("class") or ""] = current
        for child in list(node):
            if child.tag == "default":
                walk(child, current)

    top = root.find("default")
    if top is not None:
        walk(top, {})
    return classes


def _merged(tag: str, element, active_class: str, classes) -> dict[str, str]:
    chosen = active_class if active_class in classes else ""
    merged = dict(classes.get(chosen, {}).get(tag, {}))
    merged.update(element.attrib)
    return merged


def _orientation(attrib: dict, degrees: bool):
    if attrib.get("quat"):
        values = _floats(attrib["quat"])
        return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    if attrib.get("euler"):
        roll, pitch, yaw = _floats(attrib["euler"])[:3]
        if degrees:
            roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
        from palatial_sim_file_converters.frame import quat_from_rpy

        return quat_from_rpy(float(roll), float(pitch), float(yaw))
    return (1.0, 0.0, 0.0, 0.0)


def _span(text, angular: bool, degrees: bool):
    if not text:
        return None
    lower, upper = _floats(text)[:2]
    if angular and degrees:
        lower, upper = np.deg2rad(lower), np.deg2rad(upper)
    return (float(lower), float(upper))


def _angle_value(text, angular: bool, degrees: bool):
    if text is None:
        return None
    value = float(text)
    if angular and degrees:
        value = float(np.deg2rad(value))
    return value


def _optional_float(text):
    return None if text is None else float(text)


def _vector(text, default):
    if not text:
        return default
    values = _floats(text)
    while len(values) < len(default):
        values.append(0.0)
    return tuple(float(value) for value in values[: len(default)])


def _floats(text: str) -> list[float]:
    return [float(value) for value in text.split()]


def _unique(name: str, used: set[str]) -> str:
    candidate = name or "body"
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while f"{candidate}_{index}" in used:
        index += 1
    unique = f"{candidate}_{index}"
    used.add(unique)
    return unique


def _token(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned
