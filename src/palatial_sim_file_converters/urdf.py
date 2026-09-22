"""Read and write URDF.

URDF joints are revolute, continuous, prismatic, fixed, floating, or planar.
A MuJoCo ball joint becomes three revolute joints because URDF has no spherical
joint. A plane becomes a thin box. Capsules, ellipsoids, cones, and SDF meshes
become triangle meshes. Distance tendons have no URDF equivalent and are
reported instead of being turned into a parent joint.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from palatial_sim_file_converters.frame import (
    apply_pose,
    child_bodies,
    diagonalize_inertia,
    kinematic_joints,
    quat_from_rpy,
    relative_pose,
    root_bodies,
    rpy_from_quat,
    world_pose,
)
from palatial_sim_file_converters.meshes import load_surface, write_obj
from palatial_sim_file_converters.model import Asset, Body, Collider, Finding, Joint, Visual
from palatial_sim_file_converters.primitives import box_mesh, capsule_mesh, cylinder_mesh, sphere_mesh
from palatial_sim_file_converters.xform import transform_points


def load_urdf(path: str | Path) -> Asset:
    path = Path(path)
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"{path} is not a URDF robot")
    links = {element.get("name"): element for element in root.findall("link")}
    joints = [element for element in root.findall("joint")]
    child_of = {element.find("child").get("link"): element for element in joints if element.find("child") is not None}
    roots = [name for name in links if name not in child_of]
    bodies: list[Body] = []
    scene_joints: list[Joint] = []
    by_name: dict[str, Body] = {}

    def visit(name: str, parent_rigid: np.ndarray, parent: Body | None, joint_element) -> None:
        element = links[name]
        local_pos, local_quat = ( (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0) )
        if joint_element is not None:
            local_pos, local_quat = _origin(joint_element.find("origin"))
        rigid, pos, quat = world_pose(local_pos, local_quat, parent_rigid)
        body = Body(
            path=f"/World/{_token(name)}",
            name=name,
            pos=pos,
            quat=quat,
            rigid=rigid,
            mass=None,
            com=None,
            diaginertia=None,
            principal_quat=None,
            kinematic=False,
        )
        _link_inertial(element.find("inertial"), body)
        for visual in element.findall("visual"):
            _link_shape(visual, body, path.parent, visual=True)
        for collision in element.findall("collision"):
            _link_shape(collision, body, path.parent, visual=False)
        bodies.append(body)
        by_name[name] = body
        if joint_element is not None and parent is not None:
            built = _link_joint(joint_element, parent, body)
            if built is not None:
                scene_joints.extend(built if isinstance(built, list) else [built])
        for candidate in joints:
            child = candidate.find("child")
            parent_name = candidate.find("parent")
            if child is None or parent_name is None or parent_name.get("link") != name:
                continue
            visit(child.get("link"), rigid, body, candidate)

    for name in roots:
        visit(name, np.eye(4), None, None)
    return Asset(
        name=root.get("name") or path.stem,
        source=str(path),
        up_axis="Z",
        meters_per_unit=1.0,
        bodies=bodies,
        joints=scene_joints,
    )


def write_urdf(asset: Asset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = path.parent / "meshes"
    robot = ET.Element("robot", {"name": asset.name})
    incoming = kinematic_joints(asset)
    if _needs_world(asset, incoming):
        ET.SubElement(robot, "link", {"name": "world"})
    for body in root_bodies(asset, incoming):
        _write_link(robot, body, None, asset, incoming, mesh_dir, path.parent)
    if any(joint.kind == "distance" for joint in asset.joints):
        asset.findings.append(
            Finding(
                "warning",
                "distance_not_in_urdf",
                asset.source,
                "Distance limits are omitted. URDF has no tendon or distance joint.",
            )
        )
    ET.indent(robot, space="  ")
    path.write_text(ET.tostring(robot, encoding="unicode") + "\n", encoding="utf-8")
    return path


def _write_link(robot, body: Body, parent: Body | None, asset, incoming, mesh_dir: Path, root: Path) -> None:
    link = ET.SubElement(robot, "link", {"name": body.name})
    if body.mass is not None:
        inertial = ET.SubElement(link, "inertial")
        _origin_element(inertial, body.com or (0.0, 0.0, 0.0), body.principal_quat or (1.0, 0.0, 0.0, 0.0))
        ET.SubElement(inertial, "mass", {"value": f"{body.mass:.8g}"})
        inertia = body.diaginertia or (1e-6, 1e-6, 1e-6)
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": f"{max(inertia[0], 1e-12):.8g}",
                "iyy": f"{max(inertia[1], 1e-12):.8g}",
                "izz": f"{max(inertia[2], 1e-12):.8g}",
                "ixy": "0",
                "ixz": "0",
                "iyz": "0",
            },
        )
    for index, visual in enumerate(body.visuals):
        _write_geometry(link, "visual", visual, body, index, mesh_dir, root, asset)
    for index, collider in enumerate(body.colliders):
        _write_geometry(link, "collision", collider, body, index, mesh_dir, root, asset)
    joint = incoming.get(body.path)
    if parent is None and (body.kinematic or (joint is not None and joint.kind == "fixed")):
        _write_fixed(robot, "world", body.name, body.pos, body.quat, f"{body.name}_world")
    elif joint is not None and parent is not None:
        _write_joint(robot, joint, parent, body, asset)
    for child in child_bodies(asset, body, incoming):
        _write_link(robot, child, body, asset, incoming, mesh_dir, root)


def _write_joint(robot, joint: Joint, parent: Body, child: Body, asset: Asset) -> None:
    origin_pos, origin_quat = relative_pose(child, parent)
    if joint.kind == "ball":
        angle = None if joint.range is None else float(joint.range[1])
        limits = None if angle is None else (-angle, angle)
        asset.findings.append(
            Finding(
                "warning",
                "ball_as_revolute",
                joint.path,
                "URDF has no spherical joint. The ball joint is three revolute joints on X, Y, and Z.",
            )
        )
        _revolute_chain(robot, parent.name, child.name, origin_pos, origin_quat, joint.name, limits)
        return
    if joint.kind == "compound" and joint.dofs:
        _compound_chain(robot, parent.name, child.name, origin_pos, origin_quat, joint)
        return
    if joint.kind == "fixed":
        _write_fixed(robot, parent.name, child.name, origin_pos, origin_quat, joint.name)
        return
    kind = "prismatic" if joint.kind == "prismatic" else "continuous" if joint.range is None else "revolute"
    element = ET.SubElement(robot, "joint", {"name": joint.name, "type": kind})
    ET.SubElement(element, "parent", {"link": parent.name})
    ET.SubElement(element, "child", {"link": child.name})
    _origin_element(element, origin_pos, origin_quat)
    ET.SubElement(element, "axis", {"xyz": _text(joint.axis)})
    if joint.range is not None and kind != "continuous":
        lower, upper = joint.range
        if kind == "revolute":
            lower, upper = np.rad2deg(lower), np.rad2deg(upper)
        ET.SubElement(element, "limit", {"lower": f"{lower:.8g}", "upper": f"{upper:.8g}", "effort": "1", "velocity": "1"})
    if joint.damping:
        ET.SubElement(element, "dynamics", {"damping": f"{joint.damping:.8g}"})


def _revolute_chain(robot, parent, child, pos, quat, name, limits) -> None:
    middle_a = f"{name}_y"
    middle_b = f"{name}_z"
    for link_name in (middle_a, middle_b):
        link = ET.SubElement(robot, "link", {"name": link_name})
        inertial = ET.SubElement(link, "inertial")
        _origin_element(inertial, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        ET.SubElement(inertial, "mass", {"value": "1e-6"})
        ET.SubElement(inertial, "inertia", {"ixx": "1e-9", "iyy": "1e-9", "izz": "1e-9", "ixy": "0", "ixz": "0", "iyz": "0"})
    specs = (
        (f"{name}_x", parent, middle_a, (1.0, 0.0, 0.0), pos, quat),
        (f"{name}_y", middle_a, middle_b, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        (f"{name}_z", middle_b, child, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )
    for joint_name, parent_name, child_name, axis, origin, rotation in specs:
        element = ET.SubElement(robot, "joint", {"name": joint_name, "type": "revolute" if limits else "continuous"})
        ET.SubElement(element, "parent", {"link": parent_name})
        ET.SubElement(element, "child", {"link": child_name})
        _origin_element(element, origin, rotation)
        ET.SubElement(element, "axis", {"xyz": _text(axis)})
        if limits is not None:
            lower, upper = np.rad2deg(limits[0]), np.rad2deg(limits[1])
            ET.SubElement(element, "limit", {"lower": f"{lower:.8g}", "upper": f"{upper:.8g}", "effort": "1", "velocity": "1"})


def _compound_chain(robot, parent, child, pos, quat, joint: Joint) -> None:
    dofs = joint.dofs or []
    previous = parent
    for index, dof in enumerate(dofs):
        last = index == len(dofs) - 1
        child_name = child if last else f"{joint.name}_{index}"
        if not last:
            link = ET.SubElement(robot, "link", {"name": child_name})
            inertial = ET.SubElement(link, "inertial")
            _origin_element(inertial, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
            ET.SubElement(inertial, "mass", {"value": "1e-6"})
            ET.SubElement(inertial, "inertia", {"ixx": "1e-9", "iyy": "1e-9", "izz": "1e-9", "ixy": "0", "ixz": "0", "iyz": "0"})
        kind = "prismatic" if dof["type"] == "slide" else "revolute"
        element = ET.SubElement(robot, "joint", {"name": joint.name if index == 0 else f"{joint.name}_{index}", "type": kind})
        ET.SubElement(element, "parent", {"link": previous})
        ET.SubElement(element, "child", {"link": child_name})
        if index == 0:
            _origin_element(element, pos, quat)
        else:
            _origin_element(element, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        ET.SubElement(element, "axis", {"xyz": _text(dof["axis"])})
        if dof.get("range") is not None:
            lower, upper = dof["range"]
            if kind == "revolute":
                lower, upper = np.rad2deg(lower), np.rad2deg(upper)
            ET.SubElement(element, "limit", {"lower": f"{lower:.8g}", "upper": f"{upper:.8g}", "effort": "1", "velocity": "1"})
        previous = child_name


def _write_fixed(robot, parent, child, pos, quat, name) -> None:
    element = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(element, "parent", {"link": parent})
    ET.SubElement(element, "child", {"link": child})
    _origin_element(element, pos, quat)


def _write_geometry(link, tag, shape, body: Body, index: int, mesh_dir: Path, root: Path, asset: Asset) -> None:
    element = ET.SubElement(link, tag)
    _origin_element(element, shape.primitive_pos or (0.0, 0.0, 0.0), shape.primitive_quat or (1.0, 0.0, 0.0, 0.0))
    geometry = ET.SubElement(element, "geometry")
    geom_type = shape.geom_type
    size = shape.primitive_size or (0.01,)
    if geom_type == "sphere" and shape.local_points is None:
        ET.SubElement(geometry, "sphere", {"radius": f"{size[0]:.8g}"})
        return
    if geom_type == "box" and shape.local_points is None:
        ET.SubElement(geometry, "box", {"size": _text(value * 2.0 for value in size[:3])})
        return
    if geom_type == "cylinder" and shape.local_points is None:
        ET.SubElement(geometry, "cylinder", {"radius": f"{size[0]:.8g}", "length": f"{size[1] * 2.0:.8g}"})
        return
    if geom_type == "plane":
        asset.findings.append(Finding("warning", "plane_as_box", shape.path, "URDF has no plane. The plane is a thin box of the declared half extents."))
        ET.SubElement(geometry, "box", {"size": _text((size[0] * 2.0, size[1] * 2.0, 0.02))})
        return
    points, faces = _mesh_for(shape)
    if points is None:
        return
    filename = f"{_token(body.name)}_{tag}_{index}.obj"
    write_obj(mesh_dir / filename, points, faces)
    if geom_type == "sdf":
        asset.findings.append(Finding("warning", "sdf_as_mesh", shape.path, "URDF collision meshes are triangle meshes, not MuJoCo SDFs."))
    ET.SubElement(geometry, "mesh", {"filename": str(Path("meshes") / filename)})


def _mesh_for(shape):
    if shape.local_points is not None:
        return shape.local_points, shape.faces
    size = shape.primitive_size or (0.01,)
    geom_type = shape.geom_type
    if geom_type == "ellipsoid":
        points, faces = sphere_mesh(1.0)
        return points * np.asarray(size[:3], dtype=np.float64), faces
    if geom_type == "capsule":
        return capsule_mesh(float(size[0]), float(size[0]), float(size[1]) * 2.0)
    if geom_type == "sphere":
        return sphere_mesh(float(size[0]))
    if geom_type == "box":
        return box_mesh(size[:3])
    if geom_type == "cylinder":
        return cylinder_mesh(float(size[0]), float(size[0]), float(size[1]) * 2.0)
    return None, None


def _link_joint(element, parent: Body, child: Body):
    kind = element.get("type", "fixed")
    origin_pos, origin_quat = _origin(element.find("origin"))
    axis = _vector(element.find("axis").get("xyz") if element.find("axis") is not None else None, (1.0, 0.0, 0.0))
    lower, upper = _limit(element.find("limit"))
    name = element.get("name") or f"{parent.name}_{child.name}"
    if kind == "floating":
        return None
    if kind == "planar":
        normal = np.asarray(axis, dtype=np.float64)
        length = float(np.linalg.norm(normal)) or 1.0
        normal = normal / length
        tangent = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        tangent = tangent - normal * float(np.dot(tangent, normal))
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        dofs = [
            {"type": "slide", "axis": tuple(float(v) for v in tangent), "range": None},
            {"type": "slide", "axis": tuple(float(v) for v in bitangent), "range": None},
            {"type": "hinge", "axis": tuple(float(v) for v in normal), "range": None},
        ]
        return Joint(
            name=name, path=f"/World/joints/{_token(name)}", kind="compound", parent=parent.path, child=child.path,
            axis=dofs[0]["axis"], anchor=(0.0, 0.0, 0.0), range=None, stiffness=None, damping=_damping(element),
            springref=None, disable_collision=False, dofs=dofs,
        )
    if kind == "fixed":
        mapped = "fixed"
        span = None
    elif kind == "prismatic":
        mapped = "prismatic"
        span = None if lower is None else (lower, upper)
    elif kind == "continuous":
        mapped = "revolute"
        span = None
    else:
        mapped = "revolute"
        span = None if lower is None else (float(np.deg2rad(lower)), float(np.deg2rad(upper)))
    return Joint(
        name=name,
        path=f"/World/joints/{_token(name)}",
        kind=mapped,
        parent=parent.path,
        child=child.path,
        axis=axis,
        anchor=(0.0, 0.0, 0.0),
        range=span,
        stiffness=None,
        damping=_damping(element),
        springref=None,
        disable_collision=False,
    )


def _link_shape(element, body: Body, directory: Path, visual: bool) -> None:
    geometry = element.find("geometry")
    if geometry is None or len(geometry) == 0:
        return
    shape = geometry[0]
    pos, quat = _origin(element.find("origin"))
    name = element.get("name") or ("visual" if visual else "collision")
    path = f"{body.path}/{_token(name)}_{len(body.visuals) + len(body.colliders)}"
    points = faces = None
    size = None
    geom_type = shape.tag
    if shape.tag == "box":
        full = _vector(shape.get("size"), (1.0, 1.0, 1.0))
        size = tuple(value * 0.5 for value in full)
        geom_type = "box"
    elif shape.tag == "sphere":
        size = (float(shape.get("radius") or 0.01),)
        geom_type = "sphere"
    elif shape.tag == "cylinder":
        radius = float(shape.get("radius") or 0.01)
        length = float(shape.get("length") or 0.01)
        size = (radius, length * 0.5)
        geom_type = "cylinder"
    elif shape.tag == "mesh":
        filename = shape.get("filename")
        if not filename or filename.startswith("package://"):
            return
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = directory / filename
        points, faces = load_surface(mesh_path)
        scale = _vector(shape.get("scale"), (1.0, 1.0, 1.0))
        points = points * np.asarray(scale, dtype=np.float64)
        points = apply_pose(points, pos, quat)
        pos = (0.0, 0.0, 0.0)
        quat = (1.0, 0.0, 0.0, 0.0)
        geom_type = "mesh"
    else:
        return
    world_min, world_max = _shape_bounds(body.rigid, points, pos, size)
    if visual:
        body.visuals.append(Visual(path, points, faces, world_min, world_max, geom_type, pos, quat, size))
        return
    body.colliders.append(
        Collider(
            path=path, kind=geom_type, geom_type=geom_type, approximation=None, dedicated=True, unit_cube=False,
            scales=[], world_min=world_min, world_max=world_max, friction=None, local_points=points, faces=faces,
            primitive_pos=pos, primitive_quat=quat, primitive_size=size,
        )
    )


def _needs_world(asset: Asset, incoming) -> bool:
    for body in root_bodies(asset, incoming):
        joint = incoming.get(body.path)
        if body.kinematic or (joint is not None and joint.kind == "fixed"):
            return True
    return False


def _link_inertial(element, body: Body) -> None:
    if element is None:
        return
    mass = element.find("mass")
    if mass is not None and mass.get("value"):
        body.mass = float(mass.get("value"))
    pos, quat = _origin(element.find("origin"))
    body.com = pos
    inertia = element.find("inertia")
    if inertia is None:
        return
    values = {key: float(inertia.get(key) or 0.0) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    off_diagonal = any(abs(values[key]) > 1e-12 for key in ("ixy", "ixz", "iyz"))
    if off_diagonal:
        diag, principal = diagonalize_inertia(
            values["ixx"], values["iyy"], values["izz"], values["ixy"], values["ixz"], values["iyz"]
        )
        body.diaginertia = diag
        body.principal_quat = principal
        return
    body.diaginertia = (values["ixx"], values["iyy"], values["izz"])
    body.principal_quat = quat


def _shape_bounds(rigid, points, pos, size):
    if points is not None:
        world = transform_points(rigid, points)
        return world.min(axis=0), world.max(axis=0)
    half = np.array(size[:3] if size and len(size) >= 3 else [size[0], size[0], size[0]] if size else [0.01, 0.01, 0.01], dtype=np.float64)
    center = np.asarray(pos, dtype=np.float64)
    return center - half, center + half


def _origin(element):
    if element is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    pos = _vector(element.get("xyz"), (0.0, 0.0, 0.0))
    roll, pitch, yaw = _vector(element.get("rpy"), (0.0, 0.0, 0.0))
    return pos, quat_from_rpy(roll, pitch, yaw)


def _origin_element(parent, pos, quat) -> None:
    ET.SubElement(parent, "origin", {"xyz": _text(pos), "rpy": _text(rpy_from_quat(quat))})


def _limit(element):
    if element is None:
        return None, None
    lower = element.get("lower")
    upper = element.get("upper")
    if lower is None or upper is None:
        return None, None
    return float(lower), float(upper)


def _damping(element) -> float | None:
    dynamics = element.find("dynamics")
    if dynamics is None or dynamics.get("damping") is None:
        return None
    return float(dynamics.get("damping"))


def _vector(text, default):
    if not text:
        return default
    values = [float(value) for value in text.split()]
    while len(values) < len(default):
        values.append(0.0)
    return tuple(float(value) for value in values[: len(default)])


def _text(values) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _token(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned
