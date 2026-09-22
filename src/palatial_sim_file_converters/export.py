"""Write an MJCF bundle whose collision geoms use the baked collider frames."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from palatial_sim_file_converters.model import Asset, Body, Collider, Joint
from palatial_sim_file_converters.xform import invert_rigid, matrix_to_quat

def mjcf_tree(asset: Asset) -> ET.Element:
    return _mjcf(asset, None)


def export_mjcf(asset: Asset, output: str | Path) -> Path:
    output = Path(output)
    if output.suffix.lower() in {".xml", ".mjcf"}:
        directory = output.parent
        xml_path = output
    else:
        directory = output
        xml_path = directory / "model.xml"
    root = _mjcf(asset, directory / "meshes")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    _write_report(asset, directory)
    return xml_path


def _mjcf(asset: Asset, mesh_dir: Path | None) -> ET.Element:
    root = ET.Element("mujoco", {"model": asset.name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "option", {"gravity": _format(asset.gravity)})
    asset_node = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    meshes: dict[int, str] = {}

    incoming = _incoming(asset.joints)
    emitted: set[str] = set()

    def emit(body: Body, parent: ET.Element, parent_body: Body | None) -> None:
        emitted.add(body.path)
        attrs = {"name": body.name}
        pos, quat = _pose(body, parent_body)
        attrs["pos"] = _format(pos)
        attrs["quat"] = _format(quat)
        node = ET.SubElement(parent, "body", attrs)
        joint = incoming.get(body.path)
        if parent_body is None and joint is None and not body.kinematic:
            ET.SubElement(node, "freejoint", {"name": f"{body.name}_free"})
        elif joint is not None and joint.kind not in {"fixed", "distance"}:
            _add_joint(node, joint)
        _add_inertial(node, body)
        _add_sites(node, body, asset.joints)
        _add_geoms(node, body, asset_node, mesh_dir, meshes)
        for child in asset.bodies:
            child_joint = incoming.get(child.path)
            if child_joint is not None and child_joint.parent == body.path:
                emit(child, node, body)

    for body in asset.bodies:
        joint = incoming.get(body.path)
        if joint is None or joint.parent is None:
            if body.path not in emitted:
                emit(body, world, None)

    for body in asset.bodies:
        if body.path not in emitted:
            emit(body, world, None)

    names = {body.path: body.name for body in asset.bodies}
    _add_tendons(root, asset.joints)
    excludes = _contact_excludes(asset, names)
    if excludes:
        contact = ET.SubElement(root, "contact")
        for body_a, body_b in excludes:
            ET.SubElement(contact, "exclude", {"body1": body_a, "body2": body_b})

    ET.indent(root, space="  ")
    return root


def _incoming(joints: list[Joint]) -> dict[str, Joint]:
    incoming: dict[str, Joint] = {}
    for joint in joints:
        if joint.kind == "distance":
            continue
        incoming.setdefault(joint.child, joint)
    return incoming


def _pose(body: Body, parent: Body | None):
    if parent is None:
        return body.pos, body.quat
    relative = body.rigid @ invert_rigid(parent.rigid)
    position = tuple(float(value) for value in relative[3, :3])
    quat = matrix_to_quat(relative[:3, :3].T)
    return position, quat


def _add_joint(body: ET.Element, joint: Joint) -> None:
    if joint.kind == "ball":
        attrs = {"name": joint.name, "type": "ball", "pos": _format(joint.anchor)}
        if joint.range is not None:
            attrs["range"] = _format(joint.range)
            attrs["limited"] = "true"
        _spring(attrs, joint)
        ET.SubElement(body, "joint", attrs)
        return
    if joint.kind == "compound" and joint.dofs:
        for index, dof in enumerate(joint.dofs):
            attrs = {
                "name": joint.name if index == 0 else f"{joint.name}_{index}",
                "type": dof["type"],
                "pos": _format(joint.anchor),
                "axis": _format(dof["axis"]),
            }
            if dof.get("range") is not None:
                attrs["range"] = _format(dof["range"])
                attrs["limited"] = "true"
            if index == 0:
                _spring(attrs, joint)
            ET.SubElement(body, "joint", attrs)
        return
    kind = "hinge" if joint.kind == "revolute" else "slide"
    attrs = {
        "name": joint.name,
        "type": kind,
        "pos": _format(joint.anchor),
        "axis": _format(joint.axis),
    }
    if joint.range is not None:
        attrs["range"] = _format(joint.range)
        attrs["limited"] = "true"
    _spring(attrs, joint)
    ET.SubElement(body, "joint", attrs)


def _spring(attrs: dict, joint: Joint) -> None:
    if joint.stiffness is not None:
        attrs["stiffness"] = f"{joint.stiffness:.8g}"
    if joint.damping is not None:
        attrs["damping"] = f"{joint.damping:.8g}"
    if joint.springref is not None:
        attrs["springref"] = f"{joint.springref:.8g}"


def _add_sites(body: ET.Element, source: Body, joints: list[Joint]) -> None:
    for joint in joints:
        if joint.kind != "distance":
            continue
        if joint.parent == source.path:
            ET.SubElement(
                body,
                "site",
                {"name": f"{joint.name}_a", "pos": _format(joint.parent_anchor)},
            )
        if joint.child == source.path:
            ET.SubElement(
                body,
                "site",
                {"name": f"{joint.name}_b", "pos": _format(joint.anchor)},
            )


def _add_tendons(root: ET.Element, joints: list[Joint]) -> None:
    distances = [joint for joint in joints if joint.kind == "distance" and joint.range is not None]
    if not distances:
        return
    tendon = ET.SubElement(root, "tendon")
    for joint in distances:
        spatial = ET.SubElement(
            tendon,
            "spatial",
            {
                "name": joint.name,
                "limited": "true",
                "range": _format(joint.range),
            },
        )
        ET.SubElement(spatial, "site", {"site": f"{joint.name}_a"})
        ET.SubElement(spatial, "site", {"site": f"{joint.name}_b"})


def _contact_excludes(asset: Asset, names: dict[str, str]) -> list[tuple[str, str]]:
    excludes = []
    seen: set[tuple[str, str]] = set()

    def add(path_a: str | None, path_b: str | None) -> None:
        if not path_a or not path_b or path_a not in names or path_b not in names:
            return
        key = tuple(sorted((names[path_a], names[path_b])))
        if key in seen:
            return
        seen.add(key)
        excludes.append(key)

    for joint in asset.joints:
        if joint.disable_collision:
            add(joint.parent, joint.child)
    for path_a, path_b in asset.filtered_pairs:
        add(path_a, path_b)
    return excludes


def _add_inertial(body: ET.Element, source: Body) -> None:
    if source.mass is None or source.diaginertia is None:
        return
    inertia = tuple(max(value, 1e-12) for value in source.diaginertia)
    attrs = {
        "pos": _format(source.com or (0.0, 0.0, 0.0)),
        "mass": f"{source.mass:.8g}",
        "diaginertia": _format(inertia),
    }
    if source.principal_quat is not None:
        attrs["quat"] = _format(source.principal_quat)
    ET.SubElement(body, "inertial", attrs)


def _add_geoms(body, source: Body, asset_node, mesh_dir: Path, meshes: dict[int, str]) -> None:
    has_visual = bool(source.visuals)
    density = {"density": "0"} if source.mass is not None else {}
    for visual in source.visuals:
        attrs = {
            "name": f"{source.name}_{_leaf(visual.path)}_visual",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            **density,
        }
        if visual.geom_type not in {None, "mesh"} and visual.primitive_size is not None:
            attrs["type"] = visual.geom_type
            attrs["pos"] = _format(visual.primitive_pos or (0.0, 0.0, 0.0))
            attrs["quat"] = _format(visual.primitive_quat or (1.0, 0.0, 0.0, 0.0))
            attrs["size"] = _format(visual.primitive_size)
        elif visual.local_points is not None:
            name = _mesh_asset(visual.local_points, visual.faces, visual.path, asset_node, mesh_dir, meshes)
            attrs["type"] = "mesh"
            attrs["mesh"] = name
        else:
            continue
        ET.SubElement(body, "geom", attrs)
    for index, collider in enumerate(source.colliders):
        _add_collider(body, source, collider, index, has_visual, density, asset_node, mesh_dir, meshes)


def _add_collider(body, source, collider: Collider, index, has_visual, density, asset_node, mesh_dir, meshes):
    attrs = {
        "name": f"{source.name}_col_{index}",
        "group": "3" if has_visual else "0",
        **density,
    }
    if collider.friction is not None:
        static, _, _ = collider.friction
        attrs["friction"] = f"{static:.8g} 0.005 0.0001"
    if collider.local_points is not None:
        name = _mesh_asset(
            collider.local_points,
            collider.faces,
            collider.path,
            asset_node,
            mesh_dir,
            meshes,
        )
        attrs["type"] = collider.geom_type or ("sdf" if collider.approximation == "sdf" else "mesh")
        attrs["mesh"] = name
    else:
        attrs["pos"] = _format(collider.primitive_pos or (0.0, 0.0, 0.0))
        attrs["quat"] = _format(collider.primitive_quat or (1.0, 0.0, 0.0, 0.0))
        size = collider.primitive_size or (1.0,)
        attrs["type"] = collider.geom_type or "sphere"
        attrs["size"] = _format(size)
    ET.SubElement(body, "geom", attrs)


def _mesh_asset(points, faces, path, asset_node, mesh_dir: Path, meshes: dict[int, str]) -> str:
    key = id(points)
    if key in meshes:
        return meshes[key]
    name = _leaf(path)
    filename = f"{name}.obj"
    if mesh_dir is not None:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        target = mesh_dir / filename
        suffix = 2
        while target.exists():
            filename = f"{name}_{suffix}.obj"
            target = mesh_dir / filename
            suffix += 1
        _write_obj(target, points, faces)
    ET.SubElement(asset_node, "mesh", {"name": Path(filename).stem, "file": f"meshes/{filename}"})
    meshes[key] = Path(filename).stem
    return meshes[key]


def _write_obj(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {point[0]:.8g} {point[1]:.8g} {point[2]:.8g}" for point in points]
    lines.extend(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}" for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(asset: Asset, output: Path) -> None:
    payload = {
        "source": asset.source,
        "up_axis": asset.up_axis,
        "meters_per_unit": asset.meters_per_unit,
        "bodies": [
            {
                "name": body.name,
                "path": body.path,
                "colliders": [
                    {
                        "path": collider.path,
                        "kind": collider.kind,
                        "approximation": collider.approximation,
                        "dedicated": collider.dedicated,
                        "unit_cube": collider.unit_cube,
                        "scales": [
                            {"prim": op.prim, "scale": list(op.scale)} for op in collider.scales
                        ],
                        "world_extent": [
                            float(value)
                            for value in (collider.world_max - collider.world_min)
                        ],
                        "friction": list(collider.friction) if collider.friction else None,
                    }
                    for collider in body.colliders
                ],
            }
            for body in asset.bodies
        ],
        "findings": [
            {
                "severity": finding.severity,
                "code": finding.code,
                "path": finding.path,
                "message": finding.message,
            }
            for finding in asset.findings
        ],
    }
    (output / "collider_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _leaf(path: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in path.strip("/"))
    return cleaned.strip("_") or "mesh"


def _format(values) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
