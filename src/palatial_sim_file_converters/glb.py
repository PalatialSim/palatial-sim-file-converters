"""Write a binary glTF of the visual meshes and a JSON manifest of the physics."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from palatial_sim_file_converters.frame import apply_pose, child_bodies, kinematic_joints, relative_pose, root_bodies
from palatial_sim_file_converters.model import Asset
from palatial_sim_file_converters.primitives import box_mesh, capsule_mesh, cylinder_mesh, sphere_mesh

_JSON = 0x4E4F534A
_BIN = 0x004E4942


def glb_document(asset: Asset) -> tuple[bytes, dict]:
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    incoming = kinematic_joints(asset)

    def add_view(array: np.ndarray, component: int, type_name: str, target: int) -> int:
        raw = np.ascontiguousarray(array).tobytes()
        while len(blob) % 4:
            blob.append(0)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw), "target": target}
        blob.extend(raw)
        views.append(view)
        accessor: dict = {
            "bufferView": len(views) - 1,
            "componentType": component,
            "count": int(array.shape[0]) if type_name == "SCALAR" else int(len(array)),
            "type": type_name,
        }
        if type_name == "VEC3":
            accessor["min"] = [float(value) for value in array.min(axis=0)]
            accessor["max"] = [float(value) for value in array.max(axis=0)]
        accessors.append(accessor)
        return len(accessors) - 1

    def add_mesh(points, faces) -> int | None:
        if points is None or faces is None or len(points) == 0 or len(faces) == 0:
            return None
        positions = np.asarray(points, dtype=np.float32)
        indices = np.asarray(faces, dtype=np.uint32).reshape(-1)
        position = add_view(positions, 5126, "VEC3", 34962)
        index = add_view(indices, 5125, "SCALAR", 34963)
        meshes.append({"primitives": [{"attributes": {"POSITION": position}, "indices": index}]})
        return len(meshes) - 1

    def emit(body, parent) -> int:
        points, faces = _surface(body)
        mesh_index = add_mesh(points, faces)
        position, quat = relative_pose(body, parent)
        node = {
            "name": body.name,
            "translation": [float(value) for value in position],
            "rotation": [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])],
        }
        if mesh_index is not None:
            node["mesh"] = mesh_index
        nodes.append(node)
        index = len(nodes) - 1
        children = [emit(child, body) for child in child_bodies(asset, body, incoming)]
        if children:
            node["children"] = children
        return index

    roots = [emit(body, None) for body in root_bodies(asset, incoming)]
    document = {
        "asset": {"version": "2.0", "generator": "palatial-sim-file-converters"},
        "scene": 0,
        "scenes": [{"nodes": roots}],
        "nodes": nodes,
    }
    if meshes:
        document["meshes"] = meshes
        document["accessors"] = accessors
        document["bufferViews"] = views
        document["buffers"] = [{"byteLength": len(blob)}]
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    while len(encoded) % 4:
        encoded += b" "
    while len(blob) % 4:
        blob.append(0)
    chunks = [struct.pack("<II", len(encoded), _JSON) + encoded]
    if blob:
        chunks.append(struct.pack("<II", len(blob), _BIN) + blob)
    payload = b"".join(chunks)
    return struct.pack("<III", 0x46546C67, 2, 12 + len(payload)) + payload, _manifest(asset, nodes)


def write_glb(asset: Asset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload, manifest = glb_document(asset)
    path.write_bytes(payload)
    (path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _surface(body):
    shapes = list(body.visuals) or list(body.colliders)
    if not shapes:
        return None, None
    points = []
    faces = []
    for shape in shapes:
        part_points, part_faces = _shape_mesh(shape)
        if part_points is None:
            continue
        offset = sum(len(chunk) for chunk in points)
        points.append(part_points)
        faces.append(part_faces + offset)
    if not points:
        return None, None
    return np.vstack(points), np.vstack(faces)


def _shape_mesh(shape):
    if shape.local_points is not None and shape.faces is not None:
        return np.asarray(shape.local_points, dtype=np.float64), np.asarray(shape.faces, dtype=np.int32)
    size = shape.primitive_size
    if not size:
        return None, None
    geom_type = shape.geom_type
    if geom_type == "sphere":
        points, faces = sphere_mesh(float(size[0]))
    elif geom_type == "ellipsoid":
        points, faces = sphere_mesh(1.0)
        points = points * np.asarray(size[:3], dtype=np.float64)
    elif geom_type == "box":
        points, faces = box_mesh(size[:3])
    elif geom_type == "capsule":
        points, faces = capsule_mesh(float(size[0]), float(size[0]), float(size[1]) * 2.0)
    elif geom_type == "cylinder":
        points, faces = cylinder_mesh(float(size[0]), float(size[0]), float(size[1]) * 2.0)
    elif geom_type == "plane":
        points, faces = box_mesh((size[0], size[1], 0.01))
    else:
        return None, None
    posed = apply_pose(points, shape.primitive_pos or (0.0, 0.0, 0.0), shape.primitive_quat or (1.0, 0.0, 0.0, 0.0))
    return posed, faces


def _manifest(asset: Asset, nodes: list[dict]) -> dict:
    node_index = {node["name"]: index for index, node in enumerate(nodes)}
    return {
        "source": asset.source,
        "upAxis": asset.up_axis,
        "metersPerUnit": asset.meters_per_unit,
        "gravity": list(asset.gravity),
        "bodies": [
            {
                "name": body.name,
                "path": body.path,
                "node": node_index.get(body.name),
                "mass": body.mass,
                "centerOfMass": list(body.com) if body.com else None,
                "diagonalInertia": list(body.diaginertia) if body.diaginertia else None,
                "principalAxes": list(body.principal_quat) if body.principal_quat else None,
                "kinematic": body.kinematic,
                "translation": list(body.pos),
                "rotation": list(body.quat),
                "colliders": [
                    {
                        "path": collider.path,
                        "type": collider.geom_type,
                        "approximation": collider.approximation,
                        "size": list(collider.primitive_size) if collider.primitive_size else None,
                        "friction": list(collider.friction) if collider.friction else None,
                    }
                    for collider in body.colliders
                ],
            }
            for body in asset.bodies
        ],
        "joints": [
            {
                "name": joint.name,
                "type": joint.kind,
                "parent": joint.parent,
                "child": joint.child,
                "axis": list(joint.axis),
                "anchor": list(joint.anchor),
                "parentAnchor": list(joint.parent_anchor),
                "range": list(joint.range) if joint.range else None,
                "stiffness": joint.stiffness,
                "damping": joint.damping,
            }
            for joint in asset.joints
        ],
        "filteredPairs": [list(pair) for pair in asset.filtered_pairs],
        "findings": [
            {"severity": finding.severity, "code": finding.code, "path": finding.path, "message": finding.message}
            for finding in asset.findings
        ],
    }
