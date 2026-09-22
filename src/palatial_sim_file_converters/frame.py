"""Poses shared by the format readers and writers.

Body matrices use the USD row-vector convention: translation on the last row,
and the 3x3 block is the transpose of the column-vector rotation. Quaternions
are ``w x y z``. Joint ranges in the in-memory asset are radians.
"""

from __future__ import annotations

import numpy as np

from palatial_sim_file_converters.model import Asset, Body, Joint
from palatial_sim_file_converters.xform import invert_rigid, matrix_to_quat, quat_to_matrix, transform_points


def pose_to_rigid(pos, quat) -> np.ndarray:
    rigid = np.eye(4)
    rigid[:3, :3] = quat_to_matrix(quat).T
    rigid[3, :3] = np.asarray(pos, dtype=np.float64)
    return rigid


def world_pose(local_pos, local_quat, parent_rigid: np.ndarray):
    world = pose_to_rigid(local_pos, local_quat) @ parent_rigid
    position = tuple(float(value) for value in world[3, :3])
    quat = matrix_to_quat(world[:3, :3].T)
    return world, position, quat


def relative_pose(body: Body, parent: Body | None):
    if parent is None:
        return body.pos, body.quat
    relative = body.rigid @ invert_rigid(parent.rigid)
    position = tuple(float(value) for value in relative[3, :3])
    quat = matrix_to_quat(relative[:3, :3].T)
    return position, quat


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rotation: roll, then pitch, then yaw."""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(rotation[2, 0]) < 0.999999:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    return roll, pitch, yaw


def quat_from_rpy(roll: float, pitch: float, yaw: float):
    return matrix_to_quat(rpy_to_matrix(roll, pitch, yaw))


def rpy_from_quat(quat) -> tuple[float, float, float]:
    return matrix_to_rpy(quat_to_matrix(quat))


def align_axis(source: np.ndarray, target) -> tuple[float, float, float, float]:
    """Quaternion rotating ``source`` onto ``target``."""
    origin = np.asarray(source, dtype=np.float64)
    direction = np.asarray(target, dtype=np.float64)
    direction_length = float(np.linalg.norm(direction))
    if direction_length < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    direction = direction / direction_length
    cross = np.cross(origin, direction)
    dot = float(np.dot(origin, direction))
    if dot > 1.0 - 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-8:
        axis = np.array([1.0, 0.0, 0.0]) if abs(origin[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(origin, axis)
        axis = axis / np.linalg.norm(axis)
        return (0.0, float(axis[0]), float(axis[1]), float(axis[2]))
    quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def diagonalize_inertia(ixx, iyy, izz, ixy, ixz, iyz):
    matrix = np.array(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
        dtype=np.float64,
    )
    values, vectors = np.linalg.eigh(matrix)
    if np.linalg.det(vectors) < 0.0:
        vectors[:, 0] *= -1.0
    quat = matrix_to_quat(vectors)
    return tuple(float(value) for value in values), quat


def kinematic_joints(asset: Asset) -> dict[str, Joint]:
    incoming: dict[str, Joint] = {}
    for joint in asset.joints:
        if joint.kind == "distance":
            continue
        incoming.setdefault(joint.child, joint)
    return incoming


def root_bodies(asset: Asset, incoming: dict[str, Joint]) -> list[Body]:
    return [
        body
        for body in asset.bodies
        if incoming.get(body.path) is None or incoming[body.path].parent is None
    ]


def child_bodies(asset: Asset, parent: Body, incoming: dict[str, Joint]) -> list[Body]:
    return [
        body
        for body in asset.bodies
        if (joint := incoming.get(body.path)) is not None and joint.parent == parent.path
    ]


def apply_pose(points: np.ndarray, pos, quat) -> np.ndarray:
    return transform_points(pose_to_rigid(pos, quat), points)


def near_identity(quat, tol: float = 1e-6) -> bool:
    return abs(abs(float(quat[0])) - 1.0) <= tol and all(abs(float(value)) <= tol for value in quat[1:])
