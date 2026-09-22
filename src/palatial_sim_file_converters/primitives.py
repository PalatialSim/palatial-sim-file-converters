"""Analytic collider primitives that MuJoCo does not provide directly.

OpenUSD defines ``UsdGeomCone`` as centered on the origin, with the base at
``-height/2`` and the apex at ``+height/2`` along ``axis``. MuJoCo has no cone
geom, so the converter tessellates that solid. Its collision hull is the cone
because the solid is convex.

``boundingCube`` and ``boundingSphere`` are ``UsdPhysicsMeshCollisionAPI``
approximations: the mesh is not the collider. The cube is the mesh's local
axis-aligned bounds. The sphere is centered on that box and contains every
vertex.
"""

from __future__ import annotations

import numpy as np

def aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = points.min(axis=0)
    high = points.max(axis=0)
    return (low + high) * 0.5, (high - low) * 0.5


def bounding_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    center, _half = aabb(points)
    radius = float(np.linalg.norm(points - center, axis=1).max())
    return center, radius


def cone_mesh(
    radius: float,
    height: float,
    axis: str = "Z",
    segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Convex cone mesh in the prim's local frame."""
    apex_axis = height * 0.5
    base_axis = -height * 0.5
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = [
        _place(axis, base_axis, radius * np.cos(angle), radius * np.sin(angle))
        for angle in angles
    ]
    apex = _place(axis, apex_axis, 0.0, 0.0)
    center = _place(axis, base_axis, 0.0, 0.0)
    points = np.array([apex, center, *ring], dtype=np.float64)
    faces: list[tuple[int, int, int]] = []
    for index in range(segments):
        current = 2 + index
        nxt = 2 + (index + 1) % segments
        faces.append((0, current, nxt))
        faces.append((1, nxt, current))
    return points, np.array(faces, dtype=np.int32)


def cylinder_mesh(
    radius_u: float,
    radius_v: float,
    height: float,
    axis: str = "Z",
    segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Elliptic cylinder centered on the origin. Height is the full length."""
    half = height * 0.5
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    bottom = [
        _place(axis, -half, radius_u * np.cos(angle), radius_v * np.sin(angle))
        for angle in angles
    ]
    top = [
        _place(axis, half, radius_u * np.cos(angle), radius_v * np.sin(angle))
        for angle in angles
    ]
    points = np.array(
        [_place(axis, -half, 0.0, 0.0), _place(axis, half, 0.0, 0.0), *bottom, *top],
        dtype=np.float64,
    )
    faces: list[tuple[int, int, int]] = []
    for index in range(segments):
        b0 = 2 + index
        b1 = 2 + (index + 1) % segments
        t0 = 2 + segments + index
        t1 = 2 + segments + (index + 1) % segments
        faces.append((0, b1, b0))
        faces.append((1, t0, t1))
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))
    return points, np.array(faces, dtype=np.int32)


def capsule_mesh(
    radius_u: float,
    radius_v: float,
    height: float,
    axis: str = "Z",
    segments: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Capsule whose cylindrical section has full ``height`` and elliptic radii.

    Matches ``UsdGeomCapsule``: ``height`` excludes the hemispherical end caps.
    Cap rings use the mean radius so the ends close; the cylinder keeps both radii.
    """
    half = height * 0.5
    cap_radius = (radius_u + radius_v) * 0.5
    cap_rows = max(segments // 4, 2)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    latitudes: list[tuple[float, float, float]] = []
    for phi in np.linspace(0.0, np.pi / 2.0, cap_rows, endpoint=False):
        latitudes.append((half + np.cos(phi) * cap_radius, np.sin(phi) * cap_radius, np.sin(phi) * cap_radius))
    latitudes.append((half, radius_u, radius_v))
    latitudes.append((-half, radius_u, radius_v))
    for phi in np.linspace(np.pi / 2.0 + np.pi / (2.0 * cap_rows), np.pi, cap_rows, endpoint=True):
        latitudes.append((-half + np.cos(phi) * cap_radius, np.sin(phi) * cap_radius, np.sin(phi) * cap_radius))
    points = [
        _place(axis, axial, ru * np.cos(angle), rv * np.sin(angle))
        for axial, ru, rv in latitudes
        for angle in angles
    ]
    coordinates = np.array(points, dtype=np.float64)
    faces: list[tuple[int, int, int]] = []
    for row in range(len(latitudes) - 1):
        for column in range(segments):
            a = row * segments + column
            b = row * segments + (column + 1) % segments
            c = (row + 1) * segments + column
            d = (row + 1) * segments + (column + 1) % segments
            faces.append((a, c, b))
            faces.append((b, c, d))
    return coordinates, np.array(faces, dtype=np.int32)


def box_mesh(half) -> tuple[np.ndarray, np.ndarray]:
    """Box from its half extents. The box is centered on the origin."""
    hx, hy, hz = (float(value) for value in half)
    points = np.array(
        [[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)],
        dtype=np.float64,
    )
    faces = np.array(
        [
            (0, 1, 3), (0, 3, 2),
            (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
        ],
        dtype=np.int32,
    )
    return points, faces


def sphere_mesh(radius: float, segments: int = 16, rings: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """UV sphere centered on the origin."""
    points = []
    for row in range(rings + 1):
        phi = row / rings * np.pi
        for column in range(segments):
            theta = column / segments * 2.0 * np.pi
            points.append(
                (
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                )
            )
    coordinates = np.array(points, dtype=np.float64)
    faces: list[tuple[int, int, int]] = []
    for row in range(rings):
        for column in range(segments):
            a = row * segments + column
            b = row * segments + (column + 1) % segments
            c = (row + 1) * segments + column
            d = (row + 1) * segments + (column + 1) % segments
            faces.append((a, c, b))
            faces.append((b, c, d))
    return coordinates, np.array(faces, dtype=np.int32)


def _place(axis: str, axial: float, u: float, v: float) -> tuple[float, float, float]:
    if axis == "X":
        return (axial, u, v)
    if axis == "Y":
        return (u, axial, v)
    return (u, v, axial)
