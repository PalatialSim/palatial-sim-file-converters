"""Triangle mesh files used by MJCF and URDF."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_obj(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {point[0]:.8g} {point[1]:.8g} {point[2]:.8g}" for point in points]
    lines.extend(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}" for face in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return read_obj(path)
    if suffix == ".stl":
        return read_stl(path)
    raise ValueError(f"Unsupported mesh {path.name}. OBJ and STL are read; other formats are left as references.")


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            indices = []
            for token in line.split()[1:]:
                raw = token.split("/")[0]
                if not raw:
                    continue
                index = int(raw)
                indices.append(index - 1 if index > 0 else len(vertices) + index)
            for offset in range(1, len(indices) - 1):
                faces.append((indices[0], indices[offset], indices[offset + 1]))
    if not vertices or not faces:
        raise ValueError(f"{path} has no triangles")
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data) and count > 0:
            return _binary_stl(data, count)
    return _ascii_stl(data.decode("utf-8", errors="ignore"))


def _binary_stl(data: bytes, count: int) -> tuple[np.ndarray, np.ndarray]:
    points = []
    faces = []
    offset = 84
    for _ in range(count):
        base = len(points)
        for _vertex in range(3):
            vertex = struct.unpack_from("<3f", data, offset + 12 + _vertex * 12)
            points.append(vertex)
        faces.append((base, base + 1, base + 2))
        offset += 50
    return np.array(points, dtype=np.float64), np.array(faces, dtype=np.int32)


def _ascii_stl(text: str) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "vertex" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(vertices) < 3:
        raise ValueError("ASCII STL has no triangles")
    points = []
    faces = []
    for index in range(0, len(vertices) - 2, 3):
        base = len(points)
        points.extend(vertices[index : index + 3])
        faces.append((base, base + 1, base + 2))
    return np.array(points, dtype=np.float64), np.array(faces, dtype=np.int32)
