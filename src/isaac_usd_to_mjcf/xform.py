"""Row-vector transforms shared by the USD reader and the MJCF writer.

USD Gf matrices act on row vectors: ``p_world = p_local * M``, with translation
on the last row. MuJoCo quaternions are ``w x y z`` and rotate column vectors.
"""

from __future__ import annotations

import numpy as np

Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def gf_matrix(matrix) -> np.ndarray:
    return np.array(
        [[matrix[row][col] for col in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def translation_of(matrix: np.ndarray) -> np.ndarray:
    return matrix[3, :3].copy()


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    homogeneous = np.c_[points, np.ones(len(points))]
    return (homogeneous @ matrix)[:, :3]


def invert_rigid(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[3, :3] = -translation @ rotation.T
    return inverse


def scale_to_meters(matrix: np.ndarray, meters_per_unit: float) -> np.ndarray:
    """Scale a stage-unit rigid matrix into meters without touching its rotation."""
    scaled = matrix.copy()
    scaled[3, :3] *= meters_per_unit
    return scaled


def quat_to_matrix(quat) -> np.ndarray:
    """Column-vector rotation matrix from a ``w x y z`` quaternion."""
    w, x, y, z = (float(value) for value in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """``w x y z`` quaternion from a column-vector rotation matrix."""
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    quat /= norm
    if quat[0] < 0.0:
        quat = -quat
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def rotate_vector(quat, vector) -> np.ndarray:
    return quat_to_matrix(quat) @ np.asarray(vector, dtype=np.float64)


def gf_quat(value) -> tuple[float, float, float, float]:
    if value is None:
        return (1.0, 0.0, 0.0, 0.0)
    if hasattr(value, "GetReal"):
        imaginary = value.GetImaginary()
        return (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    values = tuple(float(item) for item in value)
    if len(values) != 4:
        return (1.0, 0.0, 0.0, 0.0)
    return values
