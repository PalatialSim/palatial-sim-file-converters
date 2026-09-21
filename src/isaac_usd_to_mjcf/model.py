"""In-memory asset produced from an Isaac USD stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScaleOp:
    prim: str
    scale: tuple[float, float, float]


@dataclass
class Collider:
    path: str
    kind: str
    geom_type: str
    approximation: str | None
    dedicated: bool
    unit_cube: bool
    scales: list[ScaleOp]
    world_min: np.ndarray
    world_max: np.ndarray
    friction: tuple[float, float, float] | None
    local_points: np.ndarray | None = None
    faces: np.ndarray | None = None
    primitive_pos: tuple[float, float, float] | None = None
    primitive_quat: tuple[float, float, float, float] | None = None
    primitive_size: tuple[float, ...] | None = None


@dataclass
class Visual:
    path: str
    local_points: np.ndarray
    faces: np.ndarray
    world_min: np.ndarray
    world_max: np.ndarray


@dataclass
class Body:
    path: str
    name: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    rigid: np.ndarray
    mass: float | None
    com: tuple[float, float, float] | None
    diaginertia: tuple[float, float, float] | None
    principal_quat: tuple[float, float, float, float] | None
    kinematic: bool
    colliders: list[Collider] = field(default_factory=list)
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    path: str
    kind: str
    parent: str | None
    child: str
    axis: tuple[float, float, float]
    anchor: tuple[float, float, float]
    range: tuple[float, float] | None
    stiffness: float | None
    damping: float | None
    springref: float | None
    disable_collision: bool
    parent_anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dofs: list[dict] | None = None
    note: str | None = None


@dataclass
class Finding:
    severity: str
    code: str
    path: str
    message: str


@dataclass
class Asset:
    name: str
    source: str
    up_axis: str
    meters_per_unit: float
    bodies: list[Body]
    joints: list[Joint]
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    filtered_pairs: list[tuple[str, str]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
