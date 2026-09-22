"""Collider scale is baked into MJCF, and the check reports when it misses the visual."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from palatial_sim_file_converters.check import check_asset
from palatial_sim_file_converters.export import mjcf_tree
from palatial_sim_file_converters.read import load_usda

CUBES = """
int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5]
"""

USD = f"""#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "Lid" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.0001, 0.0001, 0.0001)
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Mesh "mesh"
        {{
            {CUBES}
            point3f[] points = [(-0.02, -0.02, -0.02), (0.02, -0.02, -0.02), (0.02, 0.02, -0.02), (-0.02, 0.02, -0.02), (-0.02, -0.02, 0.02), (0.02, -0.02, 0.02), (0.02, 0.02, 0.02), (-0.02, 0.02, 0.02)]
        }}

        def Xform "Colliders"
        {{
            double3 xformOp:scale = (1.8, 1.8, 1.8)
            uniform token[] xformOpOrder = ["xformOp:scale"]

            def Mesh "Collider" (
                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
            )
            {{
                uniform token physics:approximation = "convexHull"
                {CUBES}
                point3f[] points = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
                double3 xformOp:scale = (0.04, 0.04, 0.04)
                uniform token[] xformOpOrder = ["xformOp:scale"]
            }}
        }}
    }}

    def Xform "Base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.2
        float3 physics:diagonalInertia = (0.0002, 0.0002, 0.0002)
        quatf xformOp:orient = (0.70710678118, 0, 0, 0.70710678118)
        double3 xformOp:translate = (1, 2, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

        def Mesh "shell" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            uniform token physics:approximation = "sdf"
            {CUBES}
            point3f[] points = [(-0.01, -0.01, -0.01), (0.01, -0.01, -0.01), (0.01, 0.01, -0.01), (-0.01, 0.01, -0.01), (-0.01, -0.01, 0.01), (0.01, -0.01, 0.01), (0.01, 0.01, 0.01), (-0.01, 0.01, 0.01)]
        }}

        def Sphere "IsaacSphereCollider" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double radius = 0.033
            double3 xformOp:translate = (0, 0, 0.01)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}

    def PhysicsRevoluteJoint "hinge"
    {{
        rel physics:body0 = </World/Base>
        rel physics:body1 = </World/Lid>
        uniform token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        float physics:lowerLimit = -50
        float physics:upperLimit = 110
        bool physics:collisionEnabled = 0
    }}
}}
"""


def test_collider_scale_is_baked_and_flagged():
    asset = load_usda(USD)
    assert "/" not in asset.source
    findings = check_asset(asset)
    codes = {(finding.code, finding.path) for finding in findings}
    assert ("ancestor_scale", "/World/Lid/Colliders/Collider") in codes
    assert ("unit_cube_scale", "/World/Lid/Colliders/Collider") in codes
    assert ("stacked_scale", "/World/Lid/Colliders/Collider") in codes
    assert ("bounds_mismatch", "/World/Lid") in codes
    assert ("physx_sdf", "/World/Base/shell") in codes
    assert ("double_collision", "/World/Base") in codes

    root = mjcf_tree(asset)
    lid = root.find("./worldbody/body[@name='Base']/body[@name='Lid']")
    assert lid is not None
    assert np.allclose([float(value) for value in lid.get("pos").split()], [-2, 0, 0], atol=1e-5)
    quat = [float(value) for value in lid.get("quat").split()]
    assert np.allclose(np.abs(quat), [0.70710678118, 0, 0, 0.70710678118], atol=1e-5)

    hinge = lid.find("joint")
    assert hinge.get("type") == "hinge"
    lower, upper = (float(value) for value in hinge.get("range").split())
    assert lower == pytest.approx(np.deg2rad(-50))
    assert upper == pytest.approx(np.deg2rad(110))
    assert root.find("./contact/exclude") is not None

    sphere = root.find("./worldbody/body[@name='Base']/geom[@type='sphere']")
    assert sphere.get("size") == "0.033"
    assert np.allclose([float(value) for value in sphere.get("pos").split()], [0, 0, 0.01], atol=1e-6)
    assert root.find("./worldbody/body[@name='Base']/geom[@type='sdf']") is not None

    collider = next(body for body in asset.bodies if body.name == "Lid").colliders[0]
    extent = collider.local_points.max(axis=0) - collider.local_points.min(axis=0)
    assert np.allclose(extent, [0.072, 0.072, 0.072], atol=1e-6)
