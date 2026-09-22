"""UsdPhysics shapes and joints that MuJoCo cannot take verbatim."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from palatial_sim_file_converters.check import check_asset
from palatial_sim_file_converters.export import export_mjcf
from palatial_sim_file_converters.read import load_asset

POINTS = (
    "point3f[] points = ["
    "(-0.2, -0.1, -0.05), (0.2, -0.1, -0.05), (0.2, 0.1, -0.05), (-0.2, 0.1, -0.05), "
    "(-0.2, -0.1, 0.05), (0.2, -0.1, 0.05), (0.2, 0.1, 0.05), (-0.2, 0.1, 0.05)]"
)
FACES = """
int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5]
"""

USD = f"""#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{{
    vector3f physics:gravityDirection = (0, 0, -1)
    float physics:gravityMagnitude = 9.81
}}

def Xform "World"
{{
    def Xform "Block" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsFilteredPairsAPI"]
    )
    {{
        bool physics:kinematicEnabled = 1
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
        rel physics:filteredPairs = </World/Knob>
        def Mesh "box" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            token physics:approximation = "boundingCube"
            {FACES}
            {POINTS}
        }}
    }}

    def Xform "Ball" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        bool physics:kinematicEnabled = 1
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "sphere" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            token physics:approximation = "boundingSphere"
            {FACES}
            {POINTS}
        }}
    }}

    def Xform "Tip" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        bool physics:kinematicEnabled = 1
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
        double3 xformOp:translate = (2, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cone "cone" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double radius = 0.02
            double height = 0.04
            token axis = "Z"
        }}
    }}

    def Xform "Floor" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        bool physics:kinematicEnabled = 1
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Plane "plane" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double width = 2
            double length = 4
            token axis = "Z"
        }}
    }}

    def Xform "Root" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        bool physics:kinematicEnabled = 1
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
        double3 xformOp:translate = (0, 1, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Sphere "keep" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double radius = 0.02
        }}
    }}

    def Xform "Arm" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.0001, 0.0001, 0.0001)
        double3 xformOp:translate = (0, 1.2, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Sphere "keep" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double radius = 0.02
        }}
    }}

    def Xform "Bob" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.0001, 0.0001, 0.0001)
        double3 xformOp:translate = (0, 2, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Sphere "keep" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double radius = 0.02
        }}
    }}

    def Xform "Knob" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.0001, 0.0001, 0.0001)
        double3 xformOp:translate = (0, 3, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Sphere "keep" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double radius = 0.02
        }}
    }}

    def PhysicsSphericalJoint "swing"
    {{
        rel physics:body0 = </World/Root>
        rel physics:body1 = </World/Arm>
        float physics:coneAngle0Limit = 30
        float physics:coneAngle1Limit = 30
    }}

    def PhysicsDistanceJoint "gap"
    {{
        rel physics:body0 = </World/Root>
        rel physics:body1 = </World/Bob>
        float physics:minDistance = 0.1
        float physics:maxDistance = 0.5
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
    }}

    def PhysicsJoint "twist" (
        prepend apiSchemas = ["PhysicsLimitAPI:rotZ"]
    )
    {{
        rel physics:body0 = </World/Root>
        rel physics:body1 = </World/Knob>
        float limit:rotZ:physics:low = -20
        float limit:rotZ:physics:high = 40
    }}
}}
"""


def test_generated_shapes_and_joints(tmp_path: Path):
    stage = tmp_path / "schemas.usda"
    stage.write_text(USD)
    asset = load_asset(stage)
    check_asset(asset)
    xml_path = export_mjcf(asset, tmp_path / "out")
    root = ET.parse(xml_path).getroot()

    box = _geom(root, "Block")
    assert box.get("type") == "box"
    assert np.allclose([float(v) for v in box.get("size").split()], [0.2, 0.1, 0.05])

    sphere = _geom(root, "Ball")
    assert sphere.get("type") == "sphere"
    radius = float(sphere.get("size"))
    assert radius == pytest.approx(float(np.sqrt(0.2**2 + 0.1**2 + 0.05**2)))

    cone = _geom(root, "Tip")
    assert cone.get("type") == "mesh"
    codes = {finding.code for finding in asset.findings}
    assert "generated_cone" in codes
    assert "generated_bounding_cube" in codes
    assert "generated_bounding_sphere" in codes

    plane = _geom(root, "Floor")
    assert plane.get("type") == "plane"
    assert np.allclose([float(v) for v in plane.get("size").split()], [1.0, 2.0, 0.01])

    ball = root.find(".//joint[@name='swing']")
    assert ball.get("type") == "ball"
    assert np.allclose(
        [float(v) for v in ball.get("range").split()],
        [0.0, np.deg2rad(30.0)],
    )

    parents = {child: parent for parent in root.iter() for child in list(parent)}
    bob = root.find(".//body[@name='Bob']")
    assert parents[bob].tag == "worldbody"
    tendon = root.find(".//tendon/spatial[@name='gap']")
    assert tendon is not None
    assert np.allclose([float(v) for v in tendon.get("range").split()], [0.1, 0.5])

    hinge = root.find(".//joint[@name='twist']")
    assert hinge.get("type") == "hinge"
    assert np.allclose(
        [float(v) for v in hinge.get("range").split()],
        [np.deg2rad(-20.0), np.deg2rad(40.0)],
    )

    excluded = {
        (node.get("body1"), node.get("body2"))
        for node in root.findall(".//contact/exclude")
    }
    assert ("Block", "Knob") in excluded or ("Knob", "Block") in excluded

    assert np.allclose(
        [float(v) for v in root.find("option").get("gravity").split()],
        [0.0, 0.0, -9.81],
    )

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nbody > 1


def _geom(root: ET.Element, body: str) -> ET.Element:
    geom = root.find(f".//body[@name='{body}']/geom")
    assert geom is not None
    return geom
