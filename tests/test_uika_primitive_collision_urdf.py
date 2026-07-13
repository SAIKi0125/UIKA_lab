from __future__ import annotations

import copy
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URDF_PATH = (
    ROOT / "source/himloco_lab/himloco_lab/assets/uika/urdf/uika.urdf"
)
SIMPLE_URDF_PATH = (
    ROOT / "source/himloco_lab/himloco_lab/assets/uika/urdf/uika_simple_collision.urdf"
)
ASSET_CFG_PATH = ROOT / "source/himloco_lab/himloco_lab/assets/uika.py"
GENERATOR_PATH = ROOT / "scripts/generate_uika_simple_collision_urdf.py"

FOOT_LINKS = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"}
EXPECTED_FOOT_ORIGINS = {
    "FL_foot": (-0.01259501558, -0.004777973425, 0.005901887082),
    "FR_foot": (-0.01259501558, 0.004715404473, 0.006002067123),
    "RL_foot": (-0.01259501558, -0.004777973425, 0.005901887082),
    "RR_foot": (-0.01259501558, 0.004715404473, 0.006002067123),
}
EXPECTED_FOOT_RADIUS = 0.02527480633


def _root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _normalized_without_collisions(root: ET.Element) -> bytes:
    normalized = copy.deepcopy(root)
    for link in normalized.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
    for element in normalized.iter():
        if element.text is not None and not element.text.strip():
            element.text = None
        if element.tail is not None:
            element.tail = None
    return ET.tostring(normalized, encoding="utf-8")


def test_uika_asset_uses_generated_simple_collision_urdf():
    source = ASSET_CFG_PATH.read_text(encoding="utf-8")

    assert 'asset_path=f"{UIKA_ASSETS_DIR}/urdf/uika_simple_collision.urdf"' in source


def test_generated_urdf_uses_only_boxes_and_foot_spheres_for_collisions():
    root = _root(SIMPLE_URDF_PATH)
    links = root.findall("link")

    assert len(links) == 17
    for link in links:
        link_name = link.attrib["name"]
        collisions = link.findall("collision")
        assert len(collisions) == 1, link_name

        geometry = collisions[0].find("geometry")
        assert geometry is not None, link_name
        primitives = list(geometry)
        assert len(primitives) == 1, link_name
        assert primitives[0].tag == ("sphere" if link_name in FOOT_LINKS else "box")
        assert geometry.find("mesh") is None, link_name

        visual_geometry = link.find("visual/geometry")
        assert visual_geometry is not None, link_name
        assert visual_geometry.find("mesh") is not None, link_name


def test_generated_foot_spheres_match_current_mesh_projection():
    links = {link.attrib["name"]: link for link in _root(SIMPLE_URDF_PATH).findall("link")}

    for link_name, expected_origin in EXPECTED_FOOT_ORIGINS.items():
        collision = links[link_name].find("collision")
        assert collision is not None
        origin = collision.find("origin")
        sphere = collision.find("geometry/sphere")
        assert origin is not None
        assert sphere is not None

        actual_origin = tuple(float(value) for value in origin.attrib["xyz"].split())
        assert actual_origin == pytest.approx(expected_origin, abs=1.0e-10)
        assert float(sphere.attrib["radius"]) == pytest.approx(EXPECTED_FOOT_RADIUS, abs=1.0e-10)


def test_generation_preserves_every_non_collision_urdf_element():
    source_root = _root(SOURCE_URDF_PATH)
    generated_root = _root(SIMPLE_URDF_PATH)

    assert _normalized_without_collisions(generated_root) == _normalized_without_collisions(source_root)


def test_generator_reproduces_committed_urdf_exactly():
    expected = SIMPLE_URDF_PATH.read_bytes()

    subprocess.run([sys.executable, str(GENERATOR_PATH)], cwd=ROOT, check=True, capture_output=True, text=True)

    assert SIMPLE_URDF_PATH.read_bytes() == expected
