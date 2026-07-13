"""Generate a UIKA URDF with primitive collision bodies from mesh projections."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
UIKA_DIR = REPO_ROOT / "source" / "himloco_lab" / "himloco_lab" / "assets" / "uika"
SOURCE_URDF = UIKA_DIR / "urdf" / "uika.urdf"
OUTPUT_URDF = UIKA_DIR / "urdf" / "uika_simple_collision.urdf"
MESH_DIR = UIKA_DIR / "meshes"

FOOT_LINKS = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"}
THIGH_LINKS = {"FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"}
CALF_LINKS = {"FL_calf", "FR_calf", "RL_calf", "RR_calf"}
LIMB_LINKS = THIGH_LINKS | CALF_LINKS

# Robust bounds prevent small mesh protrusions from inflating collision bodies.
# Limb links keep most of their long-axis extent while fitting their short axes
# more tightly so the collision boxes remain bar-shaped.
DEFAULT_PERCENTILES = (2.0, 98.0)
LIMB_LONG_AXIS_PERCENTILES = (1.0, 99.0)
LIMB_SHORT_AXIS_PERCENTILES = (10.0, 90.0)
BOX_MARGIN = 0.006
FOOT_RADIUS_MARGIN = 0.002


def _fmt(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1.0e-9
    if not singular:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def _principal_axes(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, np.argsort(eigenvalues)[::-1]]

    # Make the basis deterministic and right-handed. The first axis is the
    # long bar direction, so keep it roughly aligned with positive link x.
    if axes[0, 0] < 0.0:
        axes[:, 0] *= -1.0
    if axes[1, 1] < 0.0:
        axes[:, 1] *= -1.0
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    axes[:, 2] /= np.linalg.norm(axes[:, 2])
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    return axes


def _projection_bounds(vertices: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.percentile(vertices, low, axis=0)
    upper = np.percentile(vertices, high, axis=0)
    return lower, upper


def _mesh_projection(link_name: str) -> tuple[list[float], list[float], tuple[float, float, float]]:
    mesh_path = MESH_DIR / f"{link_name}.STL"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing collision source mesh for {link_name}: {mesh_path}")

    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = mesh.vertices
    if len(vertices) == 0:
        raise ValueError(f"Collision source mesh has no vertices: {mesh_path}")

    low, high = DEFAULT_PERCENTILES
    lower, upper = _projection_bounds(vertices, low, high)
    rpy = (0.0, 0.0, 0.0)

    if link_name in LIMB_LINKS:
        axes = _principal_axes(vertices)
        vertices_local = vertices @ axes
        short_low, short_high = LIMB_SHORT_AXIS_PERCENTILES
        lower, upper = _projection_bounds(vertices_local, short_low, short_high)
        long_low, long_high = LIMB_LONG_AXIS_PERCENTILES
        long_lower, long_upper = _projection_bounds(vertices_local[:, 0], long_low, long_high)
        lower[0] = long_lower
        upper[0] = long_upper
        center_local = (lower + upper) * 0.5
        center = axes @ center_local
        size = upper - lower + BOX_MARGIN
        return center.tolist(), size.tolist(), _matrix_to_rpy(axes)

    center = ((lower + upper) * 0.5).tolist()
    margin = 0.0 if link_name in FOOT_LINKS else BOX_MARGIN
    size = (upper - lower + margin).tolist()
    return center, size, rpy


def _replace_collision_geometry(link: ET.Element) -> None:
    link_name = link.attrib["name"]
    center, size, rpy = _mesh_projection(link_name)

    collisions = link.findall("collision")
    if len(collisions) != 1:
        raise ValueError(f"Expected one collision element for {link_name}, found {len(collisions)}")
    collision = collisions[0]

    origin = collision.find("origin")
    if origin is None:
        origin = ET.Element("origin")
        collision.insert(0, origin)
    origin.set("xyz", _fmt(center))
    origin.set("rpy", _fmt(rpy))

    geometry = collision.find("geometry")
    if geometry is None:
        geometry = ET.SubElement(collision, "geometry")
    for child in list(geometry):
        geometry.remove(child)

    if link_name in FOOT_LINKS:
        radius = max(size) * 0.5 + FOOT_RADIUS_MARGIN
        ET.SubElement(geometry, "sphere", {"radius": f"{radius:.10g}"})
    else:
        ET.SubElement(geometry, "box", {"size": _fmt(size)})


def main() -> None:
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(f"Missing source URDF: {SOURCE_URDF}")

    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    links = root.findall("link")
    if not links:
        raise ValueError(f"Source URDF contains no links: {SOURCE_URDF}")

    for link in links:
        _replace_collision_geometry(link)

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=False)
    print(OUTPUT_URDF)


if __name__ == "__main__":
    main()
