# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

TERRAIN_ROOT = Path(__file__).resolve().parent
HEIGHT_MAP_DIR = TERRAIN_ROOT / "height_maps"

EXTREME_PARKOUR_SPECS = {
    "T_step_stl": {
        "heightmap": "T_step.npy",
        "display_name": "T_step",
        "real_size": (5.0, 5.0),
        "rotation_k": 2,
        "flip_lr": False,
        "goals": ((80, 125), (100, 125), (120, 125), (130, 125), (150, 155), (150, 175), (150, 190), (150, 200)),
    },
    "Slope": {
        "heightmap": "Slope.npy",
        "display_name": "Slope",
        "real_size": (5.0, 5.0),
        "rotation_k": 2,
        "flip_lr": False,
        "goals": ((55, 124), (95, 124), (125, 124), (150, 124), (170, 124), (190, 124), (200, 124), (235, 124)),
    },
    "BridgeA": {
        "heightmap": "BridgeA.npy",
        "display_name": "BridgeA",
        "real_size": (5.0, 5.0),
        "rotation_k": 2,
        "flip_lr": True,
        "goals": ((20, 100), (42, 103), (70, 103), (95, 103), (122, 103), (150, 103), (190, 100), (190, 140)),
    },
    "BridgeB": {
        "heightmap": "BridgeB.npy",
        "display_name": "BridgeB",
        "real_size": (5.0, 5.0),
        "rotation_k": 2,
        "flip_lr": True,
        "goals": ((20, 106), (50, 106), (70, 106), (90, 106), (110, 106), (130, 106), (140, 150), (140, 190)),
    },
}


def resolve_heightmap_path(terrain_name: str, heightmap_path: str | None = None) -> str:
    if heightmap_path is not None:
        path = Path(heightmap_path)
        return str(path if path.is_absolute() else TERRAIN_ROOT / path)
    return str(HEIGHT_MAP_DIR / EXTREME_PARKOUR_SPECS[terrain_name]["heightmap"])


def load_extreme_parkour_heightmap(
    heightmap_path: str,
    target_shape: tuple[int, int],
    horizontal_scale: float,
    vertical_scale: float,
    real_size: tuple[float, float] | None = None,
    goal_positions: Iterable[tuple[float, float]] | None = None,
    rotation_k: int = 0,
    flip_lr: bool = False,
    flip_ud: bool = False,
    align_bottom: bool = True,
    pad_width: float = 0.1,
    pad_height: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a source Extreme Parkour heightmap into an Isaac Lab sub-terrain grid."""
    heightmap = np.load(heightmap_path).astype(np.float32)
    if flip_lr:
        heightmap = np.flip(heightmap, axis=1)
    if flip_ud:
        heightmap = np.flip(heightmap, axis=0)
    if rotation_k:
        heightmap = np.rot90(heightmap, rotation_k)
    if align_bottom:
        heightmap = heightmap - float(heightmap.min())

    scale_x = 1.0
    scale_y = 1.0
    offset_x = 0.0
    offset_y = 0.0
    if real_size is not None:
        desired_x = max(1, int(round(real_size[0] / horizontal_scale)))
        desired_y = max(1, int(round(real_size[1] / horizontal_scale)))
        scale_x = desired_x / heightmap.shape[0]
        scale_y = desired_y / heightmap.shape[1]
        if heightmap.shape != (desired_x, desired_y):
            heightmap = ndimage.zoom(heightmap, (scale_x, scale_y), order=1)
        placed = np.zeros(target_shape, dtype=np.float32)
        copy_x = min(target_shape[0], heightmap.shape[0])
        copy_y = min(target_shape[1], heightmap.shape[1])
        src_x0 = max(0, (heightmap.shape[0] - copy_x) // 2)
        src_y0 = max(0, (heightmap.shape[1] - copy_y) // 2)
        dst_x0 = max(0, (target_shape[0] - copy_x) // 2)
        dst_y0 = max(0, (target_shape[1] - copy_y) // 2)
        placed[dst_x0 : dst_x0 + copy_x, dst_y0 : dst_y0 + copy_y] = heightmap[
            src_x0 : src_x0 + copy_x, src_y0 : src_y0 + copy_y
        ]
        heightmap = placed
        offset_x = float(dst_x0)
        offset_y = float(dst_y0)
    elif heightmap.shape != target_shape:
        scale_x = target_shape[0] / heightmap.shape[0]
        scale_y = target_shape[1] / heightmap.shape[1]
        heightmap = ndimage.zoom(heightmap, (scale_x, scale_y), order=1)

    height_field = np.rint(heightmap / vertical_scale).astype(np.int16)
    pad_width_px = int(pad_width / horizontal_scale)
    if pad_width_px > 0:
        pad_height_px = int(pad_height / vertical_scale)
        height_field[:, :pad_width_px] = pad_height_px
        height_field[:, -pad_width_px:] = pad_height_px
        height_field[:pad_width_px, :] = pad_height_px
        height_field[-pad_width_px:, :] = pad_height_px

    goals_m = None
    if goal_positions is not None:
        goals = np.array(tuple(goal_positions), dtype=np.float32)
        goals[:, 0] = goals[:, 0] * scale_x + offset_x
        goals[:, 1] = goals[:, 1] * scale_y + offset_y
        goals_m = goals * horizontal_scale

    return height_field, goals_m


def load_named_extreme_parkour_heightmap(
    terrain_name: str,
    target_shape: tuple[int, int],
    horizontal_scale: float,
    vertical_scale: float,
    heightmap_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    spec = EXTREME_PARKOUR_SPECS[terrain_name]
    return load_extreme_parkour_heightmap(
        resolve_heightmap_path(terrain_name, heightmap_path),
        target_shape=target_shape,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        real_size=spec["real_size"],
        goal_positions=spec["goals"],
        rotation_k=spec["rotation_k"],
        flip_lr=spec["flip_lr"],
    )


def sample_height_field_at_positions(
    height_field: np.ndarray,
    positions_m: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    """Sample terrain surface height using the Extreme Parkour height-scan convention."""
    points_px = (positions_m / horizontal_scale).astype(np.int64)
    px = np.clip(points_px[:, 0], 0, height_field.shape[0] - 2)
    py = np.clip(points_px[:, 1], 0, height_field.shape[1] - 2)

    heights1 = height_field[px, py]
    heights2 = height_field[px + 1, py]
    heights3 = height_field[px, py + 1]
    return np.minimum(np.minimum(heights1, heights2), heights3).astype(np.float32) * vertical_scale


def build_height_field_edge_mask(
    height_field: np.ndarray,
    vertical_scale: float,
    height_threshold: float = 0.05,
    edge_width_px: int = 1,
) -> np.ndarray:
    """Build a foot-edge mask from abrupt height-field drops.

    The source Extreme Parkour code stores an edge mask derived while converting
    height fields to trimeshes. This standalone variant marks the lower side of
    abrupt height discontinuities and optionally dilates it so near-edge contacts
    are penalized too.
    """
    threshold = height_threshold / vertical_scale
    height_field = height_field.astype(np.float32)
    edge_mask = np.zeros(height_field.shape, dtype=np.bool_)

    x_delta = height_field[1:, :] - height_field[:-1, :]
    edge_mask[:-1, :] |= x_delta > threshold
    edge_mask[1:, :] |= x_delta < -threshold

    y_delta = height_field[:, 1:] - height_field[:, :-1]
    edge_mask[:, :-1] |= y_delta > threshold
    edge_mask[:, 1:] |= y_delta < -threshold

    if edge_width_px > 0:
        structure = np.ones((edge_width_px * 2 + 1, edge_width_px * 2 + 1), dtype=np.bool_)
        edge_mask = ndimage.binary_dilation(edge_mask, structure=structure)

    return edge_mask.astype(np.bool_)


def build_extreme_parkour_edge_masks(
    tile_size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float = 0.005,
    height_threshold: float = 0.05,
    edge_width: float = 0.05,
) -> dict[str, np.ndarray]:
    """Build per-terrain edge masks for imported Extreme Parkour heightmaps."""
    target_shape = (max(1, int(tile_size[0] / horizontal_scale)), max(1, int(tile_size[1] / horizontal_scale)))
    edge_width_px = int(edge_width / horizontal_scale)
    edge_masks = {}
    for terrain_name in EXTREME_PARKOUR_SPECS:
        height_field, _ = load_named_extreme_parkour_heightmap(
            terrain_name,
            target_shape=target_shape,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
        )
        edge_masks[terrain_name] = build_height_field_edge_mask(
            height_field,
            vertical_scale=vertical_scale,
            height_threshold=height_threshold,
            edge_width_px=edge_width_px,
        )
    return edge_masks


def build_extreme_parkour_routes(
    tile_size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float = 0.005,
) -> dict[str, tuple[tuple[float, float], ...]]:
    target_shape = (max(1, int(tile_size[0] / horizontal_scale)), max(1, int(tile_size[1] / horizontal_scale)))
    routes = {}
    for terrain_name in EXTREME_PARKOUR_SPECS:
        _, goals = load_named_extreme_parkour_heightmap(
            terrain_name,
            target_shape=target_shape,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
        )
        routes[terrain_name] = tuple(tuple(float(value) for value in goal) for goal in goals)
    return routes


def build_extreme_parkour_marker_routes(
    tile_size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float = 0.005,
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    target_shape = (max(1, int(tile_size[0] / horizontal_scale)), max(1, int(tile_size[1] / horizontal_scale)))
    routes = {}
    for terrain_name in EXTREME_PARKOUR_SPECS:
        height_field, goals = load_named_extreme_parkour_heightmap(
            terrain_name,
            target_shape=target_shape,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
        )
        heights = sample_height_field_at_positions(height_field, goals, horizontal_scale, vertical_scale)
        goals_xyz = np.column_stack((goals, heights))
        routes[terrain_name] = tuple(tuple(float(value) for value in goal) for goal in goals_xyz)
    return routes


def build_flat_parkour_route(
    tile_size: tuple[float, float],
    num_goals: int = 8,
    seed: int = 2026,
    margin: float = 1.0,
) -> tuple[tuple[float, float], ...]:
    rng = np.random.default_rng(seed)
    x_values = np.linspace(margin, tile_size[0] - margin, num_goals, dtype=np.float32)
    y_values = rng.uniform(margin, tile_size[1] - margin, size=num_goals).astype(np.float32)
    return tuple((float(x), float(y)) for x, y in zip(x_values, y_values))


def build_flat_parkour_marker_route(
    route: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple((float(x), float(y), 0.0) for x, y in route)
