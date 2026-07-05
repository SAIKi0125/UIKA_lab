# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable wall obstacle dimensions for parkour experiments."""

WALL_LENGTH_M = 0.05
"""Wall thickness/depth along x, converted from 50mm."""

WALL_WIDTH_M = 1.0
"""Wall span along y, converted from 1000mm."""

WALL_HEIGHT_M = 0.3
"""Wall height along z, converted from 300mm."""

WALL_SIZE_M = (WALL_LENGTH_M, WALL_WIDTH_M, WALL_HEIGHT_M)
"""Cuboid size tuple ordered as (x length, y width, z height)."""

WALL_CENTER_POS_M = (0.0, 0.0, WALL_HEIGHT_M * 0.5)
"""Default world pose translation with the wall bottom sitting on z=0."""
