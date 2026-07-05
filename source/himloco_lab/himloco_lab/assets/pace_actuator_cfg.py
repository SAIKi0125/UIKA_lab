"""Configuration for the PACE actuator model."""

from __future__ import annotations

from isaaclab.actuators import DCMotorCfg
from isaaclab.utils import configclass

from himloco_lab.assets import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """DC motor config with encoder bias and torque delay."""

    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: dict[str, float] | float | None = 0.0
    max_delay: int | None = 0
    delay_scale_range: tuple[float, float] = (1.0, 1.0)
