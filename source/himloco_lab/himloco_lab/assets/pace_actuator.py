"""PACE actuator model for UIKA motor identification parameters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import DCMotor
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from himloco_lab.assets.pace_actuator_cfg import PaceDCMotorCfg


class PaceDCMotor(DCMotor):
    """DC motor with encoder bias and delayed torque application."""

    cfg: PaceDCMotorCfg

    def __init__(self, cfg: PaceDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)) and len(cfg.encoder_bias) != self.num_joints:
            raise ValueError(
                f"encoder_bias must have {self.num_joints} elements, "
                f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
            )

        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)
        self.nominal_delay = int(cfg.max_delay or 0)
        self.delay_scale_range = cfg.delay_scale_range
        self.delay_buffer_max = int(math.ceil(self.nominal_delay * self.delay_scale_range[1]))
        self.torques_delay_buffer = DelayBuffer(self.delay_buffer_max + 1, self._num_envs, device=self._device)
        self._sample_time_lags(torch.arange(self._num_envs, device=self._device))

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self.torques_delay_buffer.reset(env_ids)
        self._sample_time_lags(env_ids)

    def _sample_time_lags(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        scales = torch.empty(len(env_ids), device=self._device).uniform_(*self.delay_scale_range)
        delays = torch.round(self.nominal_delay * scales).to(dtype=torch.int)
        delays = torch.clamp(delays, min=0, max=self.delay_buffer_max)
        self.torques_delay_buffer.set_time_lag(delays, env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(control_action_sim.joint_efforts)
        return control_action_sim
