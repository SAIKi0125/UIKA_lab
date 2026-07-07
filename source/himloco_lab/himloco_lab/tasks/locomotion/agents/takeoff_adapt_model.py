# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class TakeoffMlpAdaptModel(nn.Module):
    """History-encoder actor for fixed-point spring jump without ball observations."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        max_length: int = 10,
        cmd_dim: int = 3,
        latent_dim: int = 32,
        privileged_target_key: str = "privileged_target",
        privileged_target_dim: int = 5,
        mlp_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        **__: Any,
    ) -> None:
        super().__init__()
        del hidden_dims

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {max_length}")
        if self.obs_dim % max_length != 0:
            raise ValueError(
                f"Actor-history obs_dim ({self.obs_dim}) must be divisible by max_length ({max_length}). "
                "Check observation history flattening."
            )

        self.max_length = int(max_length)
        self.cmd_dim = int(cmd_dim)
        self.privileged_target_key = str(privileged_target_key)
        self.privileged_target_dim = int(privileged_target_dim)
        self.obs_per_step = int(self.obs_dim // self.max_length)
        if self.cmd_dim < 0 or self.cmd_dim > self.obs_per_step:
            raise ValueError(f"cmd_dim must be in [0, obs_per_step], got {self.cmd_dim} vs {self.obs_per_step}")
        self.proprioception_dim = int(self.obs_per_step - self.cmd_dim)
        if self.proprioception_dim <= 0:
            raise ValueError(f"proprioception_dim must be > 0, got {self.proprioception_dim}")

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            action_head_out_dim = self.distribution.input_dim
        else:
            self.distribution = None
            action_head_out_dim = output_dim

        self.mem_encoder = nn.Sequential(
            *MLP(self.proprioception_dim * self.max_length, latent_dim, mlp_hidden_dims, activation, "tanh")
        )
        self.state_estimator = nn.Sequential(*MLP(latent_dim, self.privileged_target_dim, [64, 32], activation))

        action_in_dim = latent_dim + self.cmd_dim + self.privileged_target_dim
        self.low_level_net = nn.Sequential(*MLP(action_in_dim, action_head_out_dim, actor_hidden_dims, activation))

        self._privileged_recon_loss = torch.tensor(0.0)
        self._last_privileged_pred: torch.Tensor | None = None

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.low_level_net)  # type: ignore[arg-type]

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        privileged_obs: TensorDict | None = None,
        **_: Any,
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs

        hist_flat = self.get_latent(obs)
        hist_flat = self.obs_normalizer(hist_flat)
        x = hist_flat.view(hist_flat.shape[0], self.max_length, self.obs_per_step)

        pro_obs_seq = x[..., : self.proprioception_dim]
        if self.cmd_dim > 0:
            cmd = x[:, -1, self.proprioception_dim : self.proprioception_dim + self.cmd_dim]
        else:
            cmd = None

        mem = self.mem_encoder(pro_obs_seq.reshape(pro_obs_seq.shape[0], -1))
        privileged_pred = self.state_estimator(mem)
        self._last_privileged_pred = privileged_pred.detach()
        self._privileged_recon_loss = self._compute_recon_loss(privileged_obs, privileged_pred)

        if cmd is not None:
            action_head_in = torch.cat((mem, cmd, privileged_pred), dim=-1)
        else:
            action_head_in = torch.cat((mem, privileged_pred), dim=-1)
        action_head_out = self.low_level_net(action_head_in)

        if self.distribution is not None:
            self.distribution.update(action_head_out)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(action_head_out)
        return action_head_out

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        del masks, hidden_state
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        return torch.cat(obs_list, dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            hist_flat = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(hist_flat)  # type: ignore[attr-defined]

    def compute_adaptation_pred_loss(self) -> torch.Tensor:
        return self._privileged_recon_loss

    def as_jit(self) -> nn.Module:
        return _TorchTakeoffMlpAdaptModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxTakeoffMlpAdaptModel(self, verbose)

    def _compute_recon_loss(self, privileged_obs: TensorDict | None, privileged_pred: torch.Tensor) -> torch.Tensor:
        if privileged_obs is None:
            return privileged_pred.new_zeros(())
        if self.privileged_target_key not in privileged_obs.keys():
            return privileged_pred.new_zeros(())
        target = privileged_obs[self.privileged_target_key]
        if target.dim() != 2 or target.shape[-1] != self.privileged_target_dim:
            raise ValueError(
                f"Expected privileged target '{self.privileged_target_key}' with shape "
                f"[N, {self.privileged_target_dim}], got {tuple(target.shape)}"
            )
        return 2.0 * (privileged_pred - target.detach()).pow(2).mean()

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"TakeoffMlpAdaptModel expects 1D (2D batched) observations, "
                    f"got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchTakeoffMlpAdaptModel(nn.Module):
    """Exportable deterministic takeoff actor."""

    def __init__(self, model: TakeoffMlpAdaptModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mem_encoder = copy.deepcopy(model.mem_encoder)
        self.state_estimator = copy.deepcopy(model.state_estimator)
        self.low_level_net = copy.deepcopy(model.low_level_net)
        self.max_length = model.max_length
        self.obs_per_step = model.obs_per_step
        self.proprioception_dim = model.proprioception_dim
        self.cmd_dim = model.cmd_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        x = x.view(x.shape[0], self.max_length, self.obs_per_step)
        pro_obs_seq = x[..., : self.proprioception_dim]
        if self.cmd_dim > 0:
            cmd = x[:, -1, self.proprioception_dim : self.proprioception_dim + self.cmd_dim]
        else:
            cmd = None
        mem = self.mem_encoder(pro_obs_seq.reshape(pro_obs_seq.shape[0], -1))
        privileged_pred = self.state_estimator(mem)
        if cmd is not None:
            out = self.low_level_net(torch.cat((mem, cmd, privileged_pred), dim=-1))
        else:
            out = self.low_level_net(torch.cat((mem, privileged_pred), dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxTakeoffMlpAdaptModel(nn.Module):
    """Exportable deterministic takeoff actor for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: TakeoffMlpAdaptModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mem_encoder = copy.deepcopy(model.mem_encoder)
        self.state_estimator = copy.deepcopy(model.state_estimator)
        self.low_level_net = copy.deepcopy(model.low_level_net)
        self.max_length = model.max_length
        self.obs_per_step = model.obs_per_step
        self.proprioception_dim = model.proprioception_dim
        self.cmd_dim = model.cmd_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim
        self.input_names = ["obs"]
        self.output_names = ["act"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        x = x.view(x.shape[0], self.max_length, self.obs_per_step)
        pro_obs_seq = x[..., : self.proprioception_dim]
        if self.cmd_dim > 0:
            cmd = x[:, -1, self.proprioception_dim : self.proprioception_dim + self.cmd_dim]
        else:
            cmd = None
        mem = self.mem_encoder(pro_obs_seq.reshape(pro_obs_seq.shape[0], -1))
        privileged_pred = self.state_estimator(mem)
        if cmd is not None:
            out = self.low_level_net(torch.cat((mem, cmd, privileged_pred), dim=-1))
        else:
            out = self.low_level_net(torch.cat((mem, privileged_pred), dim=-1))
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> torch.Tensor:
        return torch.zeros(1, self.input_size)
