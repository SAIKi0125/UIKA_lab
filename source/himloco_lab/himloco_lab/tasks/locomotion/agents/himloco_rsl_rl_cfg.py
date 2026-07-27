# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg, HIMPPPOAlgorithmCfg, HIMPPOActorCriticCfg

@configclass
class PPORunnerCfg(HIMOnPolicyRunnerCfg):
    num_steps_per_env = 100
    max_iterations = 15000
    save_interval = 100
    experiment_name = "go2_rough"
    history_length = 5
    policy = HIMPPOActorCriticCfg(
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        init_noise_std = 1.0,
        activation = 'elu', # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        normalize_obs = False,  # Enable observation normalization for numerical stability
    )
    algorithm = HIMPPPOAlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.01,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3, #5.e-4
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.0,
    )



@configclass
class UIKAPPORunnerCfg(HIMOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "uika"
    history_length = 5
    policy = HIMPPOActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        init_noise_std=1.0,
        activation='elu',
        normalize_obs=False,
    )
    algorithm = HIMPPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule='adaptive',
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UIKAFlatPPORunnerCfg(UIKAPPORunnerCfg):
    experiment_name = "uika_flat"


@configclass
class UIKALowerPPORunnerCfg(UIKAPPORunnerCfg):
    experiment_name = "uika_lower"


@configclass
class UIKARoughNormalGaitPPORunnerCfg(UIKAPPORunnerCfg):
    experiment_name = "uika_rough_normal_gait"
