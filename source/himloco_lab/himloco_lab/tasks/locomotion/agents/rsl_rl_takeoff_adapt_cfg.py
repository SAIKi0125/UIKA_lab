from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlMlpAdaptModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class UIKATakeoffPPOAdaptRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "uika_takeoff_adapt"
    empirical_normalization = False
    obs_groups = {"actor": ["actor_history"], "critic": ["critic"]}

    actor = RslRlMlpAdaptModelCfg(
        class_name="himloco_lab.tasks.locomotion.agents.takeoff_adapt_model:TakeoffMlpAdaptModel",
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        max_length=10,
        cmd_dim=3,
        latent_dim=32,
        privileged_target_key="privileged_target",
        privileged_target_dim=5,
        mlp_hidden_dims=[256, 128],
        actor_hidden_dims=[512, 256, 128],
    )

    critic = RslRlMLPModelCfg(
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        hidden_dims=[512, 256, 128],
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        adaptation_loss_coef=1.0,
    )
