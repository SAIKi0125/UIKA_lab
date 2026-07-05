import gymnasium as gym


gym.register(
    id="UIKA-Parkour-Velocity",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotParkourEnvCfg",
        "himloco_rsl_rl_cfg": "himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:UIKAPPORunnerCfg",
        "rsl_rl_cfg_entry_point": "himloco_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="UIKA-Parkour-Velocity-Play",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotParkourPlayEnvCfg",
        "himloco_rsl_rl_cfg": "himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:UIKAPPORunnerCfg",
    },
)
