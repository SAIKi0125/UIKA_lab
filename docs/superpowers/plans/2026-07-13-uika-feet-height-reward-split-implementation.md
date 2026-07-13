# UIKA Feet-Height Reward Split Implementation Plan

## Scope

Implement the approved design in the standard UIKA velocity reward configuration while preserving lower-task isolation and all unrelated training behavior.

## Steps

1. Add focused source-level tests that initially fail unless:
   - `RewardsCfg.feet_height_body` calls `mdp.feet_height_body` with weight `-0.01`, target `-0.20`, and `tanh_mult=2.0`;
   - `RewardsCfg.feet_lift_body` calls `mdp.feet_lift_body` with weight `+2.0`, minimum `-0.30`, target `-0.10`, and `tanh_mult=2.0`;
   - `LowerRewardsCfg` disables both inherited terms.
2. Run the focused tests and confirm they fail against the current misleading one-term configuration.
3. Split the standard velocity configuration into the two approved reward terms.
4. Add `feet_lift_body = None` to `LowerRewardsCfg` while retaining `feet_height_body = None`.
5. Run focused tests, the existing lower/velocity tests, Python compilation, and `git diff --check`.
6. Review the final diff to confirm no unrelated files or reward weights changed.

## Files

- Modify `source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/velocity_env_cfg.py`.
- Modify `source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/lower_env_cfg.py`.
- Add `tests/test_uika_feet_height_rewards.py`.

The existing `mdp.feet_height_body` and `mdp.feet_lift_body` implementations already match the approved split, so `mdp/rewards.py` requires no functional edit.
