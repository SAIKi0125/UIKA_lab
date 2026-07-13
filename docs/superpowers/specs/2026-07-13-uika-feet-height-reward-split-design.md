# UIKA Feet-Height Reward Split Design

## Goal

Give the standard UIKA velocity task two explicitly named, simultaneously active body-frame foot-height terms:

- `feet_height_body`: the original target-height error penalty from the `teacherstudent` branch.
- `feet_lift_body`: the current positive, capped lift-progress reward on `master`.

This change separates the two mathematical meanings without changing their implementations or folding sign behavior into a mode flag.

## Reward Definitions

The existing MDP functions remain independent:

- `mdp.feet_height_body` returns the squared body-frame foot-height error, weighted by body-frame horizontal foot speed and gated by a nonzero command. The velocity configuration applies weight `-0.01`, matching `origin/teacherstudent`, with `target_height=-0.20` and `tanh_mult=2.0`.
- `mdp.feet_lift_body` returns capped lift progress between `minimum_height=-0.30` and `target_height=-0.10`, weighted by body-frame horizontal foot speed and gated by a nonzero command. The velocity configuration retains the current positive weight `+2.0` and `tanh_mult=2.0`.

The configuration field names match the functions they call. The current misleading field named `feet_height_body` that calls `mdp.feet_lift_body` is replaced by two fields:

```python
feet_height_body = RewTerm(func=mdp.feet_height_body, ...)
feet_lift_body = RewTerm(func=mdp.feet_lift_body, ...)
```

## Task Isolation

`LowerRewardsCfg` inherits from the standard `RewardsCfg`. It already disables `feet_height_body`; it must also set `feet_lift_body = None`. This prevents the new standard-velocity positive term from leaking into the lower task through inheritance.

No Go2, parkour, action, observation, termination, deployment, checkpoint, or policy-export behavior is changed.

## Validation

Focused tests will verify:

- both MDP functions exist with distinct implementations;
- the standard UIKA velocity config registers both terms under matching names;
- the negative term uses weight `-0.01` and the teacher/student target parameters;
- the positive term preserves weight `+2.0` and the current lift-range parameters;
- `LowerRewardsCfg` disables both terms;
- the edited Python files compile successfully.

## Non-goals

This split does not claim to fix the observed velocity-policy crouching by itself. In particular, it does not enable `base_height_l2`, change the 16x moving/standing posture scale, tune either foot-height weight, or retrain/export a policy. Those behavioral changes require a separate reward-tuning decision.
