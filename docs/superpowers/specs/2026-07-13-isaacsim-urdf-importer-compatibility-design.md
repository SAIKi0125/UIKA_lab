# Isaac Sim URDF Importer Startup Compatibility Design

## Context

UIKA tasks spawn the robot from a URDF file. In the current `isaac_lab_50` Conda environment, Isaac Sim is installed from pip packages at version `5.0.0.0`. The URDF importer extension exists at `isaacsim/exts/isaacsim.asset.importer.urdf`, but the headless Isaac Lab experience does not expose its Python module before `UrdfConverter` imports `isaacsim.asset.importer.urdf._urdf`. UIKA training therefore stops with `ModuleNotFoundError: No module named 'isaacsim.asset'` during scene creation.

An explicit command-line workaround that adds the extension directory to `PYTHONPATH` and enables the Kit extension passes the original import point. The permanent solution should provide the same behavior inside this repository without modifying the external Isaac Lab checkout or reinstalling Isaac Sim.

Headless train and play then pass, but default GUI play uses the external `isaaclab.python.kit` experience. Its `isaaclab.python-2.3.2` dependency requires URDF importer 2.4.31 exactly, while this Isaac Sim 5.0 installation only provides 2.4.19. Kit exits during dependency resolution before `SimulationApp` is usable. The repository therefore also needs a version-matched GUI experience for Isaac Sim 5.0.

## Goals

- Make HimLoco train and play entry points automatically load the pip-installed URDF importer on Isaac Sim 5.0.
- Require no manual `PYTHONPATH`, `--kit_args`, or extension commands from users.
- Make default GUI play work without a manual `--experience` argument on Isaac Sim 5.0.
- Preserve Isaac Lab's importer-version selection on Isaac Sim 5.1 and newer.
- Keep configuration idempotent and preserve user-provided launch arguments.
- Cover the compatibility behavior with tests that run without Isaac Sim, Kit, or a GPU.

## Non-goals

- Reinstalling or upgrading the `isaac_lab_50` environment.
- Modifying files in the external Isaac Lab checkout.
- Converting UIKA permanently from URDF to a checked-in USD asset.
- Fixing Warp driver warnings or CPU power-profile warnings that do not cause the reported importer failure.
- Generalizing the helper to unrelated Isaac Sim extensions before a concrete need appears.

## Considered Approaches

### Repository compatibility module plus version-matched GUI experience — selected

Add a small pure-Python module that runs before `AppLauncher` is imported. It discovers the pip extension location, exposes its Python module, asks Kit to enable it, and selects a repository-owned GUI experience for Isaac Sim 5.0. This keeps external Isaac Lab files untouched and makes the version decision testable without launching Kit.

### Modify the external Isaac Lab experience

Change `/home/esd/project/HuangCang-project/IsaacLab/apps/isaaclab.python.kit` to accept importer 2.4.19. This is smaller than carrying a local experience, but modifies an external dependency checkout and would be lost or conflict when Isaac Lab is updated.

### Preconverted UIKA USD

Replace runtime URDF conversion with a checked-in USD. This avoids importer startup entirely, but introduces a second robot source of truth and risks stale USD assets after URDF or collision-model changes.

## Architecture

Create `scripts/himloco_rsl_rl/isaacsim_compat.py` with one public entry point:

```python
configure_isaacsim_urdf_importer()
```

Both `train.py` and `play.py` call this function after importing Python standard-library modules and before importing `isaaclab.app.AppLauncher`. No Isaac Sim, Omni, or Isaac Lab module may be imported by the compatibility module.

The module has focused private helpers for:

- discovering existing site-package roots;
- reading the installed Isaac Sim distribution version without importing Isaac Sim;
- locating the URDF importer extension directory;
- prepending one path to `sys.path` and `PYTHONPATH` without duplication;
- adding the extension to AppLauncher's `--kit_args` without overwriting existing Kit arguments;
- selecting a repository-owned Isaac Sim 5.0 GUI experience without overriding headless or explicit experience choices.

Add `scripts/himloco_rsl_rl/apps/isaaclab.python.isaacsim50.kit`, adapted from the external `isaaclab.python.kit`. It keeps the same GUI and Isaac Lab extension dependencies but requests the installed `isaacsim.asset.importer.urdf` without the unavailable exact 2.4.31 constraint. The file is intentionally version-specific; upgrades to the external Isaac Lab GUI experience must be compared against this local copy.

## Version Policy

For Isaac Sim versions earlier than 5.1, the helper exposes and enables the installed unversioned extension `isaacsim.asset.importer.urdf`. This covers the current pip 5.0 environment, whose importer extension is version 2.4.19.

For a non-headless Isaac Sim version earlier than 5.1, the helper also appends `--experience <repository Isaac Sim 5.0 GUI experience>` when no experience was supplied. Headless launches retain Isaac Lab's headless experience. An explicit `--experience` or `--experience=<path>` always wins.

For Isaac Sim 5.1 and newer, the helper does not activate the unversioned importer. The installed Isaac Lab `UrdfConverter` is responsible for selecting its compatible importer version, currently 2.4.31. Avoiding early activation prevents the repository helper from defeating that upstream version pin.

If the Isaac Sim distribution version cannot be determined, the helper leaves launch state unchanged and emits one concise warning. If the version requires the compatibility path but the extension directory is absent, it likewise leaves state unchanged and emits a warning containing the extension name and searched site-package roots. The later converter error remains authoritative; Go2 or other USD-only tasks must not be blocked by a missing URDF extension.

## Launch Argument Behavior

When no `--kit_args` argument exists, configuration appends:

```text
--kit_args "--enable isaacsim.asset.importer.urdf"
```

When `--kit_args` already has a value, the helper appends the enable fragment to that value. When the extension is already present in the Kit argument value, configuration makes no change. Repeated calls are therefore safe.

The helper updates both `sys.path` and `PYTHONPATH`: `sys.path` covers the current Python process, while `PYTHONPATH` preserves visibility for Python environments initialized by Kit during startup. Existing entries and their order are preserved after the newly required extension path.

For a default GUI launch on Isaac Sim 5.0, configuration also appends:

```text
--experience scripts/himloco_rsl_rl/apps/isaaclab.python.isaacsim50.kit
```

The helper resolves this to an absolute path derived from its own file location. It does not depend on the current working directory.

## Data Flow

1. The user starts `train.py` or `play.py` normally.
2. The entry point invokes `configure_isaacsim_urdf_importer()`.
3. The helper reads distribution metadata and determines whether the Isaac Sim 5.0 compatibility path applies.
4. The helper locates the installed importer extension and updates Python and Kit launch state idempotently.
5. For an Isaac Sim 5.0 GUI launch without an explicit experience, the helper selects the repository-owned compatible GUI experience.
6. `AppLauncher` starts Kit with the importer enabled and a satisfiable GUI dependency graph.
7. UIKA's `UrdfFileCfg` creates `UrdfConverter`, whose `_urdf` import now resolves normally.

## Testing Strategy

Tests use temporary site-package directory trees and monkeypatched process state; they do not import Isaac Sim or require a GPU.

Required regression cases:

- Isaac Sim 5.0 plus an installed importer prepends the extension path and adds the Kit enable argument.
- Existing `PYTHONPATH` entries remain present and in their original relative order.
- Existing `--kit_args` content is preserved when the enable fragment is appended.
- Repeated configuration does not duplicate paths or arguments.
- Isaac Sim 5.1 or newer leaves paths and Kit arguments unchanged.
- An unknown version or missing extension returns safely and emits one actionable warning.
- Static entry-point tests prove train and play invoke compatibility setup before importing `AppLauncher`.
- Isaac Sim 5.0 GUI launch selects the repository-owned experience.
- Headless launch does not select the GUI experience.
- Explicit `--experience <path>` and `--experience=<path>` remain unchanged.
- Isaac Sim 5.1 or newer does not select the Isaac Sim 5.0 experience.
- A missing repository GUI experience emits an actionable warning and leaves AppLauncher selection unchanged.

After unit tests pass, a one-environment, one-iteration `UIKA-Flat-Velocity` startup and a default GUI `UIKA-Flat-Velocity-Play` startup are the integration smoke tests. In restricted environments where GPU/NVML or a display is unavailable, integration results must distinguish infrastructure failure from importer failure. Final acceptance requires the user's normal terminal environment to reach training and visible playback without importer or Kit dependency-resolution errors.

## Acceptance Criteria

- The normal training command works without manual environment variables or Kit arguments:

  ```bash
  python scripts/himloco_rsl_rl/train.py --task UIKA-Flat-Velocity --headless
  ```

- The normal play entry point receives the same compatibility setup.
- The default GUI play command reaches visible playback without manual compatibility arguments:

  ```bash
  python scripts/himloco_rsl_rl/play.py --task UIKA-Flat-Velocity-Play
  ```

- `--headless` and explicit `--experience` choices are never overridden.
- All new unit tests and relevant existing tests pass.
- Isaac Sim 5.1+ behavior remains delegated to Isaac Lab's version pin.
- No external Isaac Lab or Conda-environment file is modified.
