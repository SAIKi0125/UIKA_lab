"""Compatibility setup for pip-based Isaac Sim environments."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except AttributeError:
        pass

    roots.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for root in roots:
        if root not in seen and root.exists():
            seen.add(root)
            unique_roots.append(root)
    return unique_roots


def _find_isaacsim_extension(extension_name: str) -> Path | None:
    for root in _site_package_roots():
        extension_path = root / "isaacsim" / "exts" / extension_name
        if extension_path.exists():
            return extension_path
    return None


def _prepend_python_path(path: Path):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [entry for entry in current_pythonpath.split(os.pathsep) if entry]
    if path_str not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([path_str, *pythonpath_entries])


def _ensure_kit_arg_enabled(extension_name: str):
    enable_fragment = f"--enable {extension_name}"

    if "--kit_args" in sys.argv:
        kit_args_index = sys.argv.index("--kit_args")
        value_index = kit_args_index + 1
        if value_index < len(sys.argv) and not sys.argv[value_index].startswith("--"):
            if enable_fragment not in sys.argv[value_index] and extension_name not in sys.argv[value_index].split():
                sys.argv[value_index] = f"{sys.argv[value_index]} {enable_fragment}".strip()
        else:
            sys.argv.insert(value_index, enable_fragment)
        return

    sys.argv.extend(["--kit_args", enable_fragment])


def _has_cli_option(option_name: str) -> bool:
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in sys.argv[1:])


def _is_headless_launch() -> bool:
    env_headless = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")
    return env_headless or _has_cli_option("--headless")


def _ensure_isaacsim50_experience():
    if _has_cli_option("--experience") or _is_headless_launch():
        return

    experience_path = Path(__file__).resolve().parent / "apps" / "isaaclab.python.isaacsim50.kit"
    if experience_path.exists():
        sys.argv.extend(["--experience", str(experience_path)])


def configure_isaacsim_pip_extensions():
    """Make pip Isaac Sim extension modules available before AppLauncher starts Kit."""

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

    for extension_name in ("isaacsim.asset.importer.urdf", "isaacsim.asset.importer.mjcf"):
        extension_path = _find_isaacsim_extension(extension_name)
        if extension_path is not None:
            _prepend_python_path(extension_path)

    if _find_isaacsim_extension("isaacsim.asset.importer.urdf") is not None:
        _ensure_kit_arg_enabled("isaacsim.asset.importer.urdf")
        _ensure_isaacsim50_experience()
