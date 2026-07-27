"""Compatibility setup for pip-installed Isaac Sim extensions."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import re
import site
import sys
import warnings

_URDF_EXTENSION_NAME = "isaacsim.asset.importer.urdf"
_ISAAC_SIM_UPSTREAM_PIN_VERSION = (5, 1)
_ISAAC_SIM_50_GUI_EXPERIENCE = Path(__file__).resolve().parent / "apps" / "isaaclab.python.isaacsim50.kit"


def _major_minor(raw_version: str) -> tuple[int, int]:
    match = re.match(r"\s*(\d+)\.(\d+)", raw_version)
    if match is None:
        raise ValueError(f"Unrecognized Isaac Sim version: {raw_version!r}")
    return int(match.group(1)), int(match.group(2))


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except AttributeError:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except AttributeError:
        pass

    roots.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")
    return list(dict.fromkeys(root for root in roots if root.exists()))


def _find_urdf_extension(roots: list[Path]) -> Path | None:
    for root in roots:
        extension_path = root / "isaacsim" / "exts" / _URDF_EXTENSION_NAME
        if extension_path.exists():
            return extension_path
    return None


def _prepend_python_path(extension_path: Path) -> None:
    path = str(extension_path)
    sys.path[:] = [path, *(entry for entry in sys.path if entry != path)]

    pythonpath = os.environ.get("PYTHONPATH")
    entries = [] if pythonpath is None else pythonpath.split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([path, *(entry for entry in entries if entry != path)])


def _append_effective_kit_arg(fragment: str, identity: str) -> None:
    for index in range(len(sys.argv) - 1, -1, -1):
        argument = sys.argv[index]
        if argument == "--kit_args":
            value_index = index + 1
            if value_index == len(sys.argv):
                sys.argv.append(fragment)
            elif identity not in sys.argv[value_index]:
                sys.argv[value_index] = f"{sys.argv[value_index]} {fragment}".strip()
            return
        if argument.startswith("--kit_args="):
            if identity not in argument:
                sys.argv[index] = f"{argument} {fragment}"
            return
    sys.argv.extend(["--kit_args", fragment])


def _enable_kit_extension() -> None:
    _append_effective_kit_arg(f"--enable {_URDF_EXTENSION_NAME}", _URDF_EXTENSION_NAME)


def _has_cli_option(option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in sys.argv)


def _last_cli_option_value(option: str) -> str | None:
    value: str | None = None
    for index, argument in enumerate(sys.argv):
        if argument == option and index + 1 < len(sys.argv):
            value = sys.argv[index + 1]
        elif argument.startswith(f"{option}="):
            value = argument.split("=", 1)[1]
    return value


def _find_isaac_lab_extension_root() -> Path | None:
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.submodule_search_locations is None:
        return None
    for location in spec.submodule_search_locations:
        package_dir = Path(location).resolve()
        for candidate in package_dir.parents:
            if (candidate / "isaaclab" / "config" / "extension.toml").is_file():
                return candidate
    return None


def _select_isaac_sim_50_gui_experience() -> None:
    headless_requested = _has_cli_option("--headless") or os.environ.get("HEADLESS") == "1"
    livestream_value = _last_cli_option_value("--livestream")
    if livestream_value is None:
        livestream_value = os.environ.get("LIVESTREAM")
    livestream_requested = livestream_value in {"1", "2"}
    specialized_experience_requested = any(
        _has_cli_option(option) for option in ("--enable_cameras", "--xr", "--video")
    ) or any(os.environ.get(name) == "1" for name in ("ENABLE_CAMERAS", "XR"))
    experience_value = _last_cli_option_value("--experience")
    if (
        (headless_requested and not livestream_requested)
        or specialized_experience_requested
        or experience_value not in {None, ""}
    ):
        return
    if not _ISAAC_SIM_50_GUI_EXPERIENCE.is_file():
        warnings.warn(
            f"Could not find the Isaac Sim 5.0 GUI experience: {_ISAAC_SIM_50_GUI_EXPERIENCE}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    isaac_lab_extension_root = _find_isaac_lab_extension_root()
    if isaac_lab_extension_root is None:
        warnings.warn(
            "Could not locate the active Isaac Lab extension root; the Isaac Sim 5.0 GUI experience was not selected.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    _append_effective_kit_arg(
        "--/app/extensions/skipPublishVerification=true",
        "skipPublishVerification",
    )
    _append_effective_kit_arg(
        f"--/app/exts/folders/11={isaac_lab_extension_root}",
        str(isaac_lab_extension_root),
    )
    sys.argv.extend(["--experience", str(_ISAAC_SIM_50_GUI_EXPERIENCE)])


def configure_isaacsim_urdf_importer() -> None:
    """Configure URDF importer availability before Isaac Lab starts Kit."""

    try:
        isaac_sim_version = _major_minor(version("isaacsim"))
    except (PackageNotFoundError, ValueError):
        warnings.warn(
            "Could not determine the installed Isaac Sim version; URDF importer compatibility was not applied.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if isaac_sim_version >= _ISAAC_SIM_UPSTREAM_PIN_VERSION:
        return

    roots = _site_package_roots()
    extension_path = _find_urdf_extension(roots)
    if extension_path is None:
        searched_roots = ", ".join(str(root) for root in roots) or "<none>"
        warnings.warn(
            f"Could not find {_URDF_EXTENSION_NAME} in site-package roots: {searched_roots}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    _prepend_python_path(extension_path)
    _enable_kit_extension()
    if isaac_sim_version == (5, 0):
        _select_isaac_sim_50_gui_experience()
