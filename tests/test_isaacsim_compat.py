from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import site
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPAT_MODULE_PATH = REPO_ROOT / "scripts" / "himloco_rsl_rl" / "isaacsim_compat.py"
URDF_EXTENSION_NAME = "isaacsim.asset.importer.urdf"
GUI_EXPERIENCE_PATH = (
    REPO_ROOT / "scripts" / "himloco_rsl_rl" / "apps" / "isaaclab.python.isaacsim50.kit"
)


def _load_compat_module():
    spec = importlib.util.spec_from_file_location("test_isaacsim_compat_module", COMPAT_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_isaac_lab_extension_root(tmp_path: Path, monkeypatch) -> Path:
    extension_root = tmp_path / "IsaacLab" / "source"
    package_dir = extension_root / "isaaclab" / "isaaclab"
    package_dir.mkdir(parents=True)
    extension_config = extension_root / "isaaclab" / "config" / "extension.toml"
    extension_config.parent.mkdir()
    extension_config.touch()
    spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec if name == "isaaclab" else None)
    return extension_root


def _top_level_statement_index(tree: ast.Module, predicate) -> int | None:
    for index, statement in enumerate(tree.body):
        if predicate(statement):
            return index
    return None


def test_gui_experience_has_no_machine_specific_paths():
    experience = GUI_EXPERIENCE_PATH.read_text(encoding="utf-8")

    assert "/home/" not in experience


def test_gui_experience_uses_isaac_sim_50_runtime_settings():
    experience = GUI_EXPERIENCE_PATH.read_text(encoding="utf-8")
    app_settings = experience.split("[settings.app]", 1)[1]

    assert 'version = "5.0.0"' in app_settings
    assert "Assets/Isaac/5.1" not in experience
    assert experience.count("Assets/Isaac/5.0") == 3
    assert "skipPublishVerification = true" in experience


@pytest.mark.parametrize("entry_point", ["train.py", "play.py"])
def test_entry_point_configures_urdf_importer_before_app_launcher(entry_point: str):
    script_path = REPO_ROOT / "scripts" / "himloco_rsl_rl" / entry_point
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    compat_import_index = _top_level_statement_index(
        tree,
        lambda statement: isinstance(statement, ast.ImportFrom)
        and statement.module == "isaacsim_compat"
        and any(alias.name == "configure_isaacsim_urdf_importer" for alias in statement.names),
    )
    compat_call_index = _top_level_statement_index(
        tree,
        lambda statement: isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "configure_isaacsim_urdf_importer",
    )
    app_launcher_import_index = _top_level_statement_index(
        tree,
        lambda statement: isinstance(statement, ast.ImportFrom)
        and statement.module == "isaaclab.app"
        and any(alias.name == "AppLauncher" for alias in statement.names),
    )

    assert compat_import_index is not None
    assert compat_call_index is not None
    assert app_launcher_import_index is not None
    assert compat_import_index < compat_call_index < app_launcher_import_index


def test_isaac_sim_50_exposes_and_enables_pip_urdf_importer(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", ["/existing/python"])
    monkeypatch.setattr(sys, "argv", ["train.py", "--headless"])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/existing/one", "/existing/two"]))

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.path[0] == str(extension_path)
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [
        str(extension_path),
        "/existing/one",
        "/existing/two",
    ]
    assert sys.argv[-2:] == ["--kit_args", f"--enable {URDF_EXTENSION_NAME}"]


def test_isaac_sim_50_gui_selects_repository_experience(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py"])
    monkeypatch.delenv("PYTHONPATH", raising=False)
    _mock_isaac_lab_extension_root(tmp_path, monkeypatch)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.argv[-2:] == ["--experience", str(GUI_EXPERIENCE_PATH)]


def test_headless_environment_does_not_select_gui_experience(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py"])
    monkeypatch.setenv("HEADLESS", "1")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert not any(argument.startswith("--experience") for argument in sys.argv)


def test_livestream_uses_normal_gui_experience_even_when_headless(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py"])
    monkeypatch.setenv("HEADLESS", "1")
    monkeypatch.setenv("LIVESTREAM", "1")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    _mock_isaac_lab_extension_root(tmp_path, monkeypatch)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.argv[-2:] == ["--experience", str(GUI_EXPERIENCE_PATH)]


@pytest.mark.parametrize("livestream_args", [["--livestream", "1"], ["--livestream=2"]])
def test_livestream_cli_uses_normal_gui_experience_even_when_headless(
    livestream_args: list[str], tmp_path: Path, monkeypatch
):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py", "--headless", *livestream_args])
    monkeypatch.delenv("PYTHONPATH", raising=False)
    _mock_isaac_lab_extension_root(tmp_path, monkeypatch)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.argv[-2:] == ["--experience", str(GUI_EXPERIENCE_PATH)]


def test_livestream_cli_zero_overrides_environment(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py", "--headless", "--livestream", "0"])
    monkeypatch.setenv("LIVESTREAM", "1")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert not any(argument.startswith("--experience") for argument in sys.argv)


@pytest.mark.parametrize("special_option", ["--enable_cameras", "--xr", "--video"])
def test_special_rendering_modes_keep_upstream_experience(
    special_option: str, tmp_path: Path, monkeypatch
):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py", special_option])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert not any(argument.startswith("--experience") for argument in sys.argv)


@pytest.mark.parametrize("environment_name", ["ENABLE_CAMERAS", "XR"])
def test_special_rendering_environment_keeps_upstream_experience(
    environment_name: str, tmp_path: Path, monkeypatch
):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py"])
    monkeypatch.setenv(environment_name, "1")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert not any(argument.startswith("--experience") for argument in sys.argv)


def test_gui_adds_discovered_isaac_lab_extension_root(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)
    isaac_lab_extension_root = _mock_isaac_lab_extension_root(tmp_path, monkeypatch)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py"])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    kit_args_index = sys.argv.index("--kit_args")
    kit_args = sys.argv[kit_args_index + 1]
    assert "--/app/extensions/skipPublishVerification=true" in kit_args
    assert f"--/app/exts/folders/11={isaac_lab_extension_root}" in kit_args
    assert "--ext-folder" not in kit_args


def test_missing_isaac_lab_extension_root_warns_and_keeps_upstream_experience(
    tmp_path: Path, monkeypatch
):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py"])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    with pytest.warns(RuntimeWarning, match="Could not locate the active Isaac Lab extension root"):
        compat.configure_isaacsim_urdf_importer()

    assert not any(argument.startswith("--experience") for argument in sys.argv)


def test_explicit_experience_is_not_overridden(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py", "--experience", "/custom/gui.kit"])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.argv.count("--experience") == 1
    assert sys.argv[sys.argv.index("--experience") + 1] == "/custom/gui.kit"


def test_equals_form_experience_is_not_overridden(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py", "--experience=/custom/gui.kit"])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert [argument for argument in sys.argv if argument.startswith("--experience")] == [
        "--experience=/custom/gui.kit"
    ]


@pytest.mark.parametrize(
    "experience_args",
    [
        ["--experience", ""],
        ["--experience="],
        ["--experience", "/custom/gui.kit", "--experience="],
    ],
)
def test_effective_empty_experience_selects_repository_experience(
    experience_args: list[str], tmp_path: Path, monkeypatch
):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)
    _mock_isaac_lab_extension_root(tmp_path, monkeypatch)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py", *experience_args])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.argv[-2:] == ["--experience", str(GUI_EXPERIENCE_PATH)]


def test_missing_gui_experience_warns_and_leaves_selection_unchanged(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)
    missing_experience = tmp_path / "missing-gui.kit"

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py"])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    monkeypatch.setattr(compat, "_ISAAC_SIM_50_GUI_EXPERIENCE", missing_experience)
    with pytest.warns(RuntimeWarning, match=rf"Could not find.*GUI experience.*{missing_experience}"):
        compat.configure_isaacsim_urdf_importer()

    assert not any(argument.startswith("--experience") for argument in sys.argv)


def test_configuration_preserves_existing_state_and_is_idempotent(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)
    extension_path_str = str(extension_path)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", ["/existing/python", extension_path_str])
    monkeypatch.setattr(sys, "argv", ["train.py", "--headless", "--kit_args", "--foo=bar"])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/existing/one", extension_path_str, "/existing/two"]))

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()
    compat.configure_isaacsim_urdf_importer()

    assert sys.path[0] == extension_path_str
    assert sys.path.count(extension_path_str) == 1
    pythonpath_entries = os.environ["PYTHONPATH"].split(os.pathsep)
    assert pythonpath_entries == [extension_path_str, "/existing/one", "/existing/two"]
    assert sys.argv == [
        "train.py",
        "--headless",
        "--kit_args",
        f"--foo=bar --enable {URDF_EXTENSION_NAME}",
    ]


def test_configuration_preserves_empty_pythonpath_segments(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["train.py", "--headless"])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["", "/one", "", "/two", ""]))

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert os.environ["PYTHONPATH"].split(os.pathsep) == [
        str(extension_path),
        "",
        "/one",
        "",
        "/two",
        "",
    ]


def test_configuration_preserves_equals_form_kit_args(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["train.py", "--headless", "--kit_args=--foo=bar"])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()
    compat.configure_isaacsim_urdf_importer()

    assert sys.argv == [
        "train.py",
        "--headless",
        f"--kit_args=--foo=bar --enable {URDF_EXTENSION_NAME}",
    ]


def test_configuration_updates_effective_last_kit_args(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--headless", "--kit_args", "--first=1", "--kit_args=--second=2"],
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.argv == [
        "train.py",
        "--headless",
        "--kit_args",
        "--first=1",
        f"--kit_args=--second=2 --enable {URDF_EXTENSION_NAME}",
    ]


def test_isaac_sim_51_leaves_importer_selection_to_isaac_lab(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.1.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", ["/existing/python"])
    monkeypatch.setattr(sys, "argv", ["train.py", "--headless"])
    monkeypatch.setenv("PYTHONPATH", "/existing/one")

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.path == ["/existing/python"]
    assert os.environ["PYTHONPATH"] == "/existing/one"
    assert sys.argv == ["train.py", "--headless"]


def test_isaac_sim_45_does_not_select_isaac_sim_50_gui_experience(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    extension_path = site_packages / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "4.5.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["play.py"])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert not any(argument.startswith("--experience") for argument in sys.argv)


def test_unknown_isaac_sim_version_warns_and_leaves_state_unchanged(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "unknown")
    monkeypatch.setattr(sys, "path", ["/existing/python"])
    monkeypatch.setattr(sys, "argv", ["train.py", "--headless"])
    monkeypatch.setenv("PYTHONPATH", "/existing/one")

    compat = _load_compat_module()
    with pytest.warns(RuntimeWarning, match="Could not determine the installed Isaac Sim version"):
        compat.configure_isaacsim_urdf_importer()

    assert sys.path == ["/existing/python"]
    assert os.environ["PYTHONPATH"] == "/existing/one"
    assert sys.argv == ["train.py", "--headless"]


def test_missing_urdf_extension_warns_with_searched_root(tmp_path: Path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(sys, "path", ["/existing/python"])
    monkeypatch.setattr(sys, "argv", ["train.py", "--headless"])
    monkeypatch.setenv("PYTHONPATH", "/existing/one")

    compat = _load_compat_module()
    expected_warning = rf"Could not find {URDF_EXTENSION_NAME}.*{site_packages}"
    with pytest.warns(RuntimeWarning, match=expected_warning):
        compat.configure_isaacsim_urdf_importer()

    assert sys.path == ["/existing/python"]
    assert os.environ["PYTHONPATH"] == "/existing/one"
    assert sys.argv == ["train.py", "--headless"]


def test_user_site_packages_are_searched_for_urdf_extension(tmp_path: Path, monkeypatch):
    user_site = tmp_path / "user-site"
    extension_path = user_site / "isaacsim" / "exts" / URDF_EXTENSION_NAME
    extension_path.mkdir(parents=True)

    monkeypatch.setattr(importlib.metadata, "version", lambda distribution: "5.0.0.0")
    monkeypatch.setattr(site, "getsitepackages", lambda: [])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "missing-prefix"))
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "argv", ["train.py", "--headless"])
    monkeypatch.delenv("PYTHONPATH", raising=False)

    compat = _load_compat_module()
    compat.configure_isaacsim_urdf_importer()

    assert sys.path[0] == str(extension_path)
