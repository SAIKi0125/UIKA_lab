import importlib.util
import sys
from pathlib import Path


def _load_isaacsim_compat_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "himloco_rsl_rl" / "isaacsim_compat.py"
    spec = importlib.util.spec_from_file_location("isaacsim_compat", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gui_launch_uses_local_isaacsim50_experience(monkeypatch, tmp_path):
    compat = _load_isaacsim_compat_module()
    extension_path = tmp_path / "isaacsim.asset.importer.urdf"

    monkeypatch.setattr(sys, "argv", ["play.py", "--task", "UIKA-Velocity-Play"])
    monkeypatch.setattr(compat, "_find_isaacsim_extension", lambda name: extension_path if name.endswith("urdf") else None)

    compat.configure_isaacsim_pip_extensions()

    assert "--experience" in sys.argv
    experience_path = Path(sys.argv[sys.argv.index("--experience") + 1])
    assert experience_path.name == "isaaclab.python.isaacsim50.kit"
