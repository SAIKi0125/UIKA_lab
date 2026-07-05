import importlib.util
from pathlib import Path


def _load_export_policy_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "utils"
        / "export_policy.py"
    )
    spec = importlib.util.spec_from_file_location("export_policy", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onnx_export_can_be_skipped_when_onnx_is_missing(monkeypatch, tmp_path):
    export_policy = _load_export_policy_module()

    monkeypatch.setattr(export_policy.importlib.util, "find_spec", lambda name: None if name == "onnx" else object())

    exported = export_policy.export_himloco_policy_as_onnx(
        actor_critic=None,
        path=str(tmp_path),
        skip_if_unavailable=True,
    )

    assert exported is False
