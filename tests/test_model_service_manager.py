from __future__ import annotations

from pathlib import Path
import signal
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import model_service_manager as manager  # noqa: E402


def test_service_specs_expose_exact_port_gpu_and_interpreter_mapping(monkeypatch):
    monkeypatch.setenv("SONGFORMER_SERVICE_PORT", "12001")
    monkeypatch.setenv("SECTION_ASR_SERVICE_PORT", "12002")
    monkeypatch.setenv("SONGFORMER_SERVICE_GPU", "3")
    monkeypatch.setenv("SECTION_ASR_SERVICE_GPU", "5")
    monkeypatch.setenv("PIPELINE_PYTHON", "/env/pipeline/python")
    monkeypatch.setenv("QWEN_PYTHON", "/env/qwen/python")

    specs = manager.service_specs("80")

    assert tuple(specs) == manager.SERVICE_ORDER
    assert (specs["songformer"].port, specs["songformer"].gpu) == (12001, "3")
    assert str(specs["songformer"].python) == "/env/pipeline/python"
    assert (specs["section-asr"].port, specs["section-asr"].gpu) == (12002, "5")
    assert str(specs["section-asr"].python) == "/env/qwen/python"
    assert specs["omni"].port == 10103
    assert specs["omni"].device_arg is False


def _spec(name="test", gpu="0"):
    return manager.ServiceSpec(
        name,
        "127.0.0.1",
        1234,
        gpu,
        8.0,
        "24",
        Path(sys.executable),
        Path(f"/x/{name}.py"),
    )


def test_managed_process_requires_pid_start_ticks_and_exact_script(monkeypatch):
    spec = _spec()
    state = {"pid": 42, "proc_start_ticks": "123"}
    monkeypatch.setattr(manager, "_proc_start_ticks", lambda pid: "123")
    monkeypatch.setattr(
        manager, "_proc_cmdline", lambda pid: [sys.executable, "/x/test.py"]
    )
    assert manager._managed_process(state, spec)

    monkeypatch.setattr(manager, "_proc_start_ticks", lambda pid: "124")
    assert not manager._managed_process(state, spec)
    monkeypatch.setattr(manager, "_proc_start_ticks", lambda pid: "123")
    monkeypatch.setattr(manager, "_proc_cmdline", lambda pid: ["other"])
    assert not manager._managed_process(state, spec)


def test_stop_does_not_signal_a_reused_or_mismatched_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_SERVICE_STATE_DIR", str(tmp_path))
    spec = _spec()
    manager._atomic_write_json(
        manager._state_path(spec), {"pid": 42, "proc_start_ticks": "old"}
    )
    monkeypatch.setattr(manager, "_proc_start_ticks", lambda pid: "new")
    calls = []
    monkeypatch.setattr(manager.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(manager, "_http_health", lambda url: (False, "down"))

    result = manager.stop_one(spec, timeout=0.01)

    assert calls == []
    assert result["state"] == "stopped"
    assert not manager._state_path(spec).exists()


def test_stop_signals_only_the_identity_captured_in_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_SERVICE_STATE_DIR", str(tmp_path))
    spec = _spec()
    manager._atomic_write_json(
        manager._state_path(spec), {"pid": 42, "proc_start_ticks": "123"}
    )
    alive = {"value": True}
    monkeypatch.setattr(
        manager,
        "_proc_start_ticks",
        lambda pid: "123" if alive["value"] else None,
    )
    monkeypatch.setattr(
        manager, "_proc_cmdline", lambda pid: [sys.executable, "/x/test.py"]
    )
    calls = []

    def fake_kill(pid, sig):
        calls.append((pid, sig))
        alive["value"] = False

    monkeypatch.setattr(manager.os, "kill", fake_kill)
    monkeypatch.setattr(manager, "_http_health", lambda url: (False, "down"))
    result = manager.stop_one(spec, timeout=0.1)

    assert calls == [(42, signal.SIGTERM)]
    assert result["state"] == "stopped"


def test_omni_upstream_status_is_health_check_only(monkeypatch):
    monkeypatch.setenv("OMNI_UPSTREAM_SERVER", "http://omni.example:10008")
    monkeypatch.setattr(manager, "_http_health", lambda url: (True, "{}"))
    result = manager.omni_upstream_status()
    assert result["managed"] is False
    assert result["healthy"] is True
    assert result["name"] == "omni-upstream"
    assert result["health_url"] == "http://omni.example:10008/v1/models"


@pytest.mark.parametrize("profile", ["24", "48", "80"])
def test_memory_profiles_cover_every_managed_service(profile, monkeypatch):
    monkeypatch.delenv("MODEL_SERVICE_MEMORY_PROFILE", raising=False)
    specs = manager.service_specs(profile)
    assert tuple(specs) == manager.SERVICE_ORDER
    assert all(spec.profile == profile for spec in specs.values())
    assert all(0 < spec.memory_gib <= float(profile) for spec in specs.values())


def test_each_service_has_independent_environment_and_cpu_is_gpu_hidden(monkeypatch):
    monkeypatch.setenv("FAST_GATE_SERVICE_GPU", "1")
    monkeypatch.setenv("DISCOGS_SERVICE_GPU", "2")
    monkeypatch.setenv("SONGFORMER_SERVICE_GPU", "3")
    monkeypatch.setenv("SECTION_ASR_SERVICE_GPU", "4")
    monkeypatch.setenv("SONGFORMER_SERVICE_MEMORY_GIB", "9")
    specs = manager.service_specs("24")

    assert [specs[name].gpu for name in manager.SERVICE_ORDER] == [
        "1",
        "2",
        "cpu",
        "3",
        "4",
        "external",
    ]
    assert specs["songformer"].memory_gib == 9
    for name, spec in specs.items():
        env = manager._child_environment(spec)
        expected_gpu = "" if name in {"cpu-mir", "omni"} else spec.gpu
        expected_quota = (
            "0" if name in {"cpu-mir", "omni"} else f"{spec.memory_gib:g}"
        )
        assert env["CUDA_VISIBLE_DEVICES"] == expected_gpu
        assert env["MODEL_SERVICE_MEMORY_QUOTA_GIB"] == f"{spec.memory_gib:g}"
        assert env["PIPELINE_TORCH_GPU_MAX_MEMORY_GIB"] == expected_quota
        assert env["PIPELINE_ORT_GPU_MAX_MEMORY_GIB"] == expected_quota
    assert specs["cpu-mir"].device_arg is False
    assert specs["omni"].device_arg is False


def test_all_selection_and_individual_selection_are_complete(monkeypatch):
    specs = manager.service_specs("80")
    assert [value.name for value in manager._selected_specs("all", specs)] == list(
        manager.SERVICE_ORDER
    )
    for name in manager.SERVICE_ORDER:
        assert manager._selected_specs(name, specs) == [specs[name]]

    monkeypatch.setenv("MODEL_SERVICE_ALL_INCLUDE_OMNI", "0")
    assert [value.name for value in manager._selected_specs("all", specs)] == list(
        manager.CORE_SERVICE_ORDER
    )


@pytest.mark.parametrize("action", ["start", "status", "stop", "restart"])
def test_all_commands_dispatch_every_managed_service(action, monkeypatch):
    specs = {name: _spec(name) for name in manager.SERVICE_ORDER}
    starts = []
    stops = []
    statuses = []
    monkeypatch.setattr(manager, "service_specs", lambda profile: specs)
    monkeypatch.setattr(
        manager,
        "start_one",
        lambda spec, wait, no_wait: starts.append(spec.name)
        or {"name": spec.name, "healthy": True, "state": "ready"},
    )
    monkeypatch.setattr(
        manager,
        "stop_one",
        lambda spec, timeout: stops.append(spec.name)
        or {"name": spec.name, "healthy": False, "state": "stopped"},
    )
    monkeypatch.setattr(
        manager,
        "status_one",
        lambda spec: statuses.append(spec.name)
        or {"name": spec.name, "healthy": True, "state": "ready"},
    )
    monkeypatch.setattr(
        manager,
        "omni_upstream_status",
        lambda: {
            "name": "omni-upstream",
            "healthy": True,
            "state": "external-ready",
        },
    )
    monkeypatch.setattr(manager, "_print_status", lambda values, as_json: None)
    monkeypatch.setattr(sys, "argv", ["manager", action, "all", "--no-wait"])

    manager.main()

    if action in {"start", "restart"}:
        assert starts == list(manager.SERVICE_ORDER)
    if action == "stop":
        assert stops == list(reversed(manager.SERVICE_ORDER))
    elif action == "restart":
        assert stops == list(manager.SERVICE_ORDER)
    if action == "status":
        assert statuses == list(manager.SERVICE_ORDER)


@pytest.mark.parametrize("action", ["start", "status", "stop", "restart"])
def test_omni_proxy_can_be_selected_for_each_lifecycle_command(action, monkeypatch):
    specs = {name: _spec(name) for name in manager.SERVICE_ORDER}
    calls = []
    monkeypatch.setattr(manager, "service_specs", lambda profile: specs)
    monkeypatch.setattr(
        manager,
        "start_one",
        lambda spec, wait, no_wait: calls.append(("start", spec.name))
        or {"name": spec.name, "healthy": True, "state": "ready"},
    )
    monkeypatch.setattr(
        manager,
        "stop_one",
        lambda spec, timeout: calls.append(("stop", spec.name))
        or {"name": spec.name, "healthy": False, "state": "stopped"},
    )
    monkeypatch.setattr(
        manager,
        "status_one",
        lambda spec: calls.append(("status", spec.name))
        or {"name": spec.name, "healthy": True, "state": "ready"},
    )
    monkeypatch.setattr(
        manager,
        "omni_upstream_status",
        lambda: {
            "name": "omni-upstream",
            "healthy": True,
            "state": "external-ready",
        },
    )
    monkeypatch.setattr(manager, "_print_status", lambda values, as_json: None)
    monkeypatch.setattr(sys, "argv", ["manager", action, "omni", "--no-wait"])

    manager.main()

    expected = {
        "start": [("start", "omni")],
        "status": [("status", "omni")],
        "stop": [("stop", "omni")],
        "restart": [("stop", "omni"), ("start", "omni")],
    }
    assert calls == expected[action]
