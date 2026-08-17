#!/usr/bin/env python3
"""Start, inspect, stop, and restart resident pipeline model services."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_PYTHON = Path(
    "/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/"
    "moss-music-pipeline/bin/python"
)
DEFAULT_QWEN_PYTHON = Path(
    "/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/qwen3-vllm/bin/python"
)


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    host: str
    port: int
    gpu: str
    memory_gib: float
    profile: str
    python: Path
    script: Path
    extra_args: tuple[str, ...] = ()
    device_arg: bool = True


CORE_SERVICE_ORDER = (
    "fast-gate",
    "discogs",
    "cpu-mir",
    "songformer",
    "section-asr",
)
SERVICE_ORDER = (*CORE_SERVICE_ORDER, "omni")
MEMORY_PROFILES: dict[str, dict[str, float]] = {
    "24": {
        "fast-gate": 3.0,
        "discogs": 4.0,
        "cpu-mir": 12.0,
        "songformer": 10.0,
        "section-asr": 24.0,
        "omni": 1.0,
    },
    "48": {
        "fast-gate": 4.0,
        "discogs": 6.0,
        "cpu-mir": 16.0,
        "songformer": 20.0,
        "section-asr": 48.0,
        "omni": 1.0,
    },
    "80": {
        "fast-gate": 4.0,
        "discogs": 8.0,
        "cpu-mir": 24.0,
        "songformer": 32.0,
        "section-asr": 80.0,
        "omni": 1.0,
    },
}


def state_root() -> Path:
    configured = os.environ.get("MODEL_SERVICE_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(f"/tmp/music-data-pipeline-model-services-{os.getuid()}")


def _env_float(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    try:
        float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error
    return value


def _profile_memory(profile: str, service: str) -> float:
    env_name = f"{service.upper().replace('-', '_')}_SERVICE_MEMORY_GIB"
    value = float(os.environ.get(env_name, MEMORY_PROFILES[profile][service]))
    if not 0 < value <= float(profile):
        raise ValueError(
            f"{env_name} must be positive and no greater than profile {profile}GiB"
        )
    return value


def _bounded_asr_batch_size() -> str:
    value = int(os.environ.get("ASR_BATCH_SIZE", "4"))
    if value <= 0:
        raise ValueError("ASR_BATCH_SIZE must be positive")
    return str(min(value, 4))


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def all_service_order() -> tuple[str, ...]:
    if _env_flag("MODEL_SERVICE_ALL_INCLUDE_OMNI", True):
        return SERVICE_ORDER
    return CORE_SERVICE_ORDER


def service_specs(profile: str | None = None) -> dict[str, ServiceSpec]:
    selected_profile = str(
        profile or os.environ.get("MODEL_SERVICE_MEMORY_PROFILE", "80")
    )
    if selected_profile not in MEMORY_PROFILES:
        raise ValueError("MODEL_SERVICE_MEMORY_PROFILE must be one of 24, 48, 80")
    song_python = Path(
        os.environ.get("PIPELINE_PYTHON", str(DEFAULT_PIPELINE_PYTHON))
    )
    qwen_python = Path(os.environ.get("QWEN_PYTHON", str(DEFAULT_QWEN_PYTHON)))
    asr_model = os.environ.get(
        "QWEN3_ASR_MODEL_PATH",
        "/mnt/data/yuyin/datasets/jinwenqing/work/JwqMusic/Evaluation/"
        "Qwen3-ASR-main/Qwen/Qwen3-ASR-1.7B",
    )
    aligner_model = os.environ.get(
        "QWEN3_ALIGNER_MODEL_PATH",
        "/mnt/data/yuyin/datasets/jinwenqing/work/JwqMusic/Evaluation/"
        "Qwen3-ASR-main/Qwen/Qwen3-ForcedAligner-0.6B",
    )
    common_host = os.environ.get("MODEL_SERVICE_HOST", "127.0.0.1")
    values = {
        "fast-gate": ServiceSpec(
            name="fast-gate",
            host=os.environ.get("FAST_GATE_SERVICE_HOST", common_host),
            port=int(os.environ.get("FAST_GATE_SERVICE_PORT", "18101")),
            gpu=os.environ.get("FAST_GATE_SERVICE_GPU", "0"),
            memory_gib=_profile_memory(selected_profile, "fast-gate"),
            profile=selected_profile,
            python=song_python,
            script=ROOT / "scripts" / "serve_fast_gate.py",
            extra_args=(
                "--decode-workers",
                os.environ.get("FAST_GATE_DECODE_WORKERS", "16"),
            ),
        ),
        "discogs": ServiceSpec(
            name="discogs",
            host=os.environ.get("DISCOGS_SERVICE_HOST", common_host),
            port=int(os.environ.get("DISCOGS_SERVICE_PORT", "18102")),
            gpu=os.environ.get("DISCOGS_SERVICE_GPU", "0"),
            memory_gib=_profile_memory(selected_profile, "discogs"),
            profile=selected_profile,
            python=song_python,
            script=ROOT / "scripts" / "serve_discogs.py",
            extra_args=(
                "--decode-workers",
                os.environ.get("DISCOGS_DECODE_WORKERS", "8"),
                "--decode-prefetch",
                os.environ.get("DISCOGS_DECODE_PREFETCH", "16"),
                "--frame-batch-size",
                os.environ.get("DISCOGS_BATCH_SIZE", "512"),
                "--buffered-frames",
                os.environ.get("DISCOGS_BUFFERED_FRAMES", "2048"),
                "--request-batch-size",
                os.environ.get("DISCOGS_SERVICE_REQUEST_BATCH_SIZE", "64"),
            ),
        ),
        "cpu-mir": ServiceSpec(
            name="cpu-mir",
            host=os.environ.get("CPU_MIR_SERVICE_HOST", common_host),
            port=int(os.environ.get("CPU_MIR_SERVICE_PORT", "18103")),
            gpu="cpu",
            memory_gib=_profile_memory(selected_profile, "cpu-mir"),
            profile=selected_profile,
            python=song_python,
            script=ROOT / "scripts" / "serve_cpu_mir.py",
            device_arg=False,
        ),
        "songformer": ServiceSpec(
            name="songformer",
            host=os.environ.get("SONGFORMER_SERVICE_HOST", common_host),
            port=int(os.environ.get("SONGFORMER_SERVICE_PORT", "10101")),
            gpu=os.environ.get("SONGFORMER_SERVICE_GPU", "0"),
            memory_gib=_profile_memory(selected_profile, "songformer"),
            profile=selected_profile,
            python=song_python,
            script=ROOT / "scripts" / "serve_songformer.py",
            extra_args=(
                "--embedding-chunk-batch-size",
                os.environ.get("SONGFORMER_EMBEDDING_BATCH_SIZE", "1"),
                "--max-batch-size",
                os.environ.get("SONGFORMER_SERVICE_MAX_BATCH_SIZE", "2"),
                "--max-wait-ms",
                os.environ.get("SONGFORMER_SERVICE_MAX_WAIT_MS", "20"),
            ),
        ),
        "section-asr": ServiceSpec(
            name="section-asr",
            host=os.environ.get("SECTION_ASR_SERVICE_HOST", common_host),
            port=int(os.environ.get("SECTION_ASR_SERVICE_PORT", "10102")),
            gpu=os.environ.get("SECTION_ASR_SERVICE_GPU", "0"),
            memory_gib=_profile_memory(selected_profile, "section-asr"),
            profile=selected_profile,
            python=qwen_python,
            script=ROOT / "scripts" / "serve_section_asr.py",
            extra_args=(
                "--model",
                asr_model,
                "--forced-aligner",
                aligner_model,
                "--section-batch-size",
                _bounded_asr_batch_size(),
                "--max-wait-ms",
                os.environ.get("SECTION_ASR_SERVICE_MAX_WAIT_MS", "200"),
                "--decode-workers",
                os.environ.get("ASR_DECODE_WORKERS", "2"),
                "--padding",
                os.environ.get("ASR_PADDING", "1.5"),
                "--vllm-max-memory-gib",
                _env_float(
                    "ASR_VLLM_MAX_MEMORY_GIB",
                    str(_profile_memory(selected_profile, "section-asr")),
                ),
                "--gpu-max-memory-gib",
                str(_profile_memory(selected_profile, "section-asr")),
                "--forced-aligner-reserve-gib",
                _env_float("ASR_FORCED_ALIGNER_RESERVE_GIB", "8"),
                "--vllm-headroom-gib",
                _env_float("VLLM_GPU_HEADROOM_GIB", "4"),
                "--minimum-vllm-memory-gib",
                _env_float("ASR_MIN_VLLM_MEMORY_GIB", "8"),
                "--max-new-tokens",
                os.environ.get("ASR_MAX_NEW_TOKENS", "512"),
            )
            + (
                (
                    "--gpu-memory-utilization",
                    os.environ["ASR_GPU_MEMORY_UTILIZATION"],
                )
                if os.environ.get("ASR_GPU_MEMORY_UTILIZATION")
                else ()
            ),
        ),
        "omni": ServiceSpec(
            name="omni",
            host=os.environ.get("OMNI_SERVICE_HOST", common_host),
            port=int(os.environ.get("OMNI_SERVICE_PORT", "10103")),
            gpu="external",
            memory_gib=_profile_memory(selected_profile, "omni"),
            profile=selected_profile,
            python=song_python,
            script=ROOT / "scripts" / "serve_omni.py",
            extra_args=(
                "--upstream",
                os.environ.get("OMNI_UPSTREAM_SERVER", "http://127.0.0.1:10008"),
                "--model",
                os.environ.get("ALM_MODEL", "Qwen3-Omni-30B-A3B-Instruct"),
                "--max-tokens",
                os.environ.get("ALM_MAX_TOKENS", "2048"),
                "--temperature",
                os.environ.get("ALM_TEMPERATURE", "0.3"),
                "--timeout",
                os.environ.get("ALM_TIMEOUT", "600"),
                "--concurrency",
                os.environ.get("ALM_CONCURRENCY", "2"),
            ),
            device_arg=False,
        ),
    }
    ports: dict[tuple[str, int], str] = {}
    for spec in values.values():
        key = (spec.host, spec.port)
        if key in ports:
            raise ValueError(
                f"service port collision: {ports[key]} and {spec.name} use "
                f"{spec.host}:{spec.port}"
            )
        ports[key] = spec.name
    return values


def _state_path(spec: ServiceSpec) -> Path:
    return state_root() / f"{spec.name}.json"


def _log_path(spec: ServiceSpec) -> Path:
    override = os.environ.get(f"{spec.name.upper().replace('-', '_')}_SERVICE_LOG")
    if override:
        return Path(override).expanduser().resolve()
    return state_root() / "logs" / f"{spec.name}.log"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, TypeError, ValueError):
        return None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _proc_start_ticks(pid: int) -> str | None:
    try:
        # Field 22 follows a parenthesized comm that may itself contain spaces.
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[
            1
        ].split()
        return fields[19]
    except (IndexError, OSError):
        return None


def _proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _managed_process(state: dict[str, Any] | None, spec: ServiceSpec) -> bool:
    if not state:
        return False
    try:
        pid = int(state["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pid <= 1 or _proc_start_ticks(pid) != str(state.get("proc_start_ticks", "")):
        return False
    expected_script = str(spec.script.resolve())
    return expected_script in _proc_cmdline(pid)


def _service_url(spec: ServiceSpec) -> str:
    host = "127.0.0.1" if spec.host in {"0.0.0.0", "::"} else spec.host
    return f"http://{host}:{spec.port}/healthz"


def _http_health(url: str, timeout: float = 1.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(4096).decode("utf-8", "replace")
            return 200 <= response.status < 300, body
    except (OSError, urllib.error.URLError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"


def _port_is_listening(host: str, port: int) -> bool:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((connect_host, port), timeout=0.3):
            return True
    except OSError:
        return False


def status_one(spec: ServiceSpec) -> dict[str, Any]:
    state = _read_json(_state_path(spec))
    managed = _managed_process(state, spec)
    healthy, detail = _http_health(_service_url(spec))
    return {
        "name": spec.name,
        "state": "ready" if managed and healthy else "starting" if managed else "stopped",
        "managed": managed,
        "healthy": healthy,
        "pid": int(state["pid"]) if managed and state else None,
        "host": spec.host,
        "port": spec.port,
        "gpu": spec.gpu,
        "memory_gib": spec.memory_gib,
        "profile": spec.profile,
        "log": str(_log_path(spec)),
        "health_url": _service_url(spec),
        "health_detail": detail[:500],
    }


def omni_upstream_status() -> dict[str, Any]:
    upstream_base = os.environ.get(
        "OMNI_UPSTREAM_SERVER", "http://127.0.0.1:10008"
    ).rstrip("/")
    upstream_url = upstream_base + "/v1/models"
    upstream_healthy, upstream_detail = _http_health(upstream_url)
    return {
        "name": "omni-upstream",
        "state": (
            "external-ready" if upstream_healthy else "external-unavailable"
        ),
        "managed": False,
        "healthy": upstream_healthy,
        "health_url": upstream_url,
        "health_detail": upstream_detail[:500],
        "note": (
            "independent vLLM node; health-check only and never started or stopped "
            "by this manager"
        ),
    }


def _child_environment(spec: ServiceSpec) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = (
        "" if spec.gpu in {"cpu", "external"} else spec.gpu
    )
    env["MODEL_SERVICE_MEMORY_QUOTA_GIB"] = f"{spec.memory_gib:g}"
    env["PIPELINE_GPU_MAX_MEMORY_GIB"] = (
        "0" if spec.gpu in {"cpu", "external"} else f"{spec.memory_gib:g}"
    )
    env["PIPELINE_TORCH_GPU_MAX_MEMORY_GIB"] = env["PIPELINE_GPU_MAX_MEMORY_GIB"]
    env["PIPELINE_ORT_GPU_MAX_MEMORY_GIB"] = env["PIPELINE_GPU_MAX_MEMORY_GIB"]
    if spec.python == Path(
        os.environ.get("PIPELINE_PYTHON", str(DEFAULT_PIPELINE_PYTHON))
    ):
        gpu_runtime = str(ROOT / "scripts" / "gpu_runtime")
        env["PYTHONPATH"] = gpu_runtime + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        expat = spec.python.parent.parent / "lib" / "libexpat.so.1"
        if expat.is_file():
            env["LD_PRELOAD"] = str(expat) + (
                ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
            )
    if spec.name == "section-asr":
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    return env


def start_one(spec: ServiceSpec, wait_timeout: float, no_wait: bool) -> dict[str, Any]:
    state_path = _state_path(spec)
    old_state = _read_json(state_path)
    if _managed_process(old_state, spec):
        return status_one(spec)
    if state_path.exists():
        state_path.unlink()
    if _port_is_listening(spec.host, spec.port):
        raise RuntimeError(
            f"{spec.name} port {spec.host}:{spec.port} is already in use by an "
            "unmanaged process"
        )
    if not spec.python.is_file():
        raise FileNotFoundError(f"missing service Python: {spec.python}")
    if not spec.script.is_file():
        raise FileNotFoundError(f"missing service entry point: {spec.script}")

    log_path = _log_path(spec)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(spec.python),
        str(spec.script),
        "--host",
        spec.host,
        "--port",
        str(spec.port),
    ]
    if spec.device_arg:
        command.extend(("--device", "cuda:0"))
    command.extend(spec.extra_args)
    with log_path.open("ab", buffering=0) as log:
        log.write(
            (f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] start {' '.join(command)}\n").encode(
                "utf-8"
            )
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=_child_environment(spec),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    started = {
        "name": spec.name,
        "pid": process.pid,
        "proc_start_ticks": _proc_start_ticks(process.pid),
        "host": spec.host,
        "port": spec.port,
        "gpu": spec.gpu,
        "memory_gib": spec.memory_gib,
        "profile": spec.profile,
        "log": str(log_path),
        "command": command,
        "started_at": time.time(),
    }
    if not started["proc_start_ticks"]:
        process.terminate()
        raise RuntimeError(f"could not capture process identity for pid {process.pid}")
    _atomic_write_json(state_path, started)
    if no_wait:
        return status_one(spec)

    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if not _managed_process(_read_json(state_path), spec):
            raise RuntimeError(
                f"{spec.name} exited during startup; inspect {log_path}"
            )
        healthy, _detail = _http_health(_service_url(spec))
        if healthy:
            return status_one(spec)
        time.sleep(1.0)
    raise TimeoutError(
        f"{spec.name} did not become healthy within {wait_timeout:g}s; "
        f"it is still managed as pid {process.pid}, log={log_path}"
    )


def stop_one(spec: ServiceSpec, timeout: float) -> dict[str, Any]:
    state_path = _state_path(spec)
    state = _read_json(state_path)
    if not _managed_process(state, spec):
        state_path.unlink(missing_ok=True)
        return status_one(spec)
    pid = int(state["pid"])
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _managed_process(state, spec):
        time.sleep(0.2)
    if _managed_process(state, spec):
        os.kill(pid, signal.SIGKILL)
        kill_deadline = time.monotonic() + 5.0
        while time.monotonic() < kill_deadline and _managed_process(state, spec):
            time.sleep(0.1)
    if _managed_process(state, spec):
        raise RuntimeError(f"failed to stop {spec.name} pid {pid}")
    state_path.unlink(missing_ok=True)
    return status_one(spec)


def _selected_specs(name: str, specs: dict[str, ServiceSpec]) -> list[ServiceSpec]:
    if name == "all":
        return [specs[value] for value in all_service_order()]
    if name not in specs:
        raise ValueError(f"unknown managed service {name!r}")
    return [specs[name]]


def _print_status(values: Sequence[dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(list(values), ensure_ascii=False, sort_keys=True, indent=2))
        return
    for value in values:
        fields = [
            value["name"],
            f"state={value['state']}",
            f"healthy={str(value['healthy']).lower()}",
        ]
        for key in (
            "pid",
            "host",
            "port",
            "gpu",
            "memory_gib",
            "profile",
            "log",
            "health_url",
        ):
            if value.get(key) is not None:
                fields.append(f"{key}={value[key]}")
        print(" ".join(fields))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop", "restart"))
    parser.add_argument(
        "service",
        nargs="?",
        default="all",
        choices=(*SERVICE_ORDER, "all"),
    )
    parser.add_argument("--profile", choices=tuple(MEMORY_PROFILES), default=None)
    parser.add_argument("--wait-timeout", type=float, default=900.0)
    parser.add_argument("--stop-timeout", type=float, default=30.0)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.wait_timeout <= 0 or args.stop_timeout <= 0:
        raise SystemExit("timeouts must be positive")
    specs = service_specs(args.profile)
    selected = _selected_specs(args.service, specs)
    results: list[dict[str, Any]] = []
    if args.action == "status":
        results.extend(status_one(spec) for spec in selected)
        if args.service in {"all", "omni"}:
            results.append(omni_upstream_status())
    elif args.action == "start":
        results.extend(start_one(spec, args.wait_timeout, args.no_wait) for spec in selected)
    elif args.action == "stop":
        # Reverse order mirrors the start order and releases ASR first.
        results.extend(stop_one(spec, args.stop_timeout) for spec in reversed(selected))
    else:
        for spec in selected:
            stop_one(spec, args.stop_timeout)
            results.append(start_one(spec, args.wait_timeout, args.no_wait))
    _print_status(results, args.json)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
