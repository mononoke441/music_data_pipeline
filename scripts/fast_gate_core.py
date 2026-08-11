#!/usr/bin/env python3
"""Fast, zero-training AudioSet music-gate primitives.

The production path consumes the native 527 AudioSet posteriors from a frozen
PANNs model.  It does not fit or load a downstream classifier.
Audio decoding, tag inference and the deterministic sampling/cascade policy
remain separate so inference batches can span tracks.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


STAGE_A_FRACTIONS: Tuple[float, ...] = (0.10, 0.50, 0.90)
STAGE_B_FRACTIONS: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
DEFAULT_WINDOW_SECONDS = 8.0
FULL_DECODE_MAX_SECONDS = 40.0
SHORT_TRACK_CANONICAL_SAMPLE_RATE = 32000
SUPPORTED_PRECISIONS = {"fp32", "bf16"}
DECODE_SCHEDULER_SCHEMA = "bounded-track-prefetch-v1"


def bounded_thread_map_as_completed(
    function: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    max_workers: int,
    prefetch_factor: int = 2,
) -> Any:
    """Yield completed thread results while keeping only a bounded queue alive."""
    if max_workers <= 0 or prefetch_factor <= 0:
        raise ValueError("max_workers and prefetch_factor must be positive")
    iterator = iter(items)
    limit = max(1, int(max_workers) * int(prefetch_factor))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for _ in range(min(len(items), limit)):
            futures.add(executor.submit(function, next(iterator)))
        while futures:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                try:
                    item = next(iterator)
                except StopIteration:
                    item = None
                if item is not None:
                    futures.add(executor.submit(function, item))
                yield future.result()


class EmbeddingBackend(Protocol):
    """Interface implemented by frozen AudioSet backends.

    ``embed`` remains available for compatibility with pre-existing tests.  The
    production gate exclusively calls ``tag_probabilities``.
    """

    name: str
    sample_rate: int
    precision: str

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        """Return one embedding row for each ``[batch, samples]`` waveform."""

    def tag_probabilities(self, waveforms: np.ndarray) -> np.ndarray:
        """Return native AudioSet probabilities shaped ``[batch, 527]``."""


class AudioSetScorer(Protocol):
    """Interface for the fixed native-AudioSet music score contract."""

    scoring_version: str

    def predict_proba(self, tag_probabilities: np.ndarray) -> np.ndarray:
        """Return per-window music probabilities."""

    def aggregate(self, window_probabilities: Sequence[float]) -> float:
        """Aggregate window-level probabilities."""


RangeDecoder = Callable[[str, float, float, int], np.ndarray]


class InvalidAudioError(RuntimeError):
    """The source cannot provide the requested decoded audio range."""


def deterministic_offsets(
    duration: float,
    fractions: Sequence[float],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> List[float]:
    """Map relative positions to valid window starts.

    Fractions are positions within the *available start range*, rather than
    absolute points in the track.  Therefore every decoded range is a full
    window whenever the source is long enough.
    """
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration must be finite and positive, got {duration!r}")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    available = max(0.0, duration - float(window_seconds))
    output: List[float] = []
    for fraction in fractions:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"sample fraction is outside [0, 1]: {value}")
        output.append(round(available * value, 6))
    return output


def ffmpeg_decode_range(
    audio_path: str,
    start: float,
    duration: float,
    sample_rate: int,
) -> np.ndarray:
    """Decode a mono float32 range through stdout without creating a file."""
    if duration <= 0:
        return np.empty(0, dtype=np.float32)
    fast_seek_command = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-ss", f"{max(0.0, float(start)):.6f}",
        "-t", f"{float(duration):.6f}",
        "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
        "-f", "f32le", "pipe:1",
    ]
    try:
        completed = subprocess.run(fast_seek_command, check=False, capture_output=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required by the fast music gate but was not found") from error
    values = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
    if completed.returncode == 0 and values.size:
        return values

    # Some valid FLAC files have no usable seek table.  Input-side ``-ss``
    # then exits successfully with an empty pipe.  Retry only those exceptional
    # assets with accurate output-side seeking; the normal fast path is unchanged.
    accurate_seek_command = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(audio_path),
        "-ss", f"{max(0.0, float(start)):.6f}",
        "-t", f"{float(duration):.6f}",
        "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
        "-f", "f32le", "pipe:1",
    ]
    fallback = subprocess.run(accurate_seek_command, check=False, capture_output=True)
    fallback_values = np.frombuffer(fallback.stdout, dtype="<f4").astype(
        np.float32, copy=True
    )
    if fallback.returncode != 0:
        first_message = completed.stderr.decode("utf-8", errors="replace").strip()
        fallback_message = fallback.stderr.decode("utf-8", errors="replace").strip()
        raise InvalidAudioError(
            f"ffmpeg range decode failed for {audio_path!r} at {start:.3f}s: "
            f"fast_seek={first_message or completed.returncode}; "
            f"accurate_seek={fallback_message or fallback.returncode}"
        )
    if fallback_values.size == 0:
        raise InvalidAudioError(
            f"ffmpeg returned no samples for {audio_path!r} at {start:.3f}s "
            "with both fast and accurate seeking"
        )
    return fallback_values


def _fit_window(values: np.ndarray, sample_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size >= sample_count:
        return np.ascontiguousarray(values[:sample_count])
    return np.pad(values, (0, sample_count - values.size)).astype(np.float32, copy=False)


class SharedFullDecodeCache:
    """One full-track decode shared by different-rate cascade backends.

    The source is decoded at the highest required sample rate.  Lower-rate
    waveforms are derived in memory with polyphase resampling and cached.
    """

    def __init__(
        self,
        audio_path: str,
        duration: float,
        decode_sample_rate: int,
        decoder: RangeDecoder = ffmpeg_decode_range,
    ) -> None:
        self.audio_path = str(audio_path)
        self.duration = float(duration)
        self.decode_sample_rate = int(decode_sample_rate)
        self.decoder = decoder
        if self.decode_sample_rate <= 0:
            raise ValueError("shared full-decode sample rate must be positive")
        self._waveforms: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def waveform(self, sample_rate: int) -> np.ndarray:
        sample_rate = int(sample_rate)
        if sample_rate <= 0 or sample_rate > self.decode_sample_rate:
            raise ValueError(
                f"requested shared sample rate {sample_rate} exceeds "
                f"decode rate {self.decode_sample_rate}"
            )
        with self._lock:
            cached = self._waveforms.get(sample_rate)
            if cached is not None:
                return cached
            source = self._waveforms.get(self.decode_sample_rate)
            if source is None:
                source = np.asarray(
                    self.decoder(
                        self.audio_path,
                        0.0,
                        self.duration,
                        self.decode_sample_rate,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                if source.size == 0:
                    raise RuntimeError(f"full decode returned no samples for {self.audio_path!r}")
                source = np.ascontiguousarray(source)
                self._waveforms[self.decode_sample_rate] = source
            if sample_rate == self.decode_sample_rate:
                return source
            try:
                from scipy.signal import resample_poly
            except ImportError as error:
                raise RuntimeError(
                    "scipy is required to share short-track decodes across sample rates"
                ) from error
            divisor = math.gcd(sample_rate, self.decode_sample_rate)
            resampled = resample_poly(
                source,
                sample_rate // divisor,
                self.decode_sample_rate // divisor,
            ).astype(np.float32, copy=False)
            expected_length = int(round(source.size * sample_rate / self.decode_sample_rate))
            resampled = _fit_window(resampled, expected_length)
            self._waveforms[sample_rate] = resampled
            return resampled


class TrackWindowSession:
    """A per-track decode and window cache shared by both cascade stages.

    Sources up to 40 seconds are decoded exactly once.  Longer sources use
    dynamic ffmpeg range decoding, with identical offsets (the 50% window is
    shared by Stage A and B) decoded/sliced only once.
    """

    def __init__(
        self,
        audio_path: str,
        duration: float,
        sample_rate: int,
        decoder: RangeDecoder = ffmpeg_decode_range,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        shared_full_decode: Optional[SharedFullDecodeCache] = None,
    ) -> None:
        self.audio_path = str(audio_path)
        self.duration = float(duration)
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError(f"invalid duration for {self.audio_path!r}: {duration!r}")
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.sample_count = int(round(self.sample_rate * self.window_seconds))
        if self.sample_rate <= 0 or self.sample_count <= 0:
            raise ValueError("sample rate and window duration must be positive")
        self.decoder = decoder
        self.shared_full_decode = shared_full_decode
        self._full_waveform: Optional[np.ndarray] = None
        self._window_cache: Dict[float, np.ndarray] = {}

    def offsets(self, fractions: Sequence[float]) -> List[float]:
        return deterministic_offsets(self.duration, fractions, self.window_seconds)

    def _load_full_waveform(self) -> np.ndarray:
        if self._full_waveform is None:
            if self.shared_full_decode is not None:
                self._full_waveform = self.shared_full_decode.waveform(self.sample_rate)
            else:
                self._full_waveform = np.asarray(
                    self.decoder(
                        self.audio_path,
                        0.0,
                        self.duration,
                        self.sample_rate,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
        return self._full_waveform

    def window(self, offset: float) -> np.ndarray:
        key = round(float(offset), 6)
        cached = self._window_cache.get(key)
        if cached is not None:
            return cached

        if self.duration <= FULL_DECODE_MAX_SECONDS:
            full = self._load_full_waveform()
            begin = max(0, int(round(key * self.sample_rate)))
            raw = full[begin : begin + self.sample_count]
        else:
            remaining = max(0.0, self.duration - key)
            raw = self.decoder(
                self.audio_path,
                key,
                min(self.window_seconds, remaining),
                self.sample_rate,
            )
        fitted = _fit_window(raw, self.sample_count)
        self._window_cache[key] = fitted
        return fitted


def _freeze_module(module: Any) -> Any:
    if hasattr(module, "eval"):
        module.eval()
    if hasattr(module, "parameters"):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return module


def _validate_precision(precision: str) -> str:
    value = str(precision).strip().lower()
    if value not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported precision {precision!r}; choose {sorted(SUPPORTED_PRECISIONS)}")
    return value


def _torch_autocast_context(torch: Any, device: Any, precision: str) -> Any:
    precision = _validate_precision(precision)
    if precision == "fp32":
        return contextlib.nullcontext()
    if getattr(device, "type", str(device).split(":", 1)[0]) != "cuda":
        raise RuntimeError("BF16 gate inference requires a CUDA device")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected gate backend requires CUDA BF16 support")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _require_weights(path: str, backend: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{backend} weights were not found: {resolved}. "
            "Download the model checkpoint and pass it with --backend-weights."
        )
    return resolved


class _TorchModuleBackend:
    """Shared frozen-module execution used by PANNs MobileNet."""

    name = "torch_audio_backend"
    sample_rate = 32000

    def __init__(self, module: Any, device: str, precision: str = "fp32") -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(f"{self.name} backend requires PyTorch") from error
        self._torch = torch
        self.device = torch.device(device)
        self.precision = _validate_precision(precision)
        # Fail during model construction, before any output/cache is touched.
        with _torch_autocast_context(torch, self.device, self.precision):
            pass
        self.model = _freeze_module(module).to(self.device)

    def _select_embedding(self, output: Any) -> Any:
        if isinstance(output, Mapping):
            for key in ("embedding", "embeddings", "features", "feature"):
                if key in output:
                    return output[key]
            raise RuntimeError(
                f"{self.name} model output has no embedding/features key; keys={list(output)}"
            )
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            return output[1]
        return output

    def _select_tag_probabilities(self, output: Any) -> Any:
        if isinstance(output, Mapping):
            for key in ("clipwise_output", "probabilities", "probs"):
                if key in output:
                    return output[key]
            raise RuntimeError(
                f"{self.name} model output has no native AudioSet probabilities; "
                f"keys={list(output)}"
            )
        raise RuntimeError(
            f"{self.name} model must expose native AudioSet probabilities as a mapping"
        )

    def _forward(self, waveforms: np.ndarray) -> Any:
        torch = self._torch
        inputs = torch.from_numpy(np.asarray(waveforms, dtype=np.float32)).to(self.device)
        with torch.inference_mode(), _torch_autocast_context(
            torch, self.device, self.precision
        ):
            if hasattr(self.model, "extract_features"):
                return self.model.extract_features(inputs)
            if hasattr(self.model, "forward_embeddings"):
                return self.model.forward_embeddings(inputs)
            return self.model(inputs)

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        output = self._forward(waveforms)
        embedding = self._select_embedding(output)
        if getattr(embedding, "ndim", 0) > 2:
            embedding = embedding.mean(dim=tuple(range(1, embedding.ndim - 1)))
        return embedding.detach().float().cpu().numpy()

    def tag_probabilities(self, waveforms: np.ndarray) -> np.ndarray:
        probabilities = self._select_tag_probabilities(self._forward(waveforms))
        values = probabilities.detach().float().cpu().numpy()
        if values.ndim != 2 or values.shape[1] != 527:
            raise RuntimeError(
                f"{self.name} returned invalid AudioSet tag shape {values.shape}; "
                "expected [batch, 527]"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError(f"{self.name} returned invalid AudioSet probabilities")
        return values


def _load_torch_checkpoint(path: Path) -> Any:
    import torch

    try:
        return torch.jit.load(str(path), map_location="cpu")
    except Exception:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint
        if isinstance(checkpoint, Mapping):
            for key in ("model_object", "module"):
                if isinstance(checkpoint.get(key), torch.nn.Module):
                    return checkpoint[key]
        return checkpoint


def _candidate_repo(
    weights: Path,
    explicit: Optional[str],
    environment_name: str,
    marker: str,
) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get(environment_name):
        candidates.append(Path(os.environ[environment_name]).expanduser())
    candidates.extend([weights.parent, *list(weights.parents)[:4]])
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate.resolve()
    return None


def backend_source_provenance(
    backend: str,
    weights_path: str,
    repo_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Fingerprint the exact Python source tree used to construct a backend."""
    weights = _require_weights(weights_path, str(backend))
    normalized = str(backend).lower()
    if normalized == "panns_mobilenet":
        repo = _candidate_repo(weights, repo_path, "PANNS_REPO", "pytorch/models.py")
        source_root = None if repo is None else repo / "pytorch"
    else:
        raise ValueError(f"unsupported fast-gate backend provenance: {backend!r}")
    if repo is None or source_root is None or not source_root.is_dir():
        raise RuntimeError(
            f"cannot resolve the Python source tree for gate backend {backend!r}"
        )
    files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not files:
        raise RuntimeError(f"gate backend source tree has no Python files: {source_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(repo).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "schema": "python-source-tree-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "root": str(repo.resolve()),
    }


def _drop_foreign_module_tree(package: str, expected_root: Path) -> None:
    """Remove a cached top-level package when it came from another checkout."""
    module = sys.modules.get(package)
    if module is None:
        return
    root = expected_root.resolve()
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(Path(module_file).resolve())
    module_path = getattr(module, "__path__", None)
    if module_path:
        locations.extend(Path(value).resolve() for value in module_path)
    belongs_to_root = any(
        location == root or root in location.parents for location in locations
    )
    if belongs_to_root and getattr(module, "__path__", None):
        return
    prefix = f"{package}."
    for name in list(sys.modules):
        if name == package or name.startswith(prefix):
            sys.modules.pop(name, None)


def _import_panns_models(weights: Path, repo_path: Optional[str]) -> Any:
    repo = _candidate_repo(weights, repo_path, "PANNS_REPO", "pytorch/models.py")
    if repo is not None:
        _drop_foreign_module_tree("pytorch", repo)
        root = str(repo)
        # The upstream repository imports ``pytorch_utils`` as a top-level
        # module from inside ``pytorch/models.py``.  Keep both paths available
        # while importing its package rather than patching vendored source.
        pytorch_root = str(repo / "pytorch")
        for candidate in (pytorch_root, root):
            if candidate in sys.path:
                sys.path.remove(candidate)
            sys.path.insert(0, candidate)
        return importlib.import_module("pytorch.models")
    try:
        return importlib.import_module("panns_inference.models")
    except ImportError:
        try:
            return importlib.import_module("panns.models")
        except ImportError as error:
            raise RuntimeError(
                "PANNs backend requires the official repo (pass backend_repo or "
                "set PANNS_REPO) or a package exposing the official model classes"
            ) from error


def _instantiate_panns_mobilenet(
    checkpoint: Mapping[str, Any],
    weights: Path,
    repo_path: Optional[str],
) -> Any:
    models = _import_panns_models(weights, repo_path)
    model_class = getattr(models, "MobileNetV1", None) or getattr(models, "MobileNetV2", None)
    if model_class is None:
        raise RuntimeError("installed PANNs package does not expose MobileNetV1/MobileNetV2")
    model = model_class(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )
    state = checkpoint.get("model") or checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("PANNs MobileNet checkpoint has no model/state_dict weights")
    model.load_state_dict(state)
    return model


class PannsMobileNetBackend(_TorchModuleBackend):
    """Adapter for PANNs MobileNet checkpoints or exported TorchScript."""

    name = "panns_mobilenet"

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda:0",
        repo_path: Optional[str] = None,
        precision: str = "fp32",
    ) -> None:
        weights = _require_weights(weights_path, self.name)
        try:
            loaded = _load_torch_checkpoint(weights)
            module = (
                loaded if hasattr(loaded, "forward")
                else _instantiate_panns_mobilenet(loaded, weights, repo_path)
            )
        except ImportError as error:
            raise RuntimeError("PANNs MobileNet backend requires PyTorch") from error
        except Exception as error:
            raise RuntimeError(f"failed to load PANNs MobileNet weights from {weights}: {error}") from error
        super().__init__(module, device, precision=precision)


def build_backend(
    name: str,
    weights_path: str,
    device: str = "cuda:0",
    repo_path: Optional[str] = None,
    precision: str = "fp32",
) -> EmbeddingBackend:
    backends = {"panns_mobilenet": PannsMobileNetBackend}
    try:
        backend_class = backends[str(name).lower()]
    except KeyError as error:
        raise ValueError(f"unsupported fast-gate backend {name!r}; choose {sorted(backends)}") from error
    kwargs: Dict[str, Any] = {
        "device": device,
        "repo_path": repo_path,
        "precision": precision,
    }
    return backend_class(weights_path, **kwargs)


PRODUCTION_PRETRAINED_GATE_SCHEMA = "music-gate-pretrained-config-v1"


def _metrics_satisfy_constraints(
    metrics: Mapping[str, Any], constraints: Mapping[str, Any]
) -> bool:
    required_metrics = (
        "song_recall", "nonmusic_false_accept_rate", "review_rate",
    )
    required_constraints = (
        "minimum_song_recall", "maximum_nonmusic_false_accept_rate",
        "maximum_review_rate",
    )
    if any(field not in metrics for field in required_metrics):
        raise RuntimeError("production gate metrics are incomplete")
    if any(field not in constraints for field in required_constraints):
        raise RuntimeError("production gate constraints are incomplete")
    values = {
        field: float(metrics[field]) for field in required_metrics
    }
    limits = {
        field: float(constraints[field]) for field in required_constraints
    }
    if not all(math.isfinite(value) for value in (*values.values(), *limits.values())):
        raise RuntimeError("production gate metrics/constraints must be finite")
    return bool(
        values["song_recall"] >= limits["minimum_song_recall"]
        and values["nonmusic_false_accept_rate"]
        <= limits["maximum_nonmusic_false_accept_rate"]
        and values["review_rate"] <= limits["maximum_review_rate"]
    )


def _validate_support(payload: Mapping[str, Any]) -> None:
    constraints = payload.get("support_constraints")
    support = payload.get("support")
    if not isinstance(constraints, Mapping) or not isinstance(support, Mapping):
        raise RuntimeError("production gate config is missing support constraints/counts")
    requirements = {
        "total": "minimum_total_per_class",
        "validation": "minimum_validation_per_class",
        "test": "minimum_test_per_class",
    }
    for split, field in requirements.items():
        minimum = constraints.get(field)
        counts = support.get(split)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
            raise RuntimeError(f"production gate config has invalid {field}")
        if not isinstance(counts, Mapping):
            raise RuntimeError(f"production gate config is missing {split} support")
        for class_name in ("music", "nonmusic"):
            count = counts.get(class_name)
            if not isinstance(count, int) or isinstance(count, bool) or count < minimum:
                raise RuntimeError(
                    f"production gate config {split}/{class_name} support "
                    f"{count!r} is below required {minimum}"
                )
    source_constraints = payload.get("source_support_constraints")
    source_support = payload.get("source_support")
    if not isinstance(source_constraints, Mapping) or not isinstance(source_support, Mapping):
        raise RuntimeError("production gate config is missing source support constraints/counts")
    required_sources = source_constraints.get("required_sources")
    minimum_per_source = source_constraints.get("minimum_total_per_source")
    if (
        not isinstance(required_sources, list)
        or not required_sources
        or any(not isinstance(source, str) or not source for source in required_sources)
    ):
        raise RuntimeError("production gate config has invalid required_sources")
    if (
        not isinstance(minimum_per_source, int)
        or isinstance(minimum_per_source, bool)
        or minimum_per_source <= 0
    ):
        raise RuntimeError("production gate config has invalid minimum_total_per_source")
    for source in required_sources:
        counts = source_support.get(source)
        count = counts.get("total") if isinstance(counts, Mapping) else None
        if not isinstance(count, int) or isinstance(count, bool) or count < minimum_per_source:
            raise RuntimeError(
                f"production gate config source {source} support {count!r} "
                f"is below required {minimum_per_source}"
            )


def load_production_gate_config(path: str) -> Dict[str, Any]:
    """Load a fail-closed zero-training gate configuration.

    The schema records only frozen checkpoint identities, the fixed AudioSet
    scoring contract, thresholds, sampler settings, and verification metrics.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"production gate config was not found: {source}")
    if source.suffix.lower() != ".json":
        raise RuntimeError("production gate config must be a JSON artifact")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid production gate config {source}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("production gate config must contain an object")
    if payload.get("schema_version") != PRODUCTION_PRETRAINED_GATE_SCHEMA:
        raise RuntimeError(
            f"unsupported gate config schema {payload.get('schema_version')!r}; "
            f"expected {PRODUCTION_PRETRAINED_GATE_SCHEMA!r}"
        )
    if payload.get("selection_status") != "passed":
        raise RuntimeError(
            "fixed gate config is not approved: "
            f"selection_status={payload.get('selection_status')!r}"
        )
    version = payload.get("config_version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError("production gate config is missing config_version")

    # A verification run is still mandatory even though no parameter is fit.
    constraints = payload.get("constraints")
    metrics_by_split = payload.get("metrics")
    if not isinstance(constraints, Mapping) or not isinstance(metrics_by_split, Mapping):
        raise RuntimeError("production gate config is missing constraints/metrics")
    for split in ("validation", "test"):
        split_metrics = metrics_by_split.get(split)
        if not isinstance(split_metrics, Mapping):
            raise RuntimeError(f"production gate config is missing {split} metrics")
        if not _metrics_satisfy_constraints(split_metrics, constraints):
            raise RuntimeError(
                f"production gate config {split} metrics do not satisfy hard constraints"
            )
    _validate_support(payload)

    forbidden = (
        "head", "scaler", "calibrator", "stage_b_head", "stage_b_scaler",
        "stage_b_calibrator",
    )
    present = [field for field in forbidden if field in payload]
    if present:
        raise RuntimeError(
            "zero-training gate config must not contain fitted parameters: "
            + ", ".join(present)
        )
    scoring = payload.get("scoring")
    if not isinstance(scoring, Mapping):
        raise RuntimeError("production gate config is missing scoring contract")
    expected_scoring = {
        "method": "native_audioset_posteriors",
        "version": AudioSetMusicScorer.VERSION,
        "class_count": AudioSetMusicScorer.CLASS_COUNT,
        "music_indices": list(AudioSetMusicScorer.MUSIC_INDICES),
        "window_aggregation": AudioSetMusicScorer.aggregation,
    }
    for field, expected in expected_scoring.items():
        if scoring.get(field) != expected:
            raise RuntimeError(
                f"production gate scoring.{field} drifted: "
                f"expected {expected!r}, got {scoring.get(field)!r}"
            )

    required_strings = (
        "backend", "backend_architecture", "backend_repo", "backend_checkpoint",
        "backend_checkpoint_sha256", "stage_b_backend", "stage_b_backend_architecture",
        "stage_b_backend_repo", "stage_b_backend_checkpoint",
        "stage_b_backend_checkpoint_sha256", "precision", "stage_b_precision",
    )
    for field in required_strings:
        if not isinstance(payload.get(field), str) or not str(payload[field]).strip():
            raise RuntimeError(f"production gate config is missing {field}")
    for field in ("backend_checkpoint_sha256", "stage_b_backend_checkpoint_sha256"):
        digest = str(payload[field]).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"production gate config has invalid {field}")
    for field in ("precision", "stage_b_precision"):
        if payload[field] not in SUPPORTED_PRECISIONS:
            raise RuntimeError(f"production gate config has unsupported {field}={payload[field]!r}")
    for field in ("batch_size", "stage_b_batch_size"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"production gate config requires a positive integer {field}")
    for field in ("backend_source", "stage_b_backend_source"):
        metadata = payload.get(field)
        if not isinstance(metadata, Mapping) or metadata.get("schema") != "python-source-tree-v1":
            raise RuntimeError(f"production gate config has invalid {field}")
        digest = str(metadata.get("sha256") or "").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"production gate config has invalid {field} SHA256")
        if not isinstance(metadata.get("file_count"), int) or metadata["file_count"] <= 0:
            raise RuntimeError(f"production gate config has invalid {field} file_count")

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise RuntimeError("production gate config is missing thresholds")
    for stage in ("stage_a", "stage_b"):
        DecisionThresholds(
            float(thresholds[f"{stage}_reject"]),
            float(thresholds[f"{stage}_accept"]),
        )
    cascade = payload.get("cascade")
    if not isinstance(cascade, Mapping) or cascade.get("enabled") is not True:
        raise RuntimeError("production gate config must enable the two-stage cascade")
    if cascade.get("kind") not in {"same_backend_more_windows", "two_backend"}:
        raise RuntimeError("production gate config has unsupported cascade.kind")
    if cascade.get("stage_b_runs_when") != "stage_a_review":
        raise RuntimeError("production gate Stage B must run only for Stage A review")

    expected_sample_rates = {"panns_mobilenet": 32000}
    for sampler_key, backend_key in (
        ("sampler", "backend"),
        ("stage_b_sampler", "stage_b_backend"),
    ):
        sampler = payload.get(sampler_key)
        if not isinstance(sampler, Mapping):
            raise RuntimeError(f"production gate config is missing {sampler_key}")
        expected = {
            "schema": "uniform-full-track-windows-v2",
            "short_track_decode_sample_rate": SHORT_TRACK_CANONICAL_SAMPLE_RATE,
            "short_track_resampler": "scipy.signal.resample_poly",
            "full_decode_max_seconds": FULL_DECODE_MAX_SECONDS,
            "decode_scheduler_schema": DECODE_SCHEDULER_SCHEMA,
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "stage_a_fractions": list(STAGE_A_FRACTIONS),
            "stage_b_fractions": list(STAGE_B_FRACTIONS),
            "aggregation": AudioSetMusicScorer.aggregation,
        }
        for field, wanted in expected.items():
            actual = sampler.get(field)
            if field in {"stage_a_fractions", "stage_b_fractions"}:
                actual = list(map(float, actual or ()))
            elif field in {"full_decode_max_seconds", "window_seconds"}:
                actual = float(actual or 0.0)
            if actual != wanted:
                raise RuntimeError(
                    f"production gate config {sampler_key}.{field} drifted: "
                    f"expected {wanted!r}, got {actual!r}"
                )
        backend_name = str(payload[backend_key])
        if backend_name not in expected_sample_rates:
            raise RuntimeError(f"production gate selected unsupported backend {backend_name!r}")
        if int(sampler.get("sample_rate", 0)) != expected_sample_rates[backend_name]:
            raise RuntimeError(
                f"production gate {sampler_key} sample rate does not match {backend_name}"
            )
    return payload


def verify_checkpoint_sha256(path: str, expected_sha256: str, label: str) -> str:
    """Refuse to run when a selected checkpoint is missing or has drifted."""
    checkpoint = _require_weights(path, label)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != str(expected_sha256).lower():
        raise RuntimeError(
            f"{label} checkpoint SHA256 mismatch for {checkpoint}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


class AudioSetMusicScorer:
    """Zero-training score derived from native AudioSet posteriors.

    The AudioSet label table is stable for the supported 527-class PANNs
    checkpoints.  A clip is considered musical when the model
    directly recognizes the Music root, a vocal-music class, or a named music
    style/use.  Isolated instrument labels are deliberately not sufficient on
    their own, which avoids accepting tuning, single hits, bells, and similar
    non-musical events.
    """

    VERSION = "audioset-direct-music-v1"
    CLASS_COUNT = 527
    MUSIC_ROOT_INDEX = 137
    VOCAL_MUSIC_INDICES = tuple(range(27, 38))
    MUSIC_STYLE_INDICES = tuple(range(216, 283))
    MUSIC_INDICES = (
        MUSIC_ROOT_INDEX,
        *VOCAL_MUSIC_INDICES,
        *MUSIC_STYLE_INDICES,
    )
    aggregation = "median_native_audioset_music_probability"

    def __init__(self, scoring_version: str = VERSION) -> None:
        self.scoring_version = str(scoring_version)
        if not self.scoring_version:
            raise ValueError("scoring_version must not be empty")

    def predict_proba(self, tag_probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(tag_probabilities, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.CLASS_COUNT:
            raise ValueError(
                f"AudioSet tag shape must be [batch, {self.CLASS_COUNT}], got {values.shape}"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("AudioSet tag probabilities must be finite and within [0, 1]")
        return np.max(values[:, self.MUSIC_INDICES], axis=1)

    def aggregate(self, window_probabilities: Sequence[float]) -> float:
        values = np.asarray(window_probabilities, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("window music probabilities must be finite and non-empty")
        return float(np.median(values))


@dataclass(frozen=True)
class DecisionThresholds:
    reject: float
    accept: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.reject < self.accept <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= reject < accept <= 1")

    def decide(self, probability: float) -> str:
        if probability >= self.accept:
            return "accepted"
        if probability <= self.reject:
            return "rejected"
        return "review"


@dataclass(frozen=True)
class CascadeResult:
    backend: str
    scoring_version: str
    stage_probabilities: Dict[str, List[float]]
    stage_scores: Dict[str, Optional[float]]
    offsets: Dict[str, List[float]]
    decision: str
    probability: float
    window_seconds: float
    sample_rate: int
    stage_backends: Optional[Dict[str, str]] = None
    stage_sample_rates: Optional[Dict[str, int]] = None
    stage_precisions: Optional[Dict[str, str]] = None
    stage_batch_sizes: Optional[Dict[str, int]] = None
    aggregation: str = "median_native_audioset_music_probability"

    def as_music_gate(self) -> Dict[str, Any]:
        output = {
            "backend": self.backend,
            "scoring_version": self.scoring_version,
            "stage_probabilities": self.stage_probabilities,
            "stage_scores": self.stage_scores,
            "aggregation": self.aggregation,
            "offsets": self.offsets,
            "decision": self.decision,
            "probability": round(float(self.probability), 6),
            "window_seconds": self.window_seconds,
            "sample_rate": self.sample_rate,
        }
        if self.stage_backends:
            output["stage_backends"] = dict(self.stage_backends)
        if self.stage_sample_rates:
            output["stage_sample_rates"] = dict(self.stage_sample_rates)
        if self.stage_precisions:
            output["stage_precisions"] = dict(self.stage_precisions)
        if self.stage_batch_sizes:
            output["stage_batch_sizes"] = dict(self.stage_batch_sizes)
        return output


class CascadeMusicGate:
    """Two-stage gate with cross-track AudioSet batches and window reuse."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        head: AudioSetScorer,
        stage_a_thresholds: DecisionThresholds = DecisionThresholds(0.20, 0.80),
        stage_b_thresholds: DecisionThresholds = DecisionThresholds(0.40, 0.60),
        batch_size: int = 32,
        stage_b_batch_size: Optional[int] = None,
        decode_workers: int = 4,
        decoder: RangeDecoder = ffmpeg_decode_range,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        stage_b_backend: Optional[EmbeddingBackend] = None,
        stage_b_head: Optional[AudioSetScorer] = None,
    ) -> None:
        self.backend = backend
        self.head = head
        self.stage_b_backend = stage_b_backend or backend
        self.stage_b_head = stage_b_head or head
        self.stage_a_thresholds = stage_a_thresholds
        self.stage_b_thresholds = stage_b_thresholds
        self.batch_size = int(batch_size)
        self.stage_b_batch_size = int(
            batch_size if stage_b_batch_size is None else stage_b_batch_size
        )
        self.decode_workers = int(decode_workers)
        if self.batch_size <= 0 or self.stage_b_batch_size <= 0:
            raise ValueError("stage A/B batch sizes must be positive")
        if self.decode_workers <= 0:
            raise ValueError("decode_workers must be positive")
        self.decoder = decoder
        self.window_seconds = float(window_seconds)

    def _infer_missing(
        self,
        sessions: Sequence[TrackWindowSession],
        indices: Sequence[int],
        fractions: Sequence[float],
        embedding_cache: Sequence[Dict[float, np.ndarray]],
        backend: EmbeddingBackend,
        batch_size: int,
    ) -> Tuple[Dict[int, List[float]], Dict[int, np.ndarray]]:
        offsets_by_index: Dict[int, List[float]] = {}
        missing_by_track: Dict[int, List[float]] = {}
        queued: set[Tuple[int, float]] = set()
        for index in indices:
            offsets = sessions[index].offsets(fractions)
            offsets_by_index[index] = offsets
            for offset in offsets:
                key = round(offset, 6)
                cache_key = (index, key)
                if key not in embedding_cache[index] and cache_key not in queued:
                    missing_by_track.setdefault(index, []).append(key)
                    queued.add(cache_key)

        # Decode tracks concurrently, but offsets within one track serially.
        # The latter is important for the <=40 s single-full-decode invariant.
        def decode_track(item: Tuple[int, List[float]]) -> List[Tuple[Tuple[int, float], np.ndarray]]:
            index, offsets = item
            return [((index, key), sessions[index].window(key)) for key in offsets]

        pending_batch: List[Tuple[int, float, np.ndarray]] = []

        def infer_pending(force: bool = False) -> None:
            while len(pending_batch) >= batch_size or (force and pending_batch):
                take = min(len(pending_batch), batch_size)
                chunk = pending_batch[:take]
                del pending_batch[:take]
                batch = np.stack([item[2] for item in chunk], axis=0).astype(
                    np.float32, copy=False
                )
                embeddings = np.asarray(backend.tag_probabilities(batch), dtype=np.float64)
                if embeddings.ndim != 2 or embeddings.shape != (len(chunk), 527):
                    raise RuntimeError(
                        f"{backend.name} returned invalid AudioSet tag shape {embeddings.shape}; "
                        f"expected [{len(chunk)}, 527]"
                    )
                for (index, key, _), embedding in zip(chunk, embeddings):
                    embedding_cache[index][key] = np.asarray(embedding, dtype=np.float64)

        if missing_by_track:
            # Keep a bounded sliding prefetch queue.  Replacement decodes are
            # submitted before yielding a completed track, so ffmpeg keeps
            # running while the main thread executes each GPU batch.
            for decoded_track in bounded_thread_map_as_completed(
                decode_track,
                list(missing_by_track.items()),
                max_workers=self.decode_workers,
            ):
                for (index, key), waveform in decoded_track:
                    pending_batch.append((index, key, waveform))
                infer_pending()
            infer_pending(force=True)

        embeddings_by_index = {
            index: np.stack([
                embedding_cache[index][round(offset, 6)]
                for offset in offsets_by_index[index]
            ], axis=0)
            for index in indices
        }
        return offsets_by_index, embeddings_by_index

    def classify_records(self, records: Sequence[Mapping[str, Any]]) -> List[CascadeResult]:
        if not records:
            return []
        if max(int(self.backend.sample_rate), int(self.stage_b_backend.sample_rate)) > SHORT_TRACK_CANONICAL_SAMPLE_RATE:
            raise RuntimeError(
                "gate backend sample rate exceeds the fixed short-track decode rate"
            )
        shared_full_decodes: List[Optional[SharedFullDecodeCache]] = [
            SharedFullDecodeCache(
                audio_path=str(record["audio_path"]),
                duration=float(record["duration"]),
                decode_sample_rate=SHORT_TRACK_CANONICAL_SAMPLE_RATE,
                decoder=self.decoder,
            )
            if float(record["duration"]) <= FULL_DECODE_MAX_SECONDS else None
            for record in records
        ]
        sessions = [
            TrackWindowSession(
                audio_path=str(record["audio_path"]),
                duration=float(record["duration"]),
                sample_rate=int(self.backend.sample_rate),
                decoder=self.decoder,
                window_seconds=self.window_seconds,
                shared_full_decode=shared_full_decodes[index],
            )
            for index, record in enumerate(records)
        ]
        if int(self.stage_b_backend.sample_rate) == int(self.backend.sample_rate):
            stage_b_sessions = sessions
        else:
            stage_b_sessions = [
                TrackWindowSession(
                    audio_path=str(record["audio_path"]),
                    duration=float(record["duration"]),
                    sample_rate=int(self.stage_b_backend.sample_rate),
                    decoder=self.decoder,
                    window_seconds=self.window_seconds,
                    shared_full_decode=shared_full_decodes[index],
                )
                for index, record in enumerate(records)
            ]
        embedding_cache: List[Dict[float, np.ndarray]] = [dict() for _ in records]
        all_indices = list(range(len(records)))
        a_offsets, a_embeddings = self._infer_missing(
            sessions, all_indices, STAGE_A_FRACTIONS, embedding_cache,
            self.backend, self.batch_size,
        )
        a_window_probabilities = {
            index: self.head.predict_proba(a_embeddings[index]).tolist()
            for index in all_indices
        }
        a_scores = {
            index: float(self.head.aggregate(a_window_probabilities[index]))
            for index in all_indices
        }
        initial_decisions = {
            index: self.stage_a_thresholds.decide(a_scores[index]) for index in all_indices
        }
        uncertain = [index for index in all_indices if initial_decisions[index] == "review"]
        b_offsets: Dict[int, List[float]] = {}
        b_embeddings: Dict[int, np.ndarray] = {}
        b_window_probabilities: Dict[int, List[float]] = {}
        b_scores: Dict[int, float] = {}
        if uncertain:
            same_embedding_backend = (
                self.stage_b_backend is self.backend
                and self.stage_b_batch_size == self.batch_size
            )
            stage_b_cache = embedding_cache if same_embedding_backend else [dict() for _ in records]
            b_offsets, b_embeddings = self._infer_missing(
                stage_b_sessions, uncertain, STAGE_B_FRACTIONS, stage_b_cache,
                self.stage_b_backend, self.stage_b_batch_size,
            )
            b_window_probabilities = {
                index: self.stage_b_head.predict_proba(b_embeddings[index]).tolist()
                for index in uncertain
            }
            b_scores = {
                index: float(self.stage_b_head.aggregate(b_window_probabilities[index]))
                for index in uncertain
            }

        results: List[CascadeResult] = []
        for index in all_indices:
            if index in b_embeddings:
                b_score: Optional[float] = b_scores[index]
                decision = self.stage_b_thresholds.decide(b_score)
                final_probability = b_score
                stage_b_values = [
                    round(value, 6) for value in b_window_probabilities[index]
                ]
                stage_b_offsets = b_offsets[index]
            else:
                b_score = None
                decision = initial_decisions[index]
                final_probability = a_scores[index]
                stage_b_values = []
                stage_b_offsets = []
            results.append(CascadeResult(
                backend=str(self.backend.name),
                scoring_version=self.head.scoring_version,
                stage_probabilities={
                    "stage_a": [
                        round(value, 6) for value in a_window_probabilities[index]
                    ],
                    "stage_b": stage_b_values,
                },
                stage_scores={
                    "stage_a": round(a_scores[index], 6),
                    "stage_b": None if b_score is None else round(b_score, 6),
                },
                offsets={
                    "stage_a": [round(value, 6) for value in a_offsets[index]],
                    "stage_b": [round(value, 6) for value in stage_b_offsets],
                },
                decision=decision,
                probability=final_probability,
                window_seconds=self.window_seconds,
                sample_rate=int(self.backend.sample_rate),
                stage_backends={
                    "stage_a": str(self.backend.name),
                    "stage_b": str(self.stage_b_backend.name),
                },
                stage_sample_rates={
                    "stage_a": int(self.backend.sample_rate),
                    "stage_b": int(self.stage_b_backend.sample_rate),
                },
                stage_precisions={
                    "stage_a": str(getattr(self.backend, "precision", "fp32")),
                    "stage_b": str(getattr(self.stage_b_backend, "precision", "fp32")),
                },
                stage_batch_sizes={
                    "stage_a": self.batch_size,
                    "stage_b": self.stage_b_batch_size,
                },
                aggregation=str(getattr(
                    self.head,
                    "aggregation",
                    "median_window_probability",
                )),
            ))
        return results
