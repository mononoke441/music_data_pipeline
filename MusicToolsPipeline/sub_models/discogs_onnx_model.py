# -*- coding: utf-8 -*-
"""GPU Discogs EffNet embeddings with shared ONNX classification heads."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

BACKBONE_EMBEDDING_DIM = 1280
HEAD_OUTPUT_DIMS = {
    "voice": 2,
    "genre": 87,
    "mood": 56,
    "instrument": 40,
    "danceability": 2,
}
def _preload_pip_cudnn() -> None:
    """Expose pip-installed cuDNN libraries to ONNX Runtime 1.20.

    PyTorch wheels can find these libraries through their own runpaths, while
    ONNX Runtime loads its CUDA provider independently.  Loading the cuDNN
    SONAMEs globally avoids requiring callers to construct LD_LIBRARY_PATH.
    """
    candidates = {
        path
        for entry in sys.path
        for path in (Path(entry) / "nvidia" / "cudnn" / "lib").glob("libcudnn*.so.9")
    }
    pending = sorted(candidates)
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for _ in range(len(pending) + 1):
        remaining: List[Path] = []
        for path in pending:
            try:
                ctypes.CDLL(str(path), mode=mode)
            except OSError:
                remaining.append(path)
        if not remaining or len(remaining) == len(pending):
            pending = remaining
            break
        pending = remaining
    if pending:
        logger.debug(
            "could not preload every pip cuDNN library: %s",
            ", ".join(path.name for path in pending),
        )


def _require_cuda_provider(
    name: str,
    actual_providers: Sequence[str],
    require_cuda: bool,
) -> None:
    if require_cuda and "CUDAExecutionProvider" not in actual_providers:
        raise RuntimeError(
            f"Discogs {name} failed to activate CUDAExecutionProvider; "
            f"actual providers: {list(actual_providers)}. "
            "Check CUDA/cuDNN shared libraries."
        )


def cuda_tensor_binding_spec(tensor: Any, device_id: int) -> Dict[str, Any]:
    """Build the zero-copy ORT binding fields for a contiguous CUDA tensor.

    This deliberately validates every property that would otherwise make ORT
    interpret a raw ``data_ptr`` incorrectly.  It is kept independent from an
    ORT session so the pointer contract can be covered by CPU-only unit tests.
    """

    if not bool(getattr(tensor, "is_cuda", False)):
        raise ValueError("ORT CUDA I/O binding requires a CUDA tensor")
    if getattr(tensor, "dtype", None) != torch.float32:
        raise ValueError("ORT CUDA I/O binding requires torch.float32")
    if not bool(tensor.is_contiguous()):
        raise ValueError("ORT CUDA I/O binding requires a contiguous tensor")
    tensor_device_id = getattr(getattr(tensor, "device", None), "index", None)
    if tensor_device_id is None or int(tensor_device_id) != int(device_id):
        raise ValueError(
            "ORT CUDA I/O binding tensor/device mismatch: "
            f"tensor={getattr(tensor, 'device', None)}, requested=cuda:{device_id}"
        )
    return {
        "device_type": "cuda",
        "device_id": int(device_id),
        "element_type": np.float32,
        "shape": tuple(int(value) for value in tensor.shape),
        "buffer_ptr": int(tensor.data_ptr()),
    }


def bind_cuda_tensor(
    io_binding: Any,
    name: str,
    tensor: Any,
    *,
    device_id: int,
    output: bool,
) -> None:
    """Bind a torch CUDA allocation to ORT without a host-side copy."""

    spec = cuda_tensor_binding_spec(tensor, device_id)
    method = io_binding.bind_output if output else io_binding.bind_input
    method(name=name, **spec)


@dataclass(frozen=True)
class DiscogsModelPaths:
    backbone: str
    voice: str
    genre: str
    mood: str
    instrument: str
    danceability: str

    @classmethod
    def from_root(cls, root: str) -> "DiscogsModelPaths":
        base = Path(root)
        return cls(
            backbone=str(base / "discogs-effnet-bsdynamic-1.onnx"),
            voice=str(base / "voice_instrumental-discogs-effnet-1.onnx"),
            genre=str(base / "mtg_jamendo_genre-discogs-effnet-1.onnx"),
            mood=str(base / "mtg_jamendo_moodtheme-discogs-effnet-1.onnx"),
            instrument=str(base / "mtg_jamendo_instrument-discogs-effnet-1.onnx"),
            danceability=str(base / "danceability-discogs-effnet-1.onnx"),
        )

    def values(self) -> Iterable[str]:
        return (
            self.backbone, self.voice, self.genre, self.mood,
            self.instrument, self.danceability,
        )


def _find_classes(value: Any) -> Optional[List[str]]:
    if isinstance(value, dict):
        direct = value.get("classes")
        if isinstance(direct, list) and all(isinstance(x, str) for x in direct):
            return list(direct)
        for child in value.values():
            found = _find_classes(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_classes(child)
            if found:
                return found
    return None


def load_model_labels(model_path: str) -> List[str]:
    metadata_path = Path(model_path).with_suffix(".json")
    if not metadata_path.exists():
        return []
    with metadata_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return _find_classes(value) or []


class DiscogsMelFrontend(torch.nn.Module):
    """PyTorch port of Essentia TensorflowInputMusiCNN.

    The constants match Essentia v2.1_beta5-1445-gb9fa6cb6: 16 kHz,
    512-sample Hann frames, 256-sample hop, 96 Slaney mel triangles and
    log10(1 + 10000 * energy). EffNet consumes 128-frame patches at hop 62.
    """

    sample_rate = 16000
    frame_size = 512
    frame_hop = 256
    mel_bands = 96
    patch_size = 128
    patch_hop = 62

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.register_buffer(
            "window",
            torch.hann_window(self.frame_size, periodic=False, device=device),
            persistent=False,
        )
        filterbank = torchaudio.functional.melscale_fbanks(
            n_freqs=self.frame_size // 2 + 1,
            f_min=0.0,
            f_max=float(self.sample_rate / 2),
            n_mels=self.mel_bands,
            sample_rate=self.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)
        self.register_buffer("filterbank", filterbank, persistent=False)

    def mel_frames(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(self.device, dtype=torch.float32).flatten()
        spectrum = torch.stft(
            waveform,
            n_fft=self.frame_size,
            hop_length=self.frame_hop,
            win_length=self.frame_size,
            window=self.window,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        # Essentia's Spectrum/MelBands chain matches a squared magnitude here;
        # this is numerically checked by validate_discogs_frontend.py.
        power = spectrum.square().transpose(0, 1)
        mel = power @ self.filterbank
        return torch.log10(1.0 + 10000.0 * torch.clamp_min(mel, 0.0))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_frames(waveform)
        if mel.shape[0] < self.patch_size:
            # Official EffNet wrapper uses lastPatchMode=discard.
            return torch.empty(
                (0, self.patch_size, self.mel_bands),
                dtype=torch.float32,
                device=self.device,
            )
        patches = [
            mel[start : start + self.patch_size]
            for start in range(0, mel.shape[0] - self.patch_size + 1, self.patch_hop)
        ]
        return torch.stack(patches, dim=0).contiguous()


class DiscogsOnnxEngine:
    """One resident EffNet backbone and five tiny shared heads."""

    def __init__(
        self,
        paths: DiscogsModelPaths,
        *,
        device_id: int = 0,
        batch_size: int = 256,
        require_cuda: bool = True,
        instrument_threshold: float = 0.5,
        voice_threshold: float = 0.5,
        top_k: int = 8,
    ) -> None:
        missing = [path for path in paths.values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError("missing Discogs ONNX model(s): " + ", ".join(missing))
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime-gpu==1.20.1 is required for Discogs inference"
            ) from error

        _preload_pip_cudnn()
        available = ort.get_available_providers()
        if require_cuda and "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable; refusing a silent CPU fallback"
            )
        providers: List[Any] = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        if not require_cuda:
            providers.append("CPUExecutionProvider")

        # ORT otherwise creates a thread pool sized to every visible host CPU
        # for each of the six sessions.  On shared GPU workers this both
        # oversubscribes the cpuset and emits affinity errors, even though the
        # actual model execution is handled by CUDA.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(
            1, int(os.environ.get("DISCOGS_ORT_INTRA_OP_THREADS", "1"))
        )
        session_options.inter_op_num_threads = max(
            1, int(os.environ.get("DISCOGS_ORT_INTER_OP_THREADS", "1"))
        )

        self.paths = paths
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device_id = int(device_id)
        self.require_cuda = bool(require_cuda)
        self.instrument_threshold = float(instrument_threshold)
        self.voice_threshold = float(voice_threshold)
        self.top_k = int(top_k)
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch CUDA is unavailable; Discogs require_cuda=True needs "
                "torch CUDA tensors for ONNX Runtime I/O binding"
            )
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.frontend = DiscogsMelFrontend(self.device)

        def create_session(name: str, model_path: str) -> Any:
            session = ort.InferenceSession(
                model_path, sess_options=session_options, providers=providers
            )
            actual_providers = session.get_providers()
            _require_cuda_provider(name, actual_providers, require_cuda)
            return session

        self.sessions = {
            "backbone": create_session("backbone", paths.backbone),
            "voice": create_session("voice", paths.voice),
            "genre": create_session("genre", paths.genre),
            "mood": create_session("mood", paths.mood),
            "instrument": create_session("instrument", paths.instrument),
            "danceability": create_session("danceability", paths.danceability),
        }
        self.labels = {
            name: load_model_labels(getattr(paths, name))
            for name in ("voice", "genre", "mood", "instrument", "danceability")
        }
        self._embedding_output_name = self._output_name_with_width(
            self.sessions["backbone"], BACKBONE_EMBEDDING_DIM
        )
        self._head_output_names = {
            name: self._output_name_with_width(self.sessions[name], width)
            for name, width in HEAD_OUTPUT_DIMS.items()
        }
        if require_cuda:
            unavailable = [
                name
                for name, session in self.sessions.items()
                if not callable(getattr(session, "io_binding", None))
                or not callable(getattr(session, "run_with_iobinding", None))
            ]
            if unavailable:
                raise RuntimeError(
                    "Discogs require_cuda=True requires ONNX Runtime CUDA I/O "
                    "binding; unavailable for sessions: " + ", ".join(unavailable)
                )

    @staticmethod
    def _input_name(session: Any) -> str:
        return session.get_inputs()[0].name

    @staticmethod
    def _output_name_with_width(session: Any, width: int) -> str:
        matches = []
        for output in session.get_outputs():
            shape = getattr(output, "shape", None) or []
            if shape and shape[-1] == width:
                matches.append(output.name)
        if len(matches) != 1:
            shapes = [
                (getattr(output, "name", "?"), tuple(getattr(output, "shape", ()) or ()))
                for output in session.get_outputs()
            ]
            raise RuntimeError(
                f"Discogs ONNX session must expose exactly one {width}-D output; "
                f"found {shapes}"
            )
        return matches[0]

    @staticmethod
    def _select_embeddings(outputs: Sequence[np.ndarray]) -> np.ndarray:
        for output in outputs:
            if output.ndim == 2 and output.shape[-1] == BACKBONE_EMBEDDING_DIM:
                return np.asarray(output, dtype=np.float32)
        shapes = [tuple(output.shape) for output in outputs]
        raise RuntimeError(f"Discogs backbone did not return 1280-D embeddings: {shapes}")

    def _run_session(self, session: Any, values: np.ndarray) -> List[np.ndarray]:
        output: List[np.ndarray] = []
        for start in range(0, len(values), self.batch_size):
            batch = np.ascontiguousarray(values[start : start + self.batch_size], dtype=np.float32)
            result = session.run(None, {self._input_name(session): batch})
            if not output:
                output = [np.asarray(value) for value in result]
            else:
                output = [np.concatenate([old, new], axis=0) for old, new in zip(output, result)]
        return output

    def _run_cuda_iobinding(
        self,
        name: str,
        session: Any,
        values: torch.Tensor,
        *,
        output_name: str,
        output_width: int,
    ) -> torch.Tensor:
        """Run one CUDA session while keeping input and output on the GPU."""

        if not self.require_cuda:
            raise RuntimeError("CUDA I/O binding is reserved for require_cuda=True")
        if not isinstance(values, torch.Tensor) or not values.is_cuda:
            raise RuntimeError(
                f"Discogs {name} require_cuda=True received a non-CUDA tensor; "
                "NumPy fallback is disabled"
            )
        outputs: List[torch.Tensor] = []
        try:
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size].contiguous()
                result = torch.empty(
                    (len(batch), int(output_width)),
                    dtype=torch.float32,
                    device=self.device,
                )
                io_binding = session.io_binding()
                bind_cuda_tensor(
                    io_binding,
                    self._input_name(session),
                    batch,
                    device_id=self.device_id,
                    output=False,
                )
                bind_cuda_tensor(
                    io_binding,
                    output_name,
                    result,
                    device_id=self.device_id,
                    output=True,
                )
                # ORT and PyTorch do not share a compute stream here.  The two
                # synchronizations establish pointer readiness and completion
                # without staging either allocation through host memory.
                torch.cuda.synchronize(self.device)
                session.run_with_iobinding(io_binding)
                torch.cuda.synchronize(self.device)
                outputs.append(result)
        except Exception as error:
            raise RuntimeError(
                f"Discogs {name} CUDA I/O binding failed; NumPy fallback is "
                "disabled because require_cuda=True"
            ) from error
        if not outputs:
            return torch.empty(
                (0, int(output_width)), dtype=torch.float32, device=self.device
            )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0).contiguous()

    def infer_patch_batch(
        self,
        patches: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, Dict[str, np.ndarray]]:
        """Run the shared backbone once, then all heads from its embedding.

        Production CUDA inference never materializes patches or embeddings on
        the CPU.  Only the five small final prediction matrices cross back to
        host memory.  ``require_cuda=False`` retains the NumPy path for CPU and
        lightweight fake-session tests.
        """

        if self.require_cuda:
            if not isinstance(patches, torch.Tensor) or not patches.is_cuda:
                raise RuntimeError(
                    "Discogs require_cuda=True needs CUDA patches for ORT I/O "
                    "binding; NumPy fallback is disabled"
                )
            patches = patches.to(dtype=torch.float32).contiguous()
            embeddings = self._run_cuda_iobinding(
                "backbone",
                self.sessions["backbone"],
                patches,
                output_name=self._embedding_output_name,
                output_width=BACKBONE_EMBEDDING_DIM,
            )
            predictions = {
                name: self._run_cuda_iobinding(
                    name,
                    self.sessions[name],
                    embeddings,
                    output_name=self._head_output_names[name],
                    output_width=HEAD_OUTPUT_DIMS[name],
                ).cpu().numpy()
                for name in HEAD_OUTPUT_DIMS
            }
            return embeddings, predictions

        values = (
            patches.detach().cpu().numpy()
            if isinstance(patches, torch.Tensor)
            else patches
        )
        values = np.ascontiguousarray(values, dtype=np.float32)
        backbone_outputs = self._run_session(self.sessions["backbone"], values)
        embeddings = self._select_embeddings(backbone_outputs)
        predictions = {
            name: np.asarray(
                self._run_session(self.sessions[name], embeddings)[0],
                dtype=np.float32,
            )
            for name in HEAD_OUTPUT_DIMS
        }
        return embeddings, predictions

    def _top_labels(self, name: str, predictions: np.ndarray) -> List[Dict[str, float]]:
        means = np.asarray(predictions, dtype=np.float32).mean(axis=0)
        labels = self.labels.get(name) or [f"class_{index}" for index in range(len(means))]
        indices = np.argsort(means)[::-1][: self.top_k]
        return [
            {"label": labels[int(index)] if int(index) < len(labels) else f"class_{index}",
             "probability": round(float(means[index]), 6)}
            for index in indices
        ]

    def _voice_summary(self, predictions: np.ndarray) -> Dict[str, Any]:
        labels = [label.lower() for label in self.labels.get("voice", [])]
        voice_index = next(
            (index for index, label in enumerate(labels) if "voice" in label or "vocal" in label),
            1 if predictions.shape[1] > 1 else 0,
        )
        probabilities = predictions[:, voice_index].astype(np.float32)
        active = probabilities >= self.voice_threshold
        longest = current = 0
        for value in active:
            current = current + 1 if bool(value) else 0
            longest = max(longest, current)
        frame_seconds = DiscogsMelFrontend.patch_hop * DiscogsMelFrontend.frame_hop / 16000.0
        return {
            "voice_mean": round(float(probabilities.mean()), 6),
            "voice_max": round(float(probabilities.max()), 6),
            "voice_coverage": round(float(active.mean()), 6),
            "longest_voice_sec": round(longest * frame_seconds, 6),
            "frame_hop_sec": round(frame_seconds, 6),
            "probabilities": probabilities.tolist(),
        }

    def _instrument_changes(self, predictions: np.ndarray, duration: float) -> List[Dict[str, Any]]:
        labels = self.labels.get("instrument") or [
            f"class_{index}" for index in range(predictions.shape[1])
        ]
        frame_seconds = DiscogsMelFrontend.patch_hop * DiscogsMelFrontend.frame_hop / 16000.0
        previous: Optional[tuple[str, ...]] = None
        changes: List[Dict[str, Any]] = []
        for index, frame in enumerate(predictions):
            active = tuple(sorted(
                labels[class_index] if class_index < len(labels) else f"class_{class_index}"
                for class_index, probability in enumerate(frame)
                if float(probability) >= self.instrument_threshold
            ))
            if active != previous:
                changes.append({
                    "time": round(min(float(duration), index * frame_seconds), 6),
                    "active": list(active),
                })
                previous = active
        return changes

    @torch.inference_mode()
    def predict_frames(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor | np.ndarray, Dict[str, np.ndarray], float]:
        waveform = waveform.float()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if sample_rate != DiscogsMelFrontend.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                DiscogsMelFrontend.sample_rate,
            )
        duration = waveform.numel() / float(DiscogsMelFrontend.sample_rate)
        patches = self.frontend(waveform).detach().contiguous()
        embeddings, predictions = self.infer_patch_batch(patches)
        return embeddings, predictions, duration

    @torch.inference_mode()
    def analyze(self, waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        embeddings, predictions, duration = self.predict_frames(waveform, sample_rate)
        return {
            "voice_analysis": self._voice_summary(predictions["voice"]),
            "genre": self._top_labels("genre", predictions["genre"]),
            "mood_theme": self._top_labels("mood", predictions["mood"]),
            "danceability": self._top_labels("danceability", predictions["danceability"]),
            "instruments": self._top_labels("instrument", predictions["instrument"]),
            "instrument_changes": self._instrument_changes(predictions["instrument"], duration),
            "discogs_frame_count": int(embeddings.shape[0]),
            "discogs_provider": self.sessions["backbone"].get_providers()[0],
        }
