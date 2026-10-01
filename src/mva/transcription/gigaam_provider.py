"""Local GigaAM-v3 inference (default: v3_e2e_rnnt with punctuation + normalisation).

The model is loaded once per application run and reused for every visit. Audio is
passed as tensors, so GigaAM's ffmpeg-based file loader is never needed.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
import typing
from typing import Any

import numpy as np

from mva.config import SAMPLE_RATE
from mva.storage.app_settings import InferenceDevice
from mva.transcription.base import (
    LoadProgress,
    ModelLoadError,
    ProviderInfo,
    TranscriptionError,
    TranscriptionProvider,
)
from mva.transcription.model_manager import ModelManager, ModelStatus
from mva.transcription.models import AudioSegment, TranscriptSegment

log = logging.getLogger(__name__)

# GigaAM documents .transcribe() for audio up to 25 s.
GIGAAM_MAX_SEGMENT_S = 25.0


def resolve_device(preference: InferenceDevice) -> str:
    import torch

    cuda = torch.cuda.is_available()
    if preference == InferenceDevice.CPU:
        return "cpu"
    if preference == InferenceDevice.GPU and not cuda:
        log.warning("GPU requested but CUDA is unavailable; falling back to CPU")
    return "cuda" if cuda else "cpu"


def _safe_globals() -> list[Any]:
    import omegaconf
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    return [
        omegaconf.dictconfig.DictConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        typing.Any,
        dict,
        collections.defaultdict,
    ]


def _check_targets(node: Any) -> None:
    """hydra.instantiate() imports whatever `_target_` names; allow only gigaam classes."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(node, DictConfig):
        for key, value in node.items_ex(resolve=False):
            if key == "_target_" and not str(value).startswith(("gigaam.", "torchaudio.")):
                raise ModelLoadError(f"unexpected instantiate target: {value}")
            _check_targets(value)
    elif isinstance(node, ListConfig):
        for value in node:
            _check_targets(value)


class GigaAMTranscriptionProvider(TranscriptionProvider):
    def __init__(
        self,
        model_manager: ModelManager,
        device: InferenceDevice = InferenceDevice.AUTO,
        num_threads: int | None = None,
    ) -> None:
        self._manager = model_manager
        self._device_pref = device
        self._num_threads = num_threads
        self._model: Any = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self.load_seconds = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def max_segment_s(self) -> float:
        return GIGAAM_MAX_SEGMENT_S

    def info(self) -> ProviderInfo:
        manifest = self._manager.manifest
        return ProviderInfo(
            name="GigaAM",
            model=manifest.model_name,
            revision=f"{manifest.revision} gigaam@{manifest.gigaam_commit[:12]}",
            device=self._device,
        )

    def load(self, progress: LoadProgress | None = None) -> None:
        if self._model is not None:
            return
        if self._manager.status() != ModelStatus.READY:
            raise ModelLoadError("model files are not downloaded/verified")
        started = time.monotonic()
        try:
            import torch
            from gigaam.model import GigaAMASR

            if self._num_threads:
                torch.set_num_threads(self._num_threads)
            device = resolve_device(self._device_pref)
            if progress:
                progress("reading_checkpoint")
            with torch.serialization.safe_globals(_safe_globals()):
                checkpoint = torch.load(
                    self._manager.checkpoint_path, map_location="cpu", weights_only=True
                )
            cfg = checkpoint["cfg"]
            _check_targets(cfg)
            tokenizer = self._manager.tokenizer_path
            if tokenizer is not None:
                cfg.decoding.model_path = str(tokenizer)
            cfg.encoder.flash_attn = False
            if progress:
                progress("building_model")
            model = GigaAMASR(cfg)
            model.load_state_dict(checkpoint["state_dict"])
            del checkpoint
            cfg.model_name = self._manager.manifest.model_name
            model = model.eval()
            if device != "cpu":
                model.encoder = model.encoder.half()
            model = model.to(device)
            self._model = model
            self._device = device
            if progress:
                progress("warmup")
            self._infer(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except ModelLoadError:
            self._model = None
            raise
        except Exception as exc:
            self._model = None
            log.exception("GigaAM load failed")
            raise ModelLoadError(type(exc).__name__) from exc
        self.load_seconds = time.monotonic() - started
        log.info(
            "GigaAM loaded: model=%s device=%s in %.1fs",
            self._manager.manifest.model_name,
            self._device,
            self.load_seconds,
        )

    def _infer(self, samples: np.ndarray) -> str:
        import torch

        model = self._model
        with self._lock, torch.inference_mode():
            param = next(model.parameters())
            wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
            wav = wav.to(param.device).to(param.dtype).unsqueeze(0)
            length = torch.full([1], wav.shape[-1], device=param.device)
            encoded, encoded_len = model.forward(wav, length)
            decoded = model.decoding.decode(model.head, encoded, encoded_len)
        return str(decoded[0][0]).strip()

    def transcribe(self, segment: AudioSegment) -> TranscriptSegment:
        if self._model is None:
            raise TranscriptionError("model not loaded")
        if segment.samples is None:
            raise TranscriptionError("segment audio not loaded")
        if len(segment.samples) > GIGAAM_MAX_SEGMENT_S * SAMPLE_RATE:
            raise TranscriptionError("segment longer than GigaAM limit")
        started = time.monotonic()
        try:
            text = self._infer(segment.samples)
        except Exception as exc:
            log.exception("GigaAM inference failed")
            raise TranscriptionError(type(exc).__name__) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "Segment %s transcribed: audio=%.2fs inference=%dms",
            segment.id,
            segment.duration_s,
            elapsed_ms,
        )
        return TranscriptSegment(
            id=segment.id,
            seq=segment.seq,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            text=text,
            provider=f"gigaam:{self._manager.manifest.model_name}",
            inference_ms=elapsed_ms,
            overlap_prev_ms=segment.overlap_prev_ms,
        )

    def unload(self) -> None:
        self._model = None
