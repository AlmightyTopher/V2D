"""
Whisper model wrapper for V2D.

Provides speech-to-text transcription using Faster-Whisper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from v2d.models.base import BaseModel

if TYPE_CHECKING:
    from v2d.resources.manager import ResourceManager


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text."""
    id: int
    start: float
    end: float
    text: str
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float


class WhisperModel(BaseModel):
    """
    Wrapper for Faster-Whisper speech recognition model.

    Provides Japanese transcription with timing information.
    """

    name = "whisper"
    version = "large-v3"
    vram_required_gb = 6.0
    supports_fp16 = True
    supports_cpu = True

    VRAM_BY_SIZE = {
        "tiny": 1.0,
        "base": 1.5,
        "small": 2.0,
        "medium": 5.0,
        "large": 6.0,
        "large-v2": 6.0,
        "large-v3": 6.0,
    }

    def __init__(
        self,
        resource_manager: ResourceManager | None = None,
        model_size: str = "large-v3",
        compute_type: str = "float16",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        """
        Initialize Whisper wrapper.

        Args:
            resource_manager: Resource manager for GPU allocation
            model_size: Whisper model size
            compute_type: Compute type (float16, float32, int8)
            beam_size: Beam size for decoding
            vad_filter: Enable voice activity detection
        """
        super().__init__(resource_manager)
        self.model_size = model_size
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        self.version = model_size
        self.vram_required_gb = self.VRAM_BY_SIZE.get(model_size, 6.0)

    def _load_model_impl(self, device: str, dtype: str) -> None:
        """Load the Whisper model."""
        from faster_whisper import WhisperModel as FasterWhisperModel

        compute_type = self.compute_type
        if dtype == "float32":
            compute_type = "float32"
        elif dtype == "float16" and compute_type not in ("int8", "int8_float16"):
            compute_type = "float16"

        device_type = "cuda" if "cuda" in device else "cpu"
        if device_type == "cpu":
            compute_type = "float32"

        self.logger.info(
            f"Loading Whisper {self.model_size} on {device_type} ({compute_type})"
        )

        self._model = FasterWhisperModel(
            self.model_size,
            device=device_type,
            compute_type=compute_type,
        )

    def _unload_model_impl(self) -> None:
        """Unload the Whisper model."""
        if self._model is not None:
            del self._model
            self._model = None

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "ja",
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (default: Japanese)

        Returns:
            TranscriptionResult with segments and metadata
        """
        self.ensure_loaded(
            device=self.resource_manager.get_device(self.vram_required_gb) if self.resource_manager else "cuda",
            dtype=self.resource_manager.get_dtype() if self.resource_manager else "float16",
        )

        self.logger.info(f"Transcribing: {audio_path}")

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )

        segments: list[TranscriptionSegment] = []

        for i, segment in enumerate(segments_iter):
            if hasattr(segment, 'words') and segment.words:
                confidence = sum(w.probability for w in segment.words) / len(segment.words)
            else:
                confidence = 0.9

            seg = TranscriptionSegment(
                id=i,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=confidence,
            )
            segments.append(seg)

        self.logger.info(f"Transcribed {len(segments)} segments")

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )
