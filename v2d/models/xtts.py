"""
XTTS model wrapper for V2D.

Provides text-to-speech synthesis using Coqui XTTS v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from v2d.models.base import BaseModel

if TYPE_CHECKING:
    from v2d.resources.manager import ResourceManager


@dataclass
class SynthesisResult:
    """Result of speech synthesis."""
    audio: np.ndarray
    sample_rate: int
    duration: float


class XTTSModel(BaseModel):
    """
    Wrapper for Coqui XTTS v2 text-to-speech model.

    Generates natural speech from text with voice cloning capability.
    """

    name = "xtts"
    version = "v2.0.2"
    vram_required_gb = 6.0
    supports_fp16 = True
    supports_cpu = True

    def __init__(
        self,
        resource_manager: ResourceManager | None = None,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        sample_rate: int = 24000,
    ):
        """
        Initialize XTTS wrapper.

        Args:
            resource_manager: Resource manager for GPU allocation
            model_name: TTS model name
            sample_rate: Output sample rate
        """
        super().__init__(resource_manager)
        self.model_name = model_name
        self.sample_rate = sample_rate

    def _load_model_impl(self, device: str, dtype: str) -> None:
        """Load the XTTS model."""
        from TTS.api import TTS

        self.logger.info(f"Loading XTTS on {device}")

        self._model = TTS(self.model_name)
        self._model.to(device)

    def _unload_model_impl(self) -> None:
        """Unload the XTTS model."""
        if self._model is not None:
            del self._model
            self._model = None

    def synthesize(
        self,
        text: str,
        speaker_wav: Path | str | None = None,
        language: str = "en",
    ) -> SynthesisResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker_wav: Reference audio for voice cloning
            language: Language code

        Returns:
            SynthesisResult with audio array
        """
        device = self.resource_manager.get_device(self.vram_required_gb) if self.resource_manager else "cuda"
        dtype = self.resource_manager.get_dtype() if self.resource_manager else "float16"

        self.ensure_loaded(device=device, dtype=dtype)

        if speaker_wav:
            wav = self._model.tts(
                text=text,
                speaker_wav=str(speaker_wav),
                language=language,
            )
        else:
            wav = self._model.tts(
                text=text,
                language=language,
            )

        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)

        duration = len(wav) / self.sample_rate

        return SynthesisResult(
            audio=wav,
            sample_rate=self.sample_rate,
            duration=duration,
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path | str,
        speaker_wav: Path | str | None = None,
        language: str = "en",
    ) -> float:
        """
        Synthesize speech and save to file.

        Args:
            text: Text to synthesize
            output_path: Output audio path
            speaker_wav: Reference audio for voice cloning
            language: Language code

        Returns:
            Duration in seconds
        """
        import torch
        import torchaudio

        result = self.synthesize(text, speaker_wav, language)

        audio_tensor = torch.from_numpy(result.audio).unsqueeze(0)
        torchaudio.save(str(output_path), audio_tensor, result.sample_rate)

        self.logger.debug(f"Saved synthesis to: {output_path}")

        return result.duration

    def synthesize_segments(
        self,
        segments: list[dict],
        speaker_wav: Path | str,
        output_dir: Path,
        language: str = "en",
    ) -> list[dict]:
        """
        Synthesize multiple segments.

        Args:
            segments: List of segment dicts with 'translated_text'
            speaker_wav: Reference audio for voice cloning
            output_dir: Directory for output files
            language: Language code

        Returns:
            List of dicts with segment info and audio paths
        """
        import torch
        import torchaudio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, seg in enumerate(segments):
            text = seg.get("translated_text", "")

            if not text or text.strip() == "...":
                continue

            result = self.synthesize(text, speaker_wav, language)

            output_path = output_dir / f"segment_{i:04d}.wav"
            audio_tensor = torch.from_numpy(result.audio).unsqueeze(0)
            torchaudio.save(str(output_path), audio_tensor, result.sample_rate)

            results.append({
                "id": seg.get("id", i),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": text,
                "audio_path": str(output_path),
                "duration": result.duration,
            })

        self.logger.info(f"Synthesized {len(results)} segments")

        return results

    def extract_speaker_embedding(
        self,
        audio_path: Path | str,
        max_duration: float = 6.0,
    ) -> Path:
        """
        Extract a speaker reference clip from audio.

        Args:
            audio_path: Source audio path
            max_duration: Maximum duration in seconds

        Returns:
            Path to extracted reference clip
        """
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        max_samples = int(max_duration * self.sample_rate)
        reference = waveform[:, :max_samples]

        output_path = Path(audio_path).parent / "voice_reference.wav"
        torchaudio.save(str(output_path), reference, self.sample_rate)

        self.logger.info(f"Extracted speaker reference: {output_path}")

        return output_path
