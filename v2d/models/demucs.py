"""
Demucs model wrapper for V2D.

Provides audio source separation to isolate vocals from background.
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
class SeparationResult:
    """Result of audio source separation."""
    vocals: np.ndarray
    background: np.ndarray
    sample_rate: int


class DemucsModel(BaseModel):
    """
    Wrapper for Demucs audio source separation model.

    Separates vocals from background music and effects.
    """

    name = "demucs"
    version = "htdemucs"
    vram_required_gb = 4.0
    supports_fp16 = True
    supports_cpu = True

    def __init__(
        self,
        resource_manager: ResourceManager | None = None,
        model_name: str = "htdemucs",
        shifts: int = 1,
        overlap: float = 0.25,
    ):
        """
        Initialize Demucs wrapper.

        Args:
            resource_manager: Resource manager for GPU allocation
            model_name: Demucs model name
            shifts: Number of random shifts for augmentation
            overlap: Overlap between chunks
        """
        super().__init__(resource_manager)
        self.model_name = model_name
        self.shifts = shifts
        self.overlap = overlap
        self.version = model_name

    def _load_model_impl(self, device: str, dtype: str) -> None:
        """Load the Demucs model."""
        from demucs.pretrained import get_model

        self.logger.info(f"Loading Demucs {self.model_name} on {device}")

        self._model = get_model(self.model_name)
        self._model.to(device)
        self._model.eval()

    def _unload_model_impl(self) -> None:
        """Unload the Demucs model."""
        if self._model is not None:
            del self._model
            self._model = None

    def separate(
        self,
        audio_path: Path | str,
        output_channels: int = 1,
    ) -> SeparationResult:
        """
        Separate vocals from background in an audio file.

        Args:
            audio_path: Path to audio file
            output_channels: Number of output channels (1=mono, 2=stereo)

        Returns:
            SeparationResult with vocals and background arrays
        """
        import torch
        import torchaudio
        from demucs.apply import apply_model

        device = self.resource_manager.get_device(self.vram_required_gb) if self.resource_manager else "cuda"
        dtype = self.resource_manager.get_dtype() if self.resource_manager else "float16"

        self.ensure_loaded(device=device, dtype=dtype)

        self.logger.info(f"Separating vocals: {audio_path}")

        waveform, sample_rate = torchaudio.load(str(audio_path))

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        waveform = waveform.unsqueeze(0).to(self._device)

        with torch.no_grad():
            sources = apply_model(
                self._model,
                waveform,
                shifts=self.shifts,
                overlap=self.overlap,
                device=self._device,
            )

        source_names = self._model.sources
        vocals_idx = source_names.index("vocals")

        vocals = sources[0, vocals_idx]

        background_indices = [i for i in range(len(source_names)) if i != vocals_idx]
        background = sources[0, background_indices].sum(dim=0)

        if output_channels == 1:
            vocals = vocals.mean(dim=0, keepdim=True)
            background = background.mean(dim=0, keepdim=True)

        vocals_np = vocals.cpu().numpy()
        background_np = background.cpu().numpy()

        self.logger.info("Vocal separation complete")

        return SeparationResult(
            vocals=vocals_np,
            background=background_np,
            sample_rate=sample_rate,
        )

    def separate_to_files(
        self,
        audio_path: Path | str,
        vocals_path: Path | str,
        background_path: Path | str,
        output_channels: int = 1,
    ) -> None:
        """
        Separate vocals and save to files.

        Args:
            audio_path: Input audio path
            vocals_path: Output path for vocals
            background_path: Output path for background
            output_channels: Number of output channels
        """
        import torch
        import torchaudio

        result = self.separate(audio_path, output_channels)

        vocals_tensor = torch.from_numpy(result.vocals)
        background_tensor = torch.from_numpy(result.background)

        torchaudio.save(str(vocals_path), vocals_tensor, result.sample_rate)
        torchaudio.save(str(background_path), background_tensor, result.sample_rate)

        self.logger.info(f"Saved vocals to: {vocals_path}")
        self.logger.info(f"Saved background to: {background_path}")
