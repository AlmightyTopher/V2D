"""
Separate vocals from background audio using Demucs.

This stage uses the Demucs neural network to isolate vocals
from music and other background sounds.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from v2d.core.job import Job
from v2d.stages.base import BaseStage, StageResult, ResourceRequirements


class SeparateVocalsStage(BaseStage):
    """
    Separate vocals from background using Demucs.

    Produces two outputs:
    - vocals.wav: Isolated vocal track
    - no_vocals.wav: Background music/effects
    """

    name = "separate_vocals"
    dependencies = ["extract_audio"]

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None

    def execute(self, job: Job) -> StageResult:
        """Separate vocals from background audio."""
        # Get input audio
        input_path = self.require_artifact(job, "full_audio")

        # Output paths
        vocals_path = self.get_artifact_path(job, "vocals.wav")
        no_vocals_path = self.get_artifact_path(job, "no_vocals.wav")

        # Get config
        config = job.config.models.demucs

        self.logger.info(f"Separating vocals using model: {config.model}")

        try:
            # Import here to avoid loading at module level
            import torch
            import torchaudio
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            # Determine device
            device = job.config.gpu.device if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")

            # Load model
            self.logger.info(f"Loading Demucs model: {config.model}")
            model = get_model(config.model)
            model.to(device)
            model.eval()

            # Load audio
            self.logger.info(f"Loading audio: {input_path}")
            waveform, sample_rate = torchaudio.load(str(input_path))

            # Demucs expects stereo, convert if mono
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            # Add batch dimension
            waveform = waveform.unsqueeze(0).to(device)

            # Apply model
            self.logger.info("Running source separation...")
            with torch.no_grad():
                sources = apply_model(
                    model,
                    waveform,
                    shifts=config.shifts,
                    overlap=config.overlap,
                    device=device,
                )

            # Demucs htdemucs outputs: drums, bass, other, vocals
            # Index 3 is vocals
            source_names = model.sources
            vocals_idx = source_names.index("vocals")

            # Extract vocals and combine others for background
            vocals = sources[0, vocals_idx]  # Shape: [2, samples]

            # Combine non-vocal sources for background
            background_indices = [i for i in range(len(source_names)) if i != vocals_idx]
            background = sources[0, background_indices].sum(dim=0)

            # Convert to mono if configured
            if job.config.pipeline.extract_audio.channels == 1:
                vocals = vocals.mean(dim=0, keepdim=True)
                background = background.mean(dim=0, keepdim=True)

            # Move to CPU for saving
            vocals = vocals.cpu()
            background = background.cpu()

            # Save outputs
            self.logger.info(f"Saving vocals to: {vocals_path}")
            torchaudio.save(
                str(vocals_path),
                vocals,
                sample_rate,
            )

            self.logger.info(f"Saving background to: {no_vocals_path}")
            torchaudio.save(
                str(no_vocals_path),
                background,
                sample_rate,
            )

            # Clear GPU memory
            del model, waveform, sources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Register artifacts
            job.add_artifact(
                name="vocals",
                path=vocals_path,
                stage=self.name,
                metadata={"sample_rate": sample_rate},
            )

            job.add_artifact(
                name="no_vocals",
                path=no_vocals_path,
                stage=self.name,
                metadata={"sample_rate": sample_rate},
            )

            return StageResult.ok(
                artifacts=["vocals", "no_vocals"],
                metrics={
                    "vocals_size_bytes": vocals_path.stat().st_size,
                    "background_size_bytes": no_vocals_path.stat().st_size,
                    "device": device,
                },
            )

        except ImportError as e:
            return StageResult.fail(f"Demucs not installed: {e}")
        except Exception as e:
            return StageResult.fail(f"Vocal separation failed: {e}")

    def validate_inputs(self, job: Job) -> StageResult:
        """Validate input audio exists."""
        try:
            self.require_artifact(job, "full_audio")
            return StageResult.ok()
        except FileNotFoundError as e:
            return StageResult.fail(str(e))

    def validate_outputs(self, job: Job) -> StageResult:
        """Validate separation outputs exist."""
        for artifact_name in ["vocals", "no_vocals"]:
            artifact = job.manifest.get_artifact(artifact_name)
            if not artifact:
                return StageResult.fail(f"{artifact_name} artifact not registered")

            path = Path(artifact.path)
            if not path.exists():
                return StageResult.fail(f"Output file missing: {path}")

            if path.stat().st_size == 0:
                return StageResult.fail(f"Output file is empty: {path}")

        return StageResult.ok()

    def estimate_resources(self, job: Job) -> ResourceRequirements:
        """Demucs requires significant GPU memory."""
        return ResourceRequirements(
            gpu_vram_gb=4.0,  # htdemucs needs ~4GB VRAM
            cpu_cores=4,
            ram_gb=8.0,
            disk_gb=1.0,
        )
