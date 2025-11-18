"""
Separate vocals from background audio using Demucs.

This stage uses the Demucs neural network to isolate vocals
from music and other background sounds.
"""

from __future__ import annotations

from pathlib import Path

from v2d.core.job import Job
from v2d.models.demucs import DemucsModel
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
            # Get or create Demucs model from registry
            demucs = job.model_registry.get_or_create(
                "demucs",
                DemucsModel,
                resource_manager=job.resource_manager,
                model_name=config.model,
                shifts=config.shifts,
                overlap=config.overlap,
            )

            # Determine output channels
            output_channels = job.config.pipeline.extract_audio.channels

            # Perform separation using the model wrapper
            result = demucs.separate(input_path, output_channels)

            # Save results to files
            import torch
            import torchaudio

            vocals_tensor = torch.from_numpy(result.vocals)
            background_tensor = torch.from_numpy(result.background)

            torchaudio.save(str(vocals_path), vocals_tensor, result.sample_rate)
            torchaudio.save(str(no_vocals_path), background_tensor, result.sample_rate)

            # Register artifacts
            job.add_artifact(
                name="vocals",
                path=vocals_path,
                stage=self.name,
                metadata={"sample_rate": result.sample_rate},
            )

            job.add_artifact(
                name="no_vocals",
                path=no_vocals_path,
                stage=self.name,
                metadata={"sample_rate": result.sample_rate},
            )

            return StageResult.ok(
                artifacts=["vocals", "no_vocals"],
                metrics={
                    "vocals_size_bytes": vocals_path.stat().st_size,
                    "background_size_bytes": no_vocals_path.stat().st_size,
                    "device": demucs.device or "cpu",
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
