"""
Extract audio from video using FFmpeg.

This is the first stage in the pipeline, extracting the audio track
from the input video for further processing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from v2d.core.job import Job
from v2d.stages.base import BaseStage, StageResult, ResourceRequirements


class ExtractAudioStage(BaseStage):
    """
    Extract audio track from input video.

    Uses FFmpeg to extract audio and convert to the configured
    format (default: WAV, mono, 24kHz).
    """

    name = "extract_audio"
    dependencies = []  # First stage, no dependencies

    def execute(self, job: Job) -> StageResult:
        """Extract audio from video file."""
        # Get input path from manifest
        input_path = Path(job.manifest.input_path)

        # Get output path
        output_path = self.get_artifact_path(job, "full_audio.wav")

        # Get config
        config = job.config.pipeline.extract_audio

        self.logger.info(f"Extracting audio from: {input_path}")

        try:
            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM format
                "-ar", str(config.sample_rate),  # Sample rate
                "-ac", str(config.channels),  # Channels
                "-y",  # Overwrite output
                str(output_path),
            ]

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown FFmpeg error"
                return StageResult.fail(f"FFmpeg failed: {error_msg}")

            # Verify output exists and has content
            if not output_path.exists():
                return StageResult.fail("FFmpeg did not create output file")

            if output_path.stat().st_size == 0:
                return StageResult.fail("FFmpeg created empty output file")

            # Get audio duration for metrics
            duration = self._get_audio_duration(output_path)

            # Register artifact with job
            job.add_artifact(
                name="full_audio",
                path=output_path,
                stage=self.name,
                metadata={
                    "sample_rate": config.sample_rate,
                    "channels": config.channels,
                    "duration_seconds": duration,
                },
            )

            self.logger.info(f"Extracted audio: {duration:.2f}s")

            return StageResult.ok(
                artifacts=["full_audio"],
                metrics={
                    "audio_duration_seconds": duration,
                    "output_size_bytes": output_path.stat().st_size,
                },
            )

        except subprocess.TimeoutExpired:
            return StageResult.fail("FFmpeg timed out after 10 minutes")
        except FileNotFoundError:
            return StageResult.fail("FFmpeg not found. Please install FFmpeg.")
        except Exception as e:
            return StageResult.fail(f"Audio extraction failed: {e}")

    def validate_inputs(self, job: Job) -> StageResult:
        """Validate input video exists."""
        input_path = Path(job.manifest.input_path)

        if not input_path.exists():
            return StageResult.fail(f"Input video not found: {input_path}")

        # Check it's a video file
        valid_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
        if input_path.suffix.lower() not in valid_extensions:
            return StageResult.fail(
                f"Invalid video format: {input_path.suffix}. "
                f"Supported: {valid_extensions}"
            )

        return StageResult.ok()

    def validate_outputs(self, job: Job) -> StageResult:
        """Validate audio was extracted."""
        artifact = job.manifest.get_artifact("full_audio")
        if not artifact:
            return StageResult.fail("full_audio artifact not registered")

        output_path = Path(artifact.path)
        if not output_path.exists():
            return StageResult.fail(f"Output file missing: {output_path}")

        return StageResult.ok()

    def estimate_resources(self, job: Job) -> ResourceRequirements:
        """FFmpeg is CPU-bound, no GPU needed."""
        return ResourceRequirements(
            gpu_vram_gb=0,
            cpu_cores=2,
            ram_gb=1.0,
            disk_gb=2.0,  # Audio file size estimate
        )

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except (ValueError, subprocess.SubprocessError):
            return 0.0
