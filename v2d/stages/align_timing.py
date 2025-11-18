"""
Align synthesized audio timing with original video.

This stage adjusts the duration of the synthesized speech to match
the original video timing, using time-stretching techniques.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from v2d.core.job import Job
from v2d.stages.base import BaseStage, StageResult, ResourceRequirements


class AlignTimingStage(BaseStage):
    """
    Align synthesized audio timing with original.

    Uses time-stretching to match the synthesized audio duration
    to the original video duration.
    """

    name = "align_timing"
    dependencies = ["synthesize", "extract_audio"]

    def execute(self, job: Job) -> StageResult:
        """Align timing of synthesized audio."""
        # Get inputs
        synthesized_path = self.require_artifact(job, "synthesized")
        full_audio_path = self.require_artifact(job, "full_audio")

        # Output path
        aligned_path = self.get_artifact_path(job, "aligned.wav")

        # Get config
        timing_config = job.config.pipeline.timing

        self.logger.info("Aligning synthesized audio timing...")

        try:
            # Get durations
            synth_duration = self._get_audio_duration(synthesized_path)
            original_duration = self._get_audio_duration(full_audio_path)

            if synth_duration <= 0 or original_duration <= 0:
                return StageResult.fail("Could not determine audio durations")

            # Calculate stretch ratio
            stretch_ratio = original_duration / synth_duration

            self.logger.info(
                f"Original: {original_duration:.2f}s, "
                f"Synthesized: {synth_duration:.2f}s, "
                f"Ratio: {stretch_ratio:.3f}"
            )

            # Check if ratio is within acceptable bounds
            if stretch_ratio > timing_config.max_stretch_ratio:
                self.logger.warning(
                    f"Stretch ratio {stretch_ratio:.2f} exceeds max "
                    f"{timing_config.max_stretch_ratio}, clamping"
                )
                stretch_ratio = timing_config.max_stretch_ratio

            if stretch_ratio < timing_config.min_stretch_ratio:
                self.logger.warning(
                    f"Stretch ratio {stretch_ratio:.2f} below min "
                    f"{timing_config.min_stretch_ratio}, clamping"
                )
                stretch_ratio = timing_config.min_stretch_ratio

            # Use rubberband for high-quality time-stretching
            # If not available, fall back to FFmpeg atempo filter
            if self._has_rubberband():
                success = self._stretch_with_rubberband(
                    synthesized_path, aligned_path, stretch_ratio
                )
            else:
                success = self._stretch_with_ffmpeg(
                    synthesized_path, aligned_path, stretch_ratio
                )

            if not success:
                return StageResult.fail("Time-stretching failed")

            # Verify output
            if not aligned_path.exists():
                return StageResult.fail("Aligned audio not created")

            aligned_duration = self._get_audio_duration(aligned_path)

            # Pad or trim to exact duration if needed
            final_path = self._adjust_to_exact_duration(
                aligned_path, original_duration, job
            )

            # Register artifact
            job.add_artifact(
                name="aligned",
                path=final_path,
                stage=self.name,
                metadata={
                    "original_duration": original_duration,
                    "synthesized_duration": synth_duration,
                    "aligned_duration": aligned_duration,
                    "stretch_ratio": stretch_ratio,
                },
            )

            self.logger.info(
                f"Aligned audio: {aligned_duration:.2f}s "
                f"(target: {original_duration:.2f}s)"
            )

            return StageResult.ok(
                artifacts=["aligned"],
                metrics={
                    "original_duration": original_duration,
                    "synthesized_duration": synth_duration,
                    "aligned_duration": aligned_duration,
                    "stretch_ratio": stretch_ratio,
                },
            )

        except Exception as e:
            return StageResult.fail(f"Timing alignment failed: {e}")

    def _get_audio_duration(self, path: Path) -> float:
        """Get duration of audio file in seconds."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except (ValueError, subprocess.SubprocessError):
            return 0.0

    def _has_rubberband(self) -> bool:
        """Check if rubberband-cli is available."""
        try:
            result = subprocess.run(
                ["rubberband", "--version"],
                capture_output=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _stretch_with_rubberband(
        self, input_path: Path, output_path: Path, ratio: float
    ) -> bool:
        """Use rubberband for high-quality time-stretching."""
        try:
            # Rubberband uses time ratio (how much to stretch)
            cmd = [
                "rubberband",
                "--time", str(ratio),
                "--pitch", "1.0",  # Preserve pitch
                "--crisp", "3",  # High quality
                str(input_path),
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0

        except Exception as e:
            self.logger.warning(f"Rubberband failed: {e}")
            return False

    def _stretch_with_ffmpeg(
        self, input_path: Path, output_path: Path, ratio: float
    ) -> bool:
        """Use FFmpeg atempo filter for time-stretching."""
        try:
            # atempo filter range is 0.5 to 2.0
            # For ratios outside this range, we need to chain filters

            # Speed factor is inverse of stretch ratio
            # stretch_ratio > 1 means slower playback (longer duration)
            # stretch_ratio < 1 means faster playback (shorter duration)
            tempo = 1.0 / ratio

            # Build atempo filter chain
            atempo_filters = []
            remaining = tempo

            # Chain atempo filters to handle values outside 0.5-2.0 range
            while remaining > 2.0:
                atempo_filters.append("atempo=2.0")
                remaining /= 2.0
            while remaining < 0.5:
                atempo_filters.append("atempo=0.5")
                remaining /= 0.5

            atempo_filters.append(f"atempo={remaining}")
            filter_str = ",".join(atempo_filters)

            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-filter:a", filter_str,
                "-y",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0

        except Exception as e:
            self.logger.warning(f"FFmpeg atempo failed: {e}")
            return False

    def _adjust_to_exact_duration(
        self, audio_path: Path, target_duration: float, job: Job
    ) -> Path:
        """Adjust audio to exact target duration by padding or trimming."""
        try:
            current_duration = self._get_audio_duration(audio_path)
            difference = abs(current_duration - target_duration)

            # If within 100ms, don't adjust
            if difference < 0.1:
                return audio_path

            final_path = self.get_artifact_path(job, "aligned_final.wav")

            if current_duration < target_duration:
                # Pad with silence
                pad_duration = target_duration - current_duration
                cmd = [
                    "ffmpeg",
                    "-i", str(audio_path),
                    "-af", f"apad=pad_dur={pad_duration}",
                    "-y",
                    str(final_path),
                ]
            else:
                # Trim to target duration
                cmd = [
                    "ffmpeg",
                    "-i", str(audio_path),
                    "-t", str(target_duration),
                    "-y",
                    str(final_path),
                ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode == 0 and final_path.exists():
                return final_path
            else:
                return audio_path

        except Exception:
            return audio_path

    def validate_inputs(self, job: Job) -> StageResult:
        """Validate required inputs exist."""
        try:
            self.require_artifact(job, "synthesized")
            self.require_artifact(job, "full_audio")
            return StageResult.ok()
        except FileNotFoundError as e:
            return StageResult.fail(str(e))

    def validate_outputs(self, job: Job) -> StageResult:
        """Validate aligned audio was created."""
        artifact = job.manifest.get_artifact("aligned")
        if not artifact:
            return StageResult.fail("aligned artifact not registered")

        path = Path(artifact.path)
        if not path.exists():
            return StageResult.fail(f"Aligned audio missing: {path}")

        if path.stat().st_size == 0:
            return StageResult.fail("Aligned audio is empty")

        return StageResult.ok()

    def estimate_resources(self, job: Job) -> ResourceRequirements:
        """Time-stretching is CPU-bound."""
        return ResourceRequirements(
            gpu_vram_gb=0,
            cpu_cores=2,
            ram_gb=2.0,
            disk_gb=1.0,
        )
