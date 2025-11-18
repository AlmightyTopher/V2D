"""
Mux final video with dubbed audio using FFmpeg.

This is the final stage in the pipeline, combining the original video
with the new dubbed audio track and background music/effects.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from v2d.core.job import Job
from v2d.stages.base import BaseStage, StageResult, ResourceRequirements


class MuxOutputStage(BaseStage):
    """
    Mux video with dubbed audio track.

    Combines:
    - Original video track
    - Aligned dubbed vocals
    - Original background music/effects

    Produces the final dubbed video file.
    """

    name = "mux_output"
    dependencies = ["align_timing", "separate_vocals"]

    def execute(self, job: Job) -> StageResult:
        """Mux video with new audio track."""
        # Get inputs
        aligned_path = self.require_artifact(job, "aligned")
        no_vocals_path = self.require_artifact(job, "no_vocals")
        input_video = Path(job.manifest.input_path)

        # Output path
        output_dir = job.project_root / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        input_stem = input_video.stem
        output_path = output_dir / f"{input_stem}_dubbed.mp4"

        # Ensure unique filename
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{input_stem}_dubbed_{counter}.mp4"
            counter += 1

        self.logger.info(f"Creating final output: {output_path}")

        try:
            # First, mix the dubbed vocals with background
            mixed_audio_path = self.get_artifact_path(job, "mixed_audio.wav")

            success = self._mix_audio_tracks(
                aligned_path, no_vocals_path, mixed_audio_path, job
            )

            if not success:
                return StageResult.fail("Audio mixing failed")

            # Then mux with video
            success = self._mux_video_audio(
                input_video, mixed_audio_path, output_path
            )

            if not success:
                return StageResult.fail("Video muxing failed")

            # Verify output
            if not output_path.exists():
                return StageResult.fail("Output video not created")

            if output_path.stat().st_size == 0:
                return StageResult.fail("Output video is empty")

            # Get output info
            duration = self._get_video_duration(output_path)
            file_size = output_path.stat().st_size

            # Register artifacts
            job.add_artifact(
                name="mixed_audio",
                path=mixed_audio_path,
                stage=self.name,
            )

            job.add_artifact(
                name="output_video",
                path=output_path,
                stage=self.name,
                metadata={
                    "duration": duration,
                    "file_size": file_size,
                },
            )

            self.logger.info(
                f"Output created: {output_path} "
                f"({file_size / 1024 / 1024:.1f} MB, {duration:.1f}s)"
            )

            return StageResult.ok(
                artifacts=["mixed_audio", "output_video"],
                metrics={
                    "output_path": str(output_path),
                    "duration": duration,
                    "file_size_bytes": file_size,
                },
            )

        except Exception as e:
            return StageResult.fail(f"Muxing failed: {e}")

    def _mix_audio_tracks(
        self,
        vocals_path: Path,
        background_path: Path,
        output_path: Path,
        job: Job,
    ) -> bool:
        """Mix dubbed vocals with background audio."""
        try:
            # Get the vocal/background balance
            # Vocals should be prominent, background slightly reduced
            vocal_volume = 1.0
            background_volume = 0.7  # Slightly reduce background

            cmd = [
                "ffmpeg",
                "-i", str(vocals_path),
                "-i", str(background_path),
                "-filter_complex",
                f"[0:a]volume={vocal_volume}[v];"
                f"[1:a]volume={background_volume}[b];"
                f"[v][b]amix=inputs=2:duration=longest:dropout_transition=2",
                "-ac", "2",  # Stereo output
                "-y",
                str(output_path),
            ]

            self.logger.debug(f"Mixing audio: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                self.logger.error(f"Audio mix failed: {result.stderr[-500:]}")
                return False

            return output_path.exists()

        except subprocess.TimeoutExpired:
            self.logger.error("Audio mixing timed out")
            return False
        except Exception as e:
            self.logger.error(f"Audio mixing error: {e}")
            return False

    def _mux_video_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> bool:
        """Mux video with new audio track."""
        try:
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",  # Copy video stream (no re-encoding)
                "-c:a", "aac",  # Encode audio as AAC
                "-b:a", "192k",  # Audio bitrate
                "-map", "0:v:0",  # Use video from first input
                "-map", "1:a:0",  # Use audio from second input
                "-shortest",  # End when shortest stream ends
                "-y",
                str(output_path),
            ]

            self.logger.debug(f"Muxing video: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                self.logger.error(f"Video mux failed: {result.stderr[-500:]}")
                return False

            return output_path.exists()

        except subprocess.TimeoutExpired:
            self.logger.error("Video muxing timed out")
            return False
        except Exception as e:
            self.logger.error(f"Video muxing error: {e}")
            return False

    def _get_video_duration(self, path: Path) -> float:
        """Get duration of video file in seconds."""
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

    def validate_inputs(self, job: Job) -> StageResult:
        """Validate required inputs exist."""
        try:
            self.require_artifact(job, "aligned")
            self.require_artifact(job, "no_vocals")

            # Check original video exists
            input_video = Path(job.manifest.input_path)
            if not input_video.exists():
                return StageResult.fail(f"Input video not found: {input_video}")

            return StageResult.ok()

        except FileNotFoundError as e:
            return StageResult.fail(str(e))

    def validate_outputs(self, job: Job) -> StageResult:
        """Validate output video was created."""
        artifact = job.manifest.get_artifact("output_video")
        if not artifact:
            return StageResult.fail("output_video artifact not registered")

        path = Path(artifact.path)
        if not path.exists():
            return StageResult.fail(f"Output video missing: {path}")

        if path.stat().st_size == 0:
            return StageResult.fail("Output video is empty")

        # Verify it's a valid video
        duration = self._get_video_duration(path)
        if duration <= 0:
            return StageResult.fail("Output video has no duration")

        return StageResult.ok()

    def estimate_resources(self, job: Job) -> ResourceRequirements:
        """Muxing is I/O and CPU bound."""
        return ResourceRequirements(
            gpu_vram_gb=0,
            cpu_cores=2,
            ram_gb=2.0,
            disk_gb=5.0,  # Output file size
        )
