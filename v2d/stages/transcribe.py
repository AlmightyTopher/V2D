"""
Transcribe Japanese audio using Faster-Whisper.

This stage uses the Whisper speech recognition model to transcribe
the isolated vocals into text with timing information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from v2d.core.job import Job
from v2d.models.whisper import WhisperModel
from v2d.stages.base import BaseStage, StageResult, ResourceRequirements


@dataclass
class TranscriptSegment:
    """A single segment of transcribed text with timing."""

    id: int
    start: float
    end: float
    text: str
    confidence: float

    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return self.end - self.start


class TranscribeStage(BaseStage):
    """
    Transcribe vocals using Faster-Whisper.

    Produces a transcript with timing information for each segment.
    """

    name = "transcribe"
    dependencies = ["separate_vocals"]

    def execute(self, job: Job) -> StageResult:
        """Transcribe the vocal track."""
        # Get input audio
        vocals_path = self.require_artifact(job, "vocals")

        # Output path
        transcript_path = self.get_artifact_path(job, "transcript.json")

        # Get config
        config = job.config.models.whisper

        self.logger.info(f"Transcribing with Whisper {config.model_size}")

        try:
            # Get or create Whisper model from registry
            whisper = job.model_registry.get_or_create(
                "whisper",
                WhisperModel,
                resource_manager=job.resource_manager,
                model_size=config.model_size,
                compute_type=config.compute_type,
                beam_size=config.beam_size,
                vad_filter=config.vad_filter,
            )

            # Transcribe using the model wrapper
            result = whisper.transcribe(vocals_path, language="ja")

            # Convert to our segment format
            segments: list[TranscriptSegment] = []
            total_confidence = 0.0

            for seg in result.segments:
                transcript_seg = TranscriptSegment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    confidence=seg.confidence,
                )
                segments.append(transcript_seg)
                total_confidence += seg.confidence

                self.logger.debug(
                    f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}"
                )

            # Calculate metrics
            avg_confidence = total_confidence / len(segments) if segments else 0
            total_duration = segments[-1].end if segments else 0

            self.logger.info(
                f"Transcribed {len(segments)} segments, "
                f"duration: {total_duration:.2f}s, "
                f"avg confidence: {avg_confidence:.2f}"
            )

            # Save transcript
            transcript_data = {
                "language": result.language,
                "language_probability": result.language_probability,
                "duration": result.duration,
                "segments": [asdict(s) for s in segments],
            }

            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)

            # Register artifact
            job.add_artifact(
                name="transcript",
                path=transcript_path,
                stage=self.name,
                metadata={
                    "segment_count": len(segments),
                    "language": result.language,
                    "duration": result.duration,
                    "average_confidence": avg_confidence,
                },
            )

            return StageResult.ok(
                artifacts=["transcript"],
                metrics={
                    "segment_count": len(segments),
                    "audio_duration": result.duration,
                    "average_confidence": avg_confidence,
                    "language": result.language,
                    "language_probability": result.language_probability,
                },
            )

        except ImportError as e:
            return StageResult.fail(f"faster-whisper not installed: {e}")
        except Exception as e:
            return StageResult.fail(f"Transcription failed: {e}")

    def validate_inputs(self, job: Job) -> StageResult:
        """Validate vocals audio exists."""
        try:
            self.require_artifact(job, "vocals")
            return StageResult.ok()
        except FileNotFoundError as e:
            return StageResult.fail(str(e))

    def validate_outputs(self, job: Job) -> StageResult:
        """Validate transcript was created."""
        artifact = job.manifest.get_artifact("transcript")
        if not artifact:
            return StageResult.fail("transcript artifact not registered")

        path = Path(artifact.path)
        if not path.exists():
            return StageResult.fail(f"Transcript file missing: {path}")

        # Verify it's valid JSON with segments
        try:
            data = json.loads(path.read_text())
            if "segments" not in data:
                return StageResult.fail("Transcript missing segments")
            if len(data["segments"]) == 0:
                return StageResult.fail("Transcript has no segments")
        except json.JSONDecodeError as e:
            return StageResult.fail(f"Invalid transcript JSON: {e}")

        return StageResult.ok()

    def estimate_resources(self, job: Job) -> ResourceRequirements:
        """Whisper large-v3 needs significant VRAM."""
        # VRAM requirements vary by model size
        vram_by_size = {
            "tiny": 1.0,
            "base": 1.5,
            "small": 2.0,
            "medium": 5.0,
            "large": 6.0,
            "large-v2": 6.0,
            "large-v3": 6.0,
        }
        model_size = job.config.models.whisper.model_size
        vram = vram_by_size.get(model_size, 6.0)

        return ResourceRequirements(
            gpu_vram_gb=vram,
            cpu_cores=4,
            ram_gb=8.0,
            disk_gb=0.1,  # Transcript is small
        )


def load_transcript(path: Path) -> list[TranscriptSegment]:
    """
    Load transcript from JSON file.

    Args:
        path: Path to transcript JSON

    Returns:
        List of TranscriptSegment objects
    """
    data = json.loads(path.read_text())
    return [
        TranscriptSegment(**seg)
        for seg in data["segments"]
    ]
