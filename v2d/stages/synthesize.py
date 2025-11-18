"""
Synthesize English speech using TTS (XTTS/StyleTTS2).

This stage generates English voice audio from the translated text,
attempting to match the timing and characteristics of the original.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v2d.core.job import Job
from v2d.stages.base import BaseStage, StageResult, ResourceRequirements


class SynthesizeStage(BaseStage):
    """
    Synthesize English speech using TTS.

    Generates audio for each translated segment and produces
    individual segment files plus a combined track.
    """

    name = "synthesize"
    dependencies = ["translate", "separate_vocals"]

    def execute(self, job: Job) -> StageResult:
        """Synthesize speech for translated segments."""
        # Get input translation
        translation_path = self.require_artifact(job, "translation")
        vocals_path = self.require_artifact(job, "vocals")

        # Output path for combined audio
        synthesized_path = self.get_artifact_path(job, "synthesized.wav")

        # Get config
        tts_config = job.config.models.tts
        sample_rate = tts_config.sample_rate

        self.logger.info(f"Synthesizing with engine: {tts_config.engine}")

        try:
            # Load translation
            with open(translation_path, "r", encoding="utf-8") as f:
                translation_data = json.load(f)

            segments = translation_data["segments"]

            if not segments:
                return StageResult.fail("No segments to synthesize")

            # Import TTS library
            import torch
            import torchaudio
            from TTS.api import TTS

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")

            # Initialize TTS model
            self.logger.info("Loading XTTS model...")

            # Use XTTS v2 for voice cloning capability
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            tts.to(device)

            # Extract a short reference clip from the original vocals for voice cloning
            # Use first few seconds of vocals as reference
            self.logger.info("Extracting voice reference from original vocals...")
            reference_path = self.get_artifact_path(job, "voice_reference.wav")

            # Load vocals and extract reference clip (first 6 seconds)
            waveform, orig_sr = torchaudio.load(str(vocals_path))

            # Resample if needed
            if orig_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
                waveform = resampler(waveform)

            # Extract first 6 seconds (or full audio if shorter)
            max_samples = 6 * sample_rate
            reference_audio = waveform[:, :max_samples]

            # Save reference clip
            torchaudio.save(str(reference_path), reference_audio, sample_rate)

            # Synthesize each segment
            segment_audios: list[tuple[float, np.ndarray]] = []
            total_generated_duration = 0.0

            self.logger.info(f"Synthesizing {len(segments)} segments...")

            for i, seg in enumerate(segments):
                text = seg["translated_text"]
                start_time = seg["start"]

                if not text or text.strip() == "...":
                    # Skip empty segments
                    continue

                self.logger.debug(f"[{i+1}/{len(segments)}] {text[:50]}...")

                # Generate speech
                wav = tts.tts(
                    text=text,
                    speaker_wav=str(reference_path),
                    language="en",
                )

                # Convert to numpy array if needed
                if isinstance(wav, list):
                    wav = np.array(wav, dtype=np.float32)

                # Track segment
                segment_duration = len(wav) / sample_rate
                total_generated_duration += segment_duration
                segment_audios.append((start_time, wav))

                # Save individual segment
                segment_path = job.files.get_segment_path(i, "wav")
                segment_tensor = torch.tensor(wav).unsqueeze(0)
                torchaudio.save(str(segment_path), segment_tensor, sample_rate)

            if not segment_audios:
                return StageResult.fail("No audio segments were generated")

            # Combine segments into single track
            self.logger.info("Combining segments into final track...")

            # Get total duration from last segment end time
            last_segment = segments[-1]
            total_duration = last_segment["end"] + 2.0  # Add 2s padding
            total_samples = int(total_duration * sample_rate)

            # Create output buffer
            combined = np.zeros(total_samples, dtype=np.float32)

            # Place each segment at its timestamp
            for start_time, audio in segment_audios:
                start_sample = int(start_time * sample_rate)
                end_sample = start_sample + len(audio)

                # Ensure we don't overflow
                if end_sample > total_samples:
                    end_sample = total_samples
                    audio = audio[:end_sample - start_sample]

                # Mix audio (in case of overlap)
                combined[start_sample:end_sample] += audio

            # Normalize to prevent clipping
            max_val = np.abs(combined).max()
            if max_val > 0:
                combined = combined / max_val * 0.9

            # Save combined audio
            combined_tensor = torch.tensor(combined).unsqueeze(0)
            torchaudio.save(str(synthesized_path), combined_tensor, sample_rate)

            # Clean up
            del tts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Register artifacts
            job.add_artifact(
                name="synthesized",
                path=synthesized_path,
                stage=self.name,
                metadata={
                    "segment_count": len(segment_audios),
                    "sample_rate": sample_rate,
                    "duration": total_duration,
                },
            )

            job.add_artifact(
                name="voice_reference",
                path=reference_path,
                stage=self.name,
                metadata={"sample_rate": sample_rate},
            )

            self.logger.info(
                f"Synthesized {len(segment_audios)} segments, "
                f"total duration: {total_duration:.2f}s"
            )

            return StageResult.ok(
                artifacts=["synthesized", "voice_reference"],
                metrics={
                    "segment_count": len(segment_audios),
                    "total_duration": total_duration,
                    "generated_duration": total_generated_duration,
                },
            )

        except ImportError as e:
            return StageResult.fail(f"TTS not installed: {e}")
        except Exception as e:
            return StageResult.fail(f"Synthesis failed: {e}")

    def validate_inputs(self, job: Job) -> StageResult:
        """Validate translation and vocals exist."""
        try:
            self.require_artifact(job, "translation")
            self.require_artifact(job, "vocals")
            return StageResult.ok()
        except FileNotFoundError as e:
            return StageResult.fail(str(e))

    def validate_outputs(self, job: Job) -> StageResult:
        """Validate synthesized audio was created."""
        artifact = job.manifest.get_artifact("synthesized")
        if not artifact:
            return StageResult.fail("synthesized artifact not registered")

        path = Path(artifact.path)
        if not path.exists():
            return StageResult.fail(f"Synthesized audio missing: {path}")

        if path.stat().st_size == 0:
            return StageResult.fail("Synthesized audio is empty")

        return StageResult.ok()

    def estimate_resources(self, job: Job) -> ResourceRequirements:
        """XTTS requires significant GPU memory."""
        return ResourceRequirements(
            gpu_vram_gb=6.0,  # XTTS v2 needs ~4-6GB
            cpu_cores=4,
            ram_gb=8.0,
            disk_gb=2.0,  # Audio files
        )
