"""
Translate Japanese text to English using MarianMT.

This stage translates the transcribed Japanese text to English
while preserving segment timing information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from v2d.core.job import Job
from v2d.stages.base import BaseStage, StageResult, ResourceRequirements


@dataclass
class TranslatedSegment:
    """A translated segment with original and translated text."""

    id: int
    start: float
    end: float
    original_text: str
    translated_text: str
    char_ratio: float  # Ratio of translated chars to original

    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return self.end - self.start


class TranslateStage(BaseStage):
    """
    Translate transcription using MarianMT.

    Translates Japanese text to English segment by segment,
    preserving timing information.
    """

    name = "translate"
    dependencies = ["transcribe"]

    def execute(self, job: Job) -> StageResult:
        """Translate the transcript."""
        # Get input transcript
        transcript_path = self.require_artifact(job, "transcript")

        # Output path
        translation_path = self.get_artifact_path(job, "translation.json")

        # Get config
        config = job.config.models.marian

        self.logger.info(f"Translating with model: {config.model}")

        try:
            # Import here to avoid loading at module level
            from transformers import MarianMTModel, MarianTokenizer
            import torch

            # Load transcript
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)

            segments = transcript_data["segments"]

            if not segments:
                return StageResult.fail("No segments to translate")

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")

            # Load model and tokenizer
            self.logger.info(f"Loading MarianMT model: {config.model}")
            tokenizer = MarianTokenizer.from_pretrained(config.model)
            model = MarianMTModel.from_pretrained(config.model)
            model.to(device)
            model.eval()

            # Translate each segment
            translated_segments: list[TranslatedSegment] = []
            total_original_chars = 0
            total_translated_chars = 0

            self.logger.info(f"Translating {len(segments)} segments...")

            # Process in batches for efficiency
            batch_size = 8
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]
                texts = [seg["text"] for seg in batch]

                # Skip empty texts
                texts = [t if t.strip() else "..." for t in texts]

                # Tokenize
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                ).to(device)

                # Generate translation
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=config.max_length,
                        num_beams=4,
                        early_stopping=True,
                    )

                # Decode
                translations = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                )

                # Create translated segments
                for j, (seg, translation) in enumerate(zip(batch, translations)):
                    original = seg["text"]
                    translated = translation.strip()

                    # Calculate character ratio
                    orig_len = len(original)
                    trans_len = len(translated)
                    ratio = trans_len / orig_len if orig_len > 0 else 1.0

                    total_original_chars += orig_len
                    total_translated_chars += trans_len

                    trans_seg = TranslatedSegment(
                        id=seg["id"],
                        start=seg["start"],
                        end=seg["end"],
                        original_text=original,
                        translated_text=translated,
                        char_ratio=ratio,
                    )
                    translated_segments.append(trans_seg)

                    self.logger.debug(
                        f"[{trans_seg.start:.2f}s] {original} -> {translated}"
                    )

            # Calculate overall metrics
            overall_ratio = (
                total_translated_chars / total_original_chars
                if total_original_chars > 0 else 1.0
            )

            self.logger.info(
                f"Translated {len(translated_segments)} segments, "
                f"char ratio: {overall_ratio:.2f}"
            )

            # Save translation
            translation_data = {
                "source_language": "ja",
                "target_language": "en",
                "model": config.model,
                "segments": [asdict(s) for s in translated_segments],
                "metrics": {
                    "total_original_chars": total_original_chars,
                    "total_translated_chars": total_translated_chars,
                    "overall_char_ratio": overall_ratio,
                },
            }

            with open(translation_path, "w", encoding="utf-8") as f:
                json.dump(translation_data, f, ensure_ascii=False, indent=2)

            # Clean up model
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Register artifact
            job.add_artifact(
                name="translation",
                path=translation_path,
                stage=self.name,
                metadata={
                    "segment_count": len(translated_segments),
                    "char_ratio": overall_ratio,
                },
            )

            return StageResult.ok(
                artifacts=["translation"],
                metrics={
                    "segment_count": len(translated_segments),
                    "original_chars": total_original_chars,
                    "translated_chars": total_translated_chars,
                    "char_ratio": overall_ratio,
                },
            )

        except ImportError as e:
            return StageResult.fail(f"transformers not installed: {e}")
        except Exception as e:
            return StageResult.fail(f"Translation failed: {e}")

    def validate_inputs(self, job: Job) -> StageResult:
        """Validate transcript exists."""
        try:
            self.require_artifact(job, "transcript")
            return StageResult.ok()
        except FileNotFoundError as e:
            return StageResult.fail(str(e))

    def validate_outputs(self, job: Job) -> StageResult:
        """Validate translation was created."""
        artifact = job.manifest.get_artifact("translation")
        if not artifact:
            return StageResult.fail("translation artifact not registered")

        path = Path(artifact.path)
        if not path.exists():
            return StageResult.fail(f"Translation file missing: {path}")

        # Verify it's valid JSON with segments
        try:
            data = json.loads(path.read_text())
            if "segments" not in data:
                return StageResult.fail("Translation missing segments")
        except json.JSONDecodeError as e:
            return StageResult.fail(f"Invalid translation JSON: {e}")

        return StageResult.ok()

    def estimate_resources(self, job: Job) -> ResourceRequirements:
        """MarianMT is relatively lightweight."""
        return ResourceRequirements(
            gpu_vram_gb=2.0,
            cpu_cores=2,
            ram_gb=4.0,
            disk_gb=0.1,
        )


def load_translation(path: Path) -> list[TranslatedSegment]:
    """
    Load translation from JSON file.

    Args:
        path: Path to translation JSON

    Returns:
        List of TranslatedSegment objects
    """
    data = json.loads(path.read_text())
    return [
        TranslatedSegment(**seg)
        for seg in data["segments"]
    ]
