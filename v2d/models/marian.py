"""
MarianMT model wrapper for V2D.

Provides Japanese to English translation using MarianMT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from v2d.models.base import BaseModel

if TYPE_CHECKING:
    from v2d.resources.manager import ResourceManager


@dataclass
class TranslationResult:
    """Result of a translation."""
    original: str
    translated: str
    char_ratio: float


class MarianModel(BaseModel):
    """
    Wrapper for MarianMT translation model.

    Translates Japanese text to English.
    """

    name = "marian"
    version = "opus-mt-ja-en"
    vram_required_gb = 2.0
    supports_fp16 = True
    supports_cpu = True

    def __init__(
        self,
        resource_manager: ResourceManager | None = None,
        model_name: str = "Helsinki-NLP/opus-mt-ja-en",
        max_length: int = 512,
    ):
        """
        Initialize MarianMT wrapper.

        Args:
            resource_manager: Resource manager for GPU allocation
            model_name: HuggingFace model name
            max_length: Maximum sequence length
        """
        super().__init__(resource_manager)
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None

    def _load_model_impl(self, device: str, dtype: str) -> None:
        """Load the MarianMT model and tokenizer."""
        from transformers import MarianMTModel, MarianTokenizer

        self.logger.info(f"Loading MarianMT {self.model_name} on {device}")

        self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self._model = MarianMTModel.from_pretrained(self.model_name)
        self._model.to(device)
        self._model.eval()

    def _unload_model_impl(self) -> None:
        """Unload the MarianMT model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

    def translate(self, text: str) -> TranslationResult:
        """
        Translate a single text.

        Args:
            text: Japanese text to translate

        Returns:
            TranslationResult with translation
        """
        results = self.translate_batch([text])
        return results[0]

    def translate_batch(self, texts: list[str]) -> list[TranslationResult]:
        """
        Translate a batch of texts.

        Args:
            texts: List of Japanese texts

        Returns:
            List of TranslationResults
        """
        import torch

        device = self.resource_manager.get_device(self.vram_required_gb) if self.resource_manager else "cuda"
        dtype = self.resource_manager.get_dtype() if self.resource_manager else "float16"

        self.ensure_loaded(device=device, dtype=dtype)

        clean_texts = [t if t.strip() else "..." for t in texts]

        inputs = self._tokenizer(
            clean_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )

        translations = self._tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        results = []
        for original, translated in zip(texts, translations):
            translated = translated.strip()
            orig_len = len(original)
            trans_len = len(translated)
            ratio = trans_len / orig_len if orig_len > 0 else 1.0

            results.append(TranslationResult(
                original=original,
                translated=translated,
                char_ratio=ratio,
            ))

        self.logger.debug(f"Translated {len(texts)} texts")

        return results

    def translate_segments(
        self,
        segments: list[dict],
        batch_size: int = 8,
    ) -> list[dict]:
        """
        Translate transcript segments.

        Args:
            segments: List of segment dicts with 'text' field
            batch_size: Batch size for translation

        Returns:
            List of dicts with 'original_text' and 'translated_text'
        """
        results = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            texts = [seg.get("text", "") for seg in batch]

            translations = self.translate_batch(texts)

            for seg, trans in zip(batch, translations):
                results.append({
                    "id": seg.get("id", i),
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "original_text": trans.original,
                    "translated_text": trans.translated,
                    "char_ratio": trans.char_ratio,
                })

        self.logger.info(f"Translated {len(results)} segments")

        return results
