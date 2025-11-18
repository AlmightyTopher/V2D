"""Model wrappers and abstractions."""

from v2d.models.base import (
    BaseModel,
    ModelState,
    ModelInfo,
    ModelRegistry,
)
from v2d.models.whisper import (
    WhisperModel,
    TranscriptionSegment,
    TranscriptionResult,
)
from v2d.models.demucs import (
    DemucsModel,
    SeparationResult,
)
from v2d.models.marian import (
    MarianModel,
    TranslationResult,
)
from v2d.models.xtts import (
    XTTSModel,
    SynthesisResult,
)

__all__ = [
    "BaseModel",
    "ModelState",
    "ModelInfo",
    "ModelRegistry",
    "WhisperModel",
    "TranscriptionSegment",
    "TranscriptionResult",
    "DemucsModel",
    "SeparationResult",
    "MarianModel",
    "TranslationResult",
    "XTTSModel",
    "SynthesisResult",
]
