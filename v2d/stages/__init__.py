"""Pipeline stages for video processing."""

from v2d.stages.base import BaseStage, StageResult, ResourceRequirements
from v2d.stages.extract_audio import ExtractAudioStage
from v2d.stages.separate_vocals import SeparateVocalsStage
from v2d.stages.transcribe import TranscribeStage, TranscriptSegment, load_transcript
from v2d.stages.translate import TranslateStage, TranslatedSegment, load_translation
from v2d.stages.synthesize import SynthesizeStage
from v2d.stages.align_timing import AlignTimingStage
from v2d.stages.mux_output import MuxOutputStage
from v2d.stages.pipelines import (
    get_phase1_stages,
    build_phase1_pipeline,
    register_phase1_pipeline,
)

__all__ = [
    "BaseStage",
    "StageResult",
    "ResourceRequirements",
    "ExtractAudioStage",
    "SeparateVocalsStage",
    "TranscribeStage",
    "TranslateStage",
    "SynthesizeStage",
    "AlignTimingStage",
    "MuxOutputStage",
    "TranscriptSegment",
    "TranslatedSegment",
    "load_transcript",
    "load_translation",
    "get_phase1_stages",
    "build_phase1_pipeline",
    "register_phase1_pipeline",
]
