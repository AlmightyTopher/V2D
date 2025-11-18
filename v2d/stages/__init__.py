"""Pipeline stages for video processing."""

from v2d.stages.base import BaseStage, StageResult, ResourceRequirements
from v2d.stages.extract_audio import ExtractAudioStage
from v2d.stages.separate_vocals import SeparateVocalsStage
from v2d.stages.transcribe import TranscribeStage, TranscriptSegment, load_transcript
from v2d.stages.translate import TranslateStage, TranslatedSegment, load_translation
from v2d.stages.synthesize import SynthesizeStage
from v2d.stages.align_timing import AlignTimingStage
from v2d.stages.mux_output import MuxOutputStage

__all__ = [
    # Base
    "BaseStage",
    "StageResult",
    "ResourceRequirements",
    # Stages
    "ExtractAudioStage",
    "SeparateVocalsStage",
    "TranscribeStage",
    "TranslateStage",
    "SynthesizeStage",
    "AlignTimingStage",
    "MuxOutputStage",
    # Data types
    "TranscriptSegment",
    "TranslatedSegment",
    # Utilities
    "load_transcript",
    "load_translation",
]


def get_phase1_stages() -> list[BaseStage]:
    """
    Get all stages for Phase 1 pipeline in execution order.

    Returns:
        List of stage instances
    """
    return [
        ExtractAudioStage(),
        SeparateVocalsStage(),
        TranscribeStage(),
        TranslateStage(),
        SynthesizeStage(),
        AlignTimingStage(),
        MuxOutputStage(),
    ]


def register_phase1_pipeline(pipeline) -> None:
    """
    Register all Phase 1 stages with a pipeline.

    Args:
        pipeline: Pipeline instance to register stages with
    """
    for stage in get_phase1_stages():
        pipeline.register_stage(stage)
