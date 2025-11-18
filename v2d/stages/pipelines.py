"""
Pipeline builders for V2D.

Provides functions to construct and configure pipelines for different phases.
"""

from __future__ import annotations

from v2d.core.config import Config
from v2d.core.pipeline import Pipeline
from v2d.stages.base import BaseStage
from v2d.stages.extract_audio import ExtractAudioStage
from v2d.stages.separate_vocals import SeparateVocalsStage
from v2d.stages.transcribe import TranscribeStage
from v2d.stages.translate import TranslateStage
from v2d.stages.synthesize import SynthesizeStage
from v2d.stages.align_timing import AlignTimingStage
from v2d.stages.mux_output import MuxOutputStage


def get_phase1_stages(config: Config) -> list[BaseStage]:
    """
    Get all stages for Phase 1 pipeline in execution order.

    Args:
        config: V2D configuration

    Returns:
        List of stage instances in correct execution order
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


def register_phase1_pipeline(pipeline: Pipeline, config: Config) -> None:
    """
    Register all Phase 1 stages with a pipeline.

    Args:
        pipeline: Pipeline instance to register stages with
        config: V2D configuration
    """
    stages = get_phase1_stages(config)
    for stage in stages:
        pipeline.register_stage(stage)


def build_phase1_pipeline(config: Config) -> Pipeline:
    """
    Build a complete Phase 1 pipeline ready for execution.

    Args:
        config: V2D configuration

    Returns:
        Configured Pipeline instance with all Phase 1 stages registered
    """
    pipeline = Pipeline(config)
    register_phase1_pipeline(pipeline, config)
    return pipeline
