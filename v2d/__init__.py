"""
V2D - Video to Dub

A fully local, open-source AI dubbing system that transforms Japanese-language
videos into high-quality English-dubbed versions.
"""

__version__ = "0.1.0"
__author__ = "V2D Contributors"

from v2d.core.config import Config, load_config
from v2d.core.job import Job, JobState
from v2d.core.pipeline import Pipeline

__all__ = [
    "Config",
    "load_config",
    "Job",
    "JobState",
    "Pipeline",
    "__version__",
]
