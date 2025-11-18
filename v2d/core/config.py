"""
Configuration loading and validation for V2D.

Provides type-safe configuration with JSON schema validation and sensible defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


@dataclass
class SystemConfig:
    """System-level configuration."""
    max_concurrent_jobs: int = 1
    temp_directory: str = "data/jobs"
    log_level: str = "INFO"
    log_retention_days: int = 30


@dataclass
class GPUConfig:
    """GPU configuration."""
    device: str = "cuda:0"
    max_vram_usage_gb: float = 12.0
    prefer_fp16: bool = True


@dataclass
class WhisperConfig:
    """Whisper model configuration."""
    model_size: str = "large-v3"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True


@dataclass
class DemucsConfig:
    """Demucs model configuration."""
    model: str = "htdemucs"
    shifts: int = 1
    overlap: float = 0.25


@dataclass
class MarianConfig:
    """MarianMT translation configuration."""
    model: str = "Helsinki-NLP/opus-mt-ja-en"
    max_length: int = 512


@dataclass
class XTTSConfig:
    """XTTS TTS configuration."""
    model_version: str = "v2.0.2"


@dataclass
class StyleTTS2Config:
    """StyleTTS2 TTS configuration."""
    config_path: str = "StyleTTS2/Models/LJSpeech/config.yml"


@dataclass
class TTSConfig:
    """TTS engine configuration."""
    engine: Literal["xtts", "styletts2"] = "xtts"
    sample_rate: int = 24000
    xtts: XTTSConfig = field(default_factory=XTTSConfig)
    styletts2: StyleTTS2Config = field(default_factory=StyleTTS2Config)


@dataclass
class ModelsConfig:
    """All model configurations."""
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    demucs: DemucsConfig = field(default_factory=DemucsConfig)
    marian: MarianConfig = field(default_factory=MarianConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


@dataclass
class ExtractAudioConfig:
    """Audio extraction configuration."""
    sample_rate: int = 24000
    channels: int = 1
    format: str = "wav"


@dataclass
class TimingConfig:
    """Timing and stretching configuration."""
    max_stretch_ratio: float = 1.3
    min_stretch_ratio: float = 0.8
    target_silence_gap_ms: int = 200


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    phases: list[str] = field(default_factory=lambda: ["phase1"])
    extract_audio: ExtractAudioConfig = field(default_factory=ExtractAudioConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)


@dataclass
class CacheConfig:
    """Cache configuration."""
    voice_cache_path: str = "data/cache/voices"
    model_cache_path: str = "data/cache/models"
    max_voice_cache_entries: int = 1000
    voice_cache_ttl_days: int = 90


@dataclass
class CleanupConfig:
    """Cleanup configuration."""
    auto_cleanup_on_success: bool = True
    keep_checkpoints: bool = False
    dry_run: bool = False


@dataclass
class Config:
    """Main V2D configuration container."""

    system: SystemConfig = field(default_factory=SystemConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)

    # Runtime metadata
    _config_path: Path | None = field(default=None, repr=False)
    _project_root: Path = field(default_factory=Path.cwd, repr=False)

    def resolve_path(self, path: str) -> Path:
        """Resolve a relative path to absolute using project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self._project_root / p

    @classmethod
    def from_dict(cls, data: dict[str, Any], project_root: Path | None = None) -> Config:
        """Create Config from dictionary."""
        root = project_root or Path.cwd()

        # Build nested configs
        system = SystemConfig(**data.get("system", {}))
        gpu = GPUConfig(**data.get("gpu", {}))

        # Models config
        models_data = data.get("models", {})
        tts_data = models_data.get("tts", {})
        tts_config = TTSConfig(
            engine=tts_data.get("engine", "xtts"),
            sample_rate=tts_data.get("sample_rate", 24000),
            xtts=XTTSConfig(**tts_data.get("xtts", {})),
            styletts2=StyleTTS2Config(**tts_data.get("styletts2", {})),
        )
        models = ModelsConfig(
            whisper=WhisperConfig(**models_data.get("whisper", {})),
            demucs=DemucsConfig(**models_data.get("demucs", {})),
            marian=MarianConfig(**models_data.get("marian", {})),
            tts=tts_config,
        )

        # Pipeline config
        pipeline_data = data.get("pipeline", {})
        pipeline = PipelineConfig(
            phases=pipeline_data.get("phases", ["phase1"]),
            extract_audio=ExtractAudioConfig(**pipeline_data.get("extract_audio", {})),
            timing=TimingConfig(**pipeline_data.get("timing", {})),
        )

        cache = CacheConfig(**data.get("cache", {}))
        cleanup = CleanupConfig(**data.get("cleanup", {}))

        return cls(
            system=system,
            gpu=gpu,
            models=models,
            pipeline=pipeline,
            cache=cache,
            cleanup=cleanup,
            _project_root=root,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "system": {
                "max_concurrent_jobs": self.system.max_concurrent_jobs,
                "temp_directory": self.system.temp_directory,
                "log_level": self.system.log_level,
                "log_retention_days": self.system.log_retention_days,
            },
            "gpu": {
                "device": self.gpu.device,
                "max_vram_usage_gb": self.gpu.max_vram_usage_gb,
                "prefer_fp16": self.gpu.prefer_fp16,
            },
            "models": {
                "whisper": {
                    "model_size": self.models.whisper.model_size,
                    "compute_type": self.models.whisper.compute_type,
                    "beam_size": self.models.whisper.beam_size,
                    "vad_filter": self.models.whisper.vad_filter,
                },
                "demucs": {
                    "model": self.models.demucs.model,
                    "shifts": self.models.demucs.shifts,
                    "overlap": self.models.demucs.overlap,
                },
                "marian": {
                    "model": self.models.marian.model,
                    "max_length": self.models.marian.max_length,
                },
                "tts": {
                    "engine": self.models.tts.engine,
                    "sample_rate": self.models.tts.sample_rate,
                    "xtts": {
                        "model_version": self.models.tts.xtts.model_version,
                    },
                    "styletts2": {
                        "config_path": self.models.tts.styletts2.config_path,
                    },
                },
            },
            "pipeline": {
                "phases": self.pipeline.phases,
                "extract_audio": {
                    "sample_rate": self.pipeline.extract_audio.sample_rate,
                    "channels": self.pipeline.extract_audio.channels,
                    "format": self.pipeline.extract_audio.format,
                },
                "timing": {
                    "max_stretch_ratio": self.pipeline.timing.max_stretch_ratio,
                    "min_stretch_ratio": self.pipeline.timing.min_stretch_ratio,
                    "target_silence_gap_ms": self.pipeline.timing.target_silence_gap_ms,
                },
            },
            "cache": {
                "voice_cache_path": self.cache.voice_cache_path,
                "model_cache_path": self.cache.model_cache_path,
                "max_voice_cache_entries": self.cache.max_voice_cache_entries,
                "voice_cache_ttl_days": self.cache.voice_cache_ttl_days,
            },
            "cleanup": {
                "auto_cleanup_on_success": self.cleanup.auto_cleanup_on_success,
                "keep_checkpoints": self.cleanup.keep_checkpoints,
                "dry_run": self.cleanup.dry_run,
            },
        }


class ConfigValidator:
    """Validates configuration against schema and semantic rules."""

    def __init__(self, schema_path: Path | None = None):
        self.schema: dict[str, Any] | None = None
        if schema_path and schema_path.exists():
            self.schema = json.loads(schema_path.read_text())

    def validate(self, config_dict: dict[str, Any]) -> list[str]:
        """
        Validate configuration dictionary.

        Returns list of error messages (empty if valid).
        """
        errors: list[str] = []

        # JSON Schema validation (if available)
        if HAS_JSONSCHEMA and self.schema:
            try:
                jsonschema.validate(config_dict, self.schema)
            except jsonschema.ValidationError as e:
                errors.append(f"Schema error: {e.message}")

        # Semantic validation
        errors.extend(self._validate_semantics(config_dict))

        return errors

    def _validate_semantics(self, config: dict[str, Any]) -> list[str]:
        """Validate semantic correctness of configuration."""
        errors: list[str] = []

        # GPU validation
        gpu = config.get("gpu", {})
        max_vram = gpu.get("max_vram_usage_gb", 12)
        if max_vram > 48:
            errors.append(f"max_vram_usage_gb ({max_vram}) exceeds reasonable limit of 48GB")
        if max_vram < 4:
            errors.append(f"max_vram_usage_gb ({max_vram}) is below minimum requirement of 4GB")

        # Timing validation
        timing = config.get("pipeline", {}).get("timing", {})
        max_stretch = timing.get("max_stretch_ratio", 1.3)
        min_stretch = timing.get("min_stretch_ratio", 0.8)

        if max_stretch < 1.0:
            errors.append(f"max_stretch_ratio ({max_stretch}) must be >= 1.0")
        if min_stretch > 1.0:
            errors.append(f"min_stretch_ratio ({min_stretch}) must be <= 1.0")
        if min_stretch <= 0:
            errors.append(f"min_stretch_ratio ({min_stretch}) must be > 0")
        if max_stretch > 2.0:
            errors.append(f"max_stretch_ratio ({max_stretch}) > 2.0 will cause severe audio artifacts")

        # System validation
        system = config.get("system", {})
        max_jobs = system.get("max_concurrent_jobs", 1)
        if max_jobs < 1:
            errors.append(f"max_concurrent_jobs ({max_jobs}) must be >= 1")
        if max_jobs > 10:
            errors.append(f"max_concurrent_jobs ({max_jobs}) > 10 may cause resource exhaustion")

        log_retention = system.get("log_retention_days", 30)
        if log_retention < 1:
            errors.append(f"log_retention_days ({log_retention}) must be >= 1")

        # Cache validation
        cache = config.get("cache", {})
        max_entries = cache.get("max_voice_cache_entries", 1000)
        if max_entries < 10:
            errors.append(f"max_voice_cache_entries ({max_entries}) must be >= 10")

        ttl_days = cache.get("voice_cache_ttl_days", 90)
        if ttl_days < 1:
            errors.append(f"voice_cache_ttl_days ({ttl_days}) must be >= 1")

        # TTS validation
        tts = config.get("models", {}).get("tts", {})
        engine = tts.get("engine", "xtts")
        if engine not in ("xtts", "styletts2"):
            errors.append(f"tts.engine must be 'xtts' or 'styletts2', got '{engine}'")

        sample_rate = tts.get("sample_rate", 24000)
        if sample_rate not in (16000, 22050, 24000, 44100, 48000):
            errors.append(f"tts.sample_rate ({sample_rate}) is non-standard; use 16000, 22050, 24000, 44100, or 48000")

        # Whisper validation
        whisper = config.get("models", {}).get("whisper", {})
        model_size = whisper.get("model_size", "large-v3")
        valid_sizes = ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
        if model_size not in valid_sizes:
            errors.append(f"whisper.model_size '{model_size}' not in {valid_sizes}")

        compute_type = whisper.get("compute_type", "float16")
        valid_compute = ("float16", "float32", "int8", "int8_float16")
        if compute_type not in valid_compute:
            errors.append(f"whisper.compute_type '{compute_type}' not in {valid_compute}")

        return errors


def load_config(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
    validate: bool = True,
) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses config/default.yaml
        project_root: Project root directory. If None, uses current directory
        validate: Whether to validate the configuration

    Returns:
        Loaded and validated Config object

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config file not found
    """
    root = project_root or Path.cwd()

    if config_path is None:
        config_path = root / "config" / "default.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    # Load YAML
    if not config_path.exists():
        # Return default config if no file exists
        return Config(_project_root=root)

    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}

    # Validate if requested
    if validate:
        schema_path = root / "config" / "schema.json"
        validator = ConfigValidator(schema_path if schema_path.exists() else None)
        errors = validator.validate(config_dict)
        if errors:
            raise ConfigValidationError(errors)

    # Build config object
    config = Config.from_dict(config_dict, root)
    config._config_path = config_path

    return config


def get_default_config() -> Config:
    """Get a Config object with all default values."""
    return Config()
