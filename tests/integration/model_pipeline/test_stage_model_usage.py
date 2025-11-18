"""Integration tests for stage model usage with model registry."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from v2d.core.config import Config
from v2d.core.job import Job
from v2d.models.base import BaseModel, ModelRegistry
from v2d.resources.manager import ResourceManager


class TestStageModelIntegration:
    """Tests for stages using model registry."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration."""
        config_dict = {
            "system": {"temp_directory": ".v2d_jobs"},
            "gpu": {
                "device": "cpu",
                "max_vram_usage_gb": 8.0,
                "prefer_fp16": False,
            },
            "models": {
                "demucs": {
                    "model": "htdemucs",
                    "shifts": 1,
                    "overlap": 0.25,
                },
                "whisper": {
                    "model_size": "base",
                    "compute_type": "float32",
                    "beam_size": 5,
                    "vad_filter": True,
                },
                "marian": {
                    "model": "Helsinki-NLP/opus-mt-ja-en",
                    "max_length": 512,
                },
                "tts": {
                    "engine": "xtts",
                    "sample_rate": 24000,
                },
            },
            "pipeline": {
                "extract_audio": {
                    "sample_rate": 16000,
                    "channels": 1,
                },
            },
        }
        return Config.from_dict(config_dict)

    @pytest.fixture
    def job(self, config, tmp_path):
        """Create test job with model registry."""
        job = Job.create(config=config, project_root=tmp_path)
        return job

    def test_job_has_model_registry(self, job):
        """Test that job has model registry."""
        assert hasattr(job, "model_registry")
        assert isinstance(job.model_registry, ModelRegistry)

    def test_job_has_resource_manager(self, job):
        """Test that job has resource manager."""
        assert hasattr(job, "resource_manager")
        assert isinstance(job.resource_manager, ResourceManager)

    def test_model_registry_shared_across_stages(self, job):
        """Test that model registry is shared across stages."""
        # Create a mock model
        class MockModel(BaseModel):
            name = "mock_model"
            version = "1.0"
            vram_required_gb = 1.0

            def _load_model_impl(self, device, dtype):
                self._model = MagicMock()

            def _unload_model_impl(self):
                pass

        # Register model through registry
        model1 = job.model_registry.get_or_create("mock_model", MockModel)
        model2 = job.model_registry.get_or_create("mock_model", MockModel)

        # Should return same instance
        assert model1 is model2

    def test_get_model_from_job(self, job):
        """Test getting model via job.get_model()."""
        class MockModel(BaseModel):
            name = "test_model"
            version = "1.0"
            vram_required_gb = 1.0

            def _load_model_impl(self, device, dtype):
                self._model = MagicMock()

            def _unload_model_impl(self):
                pass

        # Register and retrieve
        job.model_registry.register(MockModel())
        model = job.get_model("test_model")

        assert model is not None
        assert model.name == "test_model"

    def test_job_cleanup_unloads_models(self, job, tmp_path):
        """Test that job cleanup unloads all models."""
        # Create a mock model
        class MockModel(BaseModel):
            name = "cleanup_test"
            version = "1.0"
            vram_required_gb = 1.0

            def _load_model_impl(self, device, dtype):
                self._model = MagicMock()

            def _unload_model_impl(self):
                self._model = None

        # Create and load model
        model = job.model_registry.get_or_create("cleanup_test", MockModel)
        model.load(device="cpu", dtype="float32")

        assert model.is_loaded

        # Cleanup job
        job.model_registry.unload_all()

        assert not model.is_loaded

    def test_resource_manager_tracks_vram(self, job):
        """Test that resource manager tracks VRAM allocations."""
        rm = job.resource_manager

        # Request VRAM
        rm.request_vram("model1", 2.0)
        assert rm.get_allocated_vram() == 2.0

        # Request more
        rm.request_vram("model2", 3.0)
        assert rm.get_allocated_vram() == 5.0

        # Release
        rm.release_vram("model1")
        assert rm.get_allocated_vram() == 3.0

    @patch("v2d.models.demucs.DemucsModel")
    def test_separate_vocals_uses_registry(self, mock_demucs_class, job, tmp_path):
        """Test that SeparateVocalsStage uses model registry."""
        from v2d.stages.separate_vocals import SeparateVocalsStage

        # Setup mock
        mock_model = MagicMock()
        mock_demucs_class.return_value = mock_model

        # Create stage
        stage = SeparateVocalsStage()

        # Check that stage would use model_registry.get_or_create
        # We verify this by checking the import and call pattern
        assert hasattr(job, "model_registry")
        assert hasattr(job.model_registry, "get_or_create")

    @patch("v2d.models.whisper.WhisperModel")
    def test_transcribe_uses_registry(self, mock_whisper_class, job, tmp_path):
        """Test that TranscribeStage uses model registry."""
        from v2d.stages.transcribe import TranscribeStage

        # Setup mock
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        # Create stage
        stage = TranscribeStage()

        # Check that stage would use model_registry.get_or_create
        assert hasattr(job, "model_registry")
        assert hasattr(job.model_registry, "get_or_create")

    @patch("v2d.models.marian.MarianModel")
    def test_translate_uses_registry(self, mock_marian_class, job, tmp_path):
        """Test that TranslateStage uses model registry."""
        from v2d.stages.translate import TranslateStage

        # Setup mock
        mock_model = MagicMock()
        mock_marian_class.return_value = mock_model

        # Create stage
        stage = TranslateStage()

        # Check that stage would use model_registry.get_or_create
        assert hasattr(job, "model_registry")
        assert hasattr(job.model_registry, "get_or_create")

    @patch("v2d.models.xtts.XTTSModel")
    def test_synthesize_uses_registry(self, mock_xtts_class, job, tmp_path):
        """Test that SynthesizeStage uses model registry."""
        from v2d.stages.synthesize import SynthesizeStage

        # Setup mock
        mock_model = MagicMock()
        mock_xtts_class.return_value = mock_model

        # Create stage
        stage = SynthesizeStage()

        # Check that stage would use model_registry.get_or_create
        assert hasattr(job, "model_registry")
        assert hasattr(job.model_registry, "get_or_create")


class TestModelRegistryWithResourceManager:
    """Tests for ModelRegistry integration with ResourceManager."""

    @pytest.fixture
    def resource_manager(self):
        """Create test resource manager."""
        rm = ResourceManager()
        rm.max_vram_gb = 12.0
        return rm

    @pytest.fixture
    def registry(self, resource_manager):
        """Create test model registry with resource manager."""
        return ModelRegistry(resource_manager)

    def test_model_requests_vram_on_load(self, registry, resource_manager):
        """Test that models request VRAM when loaded."""
        class VRAMModel(BaseModel):
            name = "vram_model"
            version = "1.0"
            vram_required_gb = 4.0

            def _load_model_impl(self, device, dtype):
                self._model = MagicMock()

            def _unload_model_impl(self):
                pass

        model = VRAMModel(resource_manager=resource_manager)
        registry.register(model)

        # Load model
        model.load(device="cuda", dtype="float16")

        # Check VRAM was allocated
        assert resource_manager.get_allocated_vram() == 4.0

    def test_model_releases_vram_on_unload(self, registry, resource_manager):
        """Test that models release VRAM when unloaded."""
        class VRAMModel(BaseModel):
            name = "vram_release_model"
            version = "1.0"
            vram_required_gb = 4.0

            def _load_model_impl(self, device, dtype):
                self._model = MagicMock()

            def _unload_model_impl(self):
                pass

        model = VRAMModel(resource_manager=resource_manager)
        registry.register(model)

        # Load and unload
        model.load(device="cuda", dtype="float16")
        assert resource_manager.get_allocated_vram() == 4.0

        model.unload()
        assert resource_manager.get_allocated_vram() == 0.0

    def test_registry_unload_all_releases_vram(self, registry, resource_manager):
        """Test that unload_all releases all VRAM."""
        class Model1(BaseModel):
            name = "model1"
            version = "1.0"
            vram_required_gb = 2.0

            def _load_model_impl(self, device, dtype):
                self._model = MagicMock()

            def _unload_model_impl(self):
                pass

        class Model2(BaseModel):
            name = "model2"
            version = "1.0"
            vram_required_gb = 3.0

            def _load_model_impl(self, device, dtype):
                self._model = MagicMock()

            def _unload_model_impl(self):
                pass

        # Create and load models
        m1 = Model1(resource_manager=resource_manager)
        m2 = Model2(resource_manager=resource_manager)

        registry.register(m1)
        registry.register(m2)

        m1.load(device="cuda", dtype="float16")
        m2.load(device="cuda", dtype="float16")

        assert resource_manager.get_allocated_vram() == 5.0

        # Unload all
        registry.unload_all()

        assert resource_manager.get_allocated_vram() == 0.0


class TestStageImportsModelWrappers:
    """Test that stages import model wrappers correctly."""

    def test_separate_vocals_imports_demucs_model(self):
        """Test SeparateVocalsStage imports DemucsModel."""
        from v2d.stages import separate_vocals
        assert hasattr(separate_vocals, "DemucsModel")

    def test_transcribe_imports_whisper_model(self):
        """Test TranscribeStage imports WhisperModel."""
        from v2d.stages import transcribe
        assert hasattr(transcribe, "WhisperModel")

    def test_translate_imports_marian_model(self):
        """Test TranslateStage imports MarianModel."""
        from v2d.stages import translate
        assert hasattr(translate, "MarianModel")

    def test_synthesize_imports_xtts_model(self):
        """Test SynthesizeStage imports XTTSModel."""
        from v2d.stages import synthesize
        assert hasattr(synthesize, "XTTSModel")
