"""Integration tests for model pipeline."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from v2d.models import ModelRegistry
from v2d.models.base import BaseModel
from v2d.resources.manager import ResourceManager


class MockModel(BaseModel):
    """Mock model for testing."""

    name = "mock_model"
    version = "1.0"
    vram_required_gb = 2.0

    def __init__(self, resource_manager=None):
        super().__init__(resource_manager)
        self.load_count = 0
        self.unload_count = 0

    def _load_model_impl(self, device: str, dtype: str) -> None:
        self._model = MagicMock()
        self.load_count += 1

    def _unload_model_impl(self) -> None:
        self.unload_count += 1


class TestModelResourceIntegration:
    """Tests for model and resource manager integration."""

    def test_model_uses_resource_manager(self):
        """Test that model uses resource manager for allocation."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        model = MockModel(resource_manager=rm)
        model.load(device="cuda", dtype="float16")

        assert rm.get_allocated_vram() == 2.0
        assert model.is_loaded

    def test_model_releases_on_unload(self):
        """Test that model releases resources on unload."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        model = MockModel(resource_manager=rm)
        model.load(device="cuda", dtype="float16")
        model.unload()

        assert rm.get_allocated_vram() == 0.0
        assert not model.is_loaded

    def test_multiple_models_share_manager(self):
        """Test multiple models sharing resource manager."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        model1 = MockModel(resource_manager=rm)
        model1.name = "model1"
        model1.vram_required_gb = 3.0

        model2 = MockModel(resource_manager=rm)
        model2.name = "model2"
        model2.vram_required_gb = 4.0

        model1.load(device="cuda", dtype="float16")
        model2.load(device="cuda", dtype="float16")

        assert rm.get_allocated_vram() == 7.0

        model1.unload()
        assert rm.get_allocated_vram() == 4.0

    def test_registry_with_resource_manager(self):
        """Test model registry with resource manager."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        registry = ModelRegistry(resource_manager=rm)

        model = MockModel(resource_manager=rm)
        registry.register(model)

        model.load(device="cuda", dtype="float16")

        assert registry.get_total_vram_usage() == 2.0
        assert "mock_model" in registry.get_loaded_models()

    def test_registry_unload_all(self):
        """Test registry unloads all models."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        registry = ModelRegistry(resource_manager=rm)

        model1 = MockModel(resource_manager=rm)
        model1.name = "model1"
        model2 = MockModel(resource_manager=rm)
        model2.name = "model2"

        registry.register(model1)
        registry.register(model2)

        model1.load(device="cuda", dtype="float16")
        model2.load(device="cuda", dtype="float16")

        registry.unload_all()

        assert rm.get_allocated_vram() == 0.0
        assert len(registry.get_loaded_models()) == 0

    def test_context_manager_cleanup(self):
        """Test context manager cleans up resources."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        with MockModel(resource_manager=rm) as model:
            model.load(device="cuda", dtype="float16")
            assert rm.get_allocated_vram() == 2.0

        assert rm.get_allocated_vram() == 0.0

    def test_resource_session_cleanup(self):
        """Test resource session cleans up on exit."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        with rm.resource_session("test") as session:
            rm.request_vram("model1", 4.0)
            session.allocations.append("model1")

            rm.request_vram("model2", 3.0)
            session.allocations.append("model2")

            assert rm.get_allocated_vram() == 7.0

        assert rm.get_allocated_vram() == 0.0


class TestModelSequencing:
    """Tests for sequential model loading/unloading."""

    def test_sequential_loading(self):
        """Test loading models sequentially."""
        rm = ResourceManager()
        rm.max_vram_gb = 8.0

        model1 = MockModel(resource_manager=rm)
        model1.name = "model1"
        model1.vram_required_gb = 6.0

        model2 = MockModel(resource_manager=rm)
        model2.name = "model2"
        model2.vram_required_gb = 6.0

        model1.load(device="cuda", dtype="float16")
        assert model1.is_loaded
        assert rm.get_allocated_vram() == 6.0

        model1.unload()
        assert rm.get_allocated_vram() == 0.0

        model2.load(device="cuda", dtype="float16")
        assert model2.is_loaded
        assert rm.get_allocated_vram() == 6.0

    def test_model_reuse(self):
        """Test reusing a model without reloading."""
        rm = ResourceManager()
        model = MockModel(resource_manager=rm)

        model.load(device="cuda", dtype="float16")
        model.load(device="cuda", dtype="float16")

        assert model.load_count == 1
