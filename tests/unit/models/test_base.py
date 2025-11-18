"""Unit tests for base model abstraction."""

import pytest
from unittest.mock import MagicMock, patch

from v2d.models.base import BaseModel, ModelState, ModelInfo, ModelRegistry


class ConcreteModel(BaseModel):
    """Concrete implementation for testing."""

    name = "test_model"
    version = "1.0.0"
    vram_required_gb = 2.0

    def _load_model_impl(self, device: str, dtype: str) -> None:
        self._model = MagicMock()


class TestBaseModel:
    """Tests for BaseModel."""

    def test_initial_state(self):
        """Test model starts in unloaded state."""
        model = ConcreteModel()
        assert model.state == ModelState.UNLOADED
        assert not model.is_loaded

    def test_get_info(self):
        """Test model info retrieval."""
        model = ConcreteModel()
        info = model.get_info()

        assert isinstance(info, ModelInfo)
        assert info.name == "test_model"
        assert info.version == "1.0.0"
        assert info.vram_required_gb == 2.0

    def test_load_changes_state(self):
        """Test loading changes state to loaded."""
        model = ConcreteModel()
        model.load(device="cpu", dtype="float32")

        assert model.state == ModelState.LOADED
        assert model.is_loaded
        assert model.device == "cpu"

    def test_unload_changes_state(self):
        """Test unloading changes state to unloaded."""
        model = ConcreteModel()
        model.load(device="cpu", dtype="float32")
        model.unload()

        assert model.state == ModelState.UNLOADED
        assert not model.is_loaded

    def test_load_idempotent(self):
        """Test loading same config is idempotent."""
        model = ConcreteModel()
        model.load(device="cpu", dtype="float32")
        model.load(device="cpu", dtype="float32")

        assert model.is_loaded

    def test_ensure_loaded(self):
        """Test ensure_loaded loads if needed."""
        model = ConcreteModel()
        model.ensure_loaded(device="cpu", dtype="float32")

        assert model.is_loaded

    def test_context_manager(self):
        """Test context manager unloads on exit."""
        with ConcreteModel() as model:
            model.load(device="cpu", dtype="float32")
            assert model.is_loaded

        assert not model.is_loaded

    def test_load_with_resource_manager(self):
        """Test loading with resource manager."""
        rm = MagicMock()
        rm.request_vram.return_value = True

        model = ConcreteModel(resource_manager=rm)
        model.load(device="cuda", dtype="float16")

        rm.request_vram.assert_called_once_with("test_model", 2.0)

    def test_unload_releases_resources(self):
        """Test unloading releases resources."""
        rm = MagicMock()
        rm.request_vram.return_value = True

        model = ConcreteModel(resource_manager=rm)
        model.load(device="cuda", dtype="float16")
        model.unload()

        rm.release_vram.assert_called_with("test_model")


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving models."""
        registry = ModelRegistry()
        model = ConcreteModel()

        registry.register(model)
        retrieved = registry.get("test_model")

        assert retrieved is model

    def test_get_nonexistent(self):
        """Test getting nonexistent model returns None."""
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_create(self):
        """Test get_or_create creates if needed."""
        registry = ModelRegistry()
        model = registry.get_or_create("test_model", ConcreteModel)

        assert model is not None
        assert registry.get("test_model") is model

    def test_unload_all(self):
        """Test unloading all models."""
        registry = ModelRegistry()
        model = ConcreteModel()
        model.load(device="cpu", dtype="float32")

        registry.register(model)
        registry.unload_all()

        assert not model.is_loaded

    def test_get_loaded_models(self):
        """Test listing loaded models."""
        registry = ModelRegistry()
        model1 = ConcreteModel()
        model1.name = "model1"
        model2 = ConcreteModel()
        model2.name = "model2"

        model1.load(device="cpu", dtype="float32")

        registry.register(model1)
        registry.register(model2)

        loaded = registry.get_loaded_models()
        assert "model1" in loaded
        assert "model2" not in loaded

    def test_get_total_vram_usage(self):
        """Test calculating total VRAM usage."""
        registry = ModelRegistry()
        model1 = ConcreteModel()
        model1.name = "model1"
        model1.vram_required_gb = 2.0
        model2 = ConcreteModel()
        model2.name = "model2"
        model2.vram_required_gb = 3.0

        model1.load(device="cpu", dtype="float32")

        registry.register(model1)
        registry.register(model2)

        total = registry.get_total_vram_usage()
        assert total == 2.0
