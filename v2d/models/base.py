"""
Base model abstraction for V2D.

Provides the abstract interface that all model wrappers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

from v2d.observability.logger import get_logger

if TYPE_CHECKING:
    from v2d.resources.manager import ResourceManager


class ModelState(str, Enum):
    """Model loading states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    version: str
    vram_required_gb: float
    supports_fp16: bool
    supports_cpu: bool


class BaseModel(ABC):
    """
    Abstract base class for all model wrappers.

    All model wrappers must implement this interface to ensure
    consistent behavior across the system.
    """

    name: str = "base"
    version: str = "0.0.0"

    # Resource requirements
    vram_required_gb: float = 1.0
    supports_fp16: bool = True
    supports_cpu: bool = True

    def __init__(self, resource_manager: ResourceManager | None = None):
        """
        Initialize the model wrapper.

        Args:
            resource_manager: Optional resource manager for GPU allocation
        """
        self.resource_manager = resource_manager
        self.logger = get_logger(f"v2d.models.{self.name}")
        self._state = ModelState.UNLOADED
        self._model: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float32"

    @property
    def state(self) -> ModelState:
        """Get current model state."""
        return self._state

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._state == ModelState.LOADED

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device

    def get_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name=self.name,
            version=self.version,
            vram_required_gb=self.vram_required_gb,
            supports_fp16=self.supports_fp16,
            supports_cpu=self.supports_cpu,
        )

    def load(self, device: str = "cuda", dtype: str = "float16") -> None:
        """
        Load the model to the specified device.

        Args:
            device: Target device ("cuda", "cuda:0", "cpu")
            dtype: Data type ("float16", "float32")
        """
        if self._state == ModelState.LOADED:
            if self._device == device and self._dtype == dtype:
                self.logger.debug(f"Model already loaded on {device}")
                return
            self.unload()

        self._state = ModelState.LOADING
        self._device = device
        self._dtype = dtype

        try:
            if self.resource_manager and "cuda" in device:
                self.resource_manager.request_vram(self.name, self.vram_required_gb)

            self.logger.info(f"Loading {self.name} on {device} ({dtype})")
            self._load_model_impl(device, dtype)
            self._state = ModelState.LOADED
            self.logger.info(f"Model {self.name} loaded successfully")

        except Exception as e:
            self._state = ModelState.ERROR
            if self.resource_manager:
                self.resource_manager.release_vram(self.name)
            self.logger.error(f"Failed to load {self.name}: {e}")
            raise

    def unload(self) -> None:
        """Unload the model and free resources."""
        if self._state == ModelState.UNLOADED:
            return

        self.logger.info(f"Unloading {self.name}")

        try:
            self._unload_model_impl()
        except Exception as e:
            self.logger.warning(f"Error during unload: {e}")

        if self.resource_manager:
            self.resource_manager.release_vram(self.name)

        self._model = None
        self._state = ModelState.UNLOADED
        self._clear_cuda_cache()

    def ensure_loaded(self, device: str = "cuda", dtype: str = "float16") -> None:
        """Ensure the model is loaded, loading if necessary."""
        if not self.is_loaded:
            self.load(device, dtype)

    @abstractmethod
    def _load_model_impl(self, device: str, dtype: str) -> None:
        """
        Implementation-specific model loading.

        Subclasses must implement this method.
        """
        pass

    def _unload_model_impl(self) -> None:
        """
        Implementation-specific model unloading.

        Override if special cleanup is needed.
        """
        pass

    def _clear_cuda_cache(self) -> None:
        """Clear CUDA cache to free memory."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload()
        return False

    def __del__(self):
        """Destructor - ensure model is unloaded."""
        try:
            if self._state == ModelState.LOADED:
                self.unload()
        except Exception:
            pass


class ModelRegistry:
    """
    Registry for managing model instances.

    Provides centralized access to model wrappers and handles
    lifecycle management.
    """

    def __init__(self, resource_manager: ResourceManager | None = None):
        """
        Initialize the model registry.

        Args:
            resource_manager: Resource manager for GPU allocation
        """
        self.resource_manager = resource_manager
        self._models: dict[str, BaseModel] = {}
        self.logger = get_logger("v2d.models.registry")

    def register(self, model: BaseModel) -> None:
        """
        Register a model with the registry.

        Args:
            model: Model instance to register
        """
        self._models[model.name] = model
        self.logger.debug(f"Registered model: {model.name}")

    def get(self, name: str) -> BaseModel | None:
        """
        Get a model by name.

        Args:
            name: Model name

        Returns:
            Model instance or None if not found
        """
        return self._models.get(name)

    def get_or_create(self, name: str, model_class: type[BaseModel]) -> BaseModel:
        """
        Get a model or create it if not registered.

        Args:
            name: Model name
            model_class: Class to instantiate if not found

        Returns:
            Model instance
        """
        if name not in self._models:
            model = model_class(self.resource_manager)
            self.register(model)
        return self._models[name]

    def unload_all(self) -> None:
        """Unload all registered models."""
        for name, model in self._models.items():
            if model.is_loaded:
                self.logger.info(f"Unloading {name}")
                model.unload()

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model names."""
        return [name for name, model in self._models.items() if model.is_loaded]

    def get_total_vram_usage(self) -> float:
        """Get total VRAM usage of loaded models in GB."""
        return sum(
            model.vram_required_gb
            for model in self._models.values()
            if model.is_loaded
        )
