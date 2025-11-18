"""
Resource manager for V2D.

Coordinates GPU memory allocation, model lifecycle, and system resources.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

from v2d.core.config import Config
from v2d.observability.logger import get_logger
from v2d.resources.gpu import (
    is_cuda_available,
    get_free_vram,
    get_total_vram,
    get_optimal_device,
    clear_cuda_cache,
    get_memory_status as get_gpu_memory_status,
)
from v2d.resources.memory import (
    get_available_ram,
    check_memory_available,
    check_disk_available,
)


class ResourceError(Exception):
    """Base exception for resource errors."""
    pass


class InsufficientVRAMError(ResourceError):
    """Raised when there is not enough VRAM."""
    pass


class InsufficientRAMError(ResourceError):
    """Raised when there is not enough RAM."""
    pass


class InsufficientDiskError(ResourceError):
    """Raised when there is not enough disk space."""
    pass


@dataclass
class VRAMAllocation:
    """Tracks a VRAM allocation."""
    name: str
    size_gb: float
    allocated_at: datetime
    device: str


@dataclass
class ResourceSession:
    """Context for a resource allocation session."""
    session_id: str
    started_at: datetime
    allocations: list[str] = field(default_factory=list)


class ResourceManager:
    """
    Manages system resources for V2D processing.

    Handles:
    - VRAM allocation and tracking
    - Model lifecycle management
    - Automatic cleanup of unused resources
    - Resource availability checks
    """

    def __init__(self, config: Config | None = None):
        """
        Initialize the resource manager.

        Args:
            config: V2D configuration
        """
        self.config = config
        self.logger = get_logger("v2d.resources.manager")

        self._vram_allocations: dict[str, VRAMAllocation] = {}
        self._sessions: dict[str, ResourceSession] = {}

        if config:
            self.max_vram_gb = config.gpu.max_vram_usage_gb
            self.prefer_fp16 = config.gpu.prefer_fp16
            self.device = config.gpu.device
        else:
            self.max_vram_gb = 12.0
            self.prefer_fp16 = True
            self.device = "cuda:0" if is_cuda_available() else "cpu"

    def get_device(self, required_vram_gb: float = 0) -> str:
        """
        Get the appropriate device for the given VRAM requirement.

        Args:
            required_vram_gb: Required VRAM in GB

        Returns:
            Device string
        """
        if not is_cuda_available():
            return "cpu"

        if required_vram_gb > 0:
            return get_optimal_device(required_vram_gb)

        return self.device

    def get_dtype(self) -> str:
        """
        Get the recommended data type.

        Returns:
            "float16" or "float32"
        """
        return "float16" if self.prefer_fp16 else "float32"

    def request_vram(self, name: str, size_gb: float) -> bool:
        """
        Request VRAM allocation.

        Args:
            name: Name of the allocation (usually model name)
            size_gb: Size in GB

        Returns:
            True if allocation succeeded

        Raises:
            InsufficientVRAMError: If not enough VRAM available
        """
        if not is_cuda_available():
            return True

        current_usage = self.get_allocated_vram()
        available = self.max_vram_gb - current_usage

        if size_gb > available:
            free_vram = get_free_vram()
            self.logger.warning(
                f"VRAM request for {name} ({size_gb:.1f} GB) exceeds limit. "
                f"Allocated: {current_usage:.1f} GB, "
                f"Free: {free_vram:.1f} GB, "
                f"Limit: {self.max_vram_gb:.1f} GB"
            )
            raise InsufficientVRAMError(
                f"Not enough VRAM for {name}: need {size_gb:.1f} GB, "
                f"available {available:.1f} GB"
            )

        allocation = VRAMAllocation(
            name=name,
            size_gb=size_gb,
            allocated_at=datetime.now(),
            device=self.device,
        )
        self._vram_allocations[name] = allocation

        self.logger.debug(
            f"Allocated {size_gb:.1f} GB VRAM for {name}. "
            f"Total: {self.get_allocated_vram():.1f}/{self.max_vram_gb:.1f} GB"
        )

        return True

    def release_vram(self, name: str) -> None:
        """
        Release a VRAM allocation.

        Args:
            name: Name of the allocation to release
        """
        if name in self._vram_allocations:
            allocation = self._vram_allocations.pop(name)
            self.logger.debug(
                f"Released {allocation.size_gb:.1f} GB VRAM for {name}. "
                f"Total: {self.get_allocated_vram():.1f}/{self.max_vram_gb:.1f} GB"
            )

    def get_allocated_vram(self) -> float:
        """
        Get total allocated VRAM.

        Returns:
            Allocated VRAM in GB
        """
        return sum(a.size_gb for a in self._vram_allocations.values())

    def get_available_vram(self) -> float:
        """
        Get available VRAM within the limit.

        Returns:
            Available VRAM in GB
        """
        return max(0, self.max_vram_gb - self.get_allocated_vram())

    def get_allocations(self) -> list[VRAMAllocation]:
        """
        Get all current allocations.

        Returns:
            List of allocations
        """
        return list(self._vram_allocations.values())

    def check_resources(
        self,
        vram_gb: float = 0,
        ram_gb: float = 0,
        disk_gb: float = 0,
        disk_path: str = ".",
    ) -> bool:
        """
        Check if required resources are available.

        Args:
            vram_gb: Required VRAM in GB
            ram_gb: Required RAM in GB
            disk_gb: Required disk space in GB
            disk_path: Path for disk check

        Returns:
            True if all resources are available
        """
        if vram_gb > 0 and is_cuda_available():
            if vram_gb > self.get_available_vram():
                return False

        if ram_gb > 0:
            if not check_memory_available(ram_gb):
                return False

        if disk_gb > 0:
            if not check_disk_available(disk_gb, disk_path):
                return False

        return True

    def ensure_resources(
        self,
        vram_gb: float = 0,
        ram_gb: float = 0,
        disk_gb: float = 0,
        disk_path: str = ".",
    ) -> None:
        """
        Ensure required resources are available.

        Args:
            vram_gb: Required VRAM in GB
            ram_gb: Required RAM in GB
            disk_gb: Required disk space in GB
            disk_path: Path for disk check

        Raises:
            InsufficientVRAMError: If not enough VRAM
            InsufficientRAMError: If not enough RAM
            InsufficientDiskError: If not enough disk space
        """
        if vram_gb > 0 and is_cuda_available():
            available = self.get_available_vram()
            if vram_gb > available:
                raise InsufficientVRAMError(
                    f"Need {vram_gb:.1f} GB VRAM, available: {available:.1f} GB"
                )

        if ram_gb > 0:
            available = get_available_ram()
            if ram_gb > available:
                raise InsufficientRAMError(
                    f"Need {ram_gb:.1f} GB RAM, available: {available:.1f} GB"
                )

        if disk_gb > 0:
            from v2d.resources.memory import get_free_disk
            available = get_free_disk(disk_path)
            if disk_gb > available:
                raise InsufficientDiskError(
                    f"Need {disk_gb:.1f} GB disk space, available: {available:.1f} GB"
                )

    @contextmanager
    def resource_session(self, session_id: str) -> Generator[ResourceSession, None, None]:
        """
        Context manager for a resource session.

        Automatically releases allocations when the session ends.

        Args:
            session_id: Unique session identifier

        Yields:
            ResourceSession object
        """
        session = ResourceSession(
            session_id=session_id,
            started_at=datetime.now(),
        )
        self._sessions[session_id] = session

        try:
            yield session
        finally:
            for name in session.allocations:
                self.release_vram(name)

            del self._sessions[session_id]

    @contextmanager
    def timed_operation(self, name: str) -> Generator[dict[str, Any], None, None]:
        """
        Context manager for timing an operation.

        Args:
            name: Operation name

        Yields:
            Dict to store timing info
        """
        result: dict[str, Any] = {"name": name}
        start = time.perf_counter()

        try:
            yield result
        finally:
            end = time.perf_counter()
            result["duration_seconds"] = end - start
            self.logger.debug(f"{name} completed in {result['duration_seconds']:.2f}s")

    def cleanup(self) -> None:
        """Clean up all resources and clear caches."""
        self._vram_allocations.clear()
        clear_cuda_cache()
        self.logger.info("Resource cleanup complete")

    def log_status(self) -> None:
        """Log current resource status."""
        allocated = self.get_allocated_vram()
        available = self.get_available_vram()

        self.logger.info(
            f"Resources: VRAM {allocated:.1f}/{self.max_vram_gb:.1f} GB allocated, "
            f"{available:.1f} GB available"
        )

        if is_cuda_available():
            status = get_gpu_memory_status()
            if status:
                self.logger.info(
                    f"GPU memory: {status.used_gb:.1f}/{status.total_gb:.1f} GB used"
                )

        ram_available = get_available_ram()
        self.logger.info(f"RAM available: {ram_available:.1f} GB")

        for name, allocation in self._vram_allocations.items():
            self.logger.debug(
                f"  {name}: {allocation.size_gb:.1f} GB "
                f"(since {allocation.allocated_at.strftime('%H:%M:%S')})"
            )
