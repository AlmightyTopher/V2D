"""
GPU monitoring and management for V2D.

Provides utilities for monitoring GPU memory and utilization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v2d.observability.logger import get_logger

logger = get_logger("v2d.resources.gpu")


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_percent: float
    temperature_celsius: float | None


@dataclass
class GPUMemoryStatus:
    """Current GPU memory status."""
    total_gb: float
    used_gb: float
    free_gb: float
    utilization_percent: float


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device_count() -> int:
    """Get number of available CUDA devices."""
    if not is_cuda_available():
        return 0
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def get_gpu_info(device_index: int = 0) -> GPUInfo | None:
    """
    Get detailed information about a GPU.

    Args:
        device_index: GPU device index

    Returns:
        GPUInfo or None if not available
    """
    if not is_cuda_available():
        return None

    try:
        import torch

        if device_index >= torch.cuda.device_count():
            return None

        props = torch.cuda.get_device_properties(device_index)
        total_memory = props.total_memory / 1e9

        mem_info = torch.cuda.mem_get_info(device_index)
        free_memory = mem_info[0] / 1e9
        used_memory = total_memory - free_memory

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            utilization = util.gpu
            temperature = float(temp)
            pynvml.nvmlShutdown()
        except Exception:
            utilization = 0.0
            temperature = None

        return GPUInfo(
            index=device_index,
            name=props.name,
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            used_memory_gb=used_memory,
            utilization_percent=utilization,
            temperature_celsius=temperature,
        )

    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return None


def get_memory_status(device_index: int = 0) -> GPUMemoryStatus | None:
    """
    Get current GPU memory status.

    Args:
        device_index: GPU device index

    Returns:
        GPUMemoryStatus or None if not available
    """
    if not is_cuda_available():
        return None

    try:
        import torch

        if device_index >= torch.cuda.device_count():
            return None

        props = torch.cuda.get_device_properties(device_index)
        total = props.total_memory / 1e9

        mem_info = torch.cuda.mem_get_info(device_index)
        free = mem_info[0] / 1e9
        used = total - free

        return GPUMemoryStatus(
            total_gb=total,
            used_gb=used,
            free_gb=free,
            utilization_percent=(used / total) * 100 if total > 0 else 0,
        )

    except Exception as e:
        logger.warning(f"Failed to get memory status: {e}")
        return None


def get_free_vram(device_index: int = 0) -> float:
    """
    Get free VRAM in GB.

    Args:
        device_index: GPU device index

    Returns:
        Free VRAM in GB, or 0 if not available
    """
    status = get_memory_status(device_index)
    return status.free_gb if status else 0.0


def get_total_vram(device_index: int = 0) -> float:
    """
    Get total VRAM in GB.

    Args:
        device_index: GPU device index

    Returns:
        Total VRAM in GB, or 0 if not available
    """
    status = get_memory_status(device_index)
    return status.total_gb if status else 0.0


def clear_cuda_cache() -> None:
    """Clear CUDA cache to free memory."""
    if not is_cuda_available():
        return

    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Failed to clear CUDA cache: {e}")


def get_optimal_device(required_vram_gb: float = 0) -> str:
    """
    Get the optimal device for the given VRAM requirement.

    Args:
        required_vram_gb: Required VRAM in GB

    Returns:
        Device string ("cuda:0", "cpu", etc.)
    """
    if not is_cuda_available():
        return "cpu"

    device_count = get_device_count()
    if device_count == 0:
        return "cpu"

    best_device = None
    best_free = 0.0

    for i in range(device_count):
        status = get_memory_status(i)
        if status and status.free_gb >= required_vram_gb:
            if status.free_gb > best_free:
                best_free = status.free_gb
                best_device = f"cuda:{i}"

    if best_device:
        return best_device

    if required_vram_gb == 0 and device_count > 0:
        return "cuda:0"

    return "cpu"


def log_gpu_status(device_index: int = 0) -> None:
    """Log current GPU status."""
    info = get_gpu_info(device_index)
    if info:
        logger.info(
            f"GPU {info.index}: {info.name} - "
            f"{info.free_gb:.1f}/{info.total_gb:.1f} GB free, "
            f"{info.utilization_percent:.0f}% util"
        )
    else:
        logger.info("No GPU available")
