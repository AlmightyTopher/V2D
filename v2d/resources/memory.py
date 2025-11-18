"""
System memory monitoring for V2D.

Provides utilities for monitoring system RAM usage.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil

from v2d.observability.logger import get_logger

logger = get_logger("v2d.resources.memory")


@dataclass
class MemoryStatus:
    """System memory status."""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float


@dataclass
class DiskStatus:
    """Disk space status."""
    total_gb: float
    free_gb: float
    used_gb: float
    percent_used: float


def get_memory_status() -> MemoryStatus:
    """
    Get current system memory status.

    Returns:
        MemoryStatus with RAM information
    """
    mem = psutil.virtual_memory()
    return MemoryStatus(
        total_gb=mem.total / 1e9,
        available_gb=mem.available / 1e9,
        used_gb=mem.used / 1e9,
        percent_used=mem.percent,
    )


def get_available_ram() -> float:
    """
    Get available RAM in GB.

    Returns:
        Available RAM in GB
    """
    return psutil.virtual_memory().available / 1e9


def get_total_ram() -> float:
    """
    Get total RAM in GB.

    Returns:
        Total RAM in GB
    """
    return psutil.virtual_memory().total / 1e9


def get_disk_status(path: str = "/") -> DiskStatus:
    """
    Get disk space status for a path.

    Args:
        path: Path to check disk space for

    Returns:
        DiskStatus with disk information
    """
    disk = psutil.disk_usage(path)
    return DiskStatus(
        total_gb=disk.total / 1e9,
        free_gb=disk.free / 1e9,
        used_gb=disk.used / 1e9,
        percent_used=disk.percent,
    )


def get_free_disk(path: str = "/") -> float:
    """
    Get free disk space in GB.

    Args:
        path: Path to check

    Returns:
        Free disk space in GB
    """
    return psutil.disk_usage(path).free / 1e9


def check_memory_available(required_gb: float) -> bool:
    """
    Check if required RAM is available.

    Args:
        required_gb: Required RAM in GB

    Returns:
        True if sufficient RAM is available
    """
    return get_available_ram() >= required_gb


def check_disk_available(required_gb: float, path: str = "/") -> bool:
    """
    Check if required disk space is available.

    Args:
        required_gb: Required disk space in GB
        path: Path to check

    Returns:
        True if sufficient disk space is available
    """
    return get_free_disk(path) >= required_gb


def log_memory_status() -> None:
    """Log current memory status."""
    status = get_memory_status()
    logger.info(
        f"RAM: {status.available_gb:.1f}/{status.total_gb:.1f} GB available "
        f"({status.percent_used:.0f}% used)"
    )


def log_disk_status(path: str = "/") -> None:
    """Log current disk status."""
    status = get_disk_status(path)
    logger.info(
        f"Disk: {status.free_gb:.1f}/{status.total_gb:.1f} GB free "
        f"({status.percent_used:.0f}% used)"
    )
