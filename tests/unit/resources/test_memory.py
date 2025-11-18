"""Unit tests for memory monitoring."""

import pytest
from unittest.mock import patch, MagicMock

from v2d.resources.memory import (
    get_memory_status,
    get_available_ram,
    get_total_ram,
    get_disk_status,
    get_free_disk,
    check_memory_available,
    check_disk_available,
    MemoryStatus,
    DiskStatus,
)


class TestMemoryStatus:
    """Tests for memory status functions."""

    def test_get_memory_status(self):
        """Test getting memory status."""
        status = get_memory_status()

        assert isinstance(status, MemoryStatus)
        assert status.total_gb > 0
        assert status.available_gb >= 0
        assert status.used_gb >= 0
        assert 0 <= status.percent_used <= 100

    def test_get_available_ram(self):
        """Test getting available RAM."""
        available = get_available_ram()
        assert available >= 0

    def test_get_total_ram(self):
        """Test getting total RAM."""
        total = get_total_ram()
        assert total > 0

    def test_check_memory_available_true(self):
        """Test checking memory when available."""
        result = check_memory_available(0.001)
        assert result is True

    def test_check_memory_available_false(self):
        """Test checking memory when not available."""
        result = check_memory_available(1000000)
        assert result is False


class TestDiskStatus:
    """Tests for disk status functions."""

    def test_get_disk_status(self):
        """Test getting disk status."""
        status = get_disk_status("/")

        assert isinstance(status, DiskStatus)
        assert status.total_gb > 0
        assert status.free_gb >= 0
        assert status.used_gb >= 0
        assert 0 <= status.percent_used <= 100

    def test_get_free_disk(self):
        """Test getting free disk space."""
        free = get_free_disk("/")
        assert free >= 0

    def test_check_disk_available_true(self):
        """Test checking disk when available."""
        result = check_disk_available(0.001, "/")
        assert result is True

    def test_check_disk_available_false(self):
        """Test checking disk when not available."""
        result = check_disk_available(1000000, "/")
        assert result is False
