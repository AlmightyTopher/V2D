"""Unit tests for resource manager."""

import pytest
from unittest.mock import patch, MagicMock

from v2d.resources.manager import (
    ResourceManager,
    InsufficientVRAMError,
    InsufficientRAMError,
    InsufficientDiskError,
)


class TestResourceManager:
    """Tests for ResourceManager."""

    def test_init_default(self):
        """Test default initialization."""
        rm = ResourceManager()
        assert rm.max_vram_gb == 12.0
        assert rm.prefer_fp16 is True

    def test_init_with_config(self):
        """Test initialization with config."""
        config = MagicMock()
        config.gpu.max_vram_usage_gb = 8.0
        config.gpu.prefer_fp16 = False
        config.gpu.device = "cuda:1"

        rm = ResourceManager(config)
        assert rm.max_vram_gb == 8.0
        assert rm.prefer_fp16 is False
        assert rm.device == "cuda:1"

    def test_get_dtype(self):
        """Test getting recommended dtype."""
        rm = ResourceManager()
        rm.prefer_fp16 = True
        assert rm.get_dtype() == "float16"

        rm.prefer_fp16 = False
        assert rm.get_dtype() == "float32"

    def test_request_vram(self):
        """Test requesting VRAM allocation."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        result = rm.request_vram("model1", 4.0)
        assert result is True
        assert rm.get_allocated_vram() == 4.0

    def test_request_vram_exceeds_limit(self):
        """Test requesting more VRAM than available."""
        rm = ResourceManager()
        rm.max_vram_gb = 8.0

        rm.request_vram("model1", 6.0)

        with pytest.raises(InsufficientVRAMError):
            rm.request_vram("model2", 4.0)

    def test_release_vram(self):
        """Test releasing VRAM allocation."""
        rm = ResourceManager()
        rm.request_vram("model1", 4.0)
        rm.release_vram("model1")

        assert rm.get_allocated_vram() == 0.0

    def test_get_available_vram(self):
        """Test getting available VRAM."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        rm.request_vram("model1", 4.0)
        assert rm.get_available_vram() == 6.0

    def test_get_allocations(self):
        """Test getting all allocations."""
        rm = ResourceManager()
        rm.request_vram("model1", 4.0)
        rm.request_vram("model2", 2.0)

        allocations = rm.get_allocations()
        assert len(allocations) == 2

        names = [a.name for a in allocations]
        assert "model1" in names
        assert "model2" in names

    def test_check_resources_vram(self):
        """Test checking VRAM availability."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        assert rm.check_resources(vram_gb=8.0) is True
        assert rm.check_resources(vram_gb=12.0) is False

    @patch("v2d.resources.manager.check_memory_available")
    def test_check_resources_ram(self, mock_check):
        """Test checking RAM availability."""
        mock_check.return_value = True
        rm = ResourceManager()

        assert rm.check_resources(ram_gb=8.0) is True
        mock_check.assert_called_with(8.0)

    @patch("v2d.resources.manager.check_disk_available")
    def test_check_resources_disk(self, mock_check):
        """Test checking disk availability."""
        mock_check.return_value = True
        rm = ResourceManager()

        assert rm.check_resources(disk_gb=10.0, disk_path="/tmp") is True
        mock_check.assert_called_with(10.0, "/tmp")

    def test_ensure_resources_vram_ok(self):
        """Test ensure_resources passes with enough VRAM."""
        rm = ResourceManager()
        rm.max_vram_gb = 10.0

        rm.ensure_resources(vram_gb=8.0)

    def test_ensure_resources_vram_insufficient(self):
        """Test ensure_resources raises with insufficient VRAM."""
        rm = ResourceManager()
        rm.max_vram_gb = 8.0
        rm.request_vram("existing", 6.0)

        with pytest.raises(InsufficientVRAMError):
            rm.ensure_resources(vram_gb=4.0)

    def test_resource_session(self):
        """Test resource session context manager."""
        rm = ResourceManager()

        with rm.resource_session("test_session") as session:
            rm.request_vram("model1", 4.0)
            session.allocations.append("model1")
            assert rm.get_allocated_vram() == 4.0

        assert rm.get_allocated_vram() == 0.0

    def test_timed_operation(self):
        """Test timed operation context manager."""
        rm = ResourceManager()

        with rm.timed_operation("test_op") as result:
            pass

        assert "duration_seconds" in result
        assert result["name"] == "test_op"

    def test_cleanup(self):
        """Test cleanup releases all allocations."""
        rm = ResourceManager()
        rm.request_vram("model1", 4.0)
        rm.request_vram("model2", 2.0)

        rm.cleanup()

        assert rm.get_allocated_vram() == 0.0


class TestResourceManagerDevice:
    """Tests for device selection."""

    @patch("v2d.resources.manager.is_cuda_available")
    def test_get_device_no_cuda(self, mock_cuda):
        """Test device selection without CUDA."""
        mock_cuda.return_value = False
        rm = ResourceManager()

        assert rm.get_device() == "cpu"

    @patch("v2d.resources.manager.is_cuda_available")
    @patch("v2d.resources.manager.get_optimal_device")
    def test_get_device_with_requirement(self, mock_optimal, mock_cuda):
        """Test device selection with VRAM requirement."""
        mock_cuda.return_value = True
        mock_optimal.return_value = "cuda:0"
        rm = ResourceManager()

        device = rm.get_device(required_vram_gb=4.0)
        assert device == "cuda:0"
        mock_optimal.assert_called_with(4.0)
