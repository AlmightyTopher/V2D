"""
Base stage class for V2D pipeline.

All pipeline stages inherit from this base class which provides
common functionality and enforces the stage contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from v2d.core.job import Job
from v2d.observability.logger import get_logger


@dataclass
class StageResult:
    """Result of a stage execution."""

    success: bool
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        artifacts: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> StageResult:
        """Create a successful result."""
        return cls(
            success=True,
            artifacts=artifacts or [],
            metrics=metrics or {},
        )

    @classmethod
    def fail(cls, error: str) -> StageResult:
        """Create a failed result."""
        return cls(success=False, error=error)


@dataclass
class ResourceRequirements:
    """Resource requirements for a stage."""

    gpu_vram_gb: float = 0.0
    cpu_cores: int = 1
    ram_gb: float = 1.0
    disk_gb: float = 1.0


class BaseStage(ABC):
    """
    Abstract base class for all pipeline stages.

    Stages must implement:
    - name: Unique stage identifier
    - dependencies: List of stage names that must complete first
    - execute(): Main processing logic

    Stages should also implement:
    - validate_inputs(): Check required inputs exist
    - validate_outputs(): Verify outputs were created
    - estimate_resources(): Return resource requirements
    """

    name: str = "base"
    dependencies: list[str] = []

    def __init__(self):
        self.logger = get_logger(f"v2d.stages.{self.name}")

    def run(self, job: Job) -> StageResult:
        """
        Run the stage with full lifecycle management.

        This method handles:
        - Input validation
        - Execution
        - Output validation
        - Error handling

        Args:
            job: Job to process

        Returns:
            StageResult with success/failure and artifacts
        """
        self.logger.info(f"Starting stage: {self.name}")
        start_time = datetime.now()

        try:
            # Validate inputs
            validation = self.validate_inputs(job)
            if not validation.success:
                return validation

            # Execute main logic
            result = self.execute(job)

            if result.success:
                # Validate outputs
                output_validation = self.validate_outputs(job)
                if not output_validation.success:
                    return output_validation

                # Log metrics
                duration = (datetime.now() - start_time).total_seconds()
                result.metrics["duration_seconds"] = duration
                self.logger.info(
                    f"Stage {self.name} completed in {duration:.2f}s"
                )

            return result

        except Exception as e:
            self.logger.error(f"Stage {self.name} failed: {e}")
            return StageResult.fail(str(e))

    @abstractmethod
    def execute(self, job: Job) -> StageResult:
        """
        Execute the main stage logic.

        Args:
            job: Job to process

        Returns:
            StageResult with artifacts produced
        """
        pass

    def validate_inputs(self, job: Job) -> StageResult:
        """
        Validate that all required inputs are present.

        Override this method to add specific input validation.

        Args:
            job: Job to validate

        Returns:
            StageResult (success=True if valid)
        """
        return StageResult.ok()

    def validate_outputs(self, job: Job) -> StageResult:
        """
        Validate that outputs were created correctly.

        Override this method to add specific output validation.

        Args:
            job: Job to validate

        Returns:
            StageResult (success=True if valid)
        """
        return StageResult.ok()

    def estimate_resources(self, job: Job) -> ResourceRequirements:
        """
        Estimate resource requirements for this stage.

        Override this method to provide accurate estimates.

        Args:
            job: Job to estimate for

        Returns:
            ResourceRequirements
        """
        return ResourceRequirements()

    def get_artifact_path(self, job: Job, name: str) -> Path:
        """
        Get the path where an artifact should be stored.

        Args:
            job: Job context
            name: Artifact name

        Returns:
            Path for the artifact
        """
        return job.files.get_temp_path(f"{name}")

    def require_artifact(self, job: Job, name: str) -> Path:
        """
        Get a required artifact path, raising if not found.

        Args:
            job: Job context
            name: Artifact name

        Returns:
            Path to artifact

        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        path = job.get_artifact_path(name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {name}")
        return path
