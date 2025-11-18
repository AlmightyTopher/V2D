"""
Pipeline orchestration for V2D.

The Pipeline class orchestrates the execution of stages in the correct
order based on their dependencies.
"""

from __future__ import annotations

from typing import Protocol, Any
from datetime import datetime

from v2d.core.job import Job


class StageResult:
    """Result of a stage execution."""

    def __init__(
        self,
        success: bool,
        artifacts: list[str] | None = None,
        error: str | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        self.success = success
        self.artifacts = artifacts or []
        self.error = error
        self.metrics = metrics or {}


class Stage(Protocol):
    """Protocol for pipeline stages."""

    name: str
    dependencies: list[str]

    def execute(self, job: Job) -> StageResult:
        """Execute the stage."""
        ...


class Pipeline:
    """
    Orchestrates the execution of pipeline stages.

    Handles:
    - Dependency resolution
    - Stage ordering
    - Checkpoint creation
    - Error handling
    - Resume from failure
    """

    def __init__(self):
        """Initialize the pipeline."""
        self._stages: dict[str, Stage] = {}

    def register_stage(self, stage: Stage) -> None:
        """
        Register a stage with the pipeline.

        Args:
            stage: Stage to register
        """
        self._stages[stage.name] = stage

    def get_execution_order(self) -> list[str]:
        """
        Get stages in execution order based on dependencies.

        Returns:
            List of stage names in order

        Raises:
            ValueError: If circular dependency detected
        """
        # Topological sort
        visited: set[str] = set()
        result: list[str] = []
        temp_visited: set[str] = set()

        def visit(name: str) -> None:
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected at stage: {name}")
            if name in visited:
                return

            temp_visited.add(name)

            stage = self._stages.get(name)
            if stage:
                for dep in stage.dependencies:
                    visit(dep)

            temp_visited.remove(name)
            visited.add(name)
            result.append(name)

        for stage_name in self._stages:
            visit(stage_name)

        return result

    def run(self, job: Job, resume: bool = False) -> bool:
        """
        Run the pipeline for a job.

        Args:
            job: Job to process
            resume: Whether resuming from failure

        Returns:
            True if all stages completed successfully
        """
        # Get execution order
        order = self.get_execution_order()

        # Get already completed stages if resuming
        completed = set(job.get_completed_stages()) if resume else set()

        # Execute stages
        for stage_name in order:
            # Skip completed stages
            if stage_name in completed:
                continue

            stage = self._stages.get(stage_name)
            if not stage:
                continue

            # Execute stage
            started_at = datetime.now()

            try:
                result = stage.execute(job)
                completed_at = datetime.now()

                # Record result
                job.record_stage(
                    name=stage_name,
                    started_at=started_at,
                    completed_at=completed_at,
                    success=result.success,
                    artifacts_produced=result.artifacts,
                    error_message=result.error,
                    metrics=result.metrics,
                )

                if not result.success:
                    job.fail(result.error or f"Stage {stage_name} failed")
                    return False

            except Exception as e:
                completed_at = datetime.now()
                job.record_stage(
                    name=stage_name,
                    started_at=started_at,
                    completed_at=completed_at,
                    success=False,
                    error_message=str(e),
                )
                job.fail(str(e))
                return False

        return True
