"""
Pipeline orchestration for V2D.

The Pipeline class orchestrates the execution of stages in the correct
order based on their dependencies.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from v2d.core.config import Config
from v2d.core.job import Job
from v2d.core.state import JobState
from v2d.observability.logger import get_logger, set_job_context
from v2d.stages.base import BaseStage, StageResult


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class DependencyError(PipelineError):
    """Raised when stage dependencies are not satisfied."""
    pass


class StageExecutionError(PipelineError):
    """Raised when a stage fails to execute."""

    def __init__(self, stage_name: str, error: str):
        self.stage_name = stage_name
        self.error = error
        super().__init__(f"Stage '{stage_name}' failed: {error}")


class Pipeline:
    """
    Orchestrates the execution of pipeline stages.

    Handles:
    - Stage registration and ordering
    - Dependency resolution
    - Sequential execution with checkpointing
    - Error handling and recovery
    - Manifest updates
    - Logging and metrics
    """

    def __init__(self, config: Config | None = None):
        """
        Initialize the pipeline.

        Args:
            config: V2D configuration
        """
        self.config = config
        self._stages: dict[str, BaseStage] = {}
        self._execution_order: list[str] | None = None
        self.logger = get_logger("v2d.pipeline")
        self._progress_callback: Callable[[str, int, int], None] | None = None

    def register_stage(self, stage: BaseStage) -> None:
        """
        Register a stage with the pipeline.

        Args:
            stage: Stage to register
        """
        self._stages[stage.name] = stage
        self._execution_order = None
        self.logger.debug(f"Registered stage: {stage.name}")

    def set_progress_callback(
        self, callback: Callable[[str, int, int], None]
    ) -> None:
        """
        Set a callback for progress updates.

        Args:
            callback: Function called with (stage_name, current, total)
        """
        self._progress_callback = callback

    def get_execution_order(self) -> list[str]:
        """
        Get stages in execution order based on dependencies.

        Uses topological sort to resolve dependencies.

        Returns:
            List of stage names in execution order

        Raises:
            DependencyError: If circular dependency detected
        """
        if self._execution_order is not None:
            return self._execution_order

        visited: set[str] = set()
        result: list[str] = []
        temp_visited: set[str] = set()

        def visit(name: str) -> None:
            if name in temp_visited:
                raise DependencyError(
                    f"Circular dependency detected at stage: {name}"
                )
            if name in visited:
                return

            temp_visited.add(name)

            stage = self._stages.get(name)
            if stage:
                for dep in stage.dependencies:
                    if dep not in self._stages:
                        raise DependencyError(
                            f"Stage '{name}' depends on unknown stage '{dep}'"
                        )
                    visit(dep)

            temp_visited.remove(name)
            visited.add(name)
            result.append(name)

        for stage_name in self._stages:
            visit(stage_name)

        self._execution_order = result
        return result

    def run(self, job: Job, resume: bool = False) -> bool:
        """
        Run the pipeline for a job.

        Args:
            job: Job to process
            resume: Whether resuming from a previous failure

        Returns:
            True if all stages completed successfully
        """
        set_job_context(job.job_id)

        self.logger.info(f"Starting pipeline for job: {job.job_id}")
        self.logger.info(f"Input: {job.manifest.input_path}")

        try:
            execution_order = self.get_execution_order()
        except DependencyError as e:
            self.logger.error(f"Dependency resolution failed: {e}")
            job.fail(str(e))
            return False

        completed_stages = set(job.get_completed_stages()) if resume else set()

        if resume and completed_stages:
            self.logger.info(
                f"Resuming from checkpoint. Completed stages: {completed_stages}"
            )

        total_stages = len(execution_order)

        if job.state == JobState.INITIALIZING:
            job.start()

        for idx, stage_name in enumerate(execution_order):
            if stage_name in completed_stages:
                self.logger.info(f"Skipping completed stage: {stage_name}")
                if self._progress_callback:
                    self._progress_callback(stage_name, idx + 1, total_stages)
                continue

            stage = self._stages.get(stage_name)
            if not stage:
                self.logger.error(f"Stage not found: {stage_name}")
                job.fail(f"Stage not found: {stage_name}")
                return False

            success = self._execute_stage(job, stage, idx + 1, total_stages)

            if not success:
                return False

        output_artifact = job.manifest.get_artifact("output_video")
        if output_artifact:
            job.complete(output_artifact.path)
            self.logger.info(f"Pipeline completed successfully")
            self.logger.info(f"Output: {output_artifact.path}")
        else:
            job.fail("No output video produced")
            return False

        set_job_context(None)
        return True

    def _execute_stage(
        self,
        job: Job,
        stage: BaseStage,
        current: int,
        total: int,
    ) -> bool:
        """
        Execute a single stage with full lifecycle management.

        Args:
            job: Job being processed
            stage: Stage to execute
            current: Current stage number (1-indexed)
            total: Total number of stages

        Returns:
            True if stage completed successfully
        """
        stage_name = stage.name

        self.logger.info(f"[{current}/{total}] Starting stage: {stage_name}")
        started_at = datetime.now()

        if self._progress_callback:
            self._progress_callback(stage_name, current, total)

        try:
            validation = stage.validate_inputs(job)
            if not validation.success:
                error_msg = validation.error or "Input validation failed"
                self.logger.error(f"Stage {stage_name} input validation failed: {error_msg}")
                self._record_stage_failure(job, stage_name, started_at, error_msg)
                job.fail(f"Stage {stage_name}: {error_msg}")
                return False

            result = stage.execute(job)

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            if not result.success:
                error_msg = result.error or "Execution failed"
                self.logger.error(f"Stage {stage_name} failed: {error_msg}")
                self._record_stage_failure(job, stage_name, started_at, error_msg)
                job.fail(f"Stage {stage_name}: {error_msg}")
                return False

            output_validation = stage.validate_outputs(job)
            if not output_validation.success:
                error_msg = output_validation.error or "Output validation failed"
                self.logger.error(f"Stage {stage_name} output validation failed: {error_msg}")
                self._record_stage_failure(job, stage_name, started_at, error_msg)
                job.fail(f"Stage {stage_name}: {error_msg}")
                return False

            job.record_stage(
                name=stage_name,
                started_at=started_at,
                completed_at=completed_at,
                success=True,
                artifacts_produced=result.artifacts,
                metrics=result.metrics,
            )

            self.logger.info(
                f"[{current}/{total}] Stage {stage_name} completed in {duration:.2f}s"
            )

            if result.metrics:
                metrics_str = ", ".join(
                    f"{k}={v}" for k, v in result.metrics.items()
                    if k != "duration_seconds"
                )
                if metrics_str:
                    self.logger.debug(f"Stage metrics: {metrics_str}")

            return True

        except Exception as e:
            completed_at = datetime.now()
            error_msg = str(e)
            self.logger.exception(f"Stage {stage_name} raised exception: {error_msg}")
            self._record_stage_failure(job, stage_name, started_at, error_msg)
            job.fail(f"Stage {stage_name}: {error_msg}")
            return False

    def _record_stage_failure(
        self,
        job: Job,
        stage_name: str,
        started_at: datetime,
        error: str,
    ) -> None:
        """Record a failed stage in the manifest."""
        job.record_stage(
            name=stage_name,
            started_at=started_at,
            completed_at=datetime.now(),
            success=False,
            error_message=error,
        )

    def get_stage(self, name: str) -> BaseStage | None:
        """Get a registered stage by name."""
        return self._stages.get(name)

    def list_stages(self) -> list[str]:
        """List all registered stage names."""
        return list(self._stages.keys())

    def validate_pipeline(self) -> list[str]:
        """
        Validate pipeline configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []

        if not self._stages:
            errors.append("No stages registered")
            return errors

        try:
            order = self.get_execution_order()
        except DependencyError as e:
            errors.append(str(e))
            return errors

        for stage_name in order:
            stage = self._stages[stage_name]
            for dep in stage.dependencies:
                if dep not in self._stages:
                    errors.append(
                        f"Stage '{stage_name}' depends on unregistered stage '{dep}'"
                    )

        return errors


def run_pipeline(
    input_path: Path,
    config: Config,
    project_root: Path | None = None,
    resume_job_id: str | None = None,
) -> tuple[bool, Job]:
    """
    Convenience function to run the complete Phase 1 pipeline.

    Args:
        input_path: Path to input video file
        config: V2D configuration
        project_root: Project root directory
        resume_job_id: Job ID to resume (if resuming)

    Returns:
        Tuple of (success, job)
    """
    from v2d.stages.pipelines import build_phase1_pipeline

    logger = get_logger("v2d.pipeline")

    if resume_job_id:
        job = Job.load_existing(
            job_id=resume_job_id,
            config=config,
            project_root=project_root,
        )
        job.resume()
        resume = True
        logger.info(f"Resuming job: {resume_job_id}")
    else:
        job = Job.create(config=config, project_root=project_root)
        job.initialize(input_path)
        resume = False
        logger.info(f"Created job: {job.job_id}")

    pipeline = build_phase1_pipeline(config)

    validation_errors = pipeline.validate_pipeline()
    if validation_errors:
        for error in validation_errors:
            logger.error(f"Pipeline validation error: {error}")
        job.fail("Pipeline validation failed")
        return False, job

    success = pipeline.run(job, resume=resume)

    if success and config.cleanup.auto_cleanup_on_success:
        logger.info("Cleaning up temporary files...")
        job.cleanup(keep_checkpoints=config.cleanup.keep_checkpoints)

    return success, job
