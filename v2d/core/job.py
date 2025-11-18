"""
Job management for V2D.

A Job represents a single video processing task with its own
directory, manifest, and lifecycle.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from v2d.core.config import Config
from v2d.core.state import JobState, StateMachine
from v2d.io.file_manager import FileManager, JobFileManager
from v2d.io.integrity import compute_checksum
from v2d.io.manifest import Manifest


class JobError(Exception):
    """Base exception for job errors."""
    pass


class JobNotFoundError(JobError):
    """Raised when a job cannot be found."""
    pass


class Job:
    """
    Represents a single video processing job.

    Each job has:
    - Unique ID
    - Dedicated directory for all files
    - Manifest tracking all state and artifacts
    - State machine for lifecycle management
    """

    def __init__(
        self,
        job_id: str,
        config: Config,
        project_root: Path | None = None,
    ):
        """
        Initialize a job.

        Args:
            job_id: Unique job identifier
            config: Configuration for this job
            project_root: Project root directory
        """
        self.job_id = job_id
        self.config = config
        self.project_root = project_root or Path.cwd()

        # Job directories
        jobs_root = self.project_root / config.system.temp_directory
        self.job_dir = jobs_root / job_id

        # File manager scoped to this job
        self.files = JobFileManager(
            job_id=job_id,
            jobs_root=jobs_root,
            project_root=self.project_root,
        )

        # State machine
        self._state_machine = StateMachine()

        # Manifest (loaded or created)
        self._manifest: Manifest | None = None

    @property
    def state(self) -> JobState:
        """Get current job state."""
        return self._state_machine.state

    @property
    def manifest(self) -> Manifest:
        """Get job manifest."""
        if self._manifest is None:
            raise JobError("Manifest not initialized. Call initialize() first.")
        return self._manifest

    @property
    def manifest_path(self) -> Path:
        """Get path to manifest file."""
        return self.job_dir / "manifest.json"

    def initialize(
        self,
        input_path: Path | str,
    ) -> None:
        """
        Initialize the job with an input file.

        Creates job directories and manifest.

        Args:
            input_path: Path to input video file

        Raises:
            FileNotFoundError: If input file doesn't exist
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Transition to initializing
        self._state_machine.transition(JobState.INITIALIZING)

        # Create job directories
        self.files.initialize_job_directories()

        # Calculate input checksum
        input_checksum = compute_checksum(input_path)

        # Create manifest
        self._manifest = Manifest.create_new(
            job_id=self.job_id,
            input_path=str(input_path),
            input_checksum=input_checksum,
            config=self.config.to_dict(),
        )
        self._manifest.state = self.state.value

        # Save manifest atomically
        self._save_manifest()

    def _save_manifest(self) -> None:
        """Save manifest to disk atomically."""
        if self._manifest is None:
            return

        # Use global file manager for atomic write
        fm = FileManager(self.project_root)
        fm.write_json_atomic(self.manifest_path, self._manifest.to_dict())

    def load(self) -> None:
        """
        Load existing job from disk.

        Raises:
            JobNotFoundError: If job directory or manifest doesn't exist
        """
        if not self.manifest_path.exists():
            raise JobNotFoundError(f"Job not found: {self.job_id}")

        self._manifest = Manifest.load(self.manifest_path)

        # Restore state machine state
        state = JobState(self._manifest.state)
        self._state_machine.reset(state)

    def start(self) -> None:
        """
        Start job execution.

        Transitions from INITIALIZING to RUNNING.
        """
        self._state_machine.transition(JobState.RUNNING)
        self._manifest.state = self.state.value
        self._save_manifest()

    def complete(self, output_path: Path | str) -> None:
        """
        Mark job as completed.

        Args:
            output_path: Path to final output file
        """
        self._state_machine.transition(JobState.COMPLETED)
        self._manifest.state = self.state.value
        self._manifest.output_path = str(output_path)
        self._save_manifest()

    def fail(self, error: str) -> None:
        """
        Mark job as failed.

        Args:
            error: Error message
        """
        self._state_machine.transition(JobState.FAILED)
        self._manifest.state = self.state.value
        self._manifest.error = error
        self._save_manifest()

    def resume(self) -> None:
        """
        Resume a failed job.

        Transitions from FAILED to RESUMING.
        """
        self._state_machine.transition(JobState.RESUMING)
        self._manifest.state = self.state.value
        self._manifest.error = None
        self._save_manifest()

        # Then transition to running
        self._state_machine.transition(JobState.RUNNING)
        self._manifest.state = self.state.value
        self._save_manifest()

    def cancel(self) -> None:
        """Cancel the job."""
        self._state_machine.transition(JobState.CANCELLED)
        self._manifest.state = self.state.value
        self._save_manifest()

    def add_artifact(
        self,
        name: str,
        path: Path | str,
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an artifact to the job.

        Args:
            name: Artifact name
            path: Path to artifact file
            stage: Stage that produced it
            metadata: Additional metadata
        """
        path = Path(path)
        checksum = compute_checksum(path)
        size = path.stat().st_size

        self._manifest.add_artifact(
            name=name,
            path=str(path),
            checksum=checksum,
            size=size,
            stage=stage,
            metadata=metadata,
        )
        self._save_manifest()

    def get_artifact_path(self, name: str) -> Path | None:
        """
        Get path to an artifact.

        Args:
            name: Artifact name

        Returns:
            Path to artifact or None if not found
        """
        artifact = self._manifest.get_artifact(name)
        if artifact:
            return Path(artifact.path)
        return None

    def record_stage(
        self,
        name: str,
        started_at: datetime,
        completed_at: datetime,
        success: bool,
        artifacts_produced: list[str] | None = None,
        error_message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Record completion of a pipeline stage.

        Args:
            name: Stage name
            started_at: When stage started
            completed_at: When stage completed
            success: Whether stage succeeded
            artifacts_produced: Artifacts created
            error_message: Error if failed
            metrics: Performance metrics
        """
        self._manifest.add_stage(
            name=name,
            started_at=started_at,
            completed_at=completed_at,
            success=success,
            artifacts_produced=artifacts_produced,
            error_message=error_message,
            metrics=metrics,
        )
        self._save_manifest()

    def get_completed_stages(self) -> list[str]:
        """Get list of completed stage names."""
        return self._manifest.get_completed_stages()

    def cleanup(self, keep_checkpoints: bool = False) -> None:
        """
        Clean up job temporary files.

        Args:
            keep_checkpoints: Whether to preserve checkpoints
        """
        self.files.cleanup_all(keep_checkpoints=keep_checkpoints)

    @classmethod
    def create(
        cls,
        config: Config,
        project_root: Path | None = None,
    ) -> Job:
        """
        Create a new job with a generated ID.

        Args:
            config: Configuration for the job
            project_root: Project root directory

        Returns:
            New Job instance
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        return cls(job_id=job_id, config=config, project_root=project_root)

    @classmethod
    def load_existing(
        cls,
        job_id: str,
        config: Config,
        project_root: Path | None = None,
    ) -> Job:
        """
        Load an existing job from disk.

        Args:
            job_id: Job ID to load
            config: Configuration
            project_root: Project root directory

        Returns:
            Loaded Job instance
        """
        job = cls(job_id=job_id, config=config, project_root=project_root)
        job.load()
        return job
