"""
Job manifest management for V2D.

The manifest tracks all artifacts, state, and metadata for a job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ArtifactInfo:
    """Information about a single artifact."""

    name: str
    path: str
    checksum_sha256: str
    size_bytes: int
    created_at: datetime
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "checksum_sha256": self.checksum_sha256,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "stage": self.stage,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactInfo:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            checksum_sha256=data["checksum_sha256"],
            size_bytes=data["size_bytes"],
            created_at=datetime.fromisoformat(data["created_at"]),
            stage=data["stage"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class StageInfo:
    """Information about a completed stage."""

    name: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    success: bool
    error_message: str | None = None
    artifacts_produced: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "artifacts_produced": self.artifacts_produced,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageInfo:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            duration_seconds=data["duration_seconds"],
            success=data["success"],
            error_message=data.get("error_message"),
            artifacts_produced=data.get("artifacts_produced", []),
            metrics=data.get("metrics", {}),
        )


@dataclass
class Manifest:
    """
    Job manifest containing all job state and metadata.

    The manifest is the single source of truth for job state.
    It tracks:
    - Job metadata (ID, timestamps, state)
    - Input file information
    - All produced artifacts with checksums
    - Stage completion history
    - Configuration used
    """

    job_id: str
    created_at: datetime
    updated_at: datetime
    state: str
    input_path: str
    input_checksum: str
    output_path: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, ArtifactInfo] = field(default_factory=dict)
    stages: list[StageInfo] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_artifact(
        self,
        name: str,
        path: str | Path,
        checksum: str,
        size: int,
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactInfo:
        """
        Add an artifact to the manifest.

        Args:
            name: Artifact name (e.g., "vocals", "transcript")
            path: Path to artifact file
            checksum: SHA256 checksum
            size: File size in bytes
            stage: Stage that produced this artifact
            metadata: Additional metadata

        Returns:
            The created ArtifactInfo
        """
        artifact = ArtifactInfo(
            name=name,
            path=str(path),
            checksum_sha256=checksum,
            size_bytes=size,
            created_at=datetime.now(),
            stage=stage,
            metadata=metadata or {},
        )
        self.artifacts[name] = artifact
        self.updated_at = datetime.now()
        return artifact

    def get_artifact(self, name: str) -> ArtifactInfo | None:
        """Get artifact by name."""
        return self.artifacts.get(name)

    def has_artifact(self, name: str) -> bool:
        """Check if artifact exists."""
        return name in self.artifacts

    def add_stage(
        self,
        name: str,
        started_at: datetime,
        completed_at: datetime,
        success: bool,
        artifacts_produced: list[str] | None = None,
        error_message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> StageInfo:
        """
        Record completion of a stage.

        Args:
            name: Stage name
            started_at: When stage started
            completed_at: When stage completed
            success: Whether stage succeeded
            artifacts_produced: List of artifact names produced
            error_message: Error message if failed
            metrics: Stage metrics

        Returns:
            The created StageInfo
        """
        duration = (completed_at - started_at).total_seconds()

        stage = StageInfo(
            name=name,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            artifacts_produced=artifacts_produced or [],
            metrics=metrics or {},
        )
        self.stages.append(stage)
        self.updated_at = datetime.now()
        return stage

    def get_completed_stages(self) -> list[str]:
        """Get list of successfully completed stage names."""
        return [s.name for s in self.stages if s.success]

    def is_stage_completed(self, name: str) -> bool:
        """Check if a stage has completed successfully."""
        return name in self.get_completed_stages()

    def get_last_stage(self) -> StageInfo | None:
        """Get the most recently completed stage."""
        if self.stages:
            return self.stages[-1]
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "state": self.state,
            "input_path": self.input_path,
            "input_checksum": self.input_checksum,
            "output_path": self.output_path,
            "config_snapshot": self.config_snapshot,
            "artifacts": {
                name: artifact.to_dict()
                for name, artifact in self.artifacts.items()
            },
            "stages": [stage.to_dict() for stage in self.stages],
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Create manifest from dictionary."""
        artifacts = {
            name: ArtifactInfo.from_dict(artifact_data)
            for name, artifact_data in data.get("artifacts", {}).items()
        }
        stages = [
            StageInfo.from_dict(stage_data)
            for stage_data in data.get("stages", [])
        ]

        return cls(
            job_id=data["job_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            state=data["state"],
            input_path=data["input_path"],
            input_checksum=data["input_checksum"],
            output_path=data.get("output_path"),
            config_snapshot=data.get("config_snapshot", {}),
            artifacts=artifacts,
            stages=stages,
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path | str) -> None:
        """
        Save manifest to file.

        Note: For atomic saves, use FileManager.write_json_atomic instead.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        """Load manifest from file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def create_new(
        cls,
        job_id: str,
        input_path: str,
        input_checksum: str,
        config: dict[str, Any] | None = None,
    ) -> Manifest:
        """
        Create a new manifest for a job.

        Args:
            job_id: Unique job identifier
            input_path: Path to input video
            input_checksum: Checksum of input video
            config: Configuration snapshot

        Returns:
            New manifest instance
        """
        now = datetime.now()
        return cls(
            job_id=job_id,
            created_at=now,
            updated_at=now,
            state="created",
            input_path=input_path,
            input_checksum=input_checksum,
            config_snapshot=config or {},
        )
