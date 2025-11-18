"""
File manager with atomic operations for V2D.

Provides safe file operations that prevent partial writes and corruption.
All write operations use the atomic temp-then-rename pattern.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

from v2d.io.protected_paths import ProtectedPathRegistry, ProtectedPathError


class AtomicWriteError(Exception):
    """Raised when an atomic write operation fails."""

    def __init__(self, target_path: Path, reason: str):
        self.target_path = target_path
        self.reason = reason
        super().__init__(f"Atomic write failed for {target_path}: {reason}")


class FileManager:
    """
    Manages file operations with safety guarantees.

    Key features:
    - Atomic writes (temp-then-rename pattern)
    - Protected path enforcement
    - Checksum generation
    - Safe directory creation
    - Job-scoped file tracking
    """

    def __init__(
        self,
        project_root: Path | None = None,
        protected_registry: ProtectedPathRegistry | None = None,
    ):
        """
        Initialize the file manager.

        Args:
            project_root: Root directory for the project
            protected_registry: Registry for protected paths
        """
        self.project_root = (project_root or Path.cwd()).resolve()
        self.protected = protected_registry or ProtectedPathRegistry(self.project_root)

    def write_atomic(
        self,
        target_path: Path | str,
        content: bytes | str,
        make_parents: bool = True,
    ) -> str:
        """
        Write content to file atomically.

        Uses temp-then-rename pattern to ensure the file is either
        fully written or not written at all.

        Args:
            target_path: Destination path for the file
            content: Content to write (bytes or string)
            make_parents: Create parent directories if needed

        Returns:
            SHA256 checksum of written content

        Raises:
            AtomicWriteError: If write fails
            ProtectedPathError: If target is protected
        """
        target = self._resolve_path(target_path)

        # Check protection
        self.protected.check_can_modify(target)

        # Convert string to bytes
        if isinstance(content, str):
            content = content.encode("utf-8")

        # Calculate checksum before write
        checksum = hashlib.sha256(content).hexdigest()

        # Ensure parent directory exists
        if make_parents:
            target.parent.mkdir(parents=True, exist_ok=True)

        # Generate temp file path in same directory (for atomic rename)
        temp_path = target.with_suffix(f".tmp.{uuid4().hex[:8]}")

        try:
            # Write to temp file
            temp_path.write_bytes(content)

            # Atomic rename (works on both POSIX and Windows NTFS)
            temp_path.replace(target)

        except Exception as e:
            # Clean up temp file on failure
            temp_path.unlink(missing_ok=True)
            raise AtomicWriteError(target, str(e)) from e

        return checksum

    def write_json_atomic(
        self,
        target_path: Path | str,
        data: Any,
        indent: int = 2,
        make_parents: bool = True,
    ) -> str:
        """
        Write JSON data to file atomically.

        Args:
            target_path: Destination path
            data: Data to serialize to JSON
            indent: JSON indentation level
            make_parents: Create parent directories if needed

        Returns:
            SHA256 checksum of written content
        """
        content = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
        return self.write_atomic(target_path, content, make_parents)

    @contextmanager
    def atomic_write_stream(
        self,
        target_path: Path | str,
        mode: str = "wb",
        make_parents: bool = True,
    ) -> Generator[Any, None, None]:
        """
        Context manager for atomic writes with streaming.

        Useful for large files that shouldn't be held in memory.

        Args:
            target_path: Destination path
            mode: File mode ('wb' or 'w')
            make_parents: Create parent directories if needed

        Yields:
            File handle for writing

        Example:
            with file_manager.atomic_write_stream("output.wav") as f:
                f.write(audio_data)
        """
        target = self._resolve_path(target_path)
        self.protected.check_can_modify(target)

        if make_parents:
            target.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file in same directory
        temp_path = target.with_suffix(f".tmp.{uuid4().hex[:8]}")

        try:
            with open(temp_path, mode) as f:
                yield f

            # Atomic rename on successful write
            temp_path.replace(target)

        except Exception:
            # Clean up temp file on failure
            temp_path.unlink(missing_ok=True)
            raise

    def read_file(self, path: Path | str) -> bytes:
        """
        Read file contents.

        Args:
            path: Path to read

        Returns:
            File contents as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        resolved = self._resolve_path(path)
        return resolved.read_bytes()

    def read_text(self, path: Path | str, encoding: str = "utf-8") -> str:
        """
        Read file contents as text.

        Args:
            path: Path to read
            encoding: Text encoding

        Returns:
            File contents as string
        """
        resolved = self._resolve_path(path)
        return resolved.read_text(encoding=encoding)

    def read_json(self, path: Path | str) -> Any:
        """
        Read and parse JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON data
        """
        content = self.read_text(path)
        return json.loads(content)

    def copy_file(
        self,
        source: Path | str,
        destination: Path | str,
        make_parents: bool = True,
    ) -> str:
        """
        Copy file atomically.

        Args:
            source: Source file path
            destination: Destination path
            make_parents: Create parent directories if needed

        Returns:
            SHA256 checksum of copied file
        """
        src = self._resolve_path(source)
        dest = self._resolve_path(destination)

        # Check destination protection
        self.protected.check_can_modify(dest)

        if make_parents:
            dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy via temp file for atomicity
        temp_path = dest.with_suffix(f".tmp.{uuid4().hex[:8]}")

        try:
            shutil.copy2(src, temp_path)

            # Calculate checksum
            checksum = hashlib.sha256(temp_path.read_bytes()).hexdigest()

            # Atomic rename
            temp_path.replace(dest)

            return checksum

        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def move_file(
        self,
        source: Path | str,
        destination: Path | str,
        make_parents: bool = True,
    ) -> None:
        """
        Move file safely.

        Args:
            source: Source file path
            destination: Destination path
            make_parents: Create parent directories if needed
        """
        src = self._resolve_path(source)
        dest = self._resolve_path(destination)

        # Check both source (for deletion) and destination (for creation)
        self.protected.check_can_delete(src)
        self.protected.check_can_modify(dest)

        if make_parents:
            dest.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src), str(dest))

    def delete_file(self, path: Path | str) -> bool:
        """
        Delete a file safely.

        Args:
            path: Path to delete

        Returns:
            True if file was deleted, False if it didn't exist

        Raises:
            ProtectedPathError: If path is protected
        """
        resolved = self._resolve_path(path)

        # Check protection
        self.protected.check_can_delete(resolved)

        if resolved.exists():
            resolved.unlink()
            return True
        return False

    def delete_directory(
        self,
        path: Path | str,
        recursive: bool = True,
    ) -> bool:
        """
        Delete a directory safely.

        Args:
            path: Directory to delete
            recursive: Delete contents recursively

        Returns:
            True if deleted, False if didn't exist

        Raises:
            ProtectedPathError: If path or any contents are protected
        """
        resolved = self._resolve_path(path)

        if not resolved.exists():
            return False

        # Check protection on directory and all contents
        self.protected.check_can_delete(resolved)

        if recursive:
            # Check all contents
            for item in resolved.rglob("*"):
                self.protected.check_can_delete(item)

        if recursive:
            shutil.rmtree(resolved)
        else:
            resolved.rmdir()

        return True

    def ensure_directory(self, path: Path | str) -> Path:
        """
        Ensure a directory exists, creating it if needed.

        Args:
            path: Directory path

        Returns:
            Resolved path to directory
        """
        resolved = self._resolve_path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def get_checksum(self, path: Path | str) -> str:
        """
        Calculate SHA256 checksum of a file.

        Args:
            path: File path

        Returns:
            Hex-encoded SHA256 checksum
        """
        resolved = self._resolve_path(path)
        return hashlib.sha256(resolved.read_bytes()).hexdigest()

    def verify_checksum(self, path: Path | str, expected: str) -> bool:
        """
        Verify file checksum matches expected value.

        Args:
            path: File path
            expected: Expected SHA256 checksum

        Returns:
            True if checksum matches
        """
        actual = self.get_checksum(path)
        return actual == expected

    def exists(self, path: Path | str) -> bool:
        """Check if path exists."""
        return self._resolve_path(path).exists()

    def is_file(self, path: Path | str) -> bool:
        """Check if path is a file."""
        return self._resolve_path(path).is_file()

    def is_directory(self, path: Path | str) -> bool:
        """Check if path is a directory."""
        return self._resolve_path(path).is_dir()

    def get_size(self, path: Path | str) -> int:
        """
        Get file size in bytes.

        Args:
            path: File path

        Returns:
            Size in bytes
        """
        return self._resolve_path(path).stat().st_size

    def get_modified_time(self, path: Path | str) -> datetime:
        """
        Get file modification time.

        Args:
            path: File path

        Returns:
            Modification datetime
        """
        stat = self._resolve_path(path).stat()
        return datetime.fromtimestamp(stat.st_mtime)

    def list_directory(
        self,
        path: Path | str,
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[Path]:
        """
        List directory contents.

        Args:
            path: Directory path
            pattern: Glob pattern for filtering
            recursive: Search recursively

        Returns:
            List of matching paths
        """
        resolved = self._resolve_path(path)

        if recursive:
            return list(resolved.rglob(pattern))
        else:
            return list(resolved.glob(pattern))

    def create_temp_directory(self, prefix: str = "v2d_") -> Path:
        """
        Create a temporary directory.

        Args:
            prefix: Prefix for directory name

        Returns:
            Path to created directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        return temp_dir

    def cleanup_temp_files(self, directory: Path | str, pattern: str = "*.tmp.*") -> int:
        """
        Clean up temporary files in a directory.

        Args:
            directory: Directory to clean
            pattern: Pattern for temp files

        Returns:
            Number of files deleted
        """
        resolved = self._resolve_path(directory)
        count = 0

        for temp_file in resolved.glob(pattern):
            if temp_file.is_file():
                try:
                    self.delete_file(temp_file)
                    count += 1
                except ProtectedPathError:
                    pass  # Skip protected files

        return count

    def _resolve_path(self, path: Path | str) -> Path:
        """Resolve a path to absolute."""
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self.project_root / p).resolve()


class JobFileManager(FileManager):
    """
    File manager scoped to a specific job.

    Provides convenience methods for job-specific paths and
    ensures all operations stay within the job directory.
    """

    def __init__(
        self,
        job_id: str,
        jobs_root: Path,
        project_root: Path | None = None,
        protected_registry: ProtectedPathRegistry | None = None,
    ):
        super().__init__(project_root, protected_registry)
        self.job_id = job_id
        self.job_dir = jobs_root / job_id
        self.temp_dir = self.job_dir / "temp"
        self.segments_dir = self.job_dir / "segments"
        self.checkpoints_dir = self.job_dir / "checkpoints"

    def initialize_job_directories(self) -> None:
        """Create all job directories."""
        self.ensure_directory(self.job_dir)
        self.ensure_directory(self.temp_dir)
        self.ensure_directory(self.segments_dir)
        self.ensure_directory(self.checkpoints_dir)

    def get_temp_path(self, filename: str) -> Path:
        """Get path for a temporary file."""
        return self.temp_dir / filename

    def get_segment_path(self, index: int, extension: str = "wav") -> Path:
        """Get path for a segment file."""
        return self.segments_dir / f"segment_{index:04d}.{extension}"

    def get_checkpoint_path(self, stage: str) -> Path:
        """Get path for a checkpoint file."""
        return self.checkpoints_dir / f"{stage}.json"

    def cleanup_temp(self) -> int:
        """
        Clean up all temporary files for this job.

        Returns:
            Number of files deleted
        """
        if not self.temp_dir.exists():
            return 0

        count = 0
        for item in self.temp_dir.iterdir():
            if item.is_file():
                item.unlink()
                count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                count += 1
        return count

    def cleanup_segments(self) -> int:
        """
        Clean up all segment files for this job.

        Returns:
            Number of files deleted
        """
        if not self.segments_dir.exists():
            return 0

        count = 0
        for item in self.segments_dir.iterdir():
            if item.is_file():
                item.unlink()
                count += 1
        return count

    def cleanup_all(self, keep_checkpoints: bool = False) -> None:
        """
        Clean up all job files.

        Args:
            keep_checkpoints: Whether to preserve checkpoint files
        """
        self.cleanup_temp()
        self.cleanup_segments()

        if not keep_checkpoints and self.checkpoints_dir.exists():
            shutil.rmtree(self.checkpoints_dir)
