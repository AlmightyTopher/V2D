"""
Protected path registry for V2D.

Ensures that critical paths (source files, configuration, etc.) are never
deleted or modified by cleanup operations.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable


class ProtectedPathError(Exception):
    """Raised when an operation attempts to modify a protected path."""

    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Protected path violation: {path} - {reason}")


class ProtectedPathRegistry:
    """
    Registry of paths that must never be deleted or modified by cleanup operations.

    This is a critical safety component that prevents accidental deletion of:
    - Input/source video files
    - Configuration files
    - Source code
    - Documentation

    The registry uses both exact paths and glob patterns for matching.
    """

    # Hardcoded patterns that are ALWAYS protected
    # These cannot be removed from the registry
    IMMUTABLE_PATTERNS: frozenset[str] = frozenset([
        # Source code
        "v2d/**/*.py",
        "tests/**/*.py",
        "*.py",

        # Configuration
        "config/**/*.yaml",
        "config/**/*.yml",
        "config/**/*.json",
        "*.yaml",
        "*.yml",

        # Documentation
        "*.md",
        "docs/**/*",

        # Input directory (never delete source videos)
        "data/input/**/*",
        "data/input",

        # Git
        ".git/**/*",
        ".gitignore",

        # Package files
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements*.txt",

        # Model directories (protect from accidental deletion)
        "StyleTTS2/**/*",
    ])

    # Patterns that are protected by default but can be removed
    DEFAULT_PATTERNS: frozenset[str] = frozenset([
        # Output directory (protected but can be opted out)
        "data/output/**/*",

        # Cache directories
        "data/cache/models/**/*",
    ])

    def __init__(self, project_root: Path | None = None):
        """
        Initialize the protected path registry.

        Args:
            project_root: Root directory for the project. Defaults to cwd.
        """
        self.project_root = (project_root or Path.cwd()).resolve()

        # Start with default patterns
        self._patterns: set[str] = set(self.DEFAULT_PATTERNS)

        # Additional exact paths that are protected
        self._exact_paths: set[Path] = set()

    def add_pattern(self, pattern: str) -> None:
        """
        Add a glob pattern to protect.

        Args:
            pattern: Glob pattern relative to project root
        """
        self._patterns.add(pattern)

    def remove_pattern(self, pattern: str) -> bool:
        """
        Remove a pattern from protection (only for non-immutable patterns).

        Args:
            pattern: Pattern to remove

        Returns:
            True if pattern was removed, False if it was immutable
        """
        if pattern in self.IMMUTABLE_PATTERNS:
            return False

        self._patterns.discard(pattern)
        return True

    def add_path(self, path: Path | str) -> None:
        """
        Add an exact path to protect.

        Args:
            path: Path to protect (will be resolved to absolute)
        """
        resolved = self._resolve_path(path)
        self._exact_paths.add(resolved)

    def remove_path(self, path: Path | str) -> None:
        """
        Remove an exact path from protection.

        Args:
            path: Path to unprotect
        """
        resolved = self._resolve_path(path)
        self._exact_paths.discard(resolved)

    def is_protected(self, path: Path | str) -> bool:
        """
        Check if a path is protected.

        Args:
            path: Path to check

        Returns:
            True if the path matches any protection rule
        """
        resolved = self._resolve_path(path)

        # Check exact paths
        if resolved in self._exact_paths:
            return True

        # Get path relative to project root for pattern matching
        try:
            rel_path = resolved.relative_to(self.project_root)
        except ValueError:
            # Path is outside project root - not protected by our rules
            return False

        rel_str = str(rel_path)

        # Check all patterns (immutable + configured)
        all_patterns = self.IMMUTABLE_PATTERNS | self._patterns

        for pattern in all_patterns:
            if self._matches_pattern(rel_str, pattern):
                return True

        return False

    def check_can_delete(self, path: Path | str) -> None:
        """
        Check if a path can be safely deleted.

        Args:
            path: Path to check

        Raises:
            ProtectedPathError: If the path is protected
        """
        resolved = self._resolve_path(path)

        if self.is_protected(resolved):
            # Find which pattern matched for better error message
            reason = self._get_protection_reason(resolved)
            raise ProtectedPathError(resolved, reason)

        # Check that path is within project root
        try:
            resolved.relative_to(self.project_root)
        except ValueError:
            raise ProtectedPathError(
                resolved,
                f"Path is outside project root ({self.project_root})"
            )

    def check_can_modify(self, path: Path | str) -> None:
        """
        Check if a path can be safely modified.

        This is less strict than check_can_delete - allows modification
        of some protected paths that just can't be deleted.

        Args:
            path: Path to check

        Raises:
            ProtectedPathError: If the path cannot be modified
        """
        resolved = self._resolve_path(path)

        # For now, modification rules are same as deletion
        # Could be relaxed in future for specific cases
        self.check_can_delete(resolved)

    def get_protected_paths(self) -> list[str]:
        """
        Get all protection patterns currently active.

        Returns:
            List of all patterns (immutable + configured)
        """
        all_patterns = sorted(self.IMMUTABLE_PATTERNS | self._patterns)
        exact = [str(p.relative_to(self.project_root)) for p in self._exact_paths]
        return all_patterns + exact

    def filter_deletable(self, paths: Iterable[Path | str]) -> list[Path]:
        """
        Filter a list of paths to only those that can be deleted.

        Args:
            paths: Paths to filter

        Returns:
            List of paths that are safe to delete
        """
        result = []
        for path in paths:
            resolved = self._resolve_path(path)
            if not self.is_protected(resolved):
                result.append(resolved)
        return result

    def _resolve_path(self, path: Path | str) -> Path:
        """Resolve a path to absolute."""
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self.project_root / p).resolve()

    def _matches_pattern(self, path_str: str, pattern: str) -> bool:
        """
        Check if a path string matches a glob pattern.

        Args:
            path_str: Path string (relative to project root)
            pattern: Glob pattern

        Returns:
            True if matches
        """
        # Handle both forward and back slashes
        path_str = path_str.replace("\\", "/")
        pattern = pattern.replace("\\", "/")

        # Direct match
        if fnmatch.fnmatch(path_str, pattern):
            return True

        # Check if any parent directory matches (for directory patterns)
        parts = path_str.split("/")
        for i in range(len(parts)):
            partial = "/".join(parts[:i + 1])
            if fnmatch.fnmatch(partial, pattern):
                return True

        return False

    def _get_protection_reason(self, path: Path) -> str:
        """Get human-readable reason why path is protected."""
        if path in self._exact_paths:
            return "Explicitly protected path"

        try:
            rel_str = str(path.relative_to(self.project_root))
        except ValueError:
            return "Outside project root"

        # Find matching pattern
        all_patterns = self.IMMUTABLE_PATTERNS | self._patterns

        for pattern in all_patterns:
            if self._matches_pattern(rel_str, pattern):
                if pattern in self.IMMUTABLE_PATTERNS:
                    return f"Matches immutable pattern: {pattern}"
                else:
                    return f"Matches protected pattern: {pattern}"

        return "Unknown protection rule"


# Global registry instance for convenience
_default_registry: ProtectedPathRegistry | None = None


def get_default_registry(project_root: Path | None = None) -> ProtectedPathRegistry:
    """
    Get or create the default protected path registry.

    Args:
        project_root: Project root (only used if creating new registry)

    Returns:
        The default registry instance
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = ProtectedPathRegistry(project_root)

    return _default_registry


def is_protected(path: Path | str, project_root: Path | None = None) -> bool:
    """
    Convenience function to check if a path is protected.

    Args:
        path: Path to check
        project_root: Project root (optional)

    Returns:
        True if protected
    """
    registry = get_default_registry(project_root)
    return registry.is_protected(path)


def check_can_delete(path: Path | str, project_root: Path | None = None) -> None:
    """
    Convenience function to check if a path can be deleted.

    Args:
        path: Path to check
        project_root: Project root (optional)

    Raises:
        ProtectedPathError: If path is protected
    """
    registry = get_default_registry(project_root)
    registry.check_can_delete(path)
