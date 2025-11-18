"""I/O subsystem for file operations and manifests."""

from v2d.io.file_manager import FileManager, AtomicWriteError
from v2d.io.protected_paths import ProtectedPathRegistry, ProtectedPathError
from v2d.io.manifest import Manifest, ArtifactInfo
from v2d.io.integrity import compute_checksum, verify_checksum

__all__ = [
    "FileManager",
    "AtomicWriteError",
    "ProtectedPathRegistry",
    "ProtectedPathError",
    "Manifest",
    "ArtifactInfo",
    "compute_checksum",
    "verify_checksum",
]
