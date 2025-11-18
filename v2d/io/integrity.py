"""
File integrity verification for V2D.

Provides checksum computation and verification for artifacts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_checksum(
    path: Path | str,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """
    Compute checksum of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)
        chunk_size: Size of chunks to read

    Returns:
        Hex-encoded checksum
    """
    hasher = hashlib.new(algorithm)
    path = Path(path)

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_checksum(
    path: Path | str,
    expected: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify file checksum matches expected value.

    Args:
        path: Path to file
        expected: Expected checksum
        algorithm: Hash algorithm

    Returns:
        True if checksum matches
    """
    actual = compute_checksum(path, algorithm)
    return actual == expected


def compute_checksum_bytes(
    data: bytes,
    algorithm: str = "sha256",
) -> str:
    """
    Compute checksum of bytes data.

    Args:
        data: Bytes to hash
        algorithm: Hash algorithm

    Returns:
        Hex-encoded checksum
    """
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()
