"""
Command-line interface for V2D.

Provides commands for processing videos, managing jobs, and system operations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from v2d import __version__
from v2d.core.config import load_config, ConfigValidationError
from v2d.core.job import Job, JobNotFoundError
from v2d.core.pipeline import run_pipeline
from v2d.observability.logger import get_logger, setup_logging


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="v2d",
        description="V2D - Video to Dub: Local AI dubbing system",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"v2d {__version__}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the dubbing pipeline on a video file",
    )
    run_parser.add_argument(
        "input",
        type=Path,
        help="Input video file",
    )
    run_parser.add_argument(
        "--resume",
        type=str,
        metavar="JOB_ID",
        help="Resume a failed job by ID",
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Process a video file (alias for run)",
    )
    process_parser.add_argument(
        "input",
        type=Path,
        help="Input video file",
    )
    process_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory",
    )

    jobs_parser = subparsers.add_parser(
        "jobs",
        help="Manage jobs",
    )
    jobs_subparsers = jobs_parser.add_subparsers(dest="jobs_command")

    jobs_subparsers.add_parser("list", help="List all jobs")

    status_parser = jobs_subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("job_id", help="Job ID")

    resume_parser = jobs_subparsers.add_parser("resume", help="Resume failed job")
    resume_parser.add_argument("job_id", help="Job ID")

    cleanup_parser = jobs_subparsers.add_parser("cleanup", help="Cleanup job files")
    cleanup_parser.add_argument("job_id", help="Job ID")
    cleanup_parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep checkpoint files",
    )

    subparsers.add_parser(
        "health",
        help="Check system health",
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = get_logger(__name__)

    try:
        config = load_config(args.config)
    except ConfigValidationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return 1

    if args.command == "run":
        return cmd_run(args, config)
    elif args.command == "process":
        return cmd_run(args, config)
    elif args.command == "jobs":
        return cmd_jobs(args, config)
    elif args.command == "health":
        return cmd_health(args, config)
    else:
        parser.print_help()
        return 0


def cmd_run(args: argparse.Namespace, config) -> int:
    """Handle run command - execute the full dubbing pipeline."""
    logger = get_logger(__name__)

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if not input_path.is_file():
        logger.error(f"Input is not a file: {input_path}")
        return 1

    valid_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
    if input_path.suffix.lower() not in valid_extensions:
        logger.error(
            f"Invalid video format: {input_path.suffix}. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )
        return 1

    resume_job_id = getattr(args, "resume", None)

    logger.info("=" * 60)
    logger.info("V2D - Video to Dub Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if resume_job_id:
        logger.info(f"Resuming job: {resume_job_id}")

    logger.info("=" * 60)

    try:
        success, job = run_pipeline(
            input_path=input_path,
            config=config,
            project_root=Path.cwd(),
            resume_job_id=resume_job_id,
        )

        logger.info("=" * 60)

        if success:
            output_artifact = job.manifest.get_artifact("output_video")
            output_path = output_artifact.path if output_artifact else "Unknown"

            logger.info("Pipeline completed successfully!")
            logger.info(f"Job ID: {job.job_id}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            total_duration = sum(
                stage.duration_seconds
                for stage in job.manifest.stages
                if stage.success
            )
            logger.info(f"Total time: {total_duration:.1f}s")

            logger.info("=" * 60)
            return 0
        else:
            logger.error("Pipeline failed!")
            logger.error(f"Job ID: {job.job_id}")
            logger.error(f"Error: {job.manifest.error}")
            logger.error(f"Failed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            last_stage = job.manifest.get_last_stage()
            if last_stage and not last_stage.success:
                logger.error(f"Failed stage: {last_stage.name}")
                logger.error(f"Stage error: {last_stage.error_message}")

            logger.info("")
            logger.info(f"To resume this job, run:")
            logger.info(f"  v2d run {input_path} --resume {job.job_id}")

            logger.info("=" * 60)
            return 1

    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        return 1


def cmd_jobs(args: argparse.Namespace, config) -> int:
    """Handle jobs command."""
    logger = get_logger(__name__)

    if args.jobs_command == "list":
        return cmd_jobs_list(config)
    elif args.jobs_command == "status":
        return cmd_jobs_status(args.job_id, config)
    elif args.jobs_command == "resume":
        return cmd_jobs_resume(args.job_id, config)
    elif args.jobs_command == "cleanup":
        keep_checkpoints = getattr(args, "keep_checkpoints", False)
        return cmd_jobs_cleanup(args.job_id, config, keep_checkpoints)
    else:
        logger.error("Unknown jobs command. Use: list, status, resume, cleanup")
        return 1


def cmd_jobs_list(config) -> int:
    """List all jobs."""
    logger = get_logger(__name__)

    jobs_dir = Path.cwd() / config.system.temp_directory

    if not jobs_dir.exists():
        logger.info("No jobs found.")
        return 0

    jobs = []
    for job_dir in jobs_dir.iterdir():
        if job_dir.is_dir():
            manifest_path = job_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    data = json.loads(manifest_path.read_text())
                    jobs.append({
                        "id": data["job_id"],
                        "state": data["state"],
                        "created": data["created_at"],
                        "input": Path(data["input_path"]).name,
                    })
                except (json.JSONDecodeError, KeyError):
                    pass

    if not jobs:
        logger.info("No jobs found.")
        return 0

    jobs.sort(key=lambda x: x["created"], reverse=True)

    logger.info(f"Found {len(jobs)} job(s):")
    logger.info("")
    logger.info(f"{'ID':<40} {'State':<12} {'Input':<30}")
    logger.info("-" * 82)

    for job in jobs:
        job_id = job["id"][:38] + ".." if len(job["id"]) > 40 else job["id"]
        input_name = job["input"][:28] + ".." if len(job["input"]) > 30 else job["input"]
        logger.info(f"{job_id:<40} {job['state']:<12} {input_name:<30}")

    return 0


def cmd_jobs_status(job_id: str, config) -> int:
    """Show detailed status of a job."""
    logger = get_logger(__name__)

    try:
        job = Job.load_existing(
            job_id=job_id,
            config=config,
            project_root=Path.cwd(),
        )
    except JobNotFoundError:
        logger.error(f"Job not found: {job_id}")
        return 1

    manifest = job.manifest

    logger.info(f"Job: {manifest.job_id}")
    logger.info(f"State: {manifest.state}")
    logger.info(f"Created: {manifest.created_at}")
    logger.info(f"Updated: {manifest.updated_at}")
    logger.info(f"Input: {manifest.input_path}")

    if manifest.output_path:
        logger.info(f"Output: {manifest.output_path}")

    if manifest.error:
        logger.info(f"Error: {manifest.error}")

    logger.info("")
    logger.info("Stages:")

    for stage in manifest.stages:
        status = "OK" if stage.success else "FAILED"
        logger.info(
            f"  {stage.name}: {status} ({stage.duration_seconds:.1f}s)"
        )
        if not stage.success and stage.error_message:
            logger.info(f"    Error: {stage.error_message}")

    logger.info("")
    logger.info(f"Artifacts: {len(manifest.artifacts)}")

    for name, artifact in manifest.artifacts.items():
        size_kb = artifact.size_bytes / 1024
        logger.info(f"  {name}: {size_kb:.1f} KB")

    return 0


def cmd_jobs_resume(job_id: str, config) -> int:
    """Resume a failed job."""
    logger = get_logger(__name__)

    try:
        job = Job.load_existing(
            job_id=job_id,
            config=config,
            project_root=Path.cwd(),
        )
    except JobNotFoundError:
        logger.error(f"Job not found: {job_id}")
        return 1

    if job.manifest.state != "failed":
        logger.error(f"Job is not in failed state: {job.manifest.state}")
        return 1

    input_path = Path(job.manifest.input_path)

    logger.info(f"Resuming job: {job_id}")

    success, _ = run_pipeline(
        input_path=input_path,
        config=config,
        project_root=Path.cwd(),
        resume_job_id=job_id,
    )

    return 0 if success else 1


def cmd_jobs_cleanup(job_id: str, config, keep_checkpoints: bool) -> int:
    """Cleanup job temporary files."""
    logger = get_logger(__name__)

    try:
        job = Job.load_existing(
            job_id=job_id,
            config=config,
            project_root=Path.cwd(),
        )
    except JobNotFoundError:
        logger.error(f"Job not found: {job_id}")
        return 1

    logger.info(f"Cleaning up job: {job_id}")

    job.cleanup(keep_checkpoints=keep_checkpoints)

    logger.info("Cleanup complete.")

    return 0


def cmd_health(args: argparse.Namespace, config) -> int:
    """Handle health command - check system readiness."""
    logger = get_logger(__name__)

    logger.info("V2D System Health Check")
    logger.info("=" * 40)

    all_ok = True

    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.info("[OK] FFmpeg: Available")
        else:
            logger.error("[FAIL] FFmpeg: Not working")
            all_ok = False
    except FileNotFoundError:
        logger.error("[FAIL] FFmpeg: Not installed")
        all_ok = False
    except subprocess.TimeoutExpired:
        logger.error("[FAIL] FFmpeg: Timeout")
        all_ok = False

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[OK] CUDA: {device_name} ({vram:.1f} GB)")
        else:
            logger.warning("[WARN] CUDA: Not available (will use CPU)")
    except ImportError:
        logger.error("[FAIL] PyTorch: Not installed")
        all_ok = False

    try:
        from faster_whisper import WhisperModel
        logger.info("[OK] Faster-Whisper: Available")
    except ImportError:
        logger.error("[FAIL] Faster-Whisper: Not installed")
        all_ok = False

    try:
        from transformers import MarianMTModel
        logger.info("[OK] Transformers: Available")
    except ImportError:
        logger.error("[FAIL] Transformers: Not installed")
        all_ok = False

    try:
        import demucs
        logger.info("[OK] Demucs: Available")
    except ImportError:
        logger.error("[FAIL] Demucs: Not installed")
        all_ok = False

    try:
        from TTS.api import TTS
        logger.info("[OK] Coqui TTS: Available")
    except ImportError:
        logger.error("[FAIL] Coqui TTS: Not installed")
        all_ok = False

    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    ram_available = psutil.virtual_memory().available / 1e9

    if ram_available >= 8:
        logger.info(f"[OK] RAM: {ram_available:.1f} GB available / {ram_gb:.1f} GB total")
    else:
        logger.warning(f"[WARN] RAM: {ram_available:.1f} GB available (recommend 8+ GB)")

    disk = psutil.disk_usage(Path.cwd())
    disk_free = disk.free / 1e9

    if disk_free >= 10:
        logger.info(f"[OK] Disk: {disk_free:.1f} GB free")
    else:
        logger.warning(f"[WARN] Disk: {disk_free:.1f} GB free (recommend 10+ GB)")

    logger.info("=" * 40)

    if all_ok:
        logger.info("System is ready for V2D processing.")
        return 0
    else:
        logger.error("Some components are missing. Please install dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
