"""
Command-line interface for V2D.

Provides commands for processing videos, managing jobs, and system operations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from v2d import __version__
from v2d.core.config import load_config, ConfigValidationError
from v2d.core.job import Job
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

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process a video file",
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

    # Jobs command
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

    # Health command
    subparsers.add_parser(
        "health",
        help="Check system health",
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = get_logger(__name__)

    # Load configuration
    try:
        config = load_config(args.config)
    except ConfigValidationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return 1

    # Route to command handler
    if args.command == "process":
        return cmd_process(args, config)
    elif args.command == "jobs":
        return cmd_jobs(args, config)
    elif args.command == "health":
        return cmd_health(args, config)
    else:
        parser.print_help()
        return 0


def cmd_process(args: argparse.Namespace, config) -> int:
    """Handle process command."""
    logger = get_logger(__name__)

    input_path = args.input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info(f"Processing: {input_path}")

    # Create job
    job = Job.create(config=config)
    logger.info(f"Created job: {job.job_id}")

    try:
        # Initialize job
        job.initialize(input_path)
        logger.info("Job initialized")

        # TODO: Run pipeline stages
        logger.info("Pipeline execution not yet implemented")

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


def cmd_jobs(args: argparse.Namespace, config) -> int:
    """Handle jobs command."""
    logger = get_logger(__name__)

    if args.jobs_command == "list":
        logger.info("Job listing not yet implemented")
        return 0
    elif args.jobs_command == "status":
        logger.info(f"Status for job {args.job_id}")
        return 0
    elif args.jobs_command == "resume":
        logger.info(f"Resume job {args.job_id}")
        return 0
    elif args.jobs_command == "cleanup":
        logger.info(f"Cleanup job {args.job_id}")
        return 0
    else:
        logger.error("Unknown jobs command")
        return 1


def cmd_health(args: argparse.Namespace, config) -> int:
    """Handle health command."""
    logger = get_logger(__name__)
    logger.info("System health check not yet implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
