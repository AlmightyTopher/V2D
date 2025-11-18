"""
Job state machine for V2D.

Defines valid state transitions for jobs.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar


class JobState(str, Enum):
    """Valid states for a V2D job."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RESUMING = "resuming"
    CANCELLED = "cancelled"


class InvalidStateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: JobState, to_state: JobState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid state transition: {from_state.value} -> {to_state.value}"
        )


class StateMachine:
    """
    Manages job state transitions.

    Ensures only valid transitions are allowed and provides
    a consistent interface for state management.
    """

    # Valid state transitions
    TRANSITIONS: ClassVar[dict[JobState, set[JobState]]] = {
        JobState.CREATED: {JobState.INITIALIZING, JobState.CANCELLED},
        JobState.INITIALIZING: {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED},
        JobState.RUNNING: {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED},
        JobState.COMPLETED: set(),  # Terminal state
        JobState.FAILED: {JobState.RESUMING, JobState.CANCELLED},
        JobState.RESUMING: {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED},
        JobState.CANCELLED: set(),  # Terminal state
    }

    def __init__(self, initial_state: JobState = JobState.CREATED):
        """
        Initialize state machine.

        Args:
            initial_state: Starting state
        """
        self._state = initial_state
        self._history: list[JobState] = [initial_state]

    @property
    def state(self) -> JobState:
        """Get current state."""
        return self._state

    @property
    def history(self) -> list[JobState]:
        """Get state transition history."""
        return self._history.copy()

    def can_transition(self, to_state: JobState) -> bool:
        """
        Check if transition to state is valid.

        Args:
            to_state: Target state

        Returns:
            True if transition is valid
        """
        return to_state in self.TRANSITIONS.get(self._state, set())

    def transition(self, to_state: JobState) -> None:
        """
        Transition to a new state.

        Args:
            to_state: Target state

        Raises:
            InvalidStateTransitionError: If transition is invalid
        """
        if not self.can_transition(to_state):
            raise InvalidStateTransitionError(self._state, to_state)

        self._state = to_state
        self._history.append(to_state)

    def is_terminal(self) -> bool:
        """Check if current state is terminal (no further transitions)."""
        return len(self.TRANSITIONS.get(self._state, set())) == 0

    def is_failed(self) -> bool:
        """Check if job is in failed state."""
        return self._state == JobState.FAILED

    def is_complete(self) -> bool:
        """Check if job completed successfully."""
        return self._state == JobState.COMPLETED

    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self._state in (JobState.RUNNING, JobState.INITIALIZING, JobState.RESUMING)

    def reset(self, state: JobState = JobState.CREATED) -> None:
        """
        Reset state machine.

        Args:
            state: State to reset to
        """
        self._state = state
        self._history = [state]
