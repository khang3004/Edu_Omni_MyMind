"""EduMIND Utils — Retry and Circuit Breaker Decorators.

Provides reusable ``tenacity``-based retry decorators for external service calls
(LLM APIs, embedding model downloads, vector database operations).
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from edumind.core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _log_retry(retry_state: RetryCallState) -> None:
    """Logs retry attempts with structured context."""
    logger.warning(
        "retry_attempt",
        attempt=retry_state.attempt_number,
        wait_seconds=round(retry_state.next_action.sleep if retry_state.next_action else 0, 2),  # type: ignore[union-attr]
        exception=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


def retry_on_transient_error(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    ),
) -> Callable[[F], F]:
    """Creates a retry decorator with exponential backoff for transient errors.

    Args:
        max_attempts: Maximum number of attempts (including the first call).
        min_wait: Minimum wait time in seconds between retries.
        max_wait: Maximum wait time in seconds between retries.
        retryable_exceptions: Tuple of exception types that trigger a retry.

    Returns:
        A decorator function.
    """
    return retry(  # type: ignore[return-value]
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(retryable_exceptions),
        before_sleep=_log_retry,
        reraise=True,
    )
