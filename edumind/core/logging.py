"""EduMIND Core — Structured Logging Configuration.

Configures ``structlog`` for the entire application:
  - **Development mode**: Pretty-printed coloured console output.
  - **Production mode**: JSON-formatted lines for log aggregators (ELK, Datadog).

The mode is controlled by the ``EDUMIND_LOG_FORMAT`` environment variable:
  - ``"console"`` (default) → human-readable coloured output.
  - ``"json"`` → machine-parseable JSON lines.

Every log entry is enriched with:
  - Timestamp (ISO-8601)
  - Logger name
  - Log level
  - Correlation ID (when bound via ``bind_correlation_id``)

Usage::

    from edumind.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("model_loaded", model_name="whisper-tiny", dimension=384)
"""

from __future__ import annotations

import logging
import os
import sys
import uuid

import structlog


def configure_logging(log_format: str | None = None, log_level: str | None = None) -> None:
    """Configures structlog processors and stdlib logging integration.

    Args:
        log_format: Output format — ``"console"`` or ``"json"``.
            Defaults to ``EDUMIND_LOG_FORMAT`` env var, then ``"console"``.
        log_level: Minimum log level — ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.
            Defaults to ``EDUMIND_LOG_LEVEL`` env var, then ``"INFO"``.
    """
    fmt = log_format or os.getenv("EDUMIND_LOG_FORMAT", "console")
    level_name = log_level or os.getenv("EDUMIND_LOG_LEVEL", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    # Shared processors for both formats
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib root logger to use structlog formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpcore", "httpx", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Returns a structlog bound logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger with standard enrichment processors.
    """
    return structlog.get_logger(name)


def bind_correlation_id(correlation_id: str | None = None) -> str:
    """Binds a correlation ID to the current context for request tracing.

    Args:
        correlation_id: Optional pre-existing ID. If ``None``, generates a new UUID.

    Returns:
        The correlation ID that was bound.
    """
    cid = correlation_id or uuid.uuid4().hex[:12]
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return cid


def clear_correlation_id() -> None:
    """Clears the correlation ID from the current context."""
    structlog.contextvars.unbind_contextvars("correlation_id")


# Auto-configure on first import
configure_logging()
