"""EduMIND Configuration Package.

Provides thread-safe, cached lazy access to the application settings.
"""

from __future__ import annotations

import functools
from edumind.config.settings import Settings


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the cached singleton Settings instance.

    This function avoids global module-level instantiation on import.
    """
    return Settings()


__all__ = ["get_settings", "Settings"]
