"""API Key Rotator for EduMIND.

Dynamically scans environment variables matching configured prefixes
and returns key tokens sequentially or randomly to circumvent rate limits.
"""

from __future__ import annotations

import os
import random

from edumind.core.logging import get_logger

logger = get_logger(__name__)


class KeyRotator:
    """Helper class to fetch rotating API keys for various LLM/Embedding providers."""

    @staticmethod
    def get_key(prefix: str) -> str:
        """Retrieves an active API key matching the given prefix.

        Scans for variables like {prefix}1, {prefix}2, or {prefix} (without underscore).

        Args:
            prefix: Configuration prefix (e.g., "GEMINI_API_KEY_" or "GROQ_API_KEY_").

        Returns:
            The selected API key string, or an empty string if none found.
        """
        if not prefix:
            return ""

        # Normalize prefix to uppercase
        pref = prefix.upper()
        if not pref.endswith("_"):
            pref += "_"

        keys: list[str] = []

        # 1. Scan environment variables for matching indexed patterns (e.g. GEMINI_API_KEY_1)
        for key, val in os.environ.items():
            if key.startswith(pref):
                val_str = val.strip()
                if val_str:
                    keys.append(val_str)

        # 2. Check for singular form without trailing underscore (e.g. GEMINI_API_KEY or GROQ_API_KEY)
        singular_name = pref[:-1]
        val_singular = os.getenv(singular_name)
        if val_singular and val_singular.strip():
            keys.append(val_singular.strip())

        # 3. Ultimate Fallbacks based on provider name guess
        if not keys:
            if "GEMINI" in pref or "GOOGLE" in pref:
                fallback_key = os.getenv("GOOGLE_API_KEY")
                if fallback_key and fallback_key.strip():
                    keys.append(fallback_key.strip())
            elif "OPENAI" in pref:
                fallback_key = os.getenv("OPENAI_API_KEY")
                if fallback_key and fallback_key.strip():
                    keys.append(fallback_key.strip())

        if not keys:
            logger.debug("no_api_keys_found_for_prefix", prefix=prefix)
            return ""

        # Random choice provides a simple, stateless, thread-safe load distribution
        selected_key = random.choice(keys)
        logger.debug("api_key_selected_from_pool", prefix=prefix, pool_size=len(keys))
        return selected_key
