"""EduMIND Configuration System.

Uses Pydantic BaseSettings V2 for robust, typed configuration validation.
Supports automatic environment variable loading from a .env file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    SecretStr,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJ_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Pydantic settings for the application.

    Validates types, defaults, and constraint ranges at startup time.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJ_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="EDUMIND_",
    )

    # --- System & Computation ---
    DEVICE_PREFERENCE: str = Field(default="auto", alias="EDUMIND_DEVICE")

    # --- Whisper (ASR) ---
    WHISPER_MODEL: str = Field(default="tiny")

    # --- Embedding ---
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # --- Translation ---
    TRANSLATION_MODEL: str = Field(default="none")

    # --- Qdrant (explicitly not prefixed with EDUMIND_) ---
    QDRANT_MODE: Literal["memory", "server"] = Field(
        default="memory",
        validation_alias="QDRANT_MODE",
    )
    QDRANT_HOST: str = Field(default="localhost", validation_alias="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, validation_alias="QDRANT_PORT")
    QDRANT_API_KEY: SecretStr = Field(default=SecretStr(""), validation_alias="QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = Field(
        default="edumind_documents",
        validation_alias="QDRANT_COLLECTION_NAME",
    )

    # --- Google API ---
    GOOGLE_API_KEY: SecretStr = Field(default=SecretStr(""), validation_alias="GOOGLE_API_KEY")

    # --- Internal Paths ---
    PROJ_ROOT: Path = PROJ_ROOT
    DATA_DIR: Path = PROJ_ROOT / "data"
    MODELS_DIR: Path = PROJ_ROOT / "models"

    @field_validator("QDRANT_PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validates that the port falls within the legitimate range."""
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @computed_field  # type: ignore[misc]
    @property
    def DEVICE(self) -> str:
        """Dynamically detects the optimal computation device without module-level torch import.

        Returns:
            String name of the device ("cuda", "mps", or "cpu").
        """
        if self.DEVICE_PREFERENCE != "auto":
            return self.DEVICE_PREFERENCE

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    @computed_field  # type: ignore[misc]
    @property
    def TEENCODE_MAP(self) -> dict[str, str]:
        """Loads and returns the Vietnamese/technical abbreviation translation map.

        Reads dynamically from teencode_map.json.
        """
        map_path = Path(__file__).parent / "teencode_map.json"
        if not map_path.exists():
            return {}
        try:
            with open(map_path, encoding="utf-8") as f:
                return dict(json.load(f))
        except Exception:
            return {}

    def summary(self) -> dict[str, Any]:
        """Exposes standard parameters for logging or frontend display."""
        return {
            "device": self.DEVICE,
            "whisper_model": self.WHISPER_MODEL,
            "embedding_model": self.EMBEDDING_MODEL,
            "translation_model": self.TRANSLATION_MODEL,
            "qdrant_mode": self.QDRANT_MODE,
            "has_google_api": bool(self.GOOGLE_API_KEY.get_secret_value()),
        }
