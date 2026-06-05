"""EduMIND Vector Store Service — Base Protocol.

Defines the abstract interface for storage, indexing, and similarity search
over document vectors.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from edumind.models.chunks import DocumentChunk, RetrievedChunk


class VectorStore(Protocol):
    """Protocol establishing the requirements for a vector database implementation."""

    def upsert(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> int:
        """Stores a batch of text chunks and their matching embeddings in the index.

        Args:
            chunks: List of DocumentChunk domain models.
            embeddings: 2D numpy array of computed embeddings.

        Returns:
            The number of successfully indexed vectors.
        """
        ...

    def search(self, query_vector: list[float], limit: int = 5) -> list[RetrievedChunk]:
        """Queries the vector database using a semantic query vector.

        Args:
            query_vector: Embedded search query as a list of floats.
            limit: Number of matches to retrieve.

        Returns:
            A list of RetrievedChunk objects ordered by descending score.
        """
        ...

    def clear_index(self) -> bool:
        """Wipes and recreates the target vector collection completely.

        Returns:
            True if successful.
        """
        ...

    def collection_info(self) -> dict[str, Any]:
        """Retrieves operational schema and current statistics of the collection.

        Returns:
            A dict containing status, count, and collection details.
        """
        ...

    @property
    def is_ready(self) -> bool:
        """Checks if the vector database client is fully configured and connected."""
        ...
