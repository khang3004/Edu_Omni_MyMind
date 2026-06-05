"""Qdrant Vector Database Integration.

Implements the VectorStore protocol with robust connection handling, automatic
collection creation, and transient-error retries.
"""

from __future__ import annotations

from typing import Any
import uuid

import numpy as np

from edumind.core.exceptions import VectorStoreError
from edumind.core.logging import get_logger
from edumind.models.chunks import DocumentChunk, RetrievedChunk
from edumind.services.vectorstore.base import VectorStore
from edumind.utils.retry import retry_on_transient_error

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant client wrapper implementing standard vector search and upsert operations."""

    def __init__(
        self,
        mode: str = "memory",
        host: str = "localhost",
        port: int = 6333,
        api_key: str = "",
        collection_name: str = "edumind_documents",
        embedding_dim: int = 384,
        path: str = "",
    ):
        """Initializes the Qdrant connection and verifies collection schema."""
        self._mode = mode
        self._host = host
        self._port = port
        self._api_key = api_key
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._path = path
        self._client: Any = None
        self._ready = False

        self._connect()

    def _connect(self) -> None:
        """Initializes the QdrantClient and confirms/creates the collection."""
        try:
            from qdrant_client import QdrantClient

            if self._mode == "memory":
                logger.info("initializing_qdrant_in_memory")
                self._client = QdrantClient(":memory:")
            elif self._mode == "local":
                logger.info("initializing_qdrant_local_disk", path=self._path)
                self._client = QdrantClient(path=self._path)
            else:
                logger.info("connecting_to_qdrant_server", host=self._host, port=self._port)
                self._client = QdrantClient(
                    host=self._host,
                    port=self._port,
                    api_key=self._api_key if self._api_key else None,
                )

            self._ensure_collection()
            self._ready = True
            logger.info("qdrant_vectorstore_ready", collection=self._collection_name)
        except Exception as e:
            logger.error("qdrant_initialization_failed", error=str(e))
            self._ready = False
            self._client = None

    def _ensure_collection(self) -> None:
        """Verifies if the collection exists, creating it if missing."""
        assert self._client is not None
        from qdrant_client.models import Distance, VectorParams

        try:
            collections = self._client.get_collections().collections
            existing_names = [c.name for c in collections]

            if self._collection_name not in existing_names:
                logger.info("creating_qdrant_collection", name=self._collection_name, dim=self._embedding_dim)
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
            else:
                logger.debug("qdrant_collection_exists", name=self._collection_name)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to verify/create Qdrant collection '{self._collection_name}'",
                details={"collection_name": self._collection_name, "error": str(e)},
            ) from e

    @property
    def is_ready(self) -> bool:
        """Checks if the vector db connection is active."""
        return self._ready and self._client is not None

    @retry_on_transient_error(max_attempts=3)
    def upsert(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> int:
        """Indexes document chunks and vector representations into Qdrant in batches.

        Args:
            chunks: List of DocumentChunk.
            embeddings: Numpy array of matching embedding representations.

        Returns:
            The number of successfully indexed chunks.
        """
        if not self.is_ready:
            raise VectorStoreError("Cannot upsert; QdrantVectorStore is not ready.")

        if not chunks:
            return 0

        assert len(chunks) == len(embeddings)
        from qdrant_client.models import PointStruct

        logger.info("upserting_vectors", count=len(chunks), collection=self._collection_name)

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata,
                },
            ))

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            logger.info("upsert_completed", count=len(points))
            return len(points)
        except Exception as e:
            logger.error("qdrant_upsert_failed", error=str(e))
            raise VectorStoreError(
                f"Failed to upsert points into collection '{self._collection_name}'",
                details={"count": len(chunks), "error": str(e)},
            ) from e

    @retry_on_transient_error(max_attempts=3)
    def search(self, query_vector: list[float], limit: int = 5) -> list[RetrievedChunk]:
        """Performs a cosine-similarity semantic vector search over the index.

        Args:
            query_vector: Semantic vector search representation.
            limit: Maximum retrieved limit.

        Returns:
            Sorted list of RetrievedChunk objects.
        """
        if not self.is_ready:
            raise VectorStoreError("Cannot search; QdrantVectorStore is not ready.")

        logger.debug("querying_qdrant", limit=limit)

        try:
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query_vector,
                    limit=limit,
                )
                results = response.points
            else:
                results = self._client.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    limit=limit,
                )

            retrieved = []
            for hit in results:
                payload = hit.payload or {}
                text = payload.pop("text", "")
                retrieved.append(RetrievedChunk(
                    text=text,
                    metadata=payload,
                    score=float(hit.score),
                ))

            logger.debug("query_completed", found=len(retrieved))
            return retrieved
        except Exception as e:
            logger.error("qdrant_search_failed", error=str(e))
            raise VectorStoreError(
                f"Failed to perform search on collection '{self._collection_name}'",
                details={"limit": limit, "error": str(e)},
            ) from e

    def clear_index(self) -> bool:
        """Wipes the existing collection completely and recreates it."""
        if not self.is_ready:
            return False

        logger.warning("clearing_vector_index", collection=self._collection_name)
        from qdrant_client.models import Distance, VectorParams

        try:
            self._client.delete_collection(self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("cleared_and_recreated_collection", name=self._collection_name)
            return True
        except Exception as e:
            logger.error("clear_index_failed", error=str(e))
            return False

    def collection_info(self) -> dict[str, Any]:
        """Gathers schema point counts and status information."""
        if not self.is_ready:
            return {"status": "not_ready", "count": 0}

        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "status": "ready",
                "collection_name": self._collection_name,
                "vectors_count": getattr(info, "indexed_vectors_count", getattr(info, "vectors_count", 0)),
                "points_count": info.points_count,
            }
        except Exception as e:
            logger.error("get_collection_info_failed", error=str(e))
            return {"status": "error", "error": str(e), "count": 0}
