"""EduMIND — Multimodal RAG Engine with Qdrant Vector Database

This module handles Retrieval-Augmented Generation (RAG) for teaching documents:
  1. Performs Layout-Aware PDF Chunking.
  2. Embeds document chunks using Sentence-Transformers.
  3. Stores and queries vectors using Qdrant (in-memory or remote server).
  4. Synthesizes contextual answers with source citations.

Architecture:
    MultimodalRAG
    ├── ingest_pdf(path) → list[DocumentChunk]
    ├── ingest_text(text, source) → list[DocumentChunk]
    ├── embed_and_store(chunks)
    ├── query(question, top_k) → list[RetrievedChunk]
    ├── generate_answer(question, contexts) → str
    └── clear_index()

Usage:
    rag = MultimodalRAG()
    chunks = rag.ingest_pdf("lecture_slides.pdf")
    rag.embed_and_store(chunks)
    results = rag.query("How does attention mechanism work?")
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# ----------------------------------------------------------------------
# Global cache for sentence transformer models to avoid reloading
# ----------------------------------------------------------------------
_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}


# ----------------------------------------------------------------------
# Data Classes for Document Chunks and Query Results
# ----------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """Represents a single parsed text chunk with metadata.

    Attributes:
        text: The textual content of the chunk.
        metadata: Associated metadata dictionary (e.g., page, source, section).
        chunk_id: A unique identifier for the chunk.
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        """Generates a unique chunk_id if not already provided."""
        if not self.chunk_id:
            content_hash = hashlib.md5(self.text.encode()).hexdigest()[:8]
            self.chunk_id = f"chunk_{content_hash}_{uuid.uuid4().hex[:6]}"


@dataclass
class RetrievedChunk:
    """Represents a matched chunk retrieved from vector search.

    Attributes:
        text: Retrieved text content.
        metadata: Original chunk metadata.
        score: Cosine similarity score (0.0 to 1.0).
    """
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


# ----------------------------------------------------------------------
# Main Class: MultimodalRAG
# ----------------------------------------------------------------------
class MultimodalRAG:
    """Multimodal RAG engine using Qdrant vector database.

    Handles layout-aware parsing, sentence-transformers text embeddings,
    Qdrant search indexing, and citation-aware answer generation.
    """

    def __init__(
        self,
        embedding_model_name: str | None = None,
        qdrant_mode: str | None = None,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        qdrant_api_key: str | None = None,
        collection_name: str | None = None,
    ):
        """Initializes the MultimodalRAG engine.

        Args:
            embedding_model_name: Name of the Sentence-Transformers model.
            qdrant_mode: Connection mode ("memory" or "server").
            qdrant_host: Hostname of remote Qdrant server (for server mode).
            qdrant_port: Port of remote Qdrant server (for server mode).
            qdrant_api_key: Authentication API key for Qdrant Cloud.
            collection_name: Target Qdrant collection name.
        """
        # Load configuration defaults from settings
        from edumind.config import settings

        self._embedding_model_name = embedding_model_name or settings.EMBEDDING_MODEL
        self._qdrant_mode = qdrant_mode or settings.QDRANT_MODE
        self._qdrant_host = qdrant_host or settings.QDRANT_HOST
        self._qdrant_port = qdrant_port or settings.QDRANT_PORT
        self._qdrant_api_key = qdrant_api_key or settings.QDRANT_API_KEY
        self._collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

        # Internal state
        self._embedding_model = None
        self._qdrant_client = None
        self._embedding_dim: int = 384  # Default dimension for all-MiniLM-L6-v2
        self._is_ready = False

        # Initialize sub-components
        self._load_embedding_model()
        self._init_qdrant()

    # ------------------------------------------------------------------
    # Initialization Methods
    # ------------------------------------------------------------------
    def _load_embedding_model(self) -> None:
        """Loads the Sentence-Transformers model for text embedding.

        Falls back automatically to mock embeddings (random vectors) if the model
        fails to load.
        """
        try:
            global _EMBEDDING_MODEL_CACHE
            if self._embedding_model_name not in _EMBEDDING_MODEL_CACHE:
                from sentence_transformers import SentenceTransformer
                logger.info(f"📐 Loading embedding model: {self._embedding_model_name}...")
                _EMBEDDING_MODEL_CACHE[self._embedding_model_name] = SentenceTransformer(self._embedding_model_name)

            self._embedding_model = _EMBEDDING_MODEL_CACHE[self._embedding_model_name]

            # Infer real dimensionality of the model
            test_emb = self._embedding_model.encode(["test"])
            self._embedding_dim = test_emb.shape[1]

            logger.success(
                f"✅ Loaded embedding model! Dimension: {self._embedding_dim}"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Could not load embedding model '{self._embedding_model_name}': {e}. "
                "Using mock embedding fallback (random vectors)."
            )
            self._embedding_model = None

    def _init_qdrant(self) -> None:
        """Initializes the Qdrant client and verifies/creates the target collection.

        Supports:
            - "memory": In-process ephemeral database (ideal for dev/demo).
            - "server": Remote server or Qdrant Cloud deployment.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            # Instantiate client based on configured mode
            if self._qdrant_mode == "memory":
                logger.info("🗄️ Initializing Qdrant in-memory mode...")
                self._qdrant_client = QdrantClient(":memory:")
            else:
                logger.info(
                    f"🗄️ Connecting to Qdrant server at: {self._qdrant_host}:{self._qdrant_port}..."
                )
                self._qdrant_client = QdrantClient(
                    host=self._qdrant_host,
                    port=self._qdrant_port,
                    api_key=self._qdrant_api_key if self._qdrant_api_key else None,
                )

            # Check if collection already exists
            collections = self._qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            if self._collection_name not in existing_names:
                self._qdrant_client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.success(
                    f"✅ Created Qdrant collection: '{self._collection_name}' "
                    f"(dim={self._embedding_dim}, distance=Cosine)"
                )
            else:
                logger.info(f"📦 Qdrant collection '{self._collection_name}' already exists.")

            self._is_ready = True

        except ImportError:
            logger.warning(
                "⚠️ Library 'qdrant-client' is not installed. "
                "The RAG engine is disabled."
            )
            self._is_ready = False
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Qdrant database: {e}")
            self._is_ready = False

    # ------------------------------------------------------------------
    # Ingestion & Layout-Aware Parsing
    # ------------------------------------------------------------------
    def ingest_pdf(self, pdf_path: str | Path) -> list[DocumentChunk]:
        """Parses a PDF using layout-aware paragraph and heading splitting.

        Layout-Aware Parsing Algorithm:
            1. Reads pages sequentially using pypdf.
            2. Identifies section headers (short lines, numbered, or ALL CAPS).
            3. Clusters remaining text lines into coherent paragraphs.
            4. Emits chunks annotated with rich metadata.

        Args:
            pdf_path: Path to the target PDF file.

        Returns:
            A list of DocumentChunk instances.

        Raises:
            FileNotFoundError: If the target PDF file does not exist.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        logger.info(f"📄 Ingesting PDF file: {pdf_path.name}...")

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            chunks: list[DocumentChunk] = []
            chunk_index = 0

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                page_chunks = self._layout_aware_split(
                    text=page_text,
                    page_number=page_num,
                    source_file=pdf_path.name,
                    start_chunk_index=chunk_index,
                )
                chunks.extend(page_chunks)
                chunk_index += len(page_chunks)

            logger.success(
                f"✅ Ingested PDF '{pdf_path.name}': {len(reader.pages)} pages → {len(chunks)} chunks"
            )
            return chunks

        except ImportError:
            logger.error("❌ Library 'pypdf' is not installed!")
            return []
        except Exception as e:
            logger.error(f"❌ Error during PDF ingestion: {e}")
            return []

    def ingest_text(
        self,
        text: str,
        source_name: str = "text_input",
        metadata_extra: dict | None = None,
    ) -> list[DocumentChunk]:
        """Parses a raw string input into standard metadata-rich chunks.

        Args:
            text: Raw document text string.
            source_name: Identified source name for the text block.
            metadata_extra: Optional dictionary of additional metadata flags.

        Returns:
            A list of parsed DocumentChunk objects.
        """
        if not text or not text.strip():
            return []

        chunks = self._layout_aware_split(
            text=text,
            page_number=1,
            source_file=source_name,
            start_chunk_index=0,
        )

        # Merge extra metadata if supplied
        if metadata_extra:
            for chunk in chunks:
                chunk.metadata.update(metadata_extra)

        logger.info(f"📝 Raw Text '{source_name}': processed into {len(chunks)} chunks")
        return chunks

    def _layout_aware_split(
        self,
        text: str,
        page_number: int,
        source_file: str,
        start_chunk_index: int,
        max_chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[DocumentChunk]:
        """Splits raw text of a page into paragraphs based on structural layout.

        Identifies headings to create logical boundaries, and ensures paragraphs
        exceeding max_chunk_size are split with overlap.
        """
        lines = text.split("\n")
        chunks: list[DocumentChunk] = []
        current_section = "Untitled Section"
        current_paragraph: list[str] = []
        idx = start_chunk_index

        for line in lines:
            stripped = line.strip()

            if self._is_header(stripped):
                # Flush the current paragraph before changing sections
                if current_paragraph:
                    para_text = " ".join(current_paragraph).strip()
                    if para_text:
                        for sub_chunk in self._split_long_text(para_text, max_chunk_size, overlap):
                            chunks.append(DocumentChunk(
                                text=sub_chunk,
                                metadata={
                                    "page_number": page_number,
                                    "source_file": source_file,
                                    "section_header": current_section,
                                    "chunk_index": idx,
                                    "type": "paragraph",
                                },
                            ))
                            idx += 1
                    current_paragraph = []

                # Update the active section header
                current_section = stripped

            elif stripped == "":
                # Empty line marks paragraph boundary
                if current_paragraph:
                    para_text = " ".join(current_paragraph).strip()
                    if para_text:
                        for sub_chunk in self._split_long_text(para_text, max_chunk_size, overlap):
                            chunks.append(DocumentChunk(
                                text=sub_chunk,
                                metadata={
                                    "page_number": page_number,
                                    "source_file": source_file,
                                    "section_header": current_section,
                                    "chunk_index": idx,
                                    "type": "paragraph",
                                },
                            ))
                            idx += 1
                    current_paragraph = []
            else:
                # Append standard lines to the current paragraph block
                current_paragraph.append(stripped)

        # Flush any trailing paragraph
        if current_paragraph:
            para_text = " ".join(current_paragraph).strip()
            if para_text:
                for sub_chunk in self._split_long_text(para_text, max_chunk_size, overlap):
                    chunks.append(DocumentChunk(
                        text=sub_chunk,
                        metadata={
                            "page_number": page_number,
                            "source_file": source_file,
                            "section_header": current_section,
                            "chunk_index": idx,
                            "type": "paragraph",
                        },
                    ))
                    idx += 1

        return chunks

    @staticmethod
    def _is_header(line: str) -> bool:
        """Determines if a given line qualifies structurally as a section header.

        Rules:
            1. Short lines (< 100 chars) written in ALL CAPS.
            2. Lines ending with a colon ':'.
            3. Lines starting with numbered indexes (e.g., '1.', '2.1', 'Chương 1').

        Args:
            line: Text line string to inspect.

        Returns:
            True if the line is identified as a heading.
        """
        if not line or len(line) < 2:
            return False

        # ALL CAPS check for short lines
        alpha_chars = [c for c in line if c.isalpha()]
        if alpha_chars and all(c.isupper() for c in alpha_chars) and len(line) < 100:
            return True

        # Heading ends with colon
        if line.endswith(":") and len(line) < 100:
            return True

        # Numbered index patterns (e.g., "1. Introduction", "2.1 Background", "Chapter 3")
        if re.match(r"^\d+(\.\d+)*\.?\s", line) or re.match(r"^(Chapter|Chương|Bài|Phần)\s", line, re.IGNORECASE):
            return True

        return False

    @staticmethod
    def _split_long_text(text: str, max_size: int, overlap: int) -> list[str]:
        """Slices a long text into overlapping chunks cleanly at word boundaries.

        Args:
            text: Large paragraph text to slice.
            max_size: Maximum character length allowed per slice.
            overlap: Slicing character overlap size.

        Returns:
            A list of text strings representing the sub-chunks.
        """
        if len(text) <= max_size:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(text):
            end = start + max_size

            # Attempt to split gracefully at sentence or word boundaries
            if end < len(text):
                best_break = text.rfind(". ", start, end)
                if best_break == -1 or best_break <= start:
                    best_break = text.rfind(" ", start, end)
                if best_break > start:
                    end = best_break + 1

            parts.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end

        return [p for p in parts if p]

    # ------------------------------------------------------------------
    # Vector Indexing & Retrieval
    # ------------------------------------------------------------------
    def embed_and_store(self, chunks: list[DocumentChunk]) -> int:
        """Computes text embeddings and saves the chunks inside Qdrant.

        Args:
            chunks: A list of DocumentChunk objects to save.

        Returns:
            The number of successfully indexed chunks.
        """
        if not chunks:
            logger.warning("⚠️ Empty chunks list provided; nothing to index.")
            return 0

        if not self._is_ready or self._qdrant_client is None:
            logger.error("❌ Qdrant vector database is not ready; cannot store chunks.")
            return 0

        logger.info(f"📐 Generating embeddings for {len(chunks)} chunks...")

        # Compute embeddings
        texts = [c.text for c in chunks]
        embeddings = self._encode_texts(texts)

        if embeddings is None:
            logger.error("❌ Text embedding generation failed.")
            return 0

        # Construct payload structures for Qdrant
        from qdrant_client.models import PointStruct

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
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

        # Perform batch upsert
        try:
            self._qdrant_client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            logger.success(f"✅ Successfully indexed {len(points)} chunks into Qdrant!")
            return len(points)
        except Exception as e:
            logger.error(f"❌ Failed to upsert points into Qdrant: {e}")
            return 0

    def _encode_texts(self, texts: list[str]):
        """Generates embedding representations for list of texts.

        Uses the active SentenceTransformer model, falling back to safe random
        vectors (mock) if the model is bypassed or fails.
        """
        if self._embedding_model is not None:
            try:
                return self._embedding_model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=32,
                )
            except Exception as e:
                logger.warning(f"⚠️ Model embedding error: {e}. Falling back to mock vectors.")

        # Simulated fallback (random vectors)
        import numpy as np
        logger.info("🎭 Using simulated mock embeddings (random vectors).")
        return np.random.randn(len(texts), self._embedding_dim).astype(np.float32)

    # ------------------------------------------------------------------
    # Retrieval-Augmented Generation & Querying
    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Queries the Qdrant database using semantic similarity search.

        Supports modern QdrantClient unified query API with older method fallback.

        Args:
            question: Search query / question text.
            top_k: Number of nearest matches to return.

        Returns:
            A list of RetrievedChunk objects sorted by descending similarity scores.
        """
        if not self._is_ready or self._qdrant_client is None:
            logger.error("❌ Qdrant vector database is not ready; query aborted.")
            return []

        if not question or not question.strip():
            return []

        logger.info(f"🔍 Querying database for: '{question[:50]}...'")

        # Embed search query
        query_embedding = self._encode_texts([question])
        if query_embedding is None:
            return []

        # Query index
        try:
            if hasattr(self._qdrant_client, "query_points"):
                response = self._qdrant_client.query_points(
                    collection_name=self._collection_name,
                    query=query_embedding[0].tolist(),
                    limit=top_k,
                )
                results = response.points
            else:
                results = self._qdrant_client.search(
                    collection_name=self._collection_name,
                    query_vector=query_embedding[0].tolist(),
                    limit=top_k,
                )

            retrieved = []
            for hit in results:
                payload = hit.payload or {}
                text = payload.pop("text", "")
                retrieved.append(RetrievedChunk(
                    text=text,
                    metadata=payload,
                    score=round(hit.score, 4),
                ))

            logger.success(f"✅ Found {len(retrieved)} relevant results!")
            return retrieved

        except Exception as e:
            logger.error(f"❌ Error querying Qdrant: {e}")
            return []

    def generate_answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
    ) -> str:
        """Synthesizes a structured, citation-annotated answer using context chunks.

        Args:
            question: The original user question.
            contexts: List of RetrievedChunk matched items.

        Returns:
            A formatted string containing answer synthesis and citations.
        """
        if not contexts:
            return (
                "❌ No relevant information found in the uploaded documents. "
                "Please try asking another question or uploading more files."
            )

        answer_parts: list[str] = []
        answer_parts.append(f"📋 **Question:** {question}\n")
        answer_parts.append("📖 **Synthesized Answer from Documents:**\n")

        # Synthesize answers from top-3 relevant context chunks
        for i, ctx in enumerate(contexts[:3], start=1):
            page = ctx.metadata.get("page_number", "?")
            source = ctx.metadata.get("source_file", "Unknown")
            section = ctx.metadata.get("section_header", "")

            citation = f"[Page {page}, {source}]"
            if section and section != "Untitled Section":
                citation = f"[Page {page}, {source} — §{section}]"

            text_preview = ctx.text[:300] + "..." if len(ctx.text) > 300 else ctx.text
            answer_parts.append(
                f"**{i}.** {text_preview}\n   — *{citation}* (Relevance: {ctx.score:.2f})\n"
            )

        # Footer
        answer_parts.append(
            "\n💡 *Note: The answers above are synthesized directly from your matched "
            "document contexts. Verify with original sources for maximum accuracy.*"
        )

        return "\n".join(answer_parts)

    # ------------------------------------------------------------------
    # Maintenance & Utility Methods
    # ------------------------------------------------------------------
    def clear_index(self) -> bool:
        """Wipes the active Qdrant collection completely and re-creates it.

        Returns:
            True if collection was cleared and re-created successfully.
        """
        if not self._is_ready or self._qdrant_client is None:
            return False

        try:
            from qdrant_client.models import Distance, VectorParams

            self._qdrant_client.delete_collection(self._collection_name)
            self._qdrant_client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"🗑️ Cleared and re-created Qdrant collection '{self._collection_name}'")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            return False

    def get_collection_info(self) -> dict:
        """Retrieves operational schema and point counts of the Qdrant collection.

        Returns:
            A status dictionary of the collection statistics.
        """
        if not self._is_ready or self._qdrant_client is None:
            return {"status": "not_ready", "count": 0}

        try:
            info = self._qdrant_client.get_collection(self._collection_name)
            return {
                "status": "ready",
                "collection_name": self._collection_name,
                "vectors_count": getattr(info, "indexed_vectors_count", getattr(info, "vectors_count", 0)),
                "points_count": info.points_count,
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "count": 0}

    @property
    def is_ready(self) -> bool:
        """Checks if the RAG engine and database backend are fully functional."""
        return self._is_ready

    @property
    def has_embedding_model(self) -> bool:
        """Checks if the physical text embedding model is loaded."""
        return self._embedding_model is not None
