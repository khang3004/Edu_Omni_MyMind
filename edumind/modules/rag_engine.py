"""EduMIND — Multimodal RAG Engine with Qdrant Vector Database.

This module handles Retrieval-Augmented Generation (RAG) for teaching documents:
  1. Performs Layout-Aware PDF Chunking.
  2. Embeds document chunks using Strategy-based embedding providers.
  3. Stores and queries vectors using Qdrant (in-memory or remote server).
  4. Applies keyword-boosting, query expansion, and cross-encoder re-ranking.
  5. Synthesizes contextual answers utilizing generative AI model providers.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from edumind.config import get_settings
from edumind.core.exceptions import EmbeddingError, VectorStoreError
from edumind.core.logging import get_logger
from edumind.models.chunks import DocumentChunk, RetrievedChunk
from edumind.services.embedding.base import EmbeddingProvider
from edumind.services.llm.base import LLMProvider
from edumind.services.vectorstore.base import VectorStore
from edumind.utils.text import is_section_header, split_long_text

logger = get_logger(__name__)


class CrossEncoderReRanker:
    """Optional cross-encoder re-ranking step for retrieved passages."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initializes the re-ranker.

        Args:
            model_name: HuggingFace model hub path.
        """
        self.model_name = model_name
        self._model = None
        self._attempted = False

    def load(self) -> None:
        """Lazily loads the Cross-Encoder model into memory."""
        if self._attempted:
            return
        self._attempted = True
        try:
            from sentence_transformers import CrossEncoder

            logger.info("loading_cross_encoder_model", model=self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("loaded_cross_encoder_model", model=self.model_name)
        except Exception as e:
            logger.warning(
                "cross_encoder_load_failed_bypassing_reranker",
                model=self.model_name,
                error=str(e),
            )
            self._model = None

    def re_rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Re-scores retrieved passages relative to the query.

        Args:
            query: User search query.
            chunks: List of passages retrieved from vector search.

        Returns:
            Re-ordered list of chunks.
        """
        if not chunks:
            return chunks

        self.load()
        if self._model is None:
            return chunks

        try:
            pairs = [[query, c.text] for c in chunks]
            # Predict similarity scores
            scores = self._model.predict(pairs)

            # Map raw logit scores to standard [0, 1] probability range via sigmoid
            for chunk, score in zip(chunks, scores):
                val = float(score)
                sigmoid_score = 1.0 / (1.0 + np.exp(-val))
                chunk.score = round(sigmoid_score, 4)

            # Re-sort descending
            return sorted(chunks, key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.warning("cross_encoder_rerank_failed_using_original", error=str(e))
            return chunks


class MultimodalRAG:
    """Multimodal RAG engine utilizing injected vector stores, embeddings, and LLMs.

    Fully complies with Dependency Injection, allowing modular test strategies and
    caching implementations.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vectorstore: VectorStore | None = None,
        llm_provider: LLMProvider | None = None,
        rerank_model_name: str | None = None,
    ):
        """Initializes the MultimodalRAG engine.

        Args:
            embedding_provider: Embedding model strategy. If None, resolves from container.
            vectorstore: Vector database strategy. If None, resolves from container.
            llm_provider: Answer generator strategy. If None, resolves from container.
            rerank_model_name: Cross-encoder re-ranking model path (optional).
        """
        self._embedding_provider = embedding_provider
        self._vectorstore = vectorstore
        self._llm_provider = llm_provider

        # Setup cross-encoder re-ranker
        self._reranker = (
            CrossEncoderReRanker(rerank_model_name)
            if rerank_model_name
            else CrossEncoderReRanker()
        )

    def _get_embedding_provider(self) -> EmbeddingProvider:
        """Retrieves the active embedding provider, resolving from container if needed."""
        if self._embedding_provider is None:
            from edumind.core.container import get_embedding_provider
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    def _get_vectorstore(self) -> VectorStore:
        """Retrieves the active vector store, resolving from container if needed."""
        if self._vectorstore is None:
            from edumind.core.container import get_vectorstore
            self._vectorstore = get_vectorstore()
        return self._vectorstore

    def _get_llm_provider(self) -> LLMProvider:
        """Retrieves the active LLM provider, resolving from container if needed."""
        if self._llm_provider is None:
            from edumind.core.container import get_llm_provider
            self._llm_provider = get_llm_provider()
        return self._llm_provider

    # ------------------------------------------------------------------
    # Ingestion & Layout-Aware Parsing
    # ------------------------------------------------------------------
    def ingest_pdf(self, pdf_path: str | Path) -> list[DocumentChunk]:
        """Parses a PDF using layout-aware paragraph and heading splitting.

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

        logger.info("ingesting_pdf_file", file_name=pdf_path.name)

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

            logger.info("pdf_ingested_successfully", pages=len(reader.pages), chunks=len(chunks))
            return chunks

        except ImportError:
            logger.error("pypdf_library_missing", reason="pypdf is not installed")
            return []
        except Exception as e:
            logger.error("pdf_ingestion_failed", file=pdf_path.name, error=str(e))
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

        if metadata_extra:
            for chunk in chunks:
                chunk.metadata.update(metadata_extra)

        logger.info("raw_text_ingested", source=source_name, chunks=len(chunks))
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
        """Splits raw text of a page into paragraphs based on structural layout."""
        lines = text.split("\n")
        chunks: list[DocumentChunk] = []
        current_section = "Untitled Section"
        current_paragraph: list[str] = []
        idx = start_chunk_index

        for line in lines:
            stripped = line.strip()

            if is_section_header(stripped):
                # Flush existing paragraph
                if current_paragraph:
                    para_text = " ".join(current_paragraph).strip()
                    if para_text:
                        for sub_chunk in split_long_text(para_text, max_chunk_size, overlap):
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
                current_section = stripped

            elif stripped == "":
                # Empty line boundary
                if current_paragraph:
                    para_text = " ".join(current_paragraph).strip()
                    if para_text:
                        for sub_chunk in split_long_text(para_text, max_chunk_size, overlap):
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
                current_paragraph.append(stripped)

        # Flush trailing
        if current_paragraph:
            para_text = " ".join(current_paragraph).strip()
            if para_text:
                for sub_chunk in split_long_text(para_text, max_chunk_size, overlap):
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
            logger.warning("empty_chunks_list_provided_indexing_skipped")
            return 0

        try:
            emb_provider = self._get_embedding_provider()
            vectorstore = self._get_vectorstore()

            logger.info("generating_embeddings_for_ingested_chunks", count=len(chunks))
            texts = [c.text for c in chunks]
            embeddings = emb_provider.encode(texts)

            indexed_count = vectorstore.upsert(chunks, embeddings)
            logger.info("indexing_completed", indexed_count=indexed_count)
            return indexed_count
        except (EmbeddingError, VectorStoreError) as e:
            logger.error("indexing_pipeline_failed", error=str(e))
            return 0
        except Exception as e:
            logger.error("unexpected_error_during_indexing", error=str(e))
            return 0

    # ------------------------------------------------------------------
    # Query Expansion and Hybrid Retrieval
    # ------------------------------------------------------------------
    def _expand_query(self, query: str) -> str:
        """Applies query expansion by resolving abbreviations/teencode."""
        settings = get_settings()
        teencode_map = settings.TEENCODE_MAP

        expanded = query
        # Sort key lengths descending to prevent partial substitutions
        sorted_keys = sorted(teencode_map.keys(), key=len, reverse=True)

        for abbr in sorted_keys:
            pattern = r"\b" + re.escape(abbr) + r"\b"
            expanded = re.sub(pattern, teencode_map[abbr], expanded, flags=re.IGNORECASE)

        # Deduplicate multiple spaces
        expanded = re.sub(r"\s+", " ", expanded).strip()

        if expanded != query:
            logger.info("query_expanded", original=query, expanded=expanded)
        return expanded

    def _apply_keyword_boost(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Boosts RetrievedChunk cosine scores based on exact lexical keyword hits."""
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return chunks

        for chunk in chunks:
            chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
            common_words = query_words.intersection(chunk_words)
            overlap_ratio = len(common_words) / len(query_words)
            # Add up to 0.1 lexical boost
            boost = round(0.10 * overlap_ratio, 4)
            chunk.score = min(1.0, chunk.score + boost)

        return sorted(chunks, key=lambda x: x.score, reverse=True)

    def query(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Queries the vector index applying semantic search and hybrid re-ranking.

        Args:
            question: Search query.
            top_k: Number of matches to return.

        Returns:
            Sorted list of RetrievedChunk matches.
        """
        if not question or not question.strip():
            return []

        try:
            emb_provider = self._get_embedding_provider()
            vectorstore = self._get_vectorstore()

            # 1. Query Expansion (teencode resolution)
            expanded_query = self._expand_query(question)

            # 2. Embed query
            query_vector = emb_provider.encode([expanded_query])[0].tolist()

            # 3. Vector Database retrieval
            retrieved = vectorstore.search(query_vector, limit=top_k * 2)

            # 4. Keyword Boosting (Lexical Hybrid component)
            boosted = self._apply_keyword_boost(expanded_query, retrieved)

            # 5. Optional Cross-Encoder Re-ranking
            re_ranked = self._reranker.re_rank(expanded_query, boosted)

            # 6. Clamp to top-K
            return re_ranked[:top_k]

        except (EmbeddingError, VectorStoreError) as e:
            logger.error("retrieval_failed", error=str(e))
            return []
        except Exception as e:
            logger.error("unexpected_retrieval_failure", error=str(e))
            return []

    def generate_answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
    ) -> str:
        """Delegates educational answer synthesis to the injected LLM provider strategy.

        Args:
            question: Original student question.
            contexts: Retrieved relevant context chunks.

        Returns:
            A formatted string containing answer synthesis and citations.
        """
        try:
            llm_provider = self._get_llm_provider()
            return llm_provider.generate(question, contexts)
        except Exception as e:
            logger.error("answer_generation_failed", error=str(e))
            # Graceful local template fallback
            from edumind.services.llm.template import TemplateLLMProvider

            fallback = TemplateLLMProvider()
            return fallback.generate(question, contexts)

    # ------------------------------------------------------------------
    # Maintenance Methods
    # ------------------------------------------------------------------
    def clear_index(self) -> bool:
        """Wipes the vector index collection completely."""
        try:
            vectorstore = self._get_vectorstore()
            return vectorstore.clear_index()
        except Exception as e:
            logger.error("clear_index_failed", error=str(e))
            return False

    def get_collection_info(self) -> dict:
        """Retrieves collection statistics."""
        try:
            vectorstore = self._get_vectorstore()
            return vectorstore.collection_info()
        except Exception as e:
            return {"status": "error", "error": str(e), "count": 0}

    @property
    def is_ready(self) -> bool:
        """Checks if RAG vector DB is fully functional."""
        try:
            vectorstore = self._get_vectorstore()
            return vectorstore.is_ready
        except Exception:
            return False

    @property
    def has_embedding_model(self) -> bool:
        """Checks if embedding model is ready."""
        try:
            emb = self._get_embedding_provider()
            return not isinstance(emb, MockEmbeddingProvider)
        except Exception:
            return False
