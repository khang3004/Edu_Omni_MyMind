"""EduMIND — Multimodal RAG Engine with Qdrant Vector Database.

This module handles Retrieval-Augmented Generation (RAG) for teaching documents:
  1. Performs Layout-Aware PDF Chunking.
  2. Embeds document chunks using Strategy-based embedding providers.
  3. Stores and queries vectors using Qdrant (in-memory or remote server).
  4. Applies keyword-boosting, query expansion, and cross-encoder re-ranking.
  5. Synthesizes contextual answers utilizing generative AI model providers.
"""

import json
from pathlib import Path
import re

import numpy as np

from edumind.config import get_settings
from edumind.core.exceptions import EmbeddingError, VectorStoreError
from edumind.core.logging import get_logger
from edumind.models.chunks import DocumentChunk, RetrievedChunk
from edumind.services.embedding.base import EmbeddingProvider
from edumind.services.graphstore.base import GraphStore
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

        # Instantiate a Time-To-Live (TTL) cache to bypass scoring repetitive queries/contexts
        try:
            from cachetools import TTLCache
            self._cache = TTLCache(maxsize=256, ttl=300)
        except ImportError:
            self._cache = {}

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

    def re_rank(self, query: str, chunks: list[RetrievedChunk], max_candidates: int = 10) -> list[RetrievedChunk]:
        """Re-scores retrieved passages relative to the query.

        Args:
            query: User search query.
            chunks: List of passages retrieved from vector search.
            max_candidates: Limit candidate passages to avoid high CPU latency bottleneck.

        Returns:
            Re-ordered list of chunks.
        """
        if not chunks:
            return chunks

        # Slice candidates to re-rank, leaving the rest untouched to preserve speed
        candidates = chunks[:max_candidates]
        remaining = chunks[max_candidates:]

        self.load()
        if self._model is None:
            return chunks

        try:
            uncached_indices = []
            uncached_pairs = []

            for idx, c in enumerate(candidates):
                cache_key = (query, c.text)
                if cache_key in self._cache:
                    c.score = self._cache[cache_key]
                else:
                    uncached_indices.append(idx)
                    uncached_pairs.append([query, c.text])

            # Predict in batch for cache misses
            if uncached_pairs:
                scores = self._model.predict(uncached_pairs)
                for score_idx, original_idx in enumerate(uncached_indices):
                    val = float(scores[score_idx])
                    # Sigmoid maps logit score to standard probability interval [0, 1]
                    sigmoid_score = 1.0 / (1.0 + np.exp(-val))
                    rounded_score = round(sigmoid_score, 4)

                    chunk = candidates[original_idx]
                    chunk.score = rounded_score
                    # Save to cache
                    self._cache[(query, chunk.text)] = rounded_score

            all_processed = candidates + remaining
            return sorted(all_processed, key=lambda x: x.score, reverse=True)
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
        graph_store: GraphStore | None = None,
        rerank_model_name: str | None = None,
    ):
        """Initializes the MultimodalRAG engine.

        Args:
            embedding_provider: Embedding model strategy. If None, resolves from container.
            vectorstore: Vector database strategy. If None, resolves from container.
            llm_provider: Answer generator strategy. If None, resolves from container.
            graph_store: Graph database strategy. If None, resolves from container.
            rerank_model_name: Cross-encoder re-ranking model path (optional).
        """
        self._embedding_provider = embedding_provider
        self._vectorstore = vectorstore
        self._llm_provider = llm_provider
        self._graph_store = graph_store

        # Setup cross-encoder re-ranker
        self._reranker = (
            CrossEncoderReRanker(rerank_model_name)
            if rerank_model_name
            else CrossEncoderReRanker()
        )
        self._doc_converter = None

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

    def _get_graph_store(self) -> GraphStore:
        """Retrieves the active graph store, resolving from container if needed."""
        if self._graph_store is None:
            from edumind.core.container import get_graph_store
            self._graph_store = get_graph_store()
        return self._graph_store

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
        return self.ingest_file(pdf_path)

    def ingest_file(self, file_path: str | Path) -> list[DocumentChunk]:
        """Parses any supported document format using IBM Docling, or routes audio files to Whisper.

        Supported formats include: PDF, DOCX, PPTX, XLSX, HTML, images, LaTeX, TXT, WAV, MP3, WebVTT, EML, MSG.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        suffix = file_path.suffix.lower()

        # 1. Routing Audio Files (WAV, MP3, M4A, FLAC, OGG) to ASR Note-Taker
        if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
            logger.info("routing_audio_file_to_asr_pipeline", file_name=file_path.name)
            try:
                from edumind.modules.speech_processor import CodeSwitchedASR
                asr = CodeSwitchedASR()
                transcript = asr.transcribe(file_path)
                corrected_text = asr.post_process(transcript.text)
                return self.ingest_text(
                    text=corrected_text,
                    source_name=file_path.name,
                    metadata_extra={"source_type": "🎙️ Transcript"}
                )
            except Exception as e:
                logger.error("audio_routing_failed", file_name=file_path.name, error=str(e))
                return []

        # 2. Document/Visual File Parsing using IBM Docling
        logger.info("ingesting_file_via_docling", file_name=file_path.name, format=suffix)
        try:
            if self._doc_converter is None:
                from docling.document_converter import DocumentConverter
                self._doc_converter = DocumentConverter()

            result = self._doc_converter.convert(str(file_path))
            doc = result.document

            chunks: list[DocumentChunk] = []
            current_section = "Untitled Section"

            for idx, (item, level) in enumerate(doc.iterate_items()):
                text_content = item.text if hasattr(item, "text") else ""
                if not text_content or not text_content.strip():
                    continue

                label_val = str(item.label) if hasattr(item, "label") else "paragraph"

                # Deduce page number if available
                page_num = 1
                if hasattr(item, "prov") and item.prov:
                    page_num = item.prov[0].page_no

                if "heading" in label_val.lower() or "header" in label_val.lower() or "title" in label_val.lower():
                    current_section = text_content.strip()

                # Split long elements cleanly
                for sub_text in split_long_text(text_content, max_size=500, overlap=50):
                    chunks.append(DocumentChunk(
                        text=sub_text,
                        metadata={
                            "page_number": page_num,
                            "source_file": file_path.name,
                            "section_header": current_section,
                            "chunk_index": idx,
                            "type": label_val,
                            "source_type": f"📄 {suffix[1:].upper()} Document"
                        },
                    ))

            logger.info("docling_ingestion_completed", file_name=file_path.name, chunks_count=len(chunks))

            # Explicit memory cleanup to prevent RAM OOM spikes on Apple Silicon / MPS
            import gc
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

            return chunks

        except ImportError:
            logger.error("docling_library_missing_falling_back_to_txt", reason="docling is not installed")
            # Graceful local fallback to plain text parsing if docling is missing
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
                return self.ingest_text(text_content, source_name=file_path.name)
            except Exception as e:
                logger.error("fallback_plain_text_read_failed", error=str(e))
                return []
        except Exception as e:
            logger.error("docling_ingestion_failed", file_name=file_path.name, error=str(e))
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

        def flush_paragraph(paragraph_lines: list[str], section: str, chunk_idx: int) -> int:
            if not paragraph_lines:
                return chunk_idx

            joined_text_parts = []
            for i, line_text in enumerate(paragraph_lines):
                line_str = line_text.strip()
                if not line_str:
                    continue

                # Handle physical line end hyphenation (e.g. multi-\nmodal)
                if line_str.endswith("-") and len(line_str) > 1:
                    joined_text_parts.append(line_str[:-1])
                elif joined_text_parts:
                    prev_line = paragraph_lines[i - 1].strip()
                    # If previous line did not end with standard sentence punctuation, merge with space
                    if prev_line and prev_line[-1] not in (".", "?", "!", ":", ";"):
                        joined_text_parts.append(" " + line_str)
                    else:
                        joined_text_parts.append("\n" + line_str)
                else:
                    joined_text_parts.append(line_str)

            para_text = "".join(joined_text_parts).strip()
            # Normalize redundant spaces
            para_text = re.sub(r"[ \t]+", " ", para_text)

            if para_text:
                for sub_chunk in split_long_text(para_text, max_chunk_size, overlap):
                    chunks.append(DocumentChunk(
                        text=sub_chunk,
                        metadata={
                            "page_number": page_number,
                            "source_file": source_file,
                            "section_header": section,
                            "chunk_index": chunk_idx,
                            "type": "paragraph",
                        },
                    ))
                    chunk_idx += 1
            return chunk_idx

        for line in lines:
            stripped = line.strip()

            if is_section_header(stripped):
                # Flush existing paragraph
                idx = flush_paragraph(current_paragraph, current_section, idx)
                current_paragraph = []
                current_section = stripped

            elif stripped == "":
                # Empty line boundary
                idx = flush_paragraph(current_paragraph, current_section, idx)
                current_paragraph = []
            else:
                current_paragraph.append(stripped)

        # Flush trailing content
        idx = flush_paragraph(current_paragraph, current_section, idx)
        return chunks

    # ------------------------------------------------------------------
    # Vector Indexing & Retrieval
    # ------------------------------------------------------------------
    def embed_and_store(self, chunks: list[DocumentChunk]) -> int:
        """Computes text embeddings, saves the chunks inside Qdrant, and extracts Knowledge Graph in Neo4j.

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

            # Build Neo4j knowledge graph asynchronously or sequentially
            try:
                self._extract_and_store_graph(chunks)
            except Exception as e:
                logger.error("graph_extraction_failed_skipping", error=str(e))

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
        """Queries the vector index applying semantic search, graph context, and hybrid re-ranking.

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

            # 4. Graph Context Retrieval
            graph_chunks = self._retrieve_graph_context(expanded_query)

            # 5. Merge and deduplicate candidates by chunk text similarity
            seen_texts = set()
            merged_chunks = []

            for chunk in retrieved + graph_chunks:
                norm_text = chunk.text.strip().lower()
                if norm_text not in seen_texts:
                    seen_texts.add(norm_text)
                    merged_chunks.append(chunk)

            # 6. Keyword Boosting (Lexical Hybrid component)
            boosted = self._apply_keyword_boost(expanded_query, merged_chunks)

            # 7. Optional Cross-Encoder Re-ranking
            re_ranked = self._reranker.re_rank(expanded_query, boosted)

            # 8. Clamp to top-K
            return re_ranked[:top_k]

        except (EmbeddingError, VectorStoreError) as e:
            logger.error("retrieval_failed", error=str(e))
            return []
        except Exception as e:
            logger.error("unexpected_retrieval_failure", error=str(e))
            return []

    def _extract_and_store_graph(self, chunks: list[DocumentChunk]) -> None:
        """Extracts concept entities and relationships from chunks using Gemini or regex fallback, then stores them."""
        graphstore = self._get_graph_store()
        if not graphstore.is_ready:
            return

        llm = self._get_llm_provider()
        # Check configuration
        has_gemini = hasattr(llm, "is_configured") and getattr(llm, "is_configured")

        logger.info("extracting_graph_entities_and_relations", chunks_count=len(chunks), use_gemini=has_gemini)

        for chunk in chunks:
            if not chunk.text or len(chunk.text.strip()) < 20:
                continue

            extracted = False
            if has_gemini:
                try:
                    # Construct extraction query
                    question_prompt = (
                        "Extract key academic concepts (concepts, terms, theories, definitions) and their semantic relationships from the text.\n"
                        "Format the output strictly as a JSON object (no markdown, no wrap blocks, no formatting other than pure JSON) with the following structure:\n"
                        "{\n"
                        "  \"entities\": [{\"name\": \"Concept Name\", \"type\": \"Concept\"}],\n"
                        "  \"relations\": [{\"source\": \"Concept A\", \"target\": \"Concept B\", \"type\": \"DEFINES\"}]\n"
                        "}\n"
                        "Keep types simple (e.g. Concept, Definition, Term, Process). Keep relation types simple (e.g. DEFINES, PART_OF, PREREQUISITE_FOR, RELATED_TO)."
                    )

                    # Call LLM generate using the chunk as context
                    llm_res = llm.generate(question_prompt, [RetrievedChunk(text=chunk.text, metadata={}, score=1.0)])

                    # Parse JSON
                    clean_res = re.sub(r"```json|```", "", llm_res).strip()
                    data = json.loads(clean_res)

                    entities = data.get("entities", [])
                    relations = data.get("relations", [])

                    source_file = chunk.metadata.get("source_file", "Unknown Document")

                    for ent in entities:
                        name = ent.get("name", "").strip()
                        ent_type = ent.get("type", "Concept").strip()
                        if name:
                            graphstore.upsert_entity(name, ent_type, {
                                "source_file": source_file,
                                "chunk_id": chunk.chunk_id,
                                "text": chunk.text[:200]
                            })

                    for rel in relations:
                        src = rel.get("source", "").strip()
                        tgt = rel.get("target", "").strip()
                        r_type = rel.get("type", "RELATED_TO").strip()
                        if src and tgt:
                            graphstore.upsert_relationship(src, tgt, r_type, {
                                "source_file": source_file,
                                "chunk_id": chunk.chunk_id
                            })

                    extracted = True
                except Exception as e:
                    logger.warning("gemini_graph_extraction_failed_falling_back", error=str(e))

            if not extracted:
                self._fallback_extract_and_store(chunk, graphstore)

    def _fallback_extract_and_store(self, chunk: DocumentChunk, graphstore: GraphStore) -> None:
        """Regex-based fallback entity/relationship extraction when Gemini is unavailable."""
        text = chunk.text
        # Find single/multi-word capitalized terms (potential concepts)
        candidates = re.findall(r"\b[A-Z][a-zA-Z0-9_]{1,15}(?:\s+[A-Z][a-zA-Z0-9_]{1,15})*\b", text)
        unique_candidates = list(set([c.strip() for c in candidates if len(c.strip()) > 3]))

        source_file = chunk.metadata.get("source_file", "Unknown Document")

        for entity in unique_candidates:
            graphstore.upsert_entity(entity, "Concept", {
                "source_file": source_file,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:200]
            })
            # Associate to section header if available
            section = chunk.metadata.get("section_header")
            if section and section != "Untitled Section":
                graphstore.upsert_entity(section, "Section", {"source_file": source_file})
                graphstore.upsert_relationship(entity, section, "PART_OF", {
                    "source_file": source_file,
                    "chunk_id": chunk.chunk_id
                })

    def _retrieve_graph_context(self, question: str) -> list[RetrievedChunk]:
        """Scans the query for known graph concepts and fetches their relations as supplementary chunks."""
        graphstore = self._get_graph_store()
        if not graphstore.is_ready:
            return []

        matched_entities = []
        try:
            # Check if Neo4j Server is active
            if hasattr(graphstore, "_driver") and graphstore._driver is not None:
                query_cypher = (
                    "MATCH (c:Concept) "
                    "WHERE toLower($question) CONTAINS toLower(c.name) "
                    "RETURN c.name as name"
                )
                with graphstore._driver.session() as session:
                    res = session.run(query_cypher, question=question)
                    matched_entities = [record["name"] for record in res]
            # Mock graph store fallback
            elif hasattr(graphstore, "_nodes"):
                matched_entities = [
                    name for name in graphstore._nodes
                    if name.lower() in question.lower()
                ]
        except Exception as e:
            logger.warning("graph_entity_matching_failed", error=str(e))

        if not matched_entities:
            return []

        logger.info("matched_graph_entities_for_query", count=len(matched_entities), entities=matched_entities)

        graph_chunks = []
        for entity in matched_entities:
            neighbors = graphstore.query_neighborhood(entity)
            for record in neighbors:
                src = record["source"]
                rel = record["relationship"]
                tgt = record["target"]

                # Format relational context
                relation_desc = f"Đồ thị tri thức (Knowledge Graph): Khái niệm '{src}' liên kết với '{tgt}' qua quan hệ '{rel}'."

                target_props = record.get("target_properties", {})
                if target_props.get("text"):
                    relation_desc += f" Chi tiết ngữ cảnh: {target_props['text']}"

                source_file = target_props.get("source_file", "Knowledge Graph")

                graph_chunks.append(RetrievedChunk(
                    text=relation_desc,
                    metadata={
                        "source_file": source_file,
                        "source_type": "🕸️ Đồ thị tri thức (Graph)",
                        "page_number": 1,
                        "section_header": f"Graph: {src} -> {tgt}"
                    },
                    score=0.85
                ))
        return graph_chunks

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
            from edumind.services.embedding.mock import MockEmbeddingProvider
            emb = self._get_embedding_provider()
            return not isinstance(emb, MockEmbeddingProvider)
        except Exception:
            return False
