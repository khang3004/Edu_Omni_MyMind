"""
EduMIND — Multimodal RAG Engine với Qdrant Vector Database
==========================================================
Module xử lý Retrieval-Augmented Generation (RAG) cho tài liệu giảng dạy:
  1. Phân tích bố cục PDF (Layout-Aware Chunking)
  2. Embedding chunks bằng Sentence-Transformers
  3. Lưu trữ & truy vấn vector bằng Qdrant (in-memory hoặc server)
  4. Tổng hợp câu trả lời với trích dẫn nguồn

Kiến trúc:
    MultimodalRAG
    ├── ingest_pdf(path) → list[DocumentChunk]
    ├── ingest_text(text, source) → list[DocumentChunk]
    ├── embed_and_store(chunks)
    ├── query(question, top_k) → list[RetrievedChunk]
    ├── generate_answer(question, contexts) → str
    └── clear_index()

Sử dụng:
    rag = MultimodalRAG()
    chunks = rag.ingest_pdf("lecture_slides.pdf")
    rag.embed_and_store(chunks)
    results = rag.query("Attention mechanism hoạt động thế nào?")
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


# ──────────────────────────────────────────────────────────────────────
# Data Classes cho document chunks và kết quả truy vấn
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DocumentChunk:
    """Một đoạn tài liệu đã được chia nhỏ với metadata."""
    text: str                                       # Nội dung văn bản
    metadata: dict = field(default_factory=dict)    # Metadata (page, source, section, ...)
    chunk_id: str = ""                               # ID duy nhất cho chunk

    def __post_init__(self):
        """Tự sinh chunk_id nếu chưa có."""
        if not self.chunk_id:
            content_hash = hashlib.md5(self.text.encode()).hexdigest()[:8]
            self.chunk_id = f"chunk_{content_hash}_{uuid.uuid4().hex[:6]}"


@dataclass
class RetrievedChunk:
    """Kết quả truy vấn vector search — chunk kèm điểm tương đồng."""
    text: str                                       # Nội dung văn bản
    metadata: dict = field(default_factory=dict)    # Metadata gốc
    score: float = 0.0                              # Điểm cosine similarity (0.0 → 1.0)


# ──────────────────────────────────────────────────────────────────────
# Lớp chính: MultimodalRAG
# ──────────────────────────────────────────────────────────────────────
class MultimodalRAG:
    """
    Engine RAG đa phương thức sử dụng Qdrant làm vector database.

    Tính năng:
        - Phân tích PDF với layout-aware chunking (nhận diện header/paragraph)
        - Embedding văn bản bằng Sentence-Transformers (mặc định: all-MiniLM-L6-v2)
        - Lưu trữ vector trong Qdrant (in-memory hoặc server mode)
        - Truy vấn semantic search top-k với cosine similarity
        - Tổng hợp câu trả lời template-based với trích dẫn

    Args:
        embedding_model_name: Tên mô hình Sentence-Transformers cho embedding.
        qdrant_mode: "memory" (in-process) hoặc "server" (remote).
        qdrant_host: Host của Qdrant server (chỉ khi mode="server").
        qdrant_port: Port của Qdrant server (chỉ khi mode="server").
        qdrant_api_key: API key cho Qdrant Cloud (tùy chọn).
        collection_name: Tên collection trong Qdrant.
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
        # Đọc config mặc định từ settings
        from src.config import settings

        self._embedding_model_name = embedding_model_name or settings.EMBEDDING_MODEL
        self._qdrant_mode = qdrant_mode or settings.QDRANT_MODE
        self._qdrant_host = qdrant_host or settings.QDRANT_HOST
        self._qdrant_port = qdrant_port or settings.QDRANT_PORT
        self._qdrant_api_key = qdrant_api_key or settings.QDRANT_API_KEY
        self._collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

        # Trạng thái nội bộ
        self._embedding_model = None
        self._qdrant_client = None
        self._embedding_dim: int = 384  # Mặc định cho all-MiniLM-L6-v2
        self._is_ready = False

        # Khởi tạo các thành phần
        self._load_embedding_model()
        self._init_qdrant()

    # ──────────────────────────────────────────────────────────────────
    # Khởi tạo mô hình embedding và Qdrant
    # ──────────────────────────────────────────────────────────────────
    def _load_embedding_model(self) -> None:
        """
        Tải mô hình Sentence-Transformers cho embedding.
        Fallback: sử dụng vector ngẫu nhiên nếu tải thất bại.
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"📐 Đang tải mô hình embedding: {self._embedding_model_name}...")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)

            # Lấy dimension thực tế từ mô hình
            test_emb = self._embedding_model.encode(["test"])
            self._embedding_dim = test_emb.shape[1]

            logger.success(
                f"✅ Đã tải embedding model! Dimension: {self._embedding_dim}"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Không thể tải embedding model: {e}. "
                "Sẽ sử dụng mock embedding (random vectors)."
            )
            self._embedding_model = None

    def _init_qdrant(self) -> None:
        """
        Khởi tạo Qdrant client và tạo collection nếu chưa có.

        Hỗ trợ hai chế độ:
            - "memory": Chạy in-process, không cần server (lý tưởng cho demo/dev)
            - "server": Kết nối tới Qdrant server/cloud (cho production)
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            # Tạo client theo mode
            if self._qdrant_mode == "memory":
                logger.info("🗄️ Khởi tạo Qdrant in-memory mode...")
                self._qdrant_client = QdrantClient(":memory:")
            else:
                logger.info(
                    f"🗄️ Kết nối Qdrant server: {self._qdrant_host}:{self._qdrant_port}..."
                )
                self._qdrant_client = QdrantClient(
                    host=self._qdrant_host,
                    port=self._qdrant_port,
                    api_key=self._qdrant_api_key if self._qdrant_api_key else None,
                )

            # Tạo collection với cấu hình vector
            # Kiểm tra collection đã tồn tại chưa
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
                    f"✅ Đã tạo Qdrant collection: '{self._collection_name}' "
                    f"(dim={self._embedding_dim}, distance=Cosine)"
                )
            else:
                logger.info(f"📦 Collection '{self._collection_name}' đã tồn tại.")

            self._is_ready = True

        except ImportError:
            logger.warning(
                "⚠️ Thư viện 'qdrant-client' chưa được cài đặt. "
                "RAG engine sẽ không hoạt động."
            )
            self._is_ready = False
        except Exception as e:
            logger.warning(f"⚠️ Không thể khởi tạo Qdrant: {e}")
            self._is_ready = False

    # ──────────────────────────────────────────────────────────────────
    # Ingestion: Phân tích và chia nhỏ tài liệu
    # ──────────────────────────────────────────────────────────────────
    def ingest_pdf(self, pdf_path: str | Path) -> list[DocumentChunk]:
        """
        Phân tích PDF thành các chunks với layout-aware chunking.

        Thuật toán Layout-Aware Chunking:
            1. Đọc từng trang PDF bằng pypdf
            2. Nhận diện headers (dòng ALL CAPS, dòng kết thúc bằng ':', dòng ngắn đậm)
            3. Tách paragraphs theo dòng trống
            4. Gán metadata: page_number, source_file, section_header, chunk_index

        Args:
            pdf_path: Đường dẫn tới file PDF.

        Returns:
            Danh sách DocumentChunk đã được chia nhỏ.

        Raises:
            FileNotFoundError: Nếu file PDF không tồn tại.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File PDF không tồn tại: {pdf_path}")

        logger.info(f"📄 Đang phân tích PDF: {pdf_path.name}...")

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            chunks: list[DocumentChunk] = []
            chunk_index = 0

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                # Chia trang thành các đoạn theo layout
                page_chunks = self._layout_aware_split(
                    text=page_text,
                    page_number=page_num,
                    source_file=pdf_path.name,
                    start_chunk_index=chunk_index,
                )
                chunks.extend(page_chunks)
                chunk_index += len(page_chunks)

            logger.success(
                f"✅ PDF '{pdf_path.name}': {len(reader.pages)} trang → {len(chunks)} chunks"
            )
            return chunks

        except ImportError:
            logger.error("❌ Thư viện 'pypdf' chưa được cài đặt!")
            return []
        except Exception as e:
            logger.error(f"❌ Lỗi phân tích PDF: {e}")
            return []

    def ingest_text(
        self,
        text: str,
        source_name: str = "text_input",
        metadata_extra: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chia nhỏ văn bản thuần thành các chunks.

        Args:
            text: Văn bản cần chia nhỏ.
            source_name: Tên nguồn (để hiển thị trong citation).
            metadata_extra: Metadata bổ sung (tùy chọn).

        Returns:
            Danh sách DocumentChunk.
        """
        if not text or not text.strip():
            return []

        chunks = self._layout_aware_split(
            text=text,
            page_number=1,
            source_file=source_name,
            start_chunk_index=0,
        )

        # Gán metadata bổ sung nếu có
        if metadata_extra:
            for chunk in chunks:
                chunk.metadata.update(metadata_extra)

        logger.info(f"📝 Text '{source_name}': → {len(chunks)} chunks")
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
        """
        Chia văn bản thành chunks theo layout (nhận diện header/paragraph).

        Thuật toán:
            1. Tách văn bản thành các dòng
            2. Gom dòng thành paragraphs (tách bởi dòng trống)
            3. Nhận diện dòng header → tạo section boundary
            4. Nếu paragraph quá dài → chia thêm theo max_chunk_size với overlap

        Args:
            text: Nội dung văn bản của một trang.
            page_number: Số trang.
            source_file: Tên file nguồn.
            start_chunk_index: Index bắt đầu cho chunks.
            max_chunk_size: Kích thước tối đa mỗi chunk (ký tự).
            overlap: Số ký tự overlap giữa các chunks liền kề.

        Returns:
            Danh sách DocumentChunk.
        """
        lines = text.split("\n")
        chunks: list[DocumentChunk] = []
        current_section = "Untitled Section"
        current_paragraph: list[str] = []
        idx = start_chunk_index

        for line in lines:
            stripped = line.strip()

            # Nhận diện header
            if self._is_header(stripped):
                # Lưu paragraph hiện tại (nếu có) trước khi đổi section
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

                # Cập nhật section header
                current_section = stripped

            elif stripped == "":
                # Dòng trống → kết thúc paragraph
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
                # Dòng bình thường → gom vào paragraph hiện tại
                current_paragraph.append(stripped)

        # Xử lý paragraph cuối cùng
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
        """
        Nhận diện dòng là header hay không.

        Tiêu chí:
            1. Dòng viết hoa toàn bộ (ALL CAPS) và đủ ngắn (< 100 ký tự)
            2. Dòng kết thúc bằng dấu ':'
            3. Dòng bắt đầu bằng số thứ tự (VD: "1.", "1.1", "Chapter 1")
            4. Dòng ngắn (< 60 ký tự) và không kết thúc bằng dấu câu

        Args:
            line: Dòng văn bản cần kiểm tra.

        Returns:
            True nếu dòng là header.
        """
        if not line or len(line) < 2:
            return False

        # ALL CAPS (chỉ chữ cái) và đủ ngắn
        alpha_chars = [c for c in line if c.isalpha()]
        if alpha_chars and all(c.isupper() for c in alpha_chars) and len(line) < 100:
            return True

        # Kết thúc bằng ':'
        if line.endswith(":") and len(line) < 100:
            return True

        # Bắt đầu bằng số thứ tự: "1.", "1.1", "Chapter", "Chương"
        if re.match(r"^(\d+\.)+\s", line) or re.match(r"^(Chapter|Chương|Bài|Phần)\s", line, re.IGNORECASE):
            return True

        return False

    @staticmethod
    def _split_long_text(text: str, max_size: int, overlap: int) -> list[str]:
        """
        Chia văn bản dài thành các đoạn nhỏ hơn max_size với overlap.

        Args:
            text: Văn bản cần chia.
            max_size: Kích thước tối đa mỗi đoạn.
            overlap: Số ký tự overlap giữa các đoạn.

        Returns:
            Danh sách các đoạn văn bản.
        """
        if len(text) <= max_size:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(text):
            end = start + max_size

            # Cố gắng tách tại vị trí dấu câu hoặc khoảng trắng
            if end < len(text):
                # Tìm vị trí tách tốt nhất (dấu chấm, dấu phẩy, khoảng trắng)
                best_break = text.rfind(". ", start, end)
                if best_break == -1 or best_break <= start:
                    best_break = text.rfind(" ", start, end)
                if best_break > start:
                    end = best_break + 1

            parts.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end

        return [p for p in parts if p]

    # ──────────────────────────────────────────────────────────────────
    # Embedding & Lưu trữ vào Qdrant
    # ──────────────────────────────────────────────────────────────────
    def embed_and_store(self, chunks: list[DocumentChunk]) -> int:
        """
        Embed danh sách chunks và lưu vào Qdrant vector database.

        Args:
            chunks: Danh sách DocumentChunk cần lưu.

        Returns:
            Số lượng chunks đã được lưu thành công.
        """
        if not chunks:
            logger.warning("⚠️ Danh sách chunks rỗng, không có gì để lưu.")
            return 0

        if not self._is_ready or self._qdrant_client is None:
            logger.error("❌ Qdrant chưa sẵn sàng. Không thể lưu chunks.")
            return 0

        logger.info(f"📐 Đang embed {len(chunks)} chunks...")

        # Embed tất cả texts
        texts = [c.text for c in chunks]
        embeddings = self._encode_texts(texts)

        if embeddings is None:
            logger.error("❌ Embedding thất bại.")
            return 0

        # Chuẩn bị points cho Qdrant
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

        # Upsert vào Qdrant (batch)
        try:
            self._qdrant_client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            logger.success(f"✅ Đã lưu {len(points)} chunks vào Qdrant!")
            return len(points)
        except Exception as e:
            logger.error(f"❌ Lỗi lưu vào Qdrant: {e}")
            return 0

    def _encode_texts(self, texts: list[str]):
        """
        Encode danh sách texts thành embeddings.

        Sử dụng Sentence-Transformers nếu có, hoặc random vectors (mock).

        Args:
            texts: Danh sách văn bản cần embed.

        Returns:
            numpy.ndarray shape (len(texts), embedding_dim) hoặc None nếu lỗi.
        """
        if self._embedding_model is not None:
            try:
                return self._embedding_model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=32,
                )
            except Exception as e:
                logger.warning(f"⚠️ Embedding model error: {e}. Using mock embeddings.")

        # Mock: random embeddings
        import numpy as np
        logger.info("🎭 Sử dụng mock embeddings (random vectors).")
        return np.random.randn(len(texts), self._embedding_dim).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Truy vấn: Semantic Search + Answer Generation
    # ──────────────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Tìm kiếm semantic trong Qdrant và trả về top-k chunks liên quan nhất.

        Args:
            question: Câu hỏi / truy vấn tìm kiếm.
            top_k: Số lượng kết quả trả về.

        Returns:
            Danh sách RetrievedChunk sắp xếp theo điểm tương đồng giảm dần.
        """
        if not self._is_ready or self._qdrant_client is None:
            logger.error("❌ Qdrant chưa sẵn sàng. Không thể truy vấn.")
            return []

        if not question or not question.strip():
            return []

        logger.info(f"🔍 Đang tìm kiếm: '{question[:50]}...'")

        # Embed câu hỏi
        query_embedding = self._encode_texts([question])
        if query_embedding is None:
            return []

        # Tìm kiếm trong Qdrant
        try:
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

            logger.success(f"✅ Tìm thấy {len(retrieved)} kết quả!")
            return retrieved

        except Exception as e:
            logger.error(f"❌ Lỗi truy vấn Qdrant: {e}")
            return []

    def generate_answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
    ) -> str:
        """
        Tổng hợp câu trả lời từ các chunks đã truy xuất.

        Sử dụng template-based synthesis với trích dẫn nguồn [Trang X, Nguồn Y].

        Args:
            question: Câu hỏi gốc.
            contexts: Danh sách RetrievedChunk từ truy vấn.

        Returns:
            Câu trả lời tổng hợp với citations.
        """
        if not contexts:
            return (
                "❌ Không tìm thấy thông tin liên quan trong tài liệu đã tải lên. "
                "Hãy thử đặt câu hỏi khác hoặc tải thêm tài liệu."
            )

        # Xây dựng câu trả lời template-based
        answer_parts: list[str] = []
        answer_parts.append(f"📋 **Câu hỏi:** {question}\n")
        answer_parts.append("📖 **Câu trả lời tổng hợp từ tài liệu:**\n")

        # Tổng hợp nội dung từ top chunks
        for i, ctx in enumerate(contexts[:3], start=1):
            page = ctx.metadata.get("page_number", "?")
            source = ctx.metadata.get("source_file", "Unknown")
            section = ctx.metadata.get("section_header", "")

            citation = f"[Trang {page}, {source}]"
            if section and section != "Untitled Section":
                citation = f"[Trang {page}, {source} — §{section}]"

            # Trích dẫn nội dung liên quan
            text_preview = ctx.text[:300] + "..." if len(ctx.text) > 300 else ctx.text
            answer_parts.append(
                f"**{i}.** {text_preview}\n   — *{citation}* (Relevance: {ctx.score:.2f})\n"
            )

        # Tổng kết
        answer_parts.append(
            "\n💡 *Lưu ý: Câu trả lời được tổng hợp từ các đoạn tài liệu liên quan nhất. "
            "Vui lòng kiểm tra trích dẫn gốc để xác minh thông tin.*"
        )

        return "\n".join(answer_parts)

    # ──────────────────────────────────────────────────────────────────
    # Tiện ích: Xóa index, kiểm tra trạng thái
    # ──────────────────────────────────────────────────────────────────
    def clear_index(self) -> bool:
        """
        Xóa toàn bộ dữ liệu trong collection Qdrant và tạo lại.

        Returns:
            True nếu thành công, False nếu thất bại.
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
            logger.info(f"🗑️ Đã xóa và tạo lại collection '{self._collection_name}'")
            return True
        except Exception as e:
            logger.error(f"❌ Lỗi xóa collection: {e}")
            return False

    def get_collection_info(self) -> dict:
        """
        Lấy thông tin về collection hiện tại trong Qdrant.

        Returns:
            Dict chứa số lượng vectors, tên collection, v.v.
        """
        if not self._is_ready or self._qdrant_client is None:
            return {"status": "not_ready", "count": 0}

        try:
            info = self._qdrant_client.get_collection(self._collection_name)
            return {
                "status": "ready",
                "collection_name": self._collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "count": 0}

    @property
    def is_ready(self) -> bool:
        """Kiểm tra RAG engine đã sẵn sàng chưa."""
        return self._is_ready

    @property
    def has_embedding_model(self) -> bool:
        """Kiểm tra mô hình embedding đã tải chưa."""
        return self._embedding_model is not None
