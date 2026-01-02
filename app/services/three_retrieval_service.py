import fitz  # PyMuPDF
import httpx
import time
import uuid
import numpy as np
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.core.config import settings
import torch

# Optional: FAISS for efficient vector search. If unavailable, we fall back to pure NumPy.
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class RetrievalService:
    """Document ingestion + retrieval service using a local vector store.

    Key changes vs the original implementation:
    - Removed external Pinecone dependency
    - Uses an in-memory (or FAISS-based) index for vectors
    - Keeps a metadata store mapping chunk IDs to text + page
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size_words: int = 1400,
        chunk_overlap_words: int = 220,
        enable_lazy_cross_encoder: bool = True,
        crossencoder_threshold: float = 0.3,
    ) -> None:
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.crossencoder_threshold = crossencoder_threshold

        # Device selection for sentence-transformers / torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Optional cross-encoder reranker
        self.reranker_model_name = reranker_model_name
        self.cross_encoder: Optional[CrossEncoder] = None
        self.use_crossencoder = enable_lazy_cross_encoder
        if self.use_crossencoder:
            try:
                self.cross_encoder = CrossEncoder(self.reranker_model_name, device=self.device)
            except Exception as e:
                print(f"CRITICAL WARNING: could not load CrossEncoder '{self.reranker_model_name}': {e}")
                self.use_crossencoder = False

        # Text and metadata store
        self.text_chunks: List[str] = []
        self.metadata_store: Dict[int, Dict] = {}

        # Vector index: either FAISS index or simple NumPy matrix
        self.index = None  # type: ignore
        self.index_embeddings: Optional[np.ndarray] = None

    # ---------------------- Document Ingestion ----------------------
    def ingest_and_process_pdf(
        self,
        pdf_url: str,
        namespace: str,
        *,
        force_reingest: bool = False
    ) -> None:
        """Download a PDF, split into chunks, embed, and build a local vector index."""
        print(f"[Ingest] Starting ingestion for namespace='{namespace}' from url={pdf_url}")

        # 1. Download PDF
        resp = httpx.get(pdf_url, timeout=60)
        resp.raise_for_status()
        pdf_bytes = resp.content

        # 2. Extract text
        full_text = self._extract_text_from_pdf_bytes(pdf_bytes)
        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF.")

        # 3. Chunk text into overlapping windows
        self.text_chunks, self.metadata_store = self._split_text_into_chunks_with_metadata(
            full_text,
            chunk_size=self.chunk_size_words,
            overlap=self.chunk_overlap_words,
        )
        print(f"[Ingest] Split into {len(self.text_chunks)} chunks.")

        # 4. Embed all chunks
        print("[Ingest] Creating embeddings...")
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(self.text_chunks), batch_size):
            batch = self.text_chunks[i : i + batch_size]
            embs = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_embeddings.append(embs)
        if all_embeddings:
            self.index_embeddings = np.vstack(all_embeddings)
        else:
            self.index_embeddings = np.zeros((0, self.embedding_dim), dtype="float32")

        # 5. Build FAISS index (if available), otherwise rely on NumPy cosine similarity
        if self.index_embeddings is not None and self.index_embeddings.shape[0] > 0:
            if FAISS_AVAILABLE:
                print("[Ingest] Building FAISS index...")
                index = faiss.IndexFlatIP(self.embedding_dim)
                # FAISS expects float32
                index.add(self.index_embeddings.astype("float32"))
                self.index = index
            else:
                print("[Ingest] FAISS is not installed; using pure NumPy for retrieval.")
                self.index = None
        print("[Ingest] Completed vector index build.")

    # ---------------------- Search & Rerank ----------------------
    def search_and_rerank(self, query: str, top_k_retrieval: int = 20, top_n_rerank: int = 5) -> List[str]:
        """Retrieve relevant chunks for a query using the local index, with optional reranking."""
        if self.index_embeddings is None or self.index_embeddings.shape[0] == 0:
            raise RuntimeError("Document has not been ingested. Call ingest_and_process_pdf() first.")

        # Encode query
        q_vec = self.embedding_model.encode([query], show_progress_bar=False, normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")

        # Stage 1: vector search
        if self.index is not None and FAISS_AVAILABLE:
            D, I = self.index.search(q_vec, top_k_retrieval)
            indices = I[0]
        else:
            # Fallback: cosine similarity with NumPy
            def cosine(a, b):
                return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))

            scores = [cosine(q_vec[0], emb) for emb in self.index_embeddings]
            indices = np.argsort(scores)[::-1][:top_k_retrieval]

        candidate_chunks = [self.text_chunks[i] for i in indices if 0 <= i < len(self.text_chunks)]

        # Stage 2: optional reranking with cross-encoder
        return self._crossencoder_rerank(query, candidate_chunks, top_n_rerank=top_n_rerank)

    def build_global_candidate_pool(
        self,
        hypothetical_answers: List[str],
        pool_top_k: int = 200,
        group_size: int = 8,
    ) -> List[str]:
        """Build a global candidate pool over multiple hypothetical queries."""
        if self.index_embeddings is None or self.index_embeddings.shape[0] == 0:
            raise RuntimeError("Document has not been ingested.")

        if not hypothetical_answers:
            return []

        hypos_emb = np.array(
            self.embedding_model.encode(hypothetical_answers, show_progress_bar=False, normalize_embeddings=True),
            dtype="float32",
        )
        n = len(hypos_emb)
        group_size = max(1, min(group_size, n))

        seen = set()
        candidate_chunks: List[str] = []

        for i in range(0, n, group_size):
            grp = hypos_emb[i : i + group_size]
            centroid = np.mean(grp, axis=0, dtype="float32")
            centroid = centroid.reshape(1, -1)

            # Search nearest chunks for this centroid
            if self.index is not None and FAISS_AVAILABLE:
                D, I = self.index.search(centroid, pool_top_k)
                indices = I[0]
            else:
                def cosine(a, b):
                    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))
                scores = [cosine(centroid[0], emb) for emb in self.index_embeddings]
                indices = np.argsort(scores)[::-1][:pool_top_k]

            for idx in indices:
                if idx not in seen and 0 <= idx < len(self.text_chunks):
                    seen.add(int(idx))
                    candidate_chunks.append(self.text_chunks[idx])

        return candidate_chunks

    # ---------------------- Reranking Helper ----------------------
    def _crossencoder_rerank(
        self,
        query: str,
        candidate_chunks: List[str],
        top_n_rerank: int = 5,
    ) -> List[str]:
        """Local re-ranking of a shared candidate pool using CrossEncoder, with robust fallback."""
        if not candidate_chunks:
            return []

        # Default bi-encoder scoring logic, also used as a fallback
        def bi_encoder_rerank() -> List[str]:
            q_vec = np.array(self.embedding_model.encode([query])[0], dtype=float)
            chunk_embs = np.array(
                self.embedding_model.encode(candidate_chunks, show_progress_bar=False),
                dtype=float,
            )

            def cosine(a, b):
                return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))

            scores = [cosine(q_vec, emb) for emb in chunk_embs]
            scored = list(zip(scores, candidate_chunks))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:top_n_rerank]]

        if not self.use_crossencoder or not self.cross_encoder:
            return bi_encoder_rerank()

        try:
            pairs = [[query, chunk] for chunk in candidate_chunks]
            scores = self.cross_encoder.predict(pairs)
            reranked = list(zip(scores, candidate_chunks))
            reranked.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in reranked[:top_n_rerank]]
        except Exception as e:
            print(f"Warning: CrossEncoder predict failed: {e}. Falling back to bi-encoder.")
            return bi_encoder_rerank()

    # ---------------------- Helpers ----------------------
    def _extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)

    def _split_text_into_chunks_with_metadata(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ):
        if not text:
            return [], {}

        words = text.strip().split()
        chunks: List[str] = []
        metadata: Dict[int, Dict] = {}
        chunk_idx = 0

        i = 0
        while i < len(words):
            end = min(i + chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append(chunk_text)
                metadata[chunk_idx] = {
                    "text": chunk_text,
                    "page": None,
                    "chunk_id": chunk_idx,
                }
                chunk_idx += 1
            if end == len(words):
                break
            i += max(1, chunk_size - overlap)

        return chunks, metadata
