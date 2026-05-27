"""ChromaDB 초기화, Hybrid Search, 컬렉션 관리."""

from __future__ import annotations

import time
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI 무료 티어 TPM(40000) 기준: 청크 200자 × 30개 ≈ 6000 토큰/배치
_EMBED_BATCH_SIZE = 30
_EMBED_BATCH_DELAY = 2.0  # 초

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")

# K-IFRS PDF 목록: (파일명, source_id)
K_IFRS_PDFS: list[tuple[str, str]] = [
    ("K-IFRS_제1038호_무형자산.pdf", "K-IFRS_1038"),
    ("K-IFRS_제1115호_수익.pdf", "K-IFRS_1115"),
    ("K-IFRS_제2032호_무형자산_웹사이트 원가.pdf", "K-IFRS_2032"),
]

KAM_PDFS: list[tuple[str, str]] = [
    ("삼일회계법인 제약 바이오 산업 KAM 및 대응방안.pdf", "KAM_2025"),
]


# ---------------------------------------------------------------------------
# PDF 로딩 & 청킹
# ---------------------------------------------------------------------------

def _load_pdf(pdf_path: Path, source_id: str) -> list[Document]:
    """PyMuPDF로 PDF를 페이지 단위 Document 리스트로 변환."""
    doc = fitz.open(str(pdf_path))
    documents: list[Document] = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": source_id,
                    "page": page_num + 1,
                    "file": pdf_path.name,
                },
            ))
    doc.close()
    return documents


def _chunk(docs: list[Document]) -> list[Document]:
    """회계 문서 특성에 맞는 청킹: 문단 경계 우선, 300자 기준 (TPM 한도 대응)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# ChromaDB 헬퍼
# ---------------------------------------------------------------------------

def _get_collection_count(collection_name: str) -> int:
    """ChromaDB 컬렉션의 문서 수를 반환. 컬렉션이 없으면 0."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        col = client.get_collection(collection_name)
        return col.count()
    except Exception:
        return 0


def _load_all_from_chroma(collection_name: str) -> list[Document]:
    """BM25 재구성을 위해 ChromaDB에서 모든 청크를 Document로 복원."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(collection_name)
    results = col.get(include=["documents", "metadatas"])
    raw_docs: list[str] = results.get("documents") or []
    raw_metas: list[dict[str, str]] = results.get("metadatas") or []
    return [
        Document(
            page_content=text,
            metadata=meta if meta is not None else {},
        )
        for text, meta in zip(raw_docs, raw_metas)
    ]


# ---------------------------------------------------------------------------
# Retriever 생성
# ---------------------------------------------------------------------------

def _make_retriever(
    collection_name: str,
    embeddings: OpenAIEmbeddings,
    docs_to_ingest: list[Document] | None = None,
) -> EnsembleRetriever:
    """
    ChromaDB vectorstore + BM25 를 결합한 EnsembleRetriever 생성.
    docs_to_ingest 가 있으면 ChromaDB에 추가 후 해당 청크로 BM25 구성.
    없으면 기존 ChromaDB 데이터에서 BM25 재구성.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    if docs_to_ingest:
        # 배치 단위로 임베딩 API 호출 → 무료 티어 TPM 한도(40000) 준수
        for i in range(0, len(docs_to_ingest), _EMBED_BATCH_SIZE):
            batch = docs_to_ingest[i : i + _EMBED_BATCH_SIZE]
            vectorstore.add_documents(batch)
            if i + _EMBED_BATCH_SIZE < len(docs_to_ingest):
                time.sleep(_EMBED_BATCH_DELAY)
        bm25_source = docs_to_ingest
    else:
        bm25_source = _load_all_from_chroma(collection_name)

    bm25 = BM25Retriever.from_documents(bm25_source, k=4)
    semantic = vectorstore.as_retriever(search_kwargs={"k": 4})
    return EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.3, 0.7])


def get_kifrs_retriever(embeddings: OpenAIEmbeddings) -> EnsembleRetriever:
    """K-IFRS 컬렉션 retriever. 데이터 없으면 PDF 인제스트 후 반환."""
    collection = "kifrs"
    if _get_collection_count(collection) > 0:
        return _make_retriever(collection, embeddings)

    all_docs: list[Document] = []
    for filename, source_id in K_IFRS_PDFS:
        pdf_path = PROJECT_ROOT / filename
        if pdf_path.exists():
            all_docs.extend(_load_pdf(pdf_path, source_id))
        else:
            print(f"[경고] K-IFRS PDF 없음: {pdf_path}")

    chunks = _chunk(all_docs)
    return _make_retriever(collection, embeddings, docs_to_ingest=chunks)


def get_kam_retriever(embeddings: OpenAIEmbeddings) -> EnsembleRetriever:
    """KAM 컬렉션 retriever. 데이터 없으면 PDF 인제스트 후 반환."""
    collection = "kam"
    if _get_collection_count(collection) > 0:
        return _make_retriever(collection, embeddings)

    all_docs: list[Document] = []
    for filename, source_id in KAM_PDFS:
        pdf_path = PROJECT_ROOT / filename
        if pdf_path.exists():
            all_docs.extend(_load_pdf(pdf_path, source_id))
        else:
            print(f"[경고] KAM PDF 없음: {pdf_path}")

    chunks = _chunk(all_docs)
    return _make_retriever(collection, embeddings, docs_to_ingest=chunks)


def get_dart_retriever(embeddings: OpenAIEmbeddings) -> EnsembleRetriever | None:
    """DART 컬렉션 retriever. dart_ingest.py 실행 전이면 None 반환."""
    collection = "dart"
    if _get_collection_count(collection) == 0:
        return None
    return _make_retriever(collection, embeddings)
