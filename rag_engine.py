"""Retrieval engine with duplicate-safe ingest and 2-stage retrieval."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Callable

import chromadb
import fitz
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
except Exception:
    ContextualCompressionRetriever = None
    FlashrankRerank = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

PROJECT_ROOT = Path(__file__).parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
EMBED_BATCH_SIZE = 30
EMBED_BATCH_DELAY = 1.6

K_IFRS_PDFS: list[tuple[str, str]] = [
    ("K-IFRS_제1038호_무형자산.pdf", "K-IFRS_1038"),
    ("K-IFRS_제1115호_수익.pdf", "K-IFRS_1115"),
    ("K-IFRS_제2032호_무형자산_손상사례_발췌.pdf", "K-IFRS_2032"),
]

KAM_PDFS: list[tuple[str, str]] = [
    ("삼일회계법인 제약 바이오 산업 KAM 및 유의사항.pdf", "KAM_2025"),
]


def _load_pdf(pdf_path: Path, source_id: str) -> list[Document]:
    doc = fitz.open(str(pdf_path))
    out: list[Document] = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            out.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_id,
                        "page": str(page_num + 1),
                        "file": pdf_path.name,
                        "source_url": "",
                    },
                )
            )
    doc.close()
    return out


def _chunk(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def _stable_doc_id(collection_name: str, doc: Document, idx: int) -> str:
    m = doc.metadata or {}
    company = str(m.get("company", ""))
    year = str(m.get("year", ""))
    section = str(m.get("section", ""))
    if company and year and section:
        return f"{company}_{year}_{section}_{idx}"
    body = f"{collection_name}|{idx}|{doc.page_content}"
    return f"{collection_name}_{hashlib.sha1(body.encode('utf-8')).hexdigest()[:16]}"


def save_to_chroma(
    collection_name: str,
    embeddings: OpenAIEmbeddings,
    docs: list[Document],
    id_builder: Callable[[Document, int], str] | None = None,
) -> None:
    if not docs:
        return
    store = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DIR)
    ids = [(id_builder(d, i) if id_builder else _stable_doc_id(collection_name, d, i)) for i, d in enumerate(docs)]
    try:
        store.delete(ids=ids)
    except Exception:
        pass
    try:
        store.add_documents(docs, ids=ids)
    except Exception as e:
        msg = str(e).lower()
        if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg:
            print(f"[복구] {collection_name} 인덱스 손상 감지. 컬렉션 재생성 후 재시도합니다.")
            _reset_collection(collection_name)
            store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )
            store.add_documents(docs, ids=ids)
        else:
            raise


def _get_collection_count(collection_name: str) -> int:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(collection_name).count()
    except Exception:
        return 0


def _reset_collection(collection_name: str) -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # 일부 환경에서 삭제 후에도 로컬 세그먼트 파일이 남는 경우가 있어 정리
    base = Path(CHROMA_DIR)
    for p in base.glob(f"**/*{collection_name}*"):
        try:
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def _load_all_docs(collection_name: str) -> list[Document]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(collection_name)
    payload = collection.get(include=["documents", "metadatas"])
    docs = payload.get("documents") or []
    metas = payload.get("metadatas") or []
    return [Document(page_content=t, metadata=m or {}) for t, m in zip(docs, metas)]


def _make_base_hybrid_retriever(collection_name: str, embeddings: OpenAIEmbeddings, bm25_docs: list[Document]) -> BaseRetriever:
    if not bm25_docs:
        raise ValueError(f"{collection_name} 컬렉션에 검색 가능한 문서가 없습니다.")
    store = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DIR)
    bm25 = BM25Retriever.from_documents(bm25_docs, k=50)
    semantic = store.as_retriever(search_kwargs={"k": 50})
    return EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.35, 0.65])


def _build_2stage(base: BaseRetriever, rerank_top_n: int = 10) -> BaseRetriever:
    if ContextualCompressionRetriever is None or FlashrankRerank is None:
        print("[경고] FlashRank 미설치: 1단계 하이브리드 검색만 사용")
        return base
    compressor = FlashrankRerank(top_n=rerank_top_n)
    return ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)


def _make_retriever(collection_name: str, embeddings: OpenAIEmbeddings, docs_to_ingest: list[Document] | None = None) -> BaseRetriever:
    if docs_to_ingest:
        for i in range(0, len(docs_to_ingest), EMBED_BATCH_SIZE):
            save_to_chroma(collection_name, embeddings, docs_to_ingest[i : i + EMBED_BATCH_SIZE])
            if i + EMBED_BATCH_SIZE < len(docs_to_ingest):
                time.sleep(EMBED_BATCH_DELAY)
        bm25_docs = docs_to_ingest
    else:
        try:
            bm25_docs = _load_all_docs(collection_name)
        except Exception as e:
            msg = str(e).lower()
            if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg:
                print(f"[복구] {collection_name} 조회 중 인덱스 손상. 컬렉션 초기화합니다.")
                _reset_collection(collection_name)
                bm25_docs = []
            else:
                raise

    # 컬렉션이 비어 있으면 retriever 생성 불가이므로 ingest 필요를 명확히 반환하기 위해
    # 빈 BM25 소스일 때는 최소 placeholder를 두지 않고 상위 함수에서 재ingest하도록 유도
    base = _make_base_hybrid_retriever(collection_name, embeddings, bm25_docs if bm25_docs else [])
    return _build_2stage(base, rerank_top_n=10)


def get_kifrs_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever:
    collection = "kifrs"
    count = _get_collection_count(collection)
    if count == 0:
        raw: list[Document] = []
        for filename, source_id in K_IFRS_PDFS:
            p = PROJECT_ROOT / filename
            if p.exists():
                raw.extend(_load_pdf(p, source_id))
        _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
    try:
        return _make_retriever(collection, embeddings)
    except Exception as e:
        msg = str(e).lower()
        if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg:
            print("[복구] kifrs 리트리버 생성 실패 -> 컬렉션 재생성")
            _reset_collection(collection)
            raw: list[Document] = []
            for filename, source_id in K_IFRS_PDFS:
                p = PROJECT_ROOT / filename
                if p.exists():
                    raw.extend(_load_pdf(p, source_id))
            _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
            return _make_retriever(collection, embeddings)
        raise


def get_kam_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever:
    collection = "kam"
    count = _get_collection_count(collection)
    if count == 0:
        raw: list[Document] = []
        for filename, source_id in KAM_PDFS:
            p = PROJECT_ROOT / filename
            if p.exists():
                raw.extend(_load_pdf(p, source_id))
        _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
    try:
        return _make_retriever(collection, embeddings)
    except Exception as e:
        msg = str(e).lower()
        if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg:
            print("[복구] kam 리트리버 생성 실패 -> 컬렉션 재생성")
            _reset_collection(collection)
            raw: list[Document] = []
            for filename, source_id in KAM_PDFS:
                p = PROJECT_ROOT / filename
                if p.exists():
                    raw.extend(_load_pdf(p, source_id))
            _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
            return _make_retriever(collection, embeddings)
        raise


def get_dart_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever | None:
    collection = "dart"
    if _get_collection_count(collection) == 0:
        return None
    return _make_retriever(collection, embeddings)
