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
    """컬렉션 문서 개수 반환. HNSW 에러 발생 시 exception 그대로 throw."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(collection_name).count()


def _collection_exists(collection_name: str) -> bool:
    """컬렉션 존재 여부만 확인 (문서 개수 세지 않음)."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


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


from typing import Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun

TARGET_COMPANIES_LIST = [
    "삼성바이오로직스", "셀트리온", "한미약품", "유한양행", "종근당", "SK바이오팜", 
    "SK바이오사이언스", "알테오젠", "리가켐바이오", "휴젤", "GC녹십자", "대웅제약", 
    "보령", "HK이노엔", "동아에스티", "한올바이오파마", "JW중외제약", "동국제약", 
    "삼천당제약", "메디톡스", "에스티팜", "차바이오텍", "대원제약", "부광약품", 
    "한국유나이티드제약", "한독", "안국약품", "삼진제약", "제일약품", "일동제약", 
    "신풍제약", "오스코텍", "레고켐바이오", "에스케이바이오팜", "에스케이바이오사이언스", 
    "녹십자", "보령제약", "유나이티드제약"
]

class DynamicDartRetriever(BaseRetriever):
    store: Any
    global_bm25_docs: list
    global_bm25_retriever: Any
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        found_companies = [comp for comp in TARGET_COMPANIES_LIST if comp in query]
        
        if found_companies:
            alias_map = {
                "레고켐바이오": "리가켐바이오",
                "에스케이바이오팜": "SK바이오팜",
                "에스케이바이오사이언스": "SK바이오사이언스",
                "녹십자": "GC녹십자",
                "보령제약": "보령",
                "유나이티드제약": "한국유나이티드제약"
            }
            target_company = alias_map.get(found_companies[0], found_companies[0])
            print(f"[DynamicDartRetriever] 질의에서 기업명 '{target_company}' 감지. 단일 기업 필터링 수행.")
            
            search_kwargs = {"k": 30, "filter": {"company": target_company}}
            semantic_docs = self.store.similarity_search(query, **search_kwargs)
            
            filtered_bm25_docs = [d for d in self.global_bm25_docs if d.metadata.get("company") == target_company]
            if filtered_bm25_docs:
                from langchain_community.retrievers import BM25Retriever
                enriched = []
                for d in filtered_bm25_docs:
                    company = d.metadata.get("company", "")
                    year = d.metadata.get("year", "")
                    section = d.metadata.get("section", "")
                    prefix = f"[{company} {year}년 {section}] " if company else ""
                    enriched.append(Document(page_content=prefix + (d.page_content or ""), metadata=d.metadata))
                bm25 = BM25Retriever.from_documents(enriched, k=30)
                bm25_docs = bm25.invoke(query)
                combined = self._rrf([semantic_docs, bm25_docs])
            else:
                combined = semantic_docs
        else:
            print("[DynamicDartRetriever] 질의에서 기업명 미감지. 전역 검색 수행.")
            search_kwargs = {"k": 30}
            semantic_docs = self.store.similarity_search(query, **search_kwargs)
            if self.global_bm25_retriever:
                bm25_docs = self.global_bm25_retriever.invoke(query)
                combined = self._rrf([semantic_docs, bm25_docs])
            else:
                combined = semantic_docs
                
        return combined[:20]

    def _rrf(self, doc_lists: list[list[Document]], k: int = 60) -> list[Document]:
        rrf_score = {}
        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                content = doc.page_content
                score = 1.0 / (rank + k)
                if content in rrf_score:
                    rrf_score[content]["score"] += score
                else:
                    rrf_score[content] = {"score": score, "doc": doc}
        sorted_docs = sorted(rrf_score.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

def _make_base_hybrid_retriever(collection_name: str, embeddings: OpenAIEmbeddings, bm25_docs: list[Document]) -> BaseRetriever:
    if not bm25_docs:
        raise ValueError(f"{collection_name} 컬렉션에 검색 가능한 문서가 없습니다.")
    store = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=CHROMA_DIR)
    
    # BM25의 키워드 매칭(예: 기업명)을 강화하기 위해 메타데이터를 본문에 추가
    enriched_bm25_docs = []
    for d in bm25_docs:
        company = d.metadata.get("company", "")
        year = d.metadata.get("year", "")
        section = d.metadata.get("section", "")
        prefix = f"[{company} {year}년 {section}] " if company else ""
        enriched_bm25_docs.append(Document(page_content=prefix + (d.page_content or ""), metadata=d.metadata))

    bm25 = BM25Retriever.from_documents(enriched_bm25_docs, k=50)
    semantic = store.as_retriever(search_kwargs={"k": 50})
    return EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.5, 0.5])


def _build_2stage(base: BaseRetriever, rerank_top_n: int = 15) -> BaseRetriever:
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
    
    # 컬렉션 없으면 PDF에서 초기화
    if not _collection_exists(collection):
        print("[info] KIFRS 컬렉션 초기화 중...")
        raw: list[Document] = []
        for filename, source_id in K_IFRS_PDFS:
            p = PROJECT_ROOT / filename
            if p.exists():
                raw.extend(_load_pdf(p, source_id))
        _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
        print(f"[✓] KIFRS 초기화 완료: {len(raw)}개 문서")
    
    # 리트리버 로드 시도
    try:
        return _make_retriever(collection, embeddings)
    except Exception as e:
        msg = str(e).lower()
        if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg or "backfill" in msg:
            print("[!] KIFRS HNSW 인덱스 손상 감지")
            print("[복구] KIFRS 컬렉션 재생성 중...")
            _reset_collection(collection)
            raw: list[Document] = []
            for filename, source_id in K_IFRS_PDFS:
                p = PROJECT_ROOT / filename
                if p.exists():
                    raw.extend(_load_pdf(p, source_id))
            _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
            print("[✓] KIFRS 복구 완료")
            return _make_retriever(collection, embeddings)
        raise


def get_kam_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever:
    collection = "kam"
    
    # 컬렉션 없으면 PDF에서 초기화
    if not _collection_exists(collection):
        print("[info] KAM 컬렉션 초기화 중...")
        raw: list[Document] = []
        for filename, source_id in KAM_PDFS:
            p = PROJECT_ROOT / filename
            if p.exists():
                raw.extend(_load_pdf(p, source_id))
        _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
        print(f"[✓] KAM 초기화 완료: {len(raw)}개 문서")
    
    # 리트리버 로드 시도
    try:
        return _make_retriever(collection, embeddings)
    except Exception as e:
        msg = str(e).lower()
        if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg or "backfill" in msg:
            print("[!] KAM HNSW 인덱스 손상 감지")
            print("[복구] KAM 컬렉션 재생성 중...")
            _reset_collection(collection)
            raw: list[Document] = []
            for filename, source_id in KAM_PDFS:
                p = PROJECT_ROOT / filename
                if p.exists():
                    raw.extend(_load_pdf(p, source_id))
            _make_retriever(collection, embeddings, docs_to_ingest=_chunk(raw))
            print("[✓] KAM 복구 완료")
            return _make_retriever(collection, embeddings)
        raise


def get_dart_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever | None:
    collection = "dart"
    
    # Step 1: 컬렉션 존재 여부 확인
    if not _collection_exists(collection):
        print(f"[info] DART 컬렉션이 없습니다. 초기화 필요: python dart_ingest.py 실행")
        return None
    
    # Step 2: 컬렉션 로드 시도 (HNSW 에러 감지 및 복구)
    try:
        bm25_docs = _load_all_docs(collection)
        print(f"[✓] DART 리트리버 로드 성공: {len(bm25_docs)}개 문서")
    except Exception as e:
        msg = str(e).lower()
        if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg or "backfill" in msg:
            print(f"[!] DART HNSW 인덱스 손상 감지: {str(e)[:80]}")
            print(f"[복구] DART 컬렉션 재생성 중...")
            _reset_collection(collection)
            print(f"[✓] DART 컬렉션 초기화 완료. 재실행 필요: python dart_ingest.py")
            return None
        else:
            print(f"[ERROR] DART 로드 중 예기치 않은 에러: {e}")
            raise

    if not bm25_docs:
        print("[info] DART 컬렉션이 비어있습니다.")
        return None

    # Step 3: 리트리버 생성
    try:
        store = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=CHROMA_DIR)
        
        from langchain_community.retrievers import BM25Retriever
        enriched_bm25_docs = []
        for d in bm25_docs:
            company = d.metadata.get("company", "")
            year = d.metadata.get("year", "")
            section = d.metadata.get("section", "")
            prefix = f"[{company} {year}년 {section}] " if company else ""
            enriched_bm25_docs.append(Document(page_content=prefix + (d.page_content or ""), metadata=d.metadata))
        
        global_bm25 = BM25Retriever.from_documents(enriched_bm25_docs, k=30)
        
        base = DynamicDartRetriever(
            store=store, 
            global_bm25_docs=bm25_docs, 
            global_bm25_retriever=global_bm25
        )
        print("[✓] DART 하이브리드 리트리버 생성 완료")
        return _build_2stage(base, rerank_top_n=10)
        
    except Exception as e:
        msg = str(e).lower()
        if "error loading hnsw index" in msg or "constructing hnsw segment reader" in msg or "backfill" in msg:
            print(f"[!] DART 리트리버 생성 중 HNSW 에러 발생")
            print(f"[복구] DART 컬렉션 재생성 중...")
            _reset_collection(collection)
            print(f"[✓] DART 컬렉션 초기화 완료. 재실행 필요: python dart_ingest.py")
            return None
        else:
            print(f"[ERROR] DART 리트리버 생성 중 예기치 않은 에러: {e}")
            raise
