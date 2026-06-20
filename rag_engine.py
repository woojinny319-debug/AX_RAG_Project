"""Retrieval engine with duplicate-safe ingest and 2-stage retrieval."""

from __future__ import annotations

import hashlib
import pickle
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

import os

PROJECT_ROOT = Path(__file__).parent
# [근본 원인] 프로젝트가 한글 경로("바탕 화면") 아래에 있는데, chromadb 1.5.x의 Rust HNSW
# 세그먼트 로더가 비ASCII 절대경로를 못 열어 "Error loading hnsw index"가 발생했다.
# (상대경로/ASCII 경로는 정상). 따라서 벡터 DB는 ASCII 절대경로(사용자 홈 하위)에 둔다.
# 2026-06-19 디버깅 참고.
_ASCII_DATA_HOME = os.path.join(os.path.expanduser("~"), "rag_chroma_data")
CHROMA_DIR = os.path.join(_ASCII_DATA_HOME, "chroma_db_v2")
if not CHROMA_DIR.isascii():
    # 사용자명마저 비ASCII인 극단적 경우의 안전장치: 프로젝트 상대경로로 폴백
    CHROMA_DIR = str(PROJECT_ROOT / "chroma_db_v2")
EMBED_BATCH_SIZE = 30
EMBED_BATCH_DELAY = 1.6

_SHARED_CLIENT = None


def _get_client():
    """프로세스 전체가 공유하는 단일 chromadb 클라이언트.

    [중요] 한 프로세스에서 PersistentClient나 langchain Chroma가 각자 별도의 클라이언트를
    만들면, chromadb 1.5.x에서 compactor의 backfill 충돌이 일어나 HNSW 인덱스 로드가
    비결정적으로 실패한다(2026-06-19 디버깅). 모든 chroma 접근이 이 단일 클라이언트를
    공유하도록 강제하면 충돌이 사라진다.
    """
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
    return _SHARED_CLIENT


def _reset_client() -> None:
    """공유 클라이언트를 폐기해 다음 _get_client()가 *진짜로* 새 클라이언트를 만들도록 한다.

    chromadb는 PersistentClient를 path별로 내부 캐싱(SharedSystemClient)하므로, 파이썬 참조만
    None으로 두면 같은 stale 시스템을 다시 돌려준다. clear_system_cache()로 내부 캐시까지
    비워야 세그먼트 리더가 새로 초기화되어 'Nothing found on disk'가 복구된다(2026-06-20 디버깅)."""
    global _SHARED_CLIENT
    _SHARED_CLIENT = None
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        try:
            from chromadb.api.client import SharedSystemClient as _SSC
            _SSC.clear_system_cache()
        except Exception:
            pass

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
    store = Chroma(collection_name=collection_name, embedding_function=embeddings, client=_get_client())
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
                client=_get_client(),
            )
            store.add_documents(docs, ids=ids)
        else:
            raise


def _get_collection_count(collection_name: str) -> int:
    """컬렉션 문서 개수 반환. HNSW 에러 발생 시 exception 그대로 throw."""
    client = _get_client()
    return client.get_collection(collection_name).count()


def _collection_exists(collection_name: str) -> bool:
    """컬렉션 존재 여부만 확인 (문서 개수 세지 않음)."""
    client = _get_client()
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def _reset_collection(collection_name: str) -> None:
    client = _get_client()
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


DART_BM25_SIDECAR = PROJECT_ROOT / "dart_bm25.pkl"


def _load_dart_bm25_sidecar() -> list[Document] | None:
    """BM25 코퍼스를 사이드카 파일에서 로드.

    chromadb 1.5.x는 대용량 컬렉션의 전체 get()에서 backfill 레이스로 인한 HNSW 에러를
    비결정적으로 던진다(2026-06-19 디버깅). BM25는 모든 문서 텍스트가 필요하므로 이 불안정한
    경로 대신, ingest가 생성한 사이드카(dart_bm25.pkl)에서 안정적으로 로드한다.
    (벡터 검색 similarity_search는 안정적이라 chroma를 그대로 사용)
    """
    if not DART_BM25_SIDECAR.exists():
        return None
    try:
        with open(DART_BM25_SIDECAR, "rb") as f:
            rows = pickle.load(f)
        return [Document(page_content=r["text"], metadata=r.get("metadata") or {}) for r in rows]
    except Exception as e:
        print(f"[warn] BM25 사이드카 로드 실패, chroma fallback: {str(e)[:60]}")
        return None


def _load_all_docs(collection_name: str, retries: int = 8) -> list[Document]:
    # chromadb 1.5.x는 프로세스 cold-start의 첫 전체 get()에서 HNSW 인덱스를 backfill(컴팩션)하는데,
    # 이 과정이 아직 끝나지 않았으면 일시적으로 "Error loading hnsw index"를 던진다(비결정적).
    # 대응: ① count()로 backfill을 먼저 유도(워밍업) ② 실패 시 대기 후 재시도. 데이터는 삭제하지 않는다.
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            client = _get_client()
            collection = client.get_collection(collection_name)
            _ = collection.count()  # 워밍업: cold-start backfill 유도
            payload = collection.get(include=["documents", "metadatas"])
            docs = payload.get("documents") or []
            metas = payload.get("metadatas") or []
            return [Document(page_content=t, metadata=m or {}) for t, m in zip(docs, metas)]
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "hnsw" in msg or "backfill" in msg or "compactor" in msg or "segment reader" in msg:
                time.sleep(0.6 * (attempt + 1))
                continue
            raise
    raise last_err if last_err else RuntimeError("_load_all_docs 실패")


_SEGMENT_ERR_MARKERS = ("hnsw", "segment", "nothing found", "backfill", "compactor")


def _robust_query(
    collection_name: str,
    query_embedding: list[float],
    n_results: int,
    where: dict | None = None,
    retries: int = 5,
) -> list[Document]:
    """langchain Chroma 래퍼 대신 chromadb collection.query를 직접 호출(안정적).

    langchain Chroma의 similarity_search는 다중 컬렉션·반복 쿼리·장기 실행에서 세그먼트
    리더가 stale해져 'Nothing found on disk' 등을 던졌다(특히 작은 kam 컬렉션). 직접 쿼리는
    안정적이며, 그래도 세그먼트 에러가 나면 클라이언트를 새로 열어 재시도한다(2026-06-20 디버깅).
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            col = _get_client().get_collection(collection_name)
            kwargs: dict = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas"],
            }
            if where:
                kwargs["where"] = where
            res = col.query(**kwargs)
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            return [Document(page_content=t, metadata=m or {}) for t, m in zip(docs, metas)]
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if any(marker in msg for marker in _SEGMENT_ERR_MARKERS):
                _reset_client()
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    raise last_err if last_err else RuntimeError("_robust_query 실패")


def _rrf(doc_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: 여러 랭킹 리스트를 순위 기반 점수로 융합."""
    rrf_score: dict = {}
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            content = doc.page_content
            score = 1.0 / (rank + k)
            if content in rrf_score:
                rrf_score[content]["score"] += score
            else:
                rrf_score[content] = {"score": score, "doc": doc}
    return [item["doc"] for item in sorted(rrf_score.values(), key=lambda x: x["score"], reverse=True)]


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

COMPANY_ALIAS_MAP = {
    "레고켐바이오": "리가켐바이오",
    "에스케이바이오팜": "SK바이오팜",
    "에스케이바이오사이언스": "SK바이오사이언스",
    "녹십자": "GC녹십자",
    "보령제약": "보령",
    "유나이티드제약": "한국유나이티드제약",
}
# 회사명 바로 뒤에 이 표현이 나오면 "그 회사를 제외" 의도로 본다 (예: "삼천당제약이 아닌").
NEGATION_MARKERS = ("아닌", "말고", "제외", "빼고", "이외", "외의", "외에", "아니라", "아니고")


def _detect_companies(query: str) -> tuple[list[str], list[str]]:
    """질의에서 언급된 기업을 (포함 대상, 제외 대상)으로 분류. 별칭은 정식명으로 정규화."""
    positive: list[str] = []
    negated: list[str] = []
    for comp in TARGET_COMPANIES_LIST:
        idx = query.find(comp)
        if idx == -1:
            continue
        canonical = COMPANY_ALIAS_MAP.get(comp, comp)
        window = query[idx + len(comp): idx + len(comp) + 8]  # 회사명 직후 8글자
        (negated if any(m in window for m in NEGATION_MARKERS) else positive).append(canonical)
    # 중복 제거(순서 유지). 같은 회사가 양쪽이면 제외를 우선.
    positive = [c for c in dict.fromkeys(positive) if c not in negated]
    negated = list(dict.fromkeys(negated))
    return positive, negated


def _interleave(doc_lists: list[list[Document]], limit: int) -> list[Document]:
    """여러 리스트를 라운드로빈으로 섞어 각 리스트(=각 기업)의 대표성을 보장하며 합친다."""
    out: list[Document] = []
    seen: set[str] = set()
    for rank in range(max((len(dl) for dl in doc_lists), default=0)):
        for dl in doc_lists:
            if rank < len(dl):
                d = dl[rank]
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    out.append(d)
                    if len(out) >= limit:
                        return out
    return out


def _enrich(docs: list[Document]) -> list[Document]:
    """BM25 키워드 매칭 강화를 위해 [기업 연도 섹션] 메타데이터를 본문 앞에 결합."""
    out = []
    for d in docs:
        company = d.metadata.get("company", "")
        year = d.metadata.get("year", "")
        section = d.metadata.get("section", "")
        prefix = f"[{company} {year}년 {section}] " if company else ""
        out.append(Document(page_content=prefix + (d.page_content or ""), metadata=d.metadata))
    return out


class DynamicDartRetriever(BaseRetriever):
    collection_name: str
    embeddings: Any
    global_bm25_docs: list
    global_bm25_retriever: Any

    def _company_docs(self, query: str, qv: list[float], company: str, k: int) -> list[Document]:
        """단일 기업으로 필터링한 벡터+BM25 하이브리드 결과(RRF)."""
        semantic = _robust_query(self.collection_name, qv, k, where={"company": company})
        comp_bm25_docs = [d for d in self.global_bm25_docs if d.metadata.get("company") == company]
        if comp_bm25_docs:
            from langchain_community.retrievers import BM25Retriever
            bm25 = BM25Retriever.from_documents(_enrich(comp_bm25_docs), k=k)
            return _rrf([semantic, bm25.invoke(query)])
        return semantic

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        positive, negated = _detect_companies(query)
        qv = self.embeddings.embed_query(query)

        if positive:
            # [다중 기업] 각 기업별로 따로 검색해 라운드로빈으로 합쳐 *모든 기업의 대표성*을 보장.
            # (기존엔 첫 기업만 필터링해 비교 질문에서 나머지 기업이 누락됐다 — 2026-06-20 Q3 개선)
            print(f"[DynamicDartRetriever] 대상 기업 {positive} 감지. 기업별 균형 검색.")
            per = max(15, 40 // len(positive))
            per_company = [self._company_docs(query, qv, comp, per) for comp in positive]
            return _interleave(per_company, limit=24)

        # 포함 대상이 없음 → 전역 검색. 단, 제외 대상(예: "삼천당이 아닌")이 있으면 필터로 뺀다.
        where = {"company": {"$nin": negated}} if negated else None
        if negated:
            print(f"[DynamicDartRetriever] 제외 기업 {negated} → 해당 기업 제외 전역 검색.")
        else:
            print("[DynamicDartRetriever] 기업명 미감지. 전역 검색 수행.")
        semantic_docs = _robust_query(self.collection_name, qv, 30, where=where)
        if self.global_bm25_retriever:
            bm25_docs = self.global_bm25_retriever.invoke(query)
            if negated:  # BM25 결과에서도 제외 기업 제거
                bm25_docs = [d for d in bm25_docs if d.metadata.get("company") not in negated]
            combined = _rrf([semantic_docs, bm25_docs])
        else:
            combined = semantic_docs
        return combined[:20]


class SimpleHybridRetriever(BaseRetriever):
    """kifrs/kam용 하이브리드 리트리버. 직접 chromadb 쿼리(_robust_query) + BM25를 RRF로 융합.

    기존 EnsembleRetriever + langchain Chroma.as_retriever는 장기 실행에서 세그먼트 stale
    오류('Nothing found on disk')를 냈으므로, 안정적인 직접 쿼리 경로로 대체했다(2026-06-20).
    """
    collection_name: str
    embeddings: Any
    bm25_retriever: Any

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        qv = self.embeddings.embed_query(query)
        semantic_docs = _robust_query(self.collection_name, qv, 50)
        bm25_docs = self.bm25_retriever.invoke(query) if self.bm25_retriever else []
        return _rrf([semantic_docs, bm25_docs])[:30]


def _make_base_hybrid_retriever(collection_name: str, embeddings: OpenAIEmbeddings, bm25_docs: list[Document]) -> BaseRetriever:
    if not bm25_docs:
        raise ValueError(f"{collection_name} 컬렉션에 검색 가능한 문서가 없습니다.")
    bm25 = BM25Retriever.from_documents(_enrich(bm25_docs), k=50)
    return SimpleHybridRetriever(collection_name=collection_name, embeddings=embeddings, bm25_retriever=bm25)


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
        # _load_all_docs는 내부적으로 워밍업+재시도를 수행한다. 실패해도 컬렉션을 삭제(reset)하지
        # 않는다 — 일시적 backfill 오류에 데이터를 영구 삭제하던 과거 동작이 치명적이었기 때문.
        bm25_docs = _load_all_docs(collection_name)

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
    
    # 리트리버 로드 (_load_all_docs가 워밍업+재시도로 cold-start HNSW 오류를 처리. reset 없음)
    return _make_retriever(collection, embeddings)


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
    
    # 리트리버 로드 (_load_all_docs가 워밍업+재시도로 cold-start HNSW 오류를 처리. reset 없음)
    return _make_retriever(collection, embeddings)


def get_dart_retriever(embeddings: OpenAIEmbeddings) -> BaseRetriever | None:
    # 컬렉션 이름: 오염된 'dart' 대신 'dart_docs' 사용 (2026-06-19 HNSW 영속화 디버깅 참고)
    collection = "dart_docs"
    
    # Step 1: 컬렉션 존재 여부 확인
    if not _collection_exists(collection):
        print(f"[info] DART 컬렉션이 없습니다. 초기화 필요: python dart_ingest.py 실행")
        return None
    
    # Step 2: 컬렉션 로드 시도 (HNSW 에러는 _load_all_docs가 재시도로 처리)
    # [중요] 과거엔 HNSW 에러 시 _reset_collection으로 컬렉션을 삭제했는데, 이는 일시적
    # backfill 오류에도 데이터를 영구 삭제하는 치명적 동작이었다. 이제 삭제하지 않고
    # None만 반환해 데이터를 보존한다(2026-06-19 디버깅).
    # BM25 코퍼스: 안정적인 사이드카 파일 우선, 없으면 chroma full-get으로 fallback
    bm25_docs = _load_dart_bm25_sidecar()
    if bm25_docs is not None:
        print(f"[✓] DART BM25 사이드카 로드: {len(bm25_docs)}개 문서")
    else:
        try:
            bm25_docs = _load_all_docs(collection)
            print(f"[✓] DART 리트리버 로드 성공(chroma): {len(bm25_docs)}개 문서")
        except Exception as e:
            msg = str(e).lower()
            if "hnsw" in msg or "constructing hnsw segment reader" in msg or "backfill" in msg or "compactor" in msg:
                print(f"[!] DART HNSW 로드 실패(재시도 후에도): {str(e)[:80]}")
                print(f"[info] 데이터는 보존됨. dart_bm25.pkl 사이드카를 생성하면 안정화됩니다.")
                return None
            else:
                print(f"[ERROR] DART 로드 중 예기치 않은 에러: {e}")
                raise

    if not bm25_docs:
        print("[info] DART 컬렉션이 비어있습니다.")
        return None

    # Step 3: 리트리버 생성 (직접 chromadb 쿼리 기반 — langchain Chroma store 불필요)
    from langchain_community.retrievers import BM25Retriever
    global_bm25 = BM25Retriever.from_documents(_enrich(bm25_docs), k=30)
    base = DynamicDartRetriever(
        collection_name=collection,
        embeddings=embeddings,
        global_bm25_docs=bm25_docs,
        global_bm25_retriever=global_bm25,
    )
    print("[✓] DART 하이브리드 리트리버 생성 완료")
    return _build_2stage(base, rerank_top_n=10)
