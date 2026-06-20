"""Streamlit UI for Finance RAG with explicit citations."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts import SYSTEM_PROMPT, build_user_prompt, format_cited_docs
from rag_engine import get_dart_retriever, get_kam_retriever, get_kifrs_retriever

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

st.set_page_config(page_title="금융 RAG 어시스턴트", page_icon="📘", layout="wide")
st.title("📘 금융 RAG 어시스턴트")
st.caption("답변 본문에 [S#] 근거를 붙이고, 출처 원문을 바로 확인할 수 있습니다.")

openai_key = os.getenv("OPENAI_API_KEY", "")
if not openai_key or openai_key == "your_openai_api_key_here":
    st.error("OPENAI_API_KEY가 없습니다. .env를 확인해 주세요.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_embeddings() -> OpenAIEmbeddings:
    """임베딩 모델은 캐시 (가장 무거운 리소스)"""
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource(show_spinner="KIFRS 로딩 중...")
def load_kifrs(emb: OpenAIEmbeddings) -> BaseRetriever:
    """KIFRS 리트리버 캐시"""
    return get_kifrs_retriever(emb)


@st.cache_resource(show_spinner="KAM 로딩 중...")
def load_kam(emb: OpenAIEmbeddings) -> BaseRetriever:
    """KAM 리트리버 캐시"""
    return get_kam_retriever(emb)


@st.cache_resource(show_spinner="LLM 로딩 중...")
def load_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0)


def load_dart(emb: OpenAIEmbeddings) -> BaseRetriever | None:
    """DART 리트리버는 캐시하지 않음 (자주 변경 가능, 복구 필요)"""
    return get_dart_retriever(emb)


# 리트리버 로드
embeddings = load_embeddings()
kifrs_retriever = load_kifrs(embeddings)
kam_retriever = load_kam(embeddings)
dart_retriever = load_dart(embeddings)  # 매번 새로 로드
llm = load_llm()

with st.sidebar:
    st.header("아키텍처 상태")
    st.markdown("- UI: `app.py`")
    st.markdown("- 검색엔진: `rag_engine.py`")
    st.markdown("- 수집기: `dart_ingest.py`")
    st.markdown("- 프롬프트: `prompts.py`")
    
    if dart_retriever is None:
        st.warning("⚠️ DART 데이터가 비어 있습니다.")
        st.caption("실행 필요: `python dart_ingest.py`")
    else:
        st.success("✅ DART 데이터 로드됨")
    
    if st.button("🔄 리트리버 새로고침", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


def retrieve_parallel(query: str) -> tuple[list[Document], list[Document], list[Document]]:
    # [중요] 순차 검색. chromadb 1.5.x는 단일 클라이언트의 동시(멀티스레드) 쿼리에서
    # "Nothing found on disk"(HNSW 세그먼트 레이스)를 던진다. 검색은 빠르므로 순차로 충분.
    # (2026-06-20 디버깅: 병렬 ThreadPoolExecutor → 순차로 변경)
    kifrs_docs = kifrs_retriever.invoke(query)
    kam_docs = kam_retriever.invoke(query)
    dart_docs = dart_retriever.invoke(query) if dart_retriever is not None else []
    return kifrs_docs, dart_docs, kam_docs


def _trim_docs(docs: list[Document], max_docs: int = 3, max_chars: int = 1200) -> list[Document]:
    trimmed: list[Document] = []
    for doc in docs[:max_docs]:
        text = (doc.page_content or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)"
        trimmed.append(Document(page_content=text, metadata=doc.metadata))
    return trimmed


def _build_source_context(
    kifrs_docs: list[Document], dart_docs: list[Document], kam_docs: list[Document]
) -> tuple[str, list[dict[str, str]]]:
    # gpt-4o의 넉넉한 컨텍스트 윈도우를 활용하여 문서 길이 및 개수 확대
    kifrs_docs = _trim_docs(kifrs_docs, max_docs=4, max_chars=2000)
    dart_docs = _trim_docs(dart_docs, max_docs=15, max_chars=3500)
    kam_docs = _trim_docs(kam_docs, max_docs=3, max_chars=1500)

    p1, c1 = format_cited_docs(kifrs_docs, 1)
    p2, c2 = format_cited_docs(dart_docs, len(c1) + 1)
    p3, c3 = format_cited_docs(kam_docs, len(c1) + len(c2) + 1)
    context = "\n\n".join([x for x in [p1, p2, p3] if x])
    return context, c1 + c2 + c3


def build_answer(query: str) -> tuple[str, list[dict[str, str]], list[Document], list[Document], list[Document]]:
    kifrs_docs, dart_docs, kam_docs = retrieve_parallel(query)
    source_context, catalog = _build_source_context(kifrs_docs, dart_docs, kam_docs)
    prompt = build_user_prompt(query, source_context)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = None
    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            break
        except Exception as e:
            err = str(e)
            is_tpm = ("rate_limit_exceeded" in err) or ("tokens per min" in err) or ("Error code: 429" in err)
            if not is_tpm or attempt == 2:
                raise
            wait_s = 2 * (attempt + 1)
            time.sleep(wait_s)
    if response is None:
        raise RuntimeError("LLM 응답 생성 실패")
    return str(response.content), catalog, kifrs_docs, dart_docs, kam_docs


def render_catalog(catalog: list[dict[str, str]]) -> None:
    st.markdown("### 출처 목록")
    if not catalog:
        st.caption("출처 없음")
        return
    for item in catalog:
        sid = item.get("sid", "")
        source = item.get("source", "")
        company = item.get("company", "")
        section = item.get("section", "")
        page = item.get("page", "")
        url = item.get("url", "")
        parts = [f"**[{sid}]**", source]
        if company:
            parts.append(company)
        if section:
            parts.append(f"섹션:{section}")
        if page:
            parts.append(f"p.{page}")
        st.write(" | ".join(parts))
        if url:
            st.markdown(f"- 원문 링크: {url}")


def _extract_cited_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[(S\d+)\]", answer))


def render_compact_catalog(answer: str, catalog: list[dict[str, str]], limit: int = 6) -> None:
    cited = _extract_cited_ids(answer)
    filtered = [x for x in catalog if x.get("sid", "") in cited] if cited else catalog
    st.markdown("### 핵심 출처")
    if not filtered:
        st.caption("핵심 출처를 찾지 못했습니다.")
        return
    head = filtered[:limit]
    tail = filtered[limit:]
    for item in head:
        sid = item.get("sid", "")
        source = item.get("source", "")
        page = item.get("page", "")
        section = item.get("section", "")
        url = item.get("url", "")
        st.write(f"**[{sid}]** {source} {f'| {section}' if section else ''} {f'| p.{page}' if page else ''}")
        if url:
            st.markdown(f"- 원문 링크: {url}")
    if tail:
        with st.expander(f"나머지 출처 {len(tail)}건 보기"):
            for item in tail:
                sid = item.get("sid", "")
                source = item.get("source", "")
                page = item.get("page", "")
                section = item.get("section", "")
                url = item.get("url", "")
                st.write(f"**[{sid}]** {source} {f'| {section}' if section else ''} {f'| p.{page}' if page else ''}")
                if url:
                    st.markdown(f"- 원문 링크: {url}")


query = st.chat_input("질문을 입력하세요.")
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("근거 검색 및 답변 생성 중..."):
            try:
                answer, catalog, kifrs_docs, dart_docs, kam_docs = build_answer(query)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.stop()
        st.markdown(answer)
        render_compact_catalog(answer, catalog, limit=6)
        with st.expander("검색된 원문 미리보기"):
            cols = st.columns(3)
            data = [("K-IFRS", kifrs_docs), ("DART", dart_docs), ("KAM", kam_docs)]
            for col, (name, docs) in zip(cols, data):
                with col:
                    st.markdown(f"**{name}**")
                    for d in docs[:5]:
                        st.caption(str(d.metadata))
                        st.text(d.page_content[:280] + "...")

        report = (
            f"# 금융 RAG 답변 보고서\n\n"
            f"- 생성시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"- 질문: {query}\n\n"
            f"## 답변\n{answer}\n\n"
            "## 출처 목록\n"
            + "\n".join([f"- [{x['sid']}] {x['source']} {x.get('url','')}" for x in catalog])
        )
        st.download_button(
            label="보고서 다운로드(.md)",
            data=report.encode("utf-8"),
            file_name=f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

    st.session_state["messages"].append({"role": "assistant", "content": answer})
