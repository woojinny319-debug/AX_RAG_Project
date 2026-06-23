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

# ── 브랜드 (자유롭게 수정하세요) ─────────────────────────────
APP_NAME = "KAM Lens"
APP_TAGLINE = "제약·바이오 감사 RAG · K-IFRS · DART 공시 · 삼일 KAM"
ASSISTANT_INTRO = (
    "RAG 및 Langchain 기술을 적용하여 만든 학습용 chat-bot 입니다.\n\n"
    "K-IFRS·DART 공시·삼일 KAM 자료를 근거로, 제약·바이오 기업의 회계·감사 질문에 답해 드려요. "
    "연구개발비 자산화 금액, 신약 파이프라인, 기업 간 비교 등 무엇이든 물어보세요."
)
EXAMPLES = [
    "삼천당제약 연구개발비 자산화 금액은?",
    "셀트리온과 한미약품 자산화 정책 비교",
    "연구개발비를 무형자산으로 공시한 기업 사례",
]

st.set_page_config(page_title=APP_NAME, layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Noto+Serif+KR:wght@600;700&display=swap');
    #MainMenu, footer {visibility:hidden;}
    [data-testid="stToolbar"]{display:none;}
    html, body, [class*="css"], textarea, input { font-family:'Noto Sans KR', sans-serif; }
    .block-container{ max-width: 880px; padding-top:3rem; padding-bottom:7rem; }

    /* 상단 브랜드 헤더 */
    .app-header{ display:flex; align-items:center; gap:.6rem; }
    .app-mark{ width:32px; height:32px; border-radius:9px;
        background:linear-gradient(135deg,#6FA0DA,#39507D);
        display:inline-flex; align-items:center; justify-content:center;
        color:#fff; font-size:.95rem; }
    .app-name{ font-family:'Noto Serif KR',serif; font-size:1.4rem; font-weight:700; color:#EAF0F8; }
    .app-tag{ color:#8A98AD; font-size:.85rem; margin:.2rem 0 .9rem 0; }
    .app-rule{ border-bottom:1px solid #283450; margin-bottom:1.3rem; }

    /* 채팅 — 카톡식: 아바타 말풍선 밖, content에만 배경+여백 */
    [data-testid="stChatMessage"]{
        background:transparent; border:none; padding:0;
        gap:.5rem; align-items:flex-start; width:100%;
    }
    /* 봇 아바타: 흰색 박스 + 네이비 아이콘 */
    [data-testid="stChatMessageAvatarAssistant"]{
        background:#FFFFFF !important; color:#1B2436 !important;
        border-radius:8px; margin-top:3px;
    }
    /* 유저 아바타: 삭제 */
    [data-testid="stChatMessageAvatarUser"]{ display:none !important; }
    /* 말풍선 = content. 내용 크기에 맞게 + Streamlit 기본 중앙정렬(margin auto) 제거 */
    [data-testid="stChatMessageContent"]{
        background:#19233A; border:1px solid #243049; border-radius:14px;
        padding:.85rem 1.15rem;
        flex:0 1 auto !important; width:fit-content !important; max-width:80%;
        margin:0 !important;   /* 중앙정렬 해제 → 봇 왼쪽 밀착 */
    }
    /* user 메시지: 오른쪽 끝 밀착 + 다른 색 */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
        flex-direction:row-reverse;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
        background:#2E5984; border-color:#3C6EA5; max-width:74%;
    }

    /* 입력창 위 힌트 말풍선 */
    .hint-pill{ display:inline-block; background:#1F2A44; color:#D7E0EE;
        border:1px solid #30406A; padding:.55rem 1rem; border-radius:14px;
        font-weight:600; font-size:.92rem; position:relative; margin:.2rem 0 1rem; }
    .hint-pill:after{ content:""; position:absolute; left:26px; bottom:-7px; width:13px; height:13px;
        background:#1F2A44; border-right:1px solid #30406A; border-bottom:1px solid #30406A;
        transform:rotate(45deg); }

    /* 입력창 */
    [data-testid="stChatInput"]{ background:#19233A; border:1px solid #30406A; border-radius:16px; }
    [data-testid="stChatInput"] textarea{ color:#E4E9F1; }

    /* 하단 안내 문구 */
    .disc{ color:#6B7689; font-size:.8rem; margin:.5rem 0 .2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

_hc1, _hc2 = st.columns([5, 1.3], vertical_alignment="center")
with _hc1:
    st.markdown(
        f"<div class='app-header'><span class='app-name'>{APP_NAME}</span></div>"
        f"<div class='app-tag'>{APP_TAGLINE}</div>",
        unsafe_allow_html=True,
    )
with _hc2:
    _restart = st.button("다시 시작", use_container_width=True)
st.markdown("<div class='app-rule'></div>", unsafe_allow_html=True)
if _restart:
    st.session_state["messages"] = []
    st.cache_resource.clear()
    st.rerun()

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


@st.cache_resource(show_spinner="DART 로딩 중...")
def load_dart(emb: OpenAIEmbeddings) -> BaseRetriever | None:
    """DART 리트리버 캐시 (파괴적 reset 제거로 안전하게 캐시 가능). 갱신은 '다시 시작' 버튼이 캐시 클리어."""
    return get_dart_retriever(emb)


# 리트리버 로드
embeddings = load_embeddings()
kifrs_retriever = load_kifrs(embeddings)
kam_retriever = load_kam(embeddings)
dart_retriever = load_dart(embeddings)  # 매번 새로 로드
llm = load_llm()

with st.sidebar:
    st.markdown("#### 상태")
    if dart_retriever is None:
        st.warning("DART 데이터 준비 안 됨")
        st.caption("`python dart_ingest.py` 실행 필요")
    else:
        st.markdown("DART 공시 연결됨")
    st.markdown("#### 이렇게 물어보세요")
    for _ex in EXAMPLES:
        st.caption(f"· {_ex}")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


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


# 대화 기록 렌더 (user 오른쪽 / assistant 왼쪽) — 모든 assistant 답변에 핵심 출처 표시
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("catalog"):
            render_compact_catalog(m["content"], m["catalog"])

# 대화 시작 전: 환영 메시지 + 입력 유도 힌트
if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown(ASSISTANT_INTRO)
    st.markdown(
        f"<div class='hint-pill'>{APP_NAME}가 무엇을 답해줄 수 있는지 물어보세요</div>"
        "<div class='disc'>AI는 한정된 데이터에 기반하니, 중요한 정보는 추가 확인을 권장해요.</div>",
        unsafe_allow_html=True,
    )

query = st.chat_input("AI에게 질문해 주세요.")
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

    st.session_state["messages"].append({"role": "assistant", "content": answer, "catalog": catalog})
