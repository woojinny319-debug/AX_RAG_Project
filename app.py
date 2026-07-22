"""Streamlit UI for Finance RAG with explicit citations."""

from __future__ import annotations

import base64
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts import SYSTEM_PROMPT, build_user_prompt, format_cited_docs
from rag_engine import get_dart_retriever, get_kam_retriever, get_kifrs_retriever

_PAGE_CSS = """
<style>
/* ── Streamlit 테마 primary 색상(빨강) 중립색으로 재정의 ── */
:root {
    --primary-color: #1a73e8 !important;
}

/* ── Streamlit chrome 숨김 ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden; height: 0; }

/* ── 전체 배경 ── */
html, body, .stApp, .main, [data-testid="stAppViewContainer"] {
    background: #ffffff !important;
    font-family: "Google Sans", "Noto Sans KR", sans-serif !important;
}

/* ── 콘텐츠 폭·여백 ── */
.block-container {
    max-width: 70% !important;
    padding: 2.5rem 5vw 7rem 5vw !important;
}

/* ── 페이지 제목 ── */
h1 {
    font-size: 1.3rem !important;
    font-weight: 500 !important;
    color: #1f1f1f !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.1rem !important;
}
.stCaption { color: #aaa !important; font-size: 12.5px !important; }

/* ═══════════════════════════════════════
   채팅 메시지 레이아웃
═══════════════════════════════════════ */
[data-testid="stChatMessage"] {
    display: flex !important;
    align-items: flex-start !important;
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    gap: 0 !important;
}

/* ── 아바타 숨김 ── */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

/* ═══════════════════════════════════════
   사용자 말풍선 — Gemini 스타일
   흰 배경 + 테두리 + 그림자, 오른쪽 정렬
═══════════════════════════════════════ */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    justify-content: flex-end !important;
    margin-bottom: 6px !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
    background: #e3f4ff !important;
    border: none !important;
    border-radius: 22px !important;
    padding: 12px 18px !important;
    width: fit-content !important;
    max-width: 65% !important;
    margin-left: auto !important; /* 블록 요소 정렬을 우측 끝으로 밀어내어 해결 */
    font-size: 15px !important;
    font-weight: 400 !important;
    line-height: 1.4 !important;
    color: #1f1f1f !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* 말풍선 내부의 모든 기본 마진/패딩 제거 및 가로너비 수축 강제 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] * {
    margin: 0 !important;
    padding: 0 !important;
    width: fit-content !important; /* 마크다운 컨테이너들이 100% 늘어나는 현상 방지 */
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] p {
    text-align: left !important;
}

/* ═══════════════════════════════════════
   어시스턴트 답변 — 배경 없는 순수 텍스트
═══════════════════════════════════════ */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    justify-content: flex-start !important;
    margin-bottom: 2px !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] {
    background: transparent !important;
    border: none !important;
    max-width: 88% !important;
    font-size: 13px !important;
    line-height: 1.9 !important;
    color: #1f1f1f !important;
}
/* 단락 간격 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] p {
    margin-top: 0 !important;
    margin-bottom: 0.8em !important;
}
/* 헤딩 스타일 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] h1 {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #1f1f1f !important;
    margin: 1.2em 0 0.4em 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] h2 {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #1f1f1f !important;
    margin: 1.1em 0 0.35em 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] h3 {
    font-size: 20.5px !important;
    font-weight: 600 !important;
    color: #1f1f1f !important;
    margin: 1em 0 0.3em 0 !important;
    letter-spacing: -0.01em !important;
}
/* 불릿 리스트 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] ol {
    padding-left: 1.4em !important;
    margin: 0.3em 0 0.8em 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] li {
    margin-bottom: 0.35em !important;
    line-height: 1.75 !important;
}

/* ═══════════════════════════════════════
   입력창 — Gemini 스타일 pill
═══════════════════════════════════════ */
[data-testid="stBottom"],
[data-testid="stBottom"] > *,
[data-testid="stBottom"] > * > *,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > *,
section[data-testid="stBottom"] {
    background: #ffffff !important;
    border-top: none !important;
}
[data-testid="stBottom"] {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 0 20px 16px 20px !important;
    left: 0 !important;
    right: 0 !important;
}
[data-testid="stChatInput"] {
    border-radius: 30px !important;
    border: 1px solid #dde3ea !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
    background: #f0f4f9 !important;
    overflow: hidden !important;
}
/* 질문 입력창 내부 전송(화살표) 버튼 커스텀 */
[data-testid="stChatInput"] button {
    transition: background-color 0.2s, color 0.2s;
}
[data-testid="stChatInput"] button:not(:disabled) {
    background-color: #c2e7ff !important; /* 구글/Gemini 스타일 파스텔톤 블루 */
}
[data-testid="stChatInput"] button:not(:disabled):hover {
    background-color: #a8d7ff !important; /* 호버 시 다소 짙은 파스텔 블루 */
}
[data-testid="stChatInput"] button:not(:disabled) svg {
    color: #004a77 !important; /* 아이콘을 짙은 블루로 대비감 주기 */
    fill: #004a77 !important;
}
[data-testid="stChatInput"] button:disabled {
    background-color: transparent !important;
}
[data-testid="stChatInput"] button:disabled svg {
    color: #9aa0a6 !important;
    fill: #9aa0a6 !important;
}
[data-testid="stChatInput"] textarea {
    border: none !important;
    box-shadow: none !important;
    padding: 16px 22px !important;
    font-size: 15px !important;
    background: transparent !important;
    color: #1f1f1f !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9aa0a6 !important;
}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stChatInput"]:focus-within {
    outline: none !important;
    border-color: #dde3ea !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
}
/* stChatInputContainer의 자체 테두리 제거 (둥근 모서리 뒤로 삐져나오는 청색 테두리 방지) */
[data-testid="stChatInputContainer"] {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
/* 포커스 시 생기는 모든 기본 아웃라인 및 강조 테두리 제거 */
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-visible,
[data-testid="stChatInput"] *:focus-within,
[data-testid="stChatInputContainer"] *,
[data-testid="stChatInputContainer"] *:focus,
[data-testid="stChatInputContainer"] *:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    border-color: transparent !important;
}
/* Streamlit 내장 div의 포커스 보더 억제 */
[data-testid="stChatInput"] div:focus-within,
[data-testid="stChatInput"] div:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}


/* ═══════════════════════════════════════
   출처 박스
═══════════════════════════════════════ */
.src-cards {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 2px 0 12px 0;
}
.src-card { position: relative; display: inline-block; }
.src-label {
    border: 1px solid #dde3ea;
    border-radius: 20px;
    padding: 4px 12px;
    background: #f0f4f9;
    font-size: 11.5px;
    color: #444746;
    cursor: default;
    white-space: nowrap;
    transition: background 0.15s;
}
.src-label:hover { background: #e2e8f0; border-color: #b0bec5; }
.src-tooltip {
    display: none;
    position: absolute;
    z-index: 9999;
    bottom: calc(100% + 8px);
    left: 0;
    background: #fff;
    border: 1px solid #dde3ea;
    border-radius: 12px;
    padding: 14px 16px;
    width: 420px;
    max-height: 240px;
    overflow-y: auto;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    font-size: 12.5px;
    line-height: 1.7;
    color: #3c4043;
    white-space: pre-wrap;
    word-break: break-word;
}
.src-card:hover .src-tooltip { display: block; }
</style>
"""

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


def get_image_base64(img_path):
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        return ""


st.set_page_config(
    page_title="AuditGPT",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(_PAGE_CSS, unsafe_allow_html=True)
# logo.png가 프로젝트 루트 폴더에 있다고 가정하고 base64 변환
logo_base64 = get_image_base64("logo.png")
if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 2.3rem; width: auto; margin-right: 12px; vertical-align: middle;">'
else:
    logo_html = ""

st.markdown(f"""
<div style="
    padding: 18px 0 14px 0;
    border-bottom: 1.5px solid #e8eaed;
    margin-bottom: 24px;
">
    <div style="
        display: flex;
        align-items: center; /* 로고와 글씨의 세로축 정렬 */
        font-size: 2.3rem;
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: -0.03em;
        line-height: 1.15;
    ">
        {logo_html}
        <span>AuditGPT</span>
    </div>
    <div style="
        margin-top: 4px;
        font-size: 12.5px;
        color: #5f6368;
        letter-spacing: 0.01em;
    ">K-IFRS · DART · KAM &nbsp;|&nbsp; 3-Source RAG &nbsp;|&nbsp; 제약·바이오 감사 특화</div>
</div>
""", unsafe_allow_html=True)


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
        st.warning("DART 데이터가 비어 있습니다.")
        st.caption("실행 필요: `python dart_ingest.py`")
    else:
        st.success("DART 데이터 로드됨")

    if st.button("리트리버 새로고침", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

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


def _trim_to_complete_sentences(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    matches = list(re.finditer(r'[다요임음\.!?。]\s', chunk))
    if matches:
        return chunk[:matches[-1].end()].rstrip()
    last_period = chunk.rfind('.')
    if last_period > len(chunk) // 4:
        return chunk[:last_period + 1]
    return chunk


def _build_cards_html(sids: list[str], catalog_map: dict) -> str:
    cards = ""
    for sid in sids:
        item = catalog_map.get(sid)
        if not item:
            continue
        source = item.get("source", "")
        company = item.get("company", "")
        section = item.get("section", "")
        page = item.get("page", "")
        content = _trim_to_complete_sentences(item.get("content", "") or "")
        content_escaped = (
            content.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
        )
        is_dart = source.upper().startswith("DART")
        label_parts = [source]
        if company and not is_dart:
            label_parts.append(company)
        if section:
            label_parts.append(section)
        if page and not is_dart:
            label_parts.append(f"p.{page}")
        label = " | ".join(label_parts)
        cards += f'<div class="src-card"><div class="src-label">{label}</div><div class="src-tooltip">{content_escaped}</div></div>'
    return f'<div class="src-cards">{cards}</div>' if cards else ""


def render_answer_with_inline_sources(answer: str, catalog: list[dict]) -> None:
    catalog_map = {item.get("sid"): item for item in catalog}
    for para in answer.split('\n\n'):
        if not para.strip():
            continue
        sids = list(dict.fromkeys(re.findall(r'\[(S\d+)\]', para)))
        clean_para = re.sub(r'\s*\[(S\d+)\]', '', para).strip()
        if not clean_para:
            continue
        st.markdown(clean_para)
        if sids:
            cards_html = _build_cards_html(sids, catalog_map)
            if cards_html:
                st.markdown(cards_html, unsafe_allow_html=True)


for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("AI에게 질문해 주세요.")
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Finding sources & generating answer..."):
            try:
                answer, catalog, kifrs_docs, dart_docs, kam_docs = build_answer(query)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.stop()
        render_answer_with_inline_sources(answer, catalog)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
