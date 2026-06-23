"""Streamlit UI for Finance RAG with explicit citations."""

from __future__ import annotations

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
    max-width: 100% !important;
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
    background: #d6eeff !important;
    border: none !important;
    border-radius: 22px !important;
    padding: 28px 28px !important;
    width: fit-content !important;
    max-width: 70% !important;
    font-size: 15px !important;
    font-weight: 400 !important;
    line-height: 1.65 !important;
    color: #1f1f1f !important;
    text-align: left !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07) !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] p {
    margin: 0 !important;
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
    font-size: 15px !important;
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
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1f1f1f !important;
    margin: 1.2em 0 0.4em 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] h2 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #1f1f1f !important;
    margin: 1.1em 0 0.35em 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] h3 {
    font-size: 22.5px !important;
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
[data-testid="stBottom"] {
    background: #ffffff !important;
    border-top: none !important;
    padding: 0 5vw 16px 5vw !important;
}
[data-testid="stChatInput"] {
    border-radius: 28px !important;
    border: 1px solid #dde3ea !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
    background: #f0f4f9 !important;
    overflow: hidden !important;
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
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInputContainer"]:focus-within {
    outline: none !important;
    border-color: #dde3ea !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
}
/* 빨간 테두리 완전 제거 — 모든 하위 요소 */
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-visible,
[data-testid="stChatInput"] *:focus-within,
[data-testid="stChatInputContainer"],
[data-testid="stChatInputContainer"]:focus,
[data-testid="stChatInputContainer"]:focus-visible,
[data-testid="stChatInputContainer"]:focus-within,
[data-testid="stChatInputContainer"] *,
[data-testid="stChatInputContainer"] *:focus,
[data-testid="stChatInputContainer"] *:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    border-color: transparent !important;
}
/* Streamlit 테마 accent 색상으로 인한 border 억제 */
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

<<<<<<< Updated upstream
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
=======
st.set_page_config(
    page_title="회계 챗봇",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(_PAGE_CSS, unsafe_allow_html=True)
st.markdown("""
<div style="
    padding: 18px 0 14px 0;
    border-bottom: 1.5px solid #e8eaed;
    margin-bottom: 24px;
">
    <div style="
        font-size: 2.3rem;
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: -0.03em;
        line-height: 1.15;
    ">회계 챗봇</div>
    <div style="
        margin-top: 4px;
        font-size: 12.5px;
        color: #5f6368;
        letter-spacing: 0.01em;
    ">K-IFRS · DART · KAM &nbsp;|&nbsp; 3-Source RAG &nbsp;|&nbsp; 제약·바이오 감사 특화</div>
</div>
""", unsafe_allow_html=True)
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
        st.warning("DART 데이터 준비 안 됨")
        st.caption("`python dart_ingest.py` 실행 필요")
    else:
        st.markdown("DART 공시 연결됨")
    st.markdown("#### 이렇게 물어보세요")
    for _ex in EXAMPLES:
        st.caption(f"· {_ex}")
=======
        st.warning("DART 데이터가 비어 있습니다.")
        st.caption("실행 필요: `python dart_ingest.py`")
    else:
        st.success("DART 데이터 로드됨")

    if st.button("리트리버 새로고침", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
>>>>>>> Stashed changes

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
        render_answer_with_inline_sources(answer, catalog)

    st.session_state["messages"].append({"role": "assistant", "content": answer, "catalog": catalog})
