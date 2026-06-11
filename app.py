from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts import SYSTEM_PROMPT, build_user_prompt, format_docs
from rag_engine import get_dart_retriever, get_kam_retriever, get_kifrs_retriever

load_dotenv()

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AX RAG — 회계처리 어시스턴트",
    page_icon="📊",
    layout="wide",
)

st.title("📊 제약·바이오 회계처리 AI 어시스턴트")
st.caption(
    "K-IFRS 기준서 · DART 기업 공시 · 삼일회계법인의 KAM을 통합 검색합니다."
)

# ---------------------------------------------------------------------------
# API 키 검증
# ---------------------------------------------------------------------------

openai_key = os.getenv("OPENAI_API_KEY", "")
if not openai_key or openai_key == "your_openai_api_key_here":
    st.error(
        "OPENAI_API_KEY가 설정되지 않았습니다.  \n"
        ".env 파일에 `OPENAI_API_KEY=sk-...` 를 입력하고 앱을 재시작하세요."
    )
    st.stop()

# ---------------------------------------------------------------------------
# 리소스 초기화 (앱 시작 시 1회 실행)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="K-IFRS · KAM 데이터 로딩 중...")
def load_retrievers() -> tuple[EnsembleRetriever, EnsembleRetriever, EnsembleRetriever | None]:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    kifrs = get_kifrs_retriever(embeddings)
    kam = get_kam_retriever(embeddings)
    dart = get_dart_retriever(embeddings)
    return kifrs, kam, dart


@st.cache_resource(show_spinner=False)
def load_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)


kifrs_retriever, kam_retriever, dart_retriever = load_retrievers()
llm = load_llm()

# DART 데이터 없음 경고
if dart_retriever is None:
    st.warning(
        "⚠️ DART 기업 공시 데이터가 없습니다.  \n"
        "`python dart_ingest.py` 를 실행하면 ② 기업 사례 섹션이 활성화됩니다."
    )

# ---------------------------------------------------------------------------
# 사이드바 — 정보 패널
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("데이터 소스")

    st.subheader("① K-IFRS 기준서")
    st.markdown(
        "- K-IFRS 제1038호 — 무형자산\n"
        "- K-IFRS 제1115호 — 수익\n"
        "- K-IFRS 제2032호 — 웹사이트 원가"
    )

    st.subheader("② DART 기업 공시")
    dart_status = "✅ 로드됨" if dart_retriever is not None else "❌ 미인제스트 (dart_ingest.py 실행 필요)"
    st.markdown(
        f"상태: {dart_status}  \n"
        "대상: 삼성바이오로직스 · 셀트리온 · 한미약품 · 유한양행 · 종근당"
    )

    st.subheader("③ KAM 보고서")
    st.markdown("- 삼일PwC 제약·바이오 KAM (2025)")

    st.divider()
    st.subheader("예시 질문")
    sample_questions: list[str] = [
        "제약사의 연구개발비 자산화 요건은?",
        "기술이전 마일스톤 수익인식 시점은?",
        "바이오시밀러 개발비의 손상 검토 기준은?",
        "CDMO 계약의 수익인식 방법은?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

# ---------------------------------------------------------------------------
# 채팅 히스토리 초기화
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 히스토리 렌더링
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# 검색 + LLM 합성 핵심 로직
# ---------------------------------------------------------------------------

def retrieve_parallel(query: str) -> tuple[list[Document], list[Document], list[Document]]:
    """3개 컬렉션을 ThreadPoolExecutor로 병렬 조회."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_kifrs: Future[list[Document]] = ex.submit(kifrs_retriever.invoke, query)
        f_kam: Future[list[Document]] = ex.submit(kam_retriever.invoke, query)

        if dart_retriever is not None:
            f_dart: Future[list[Document]] = ex.submit(dart_retriever.invoke, query)
            dart_docs = f_dart.result()
        else:
            dart_docs = []

        kifrs_docs = f_kifrs.result()
        kam_docs = f_kam.result()

    return kifrs_docs, dart_docs, kam_docs


def build_answer(query: str) -> tuple[str, list[Document], list[Document], list[Document]]:
    """검색 → LLM 합성 → (답변, K-IFRS 문서, DART 문서, KAM 문서) 반환."""
    kifrs_docs, dart_docs, kam_docs = retrieve_parallel(query)

    ctx_kifrs = format_docs(kifrs_docs)
    ctx_dart = format_docs(dart_docs)
    ctx_kam = format_docs(kam_docs)

    user_prompt = build_user_prompt(query, ctx_kifrs, ctx_dart, ctx_kam)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return str(response.content), kifrs_docs, dart_docs, kam_docs


def format_md_report(query: str, answer: str) -> str:
    """다운로드용 Markdown 리포트 생성."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# 회계처리 분석 리포트\n\n"
        f"> **생성일시**: {ts}  \n"
        f"> **질문**: {query}\n\n"
        "---\n\n"
        f"{answer}\n\n"
        "---\n\n"
        "*본 내용은 K-IFRS 기준서, DART 공시, 삼일회계법인 핵심감사사항 보고서를 기반으로 AI가 생성한 참고자료입니다.*"
    )


# ---------------------------------------------------------------------------
# 질문 입력 처리
# ---------------------------------------------------------------------------

# 사이드바 버튼에서 넘어온 질문 처리
pending = st.session_state.pop("pending_question", None)
query = st.chat_input("회계처리 질문을 입력하세요...") or pending

if query:
    # 사용자 메시지 표시 & 히스토리 추가
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        answer: str = ""
        kifrs_docs: list[Document] = []
        dart_docs: list[Document] = []
        kam_docs: list[Document] = []

        with st.spinner("분석 중..."):
            try:
                answer, kifrs_docs, dart_docs, kam_docs = build_answer(query)
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
                st.stop()

        st.markdown(answer)

        # 출처 Expander
        with st.expander("📎 검색된 원문 소스 보기"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**K-IFRS 기준서**")
                for doc in kifrs_docs:
                    src = doc.metadata.get("source", "")
                    pg = doc.metadata.get("page", "")
                    st.caption(f"{src} p.{pg}")
                    st.text(doc.page_content[:300] + "...")

            with col2:
                st.markdown("**DART 기업 공시**")
                if dart_docs:
                    for doc in dart_docs:
                        company = doc.metadata.get("company", "")
                        year = doc.metadata.get("year", "")
                        section = doc.metadata.get("section", "")
                        st.caption(f"{company} {year} — {section}")
                        st.text(doc.page_content[:300] + "...")
                else:
                    st.caption("DART 데이터 없음")

            with col3:
                st.markdown("**삼일회계법인 핵심감사사항(KAM) 보고서**")
                for doc in kam_docs:
                    src = doc.metadata.get("source", "")
                    pg = doc.metadata.get("page", "")
                    st.caption(f"{src} p.{pg}")
                    st.text(doc.page_content[:300] + "...")

        # .md 다운로드 버튼
        md_report = format_md_report(query, answer)
        st.download_button(
            label="📥 분석 결과 다운로드 (.md)",
            data=md_report.encode("utf-8"),
            file_name=f"회계처리_분석_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

    # 히스토리에 어시스턴트 응답 추가
    st.session_state["messages"].append({"role": "assistant", "content": answer})
