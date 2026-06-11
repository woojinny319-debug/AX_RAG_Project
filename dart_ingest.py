from __future__ import annotations

import io
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

DART_API_BASE = "https://opendart.fss.or.kr/api"
DART_VIEWER_BASE = "https://dart.fss.or.kr"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION = "dart"

_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://dart.fss.or.kr/",
}

TARGET_COMPANIES: list[tuple[str, str]] = [
    ("207940", "삼성바이오로직스"),
    ("068270", "셀트리온"),
    ("128940", "한미약품"),
    ("000100", "유한양행"),
    ("185750", "종근당"),
]

ACCOUNTING_TOPICS: list[str] = [
    "연구개발비 자산화",
    "무형자산 인식 및 상각",
    "수익인식 회계정책",
    "공정가치 평가",
]


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class DocNode:
    dcm_no: str
    ele_id: str
    offset: str
    length: str
    dtd: str
    title: str

    @property
    def length_int(self) -> int:
        return int(self.length)

    @property
    def offset_int(self) -> int:
        return int(self.offset)


# ---------------------------------------------------------------------------
# DART Open API (corp_code / rcpNo 조회)
# ---------------------------------------------------------------------------

_CORP_CODE_MAP: dict[str, str] = {}


def _load_corp_code_map(api_key: str) -> dict[str, str]:
    url = f"{DART_API_BASE}/corpCode.xml"
    r = requests.get(url, params={"crtfc_key": api_key}, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_bytes = zf.read("CORPCODE.xml")
    root = ET.fromstring(xml_bytes)
    mapping: dict[str, str] = {}
    for item in root.findall("list"):
        sc = (item.findtext("stock_code") or "").strip()
        cc = (item.findtext("corp_code") or "").strip()
        if sc and cc:
            mapping[sc] = cc
    return mapping


def _get_corp_code(api_key: str, stock_code: str) -> str | None:
    global _CORP_CODE_MAP
    if not _CORP_CODE_MAP:
        print("  corp_code 전체 목록 다운로드 중...")
        try:
            _CORP_CODE_MAP = _load_corp_code_map(api_key)
            print(f"  총 {len(_CORP_CODE_MAP)}개 기업 로드 완료")
        except Exception as e:
            print(f"  [오류] corpCode.xml 다운로드 실패: {e}")
            return None
    return _CORP_CODE_MAP.get(stock_code)


def _get_recent_rcp_no(api_key: str, corp_code: str) -> str | None:
    url = f"{DART_API_BASE}/list.json"
    params: dict[str, str] = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "pblntf_ty": "A",
        "bgn_de": "20230101",
        "end_de": "20251231",
        "page_count": "10",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "000":
            items: list[dict[str, str]] = data.get("list", [])
            annual = [
                it for it in items
                if "사업보고서" in it.get("report_nm", "")
                and "분기" not in it.get("report_nm", "")
                and "반기" not in it.get("report_nm", "")
            ]
            if annual:
                return annual[0].get("rcept_no")
    except Exception as e:
        print(f"  [오류] 공시 목록 조회 실패: {e}")
    return None


# ---------------------------------------------------------------------------
# DART 뷰어 HTML 스크래핑
# ---------------------------------------------------------------------------

def _fetch_left_panel(rcp_no: str) -> str:
    url = f"{DART_VIEWER_BASE}/dsaf001/main.do"
    r = requests.get(
        url,
        params={"rcpNo": rcp_no, "leftFrameDiv": "X"},
        headers=_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.text


def _parse_doc_nodes(left_html: str) -> list[DocNode]:
    """좌측 패널 JS 변수에서 (dcmNo, eleId, offset, length, dtd, title) 추출."""
    pattern = re.compile(
        r"node\d+\['dcmNo'\]\s*=\s*\"([0-9]+)\".*?"
        r"node\d+\['eleId'\]\s*=\s*\"([0-9]+)\".*?"
        r"node\d+\['offset'\]\s*=\s*\"([0-9]+)\".*?"
        r"node\d+\['length'\]\s*=\s*\"([0-9]+)\".*?"
        r"node\d+\['dtd'\]\s*=\s*\"([^\"]+)\".*?"
        r"node\d+\['text'\]\s*=\s*\"([^\"]+)\"",
        re.DOTALL,
    )
    return [
        DocNode(
            dcm_no=m.group(1),
            ele_id=m.group(2),
            offset=m.group(3),
            length=m.group(4),
            dtd=m.group(5),
            title=m.group(6),
        )
        for m in pattern.finditer(left_html)
    ]


def _find_notes_parent_node(nodes: list[DocNode]) -> DocNode | None:
   
   
    for node in nodes:
        if "일반사항" in node.title and node.length_int > 500_000:
            return node

  
    after_jujuk = False
    for node in nodes:
        if "주석" in node.title and node.length_int < 200_000:
            after_jujuk = True
            continue
        if after_jujuk and node.length_int > 500_000:
            return node

    candidates = [n for n in nodes if "주석" in n.title or "일반사항" in n.title]
    if candidates:
        return max(candidates, key=lambda n: n.length_int)

    return None


def _fetch_section_text(rcp_no: str, node: DocNode) -> str:

    url = f"{DART_VIEWER_BASE}/report/viewer.do"
    params: dict[str, str] = {
        "rcpNo": rcp_no,
        "dcmNo": node.dcm_no,
        "eleId": node.ele_id,
        "offset": node.offset,
        "length": node.length,
        "dtd": node.dtd,
    }
    r = requests.get(url, params=params, headers=_HEADERS, timeout=60)
    r.raise_for_status()
    if not r.text.strip():
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _collect_notes_text(rcp_no: str) -> str:
    """DART 뷰어에서 사업보고서 주석 전체 텍스트를 수집한다."""
    print("  좌측 패널 목차 수집 중...")
    try:
        left_html = _fetch_left_panel(rcp_no)
    except Exception as e:
        print(f"  [오류] 좌측 패널 수집 실패: {e}")
        return ""

    nodes = _parse_doc_nodes(left_html)
    print(f"  전체 섹션 수: {len(nodes)}")

    parent = _find_notes_parent_node(nodes)
    if not parent:
        print("  [오류] 주석 부모 노드를 찾지 못했습니다.")
        return ""

    print(f"  주석 부모 노드: '{parent.title}' (eleId={parent.ele_id}, length={parent.length})")

    try:
        text = _fetch_section_text(rcp_no, parent)
    except Exception as e:
        print(f"  [오류] 주석 텍스트 수집 실패: {e}")
        return ""

    return text


# ---------------------------------------------------------------------------
# 텍스트 섹션 분할
# ---------------------------------------------------------------------------

_SECTION_HEADING_RE = re.compile(
    r"^(?:\d+[\.\d]*\s+)?(?:중요한\s*)?(?:회계정책|연구개발|무형자산|수익인식|공정가치|"
    r"회계추정|개발비|기술적.*타당성|자산화).*$",
    re.MULTILINE,
)


def _extract_topic_chunk(full_text: str, topic: str) -> str:

    topic_keywords: dict[str, list[str]] = {
        "연구개발비 자산화": ["연구개발", "개발비", "자산화", "임상"],
        "무형자산 인식 및 상각": ["무형자산", "상각", "내용연수"],
        "수익인식 회계정책": ["수익인식", "수익", "거래가격", "수행의무"],
        "공정가치 평가": ["공정가치", "공정 가치", "평가기법"],
    }
    keywords = topic_keywords.get(topic, [topic])

    # 키워드 주변 ±2000자 조각을 모아 합산
    chunks: list[str] = []
    for kw in keywords:
        idx = 0
        while True:
            pos = full_text.find(kw, idx)
            if pos < 0:
                break
            start = max(0, pos - 500)
            end = min(len(full_text), pos + 2000)
            chunks.append(full_text[start:end])
            idx = pos + 1

    if not chunks:
        return ""

    # 중복 제거 후 합산 (최대 8000자)
    combined = "\n\n---\n\n".join(dict.fromkeys(chunks))
    return combined[:8000]


# ---------------------------------------------------------------------------
# LLM 정제
# ---------------------------------------------------------------------------

def _refine_with_llm(llm: ChatOpenAI, chunk: str, topic: str, company: str) -> str:
    messages = [
        SystemMessage(content=(
            "당신은 재무회계 전문가입니다. "
            "다음 사업보고서 주석 텍스트에서 지정된 회계 주제와 관련된 "
            "회계정책 및 처리 방법 서술 부분만 추출하십시오. "
            "원문 표현을 최대한 유지하고, 관련 없는 내용은 제거하십시오. "
            "관련 내용이 없으면 '해당 주제 관련 내용 없음'이라고 답하십시오."
        )),
        HumanMessage(content=f"[회사] {company}\n[주제] {topic}\n\n[주석 원문]\n{chunk}"),
    ]
    try:
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as e:
        print(f"  [오류] LLM 정제 실패 ({topic}): {e}")
        return ""


# ---------------------------------------------------------------------------
# ChromaDB 저장
# ---------------------------------------------------------------------------

def _save_to_chroma(embeddings: OpenAIEmbeddings, docs: list[Document]) -> None:
    if not docs:
        return
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.add_documents(docs)


# ---------------------------------------------------------------------------
# 기업별 인제스트
# ---------------------------------------------------------------------------

def ingest_company(
    api_key: str,
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    stock_code: str,
    company_name: str,
) -> bool:
    print(f"\n{'='*50}")
    print(f"처리 중: {company_name} ({stock_code})")

    corp_code = _get_corp_code(api_key, stock_code)
    if not corp_code:
        print("  [스킵] corp_code 조회 실패")
        return False
    print(f"  corp_code: {corp_code}")

    rcp_no = _get_recent_rcp_no(api_key, corp_code)
    if not rcp_no:
        print("  [스킵] 최근 사업보고서 없음")
        return False
    print(f"  rcpNo: {rcp_no}")

    time.sleep(0.5)

    notes_text = _collect_notes_text(rcp_no)
    if not notes_text.strip():
        print("  [스킵] 주석 텍스트 추출 실패")
        return False
    print(f"  수집된 주석 텍스트 길이: {len(notes_text):,} 자")

    year = "2024"
    docs: list[Document] = []
    saved_count = 0

    for topic in ACCOUNTING_TOPICS:
        print(f"  주제 정제: {topic}")
        chunk = _extract_topic_chunk(notes_text, topic)
        if not chunk:
            print("    → 관련 키워드 없음, 스킵")
            continue

        refined = _refine_with_llm(llm, chunk, topic, company_name)
        if not refined or "해당 주제 관련 내용 없음" in refined:
            print("    → LLM: 관련 내용 없음, 스킵")
            continue

        docs.append(Document(
            page_content=refined,
            metadata={
                "company": company_name,
                "ticker": stock_code,
                "corp_code": corp_code,
                "rcpNo": rcp_no,
                "year": year,
                "section": topic,
                "source": f"DART_{company_name}_{year}",
            },
        ))
        saved_count += 1
        time.sleep(0.3)

    if docs:
        _save_to_chroma(embeddings, docs)
        print(f"  저장 완료: {saved_count}개 섹션")
        return True

    print("  [스킵] 저장할 내용 없음")
    return False


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    dart_key = os.getenv("DART_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if not dart_key or dart_key == "your_dart_api_key_here":
        print("[오류] DART_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    if not openai_key or openai_key == "your_openai_api_key_here":
        print("[오류] OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    print("DART 인제스트 시작 (뷰어 스크래핑 방식)")
    print(f"대상 기업: {[name for _, name in TARGET_COMPANIES]}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    success_count = 0
    for stock_code, company_name in TARGET_COMPANIES:
        try:
            ok = ingest_company(dart_key, llm, embeddings, stock_code, company_name)
            if ok:
                success_count += 1
        except Exception as e:
            print(f"  [오류] {company_name} 처리 중 예외: {e}")

    print(f"\n{'='*50}")
    print(f"인제스트 완료: {success_count}/{len(TARGET_COMPANIES)} 기업 성공")
    if success_count == 0:
        print("[경고] 저장된 기업이 없습니다. 네트워크 연결을 확인하세요.")


if __name__ == "__main__":
    main()
