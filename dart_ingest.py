"""DART ingest with broader note extraction and higher document volume."""

from __future__ import annotations

import html
import io
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DART_API_BASE = "https://opendart.fss.or.kr/api"
DART_VIEWER_BASE = "https://dart.fss.or.kr"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "dart"


@dataclass(frozen=True)
class TargetCompany:
    company_name: str
    stock_code: str | None = None
    aliases: tuple[str, ...] = ()


TARGET_COMPANIES: list[TargetCompany] = [
    TargetCompany("삼성바이오로직스", "207940"),
    TargetCompany("셀트리온", "068270"),
    TargetCompany("한미약품", "128940"),
    TargetCompany("유한양행", "000100"),
    TargetCompany("종근당", "185750"),
    TargetCompany("SK바이오팜", aliases=("에스케이바이오팜",)),
    TargetCompany("SK바이오사이언스", aliases=("에스케이바이오사이언스",)),
    TargetCompany("알테오젠"),
    TargetCompany("리가켐바이오", aliases=("레고켐바이오",)),
    TargetCompany("휴젤"),
    TargetCompany("GC녹십자", aliases=("녹십자",)),
    TargetCompany("대웅제약"),
    TargetCompany("보령", aliases=("보령제약",)),
    TargetCompany("HK이노엔"),
    TargetCompany("동아에스티"),
    TargetCompany("한올바이오파마"),
    TargetCompany("JW중외제약"),
    TargetCompany("동국제약"),
    TargetCompany("삼천당제약"),
    TargetCompany("메디톡스"),
    TargetCompany("에스티팜"),
    TargetCompany("차바이오텍"),
    TargetCompany("대원제약"),
    TargetCompany("부광약품"),
    TargetCompany("한국유나이티드제약", aliases=("유나이티드제약",)),
    TargetCompany("한독"),
    TargetCompany("안국약품"),
    TargetCompany("삼진제약"),
    TargetCompany("제일약품"),
    TargetCompany("일동제약"),
    TargetCompany("신풍제약"),
    TargetCompany("오스코텍"),
]

ACCOUNTING_TOPICS: list[str] = [
    "연구개발비 자산화",
    "무형자산 인식 및 상각",
    "수익인식 회계정책",
    "공정가치 평가",
]

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "연구개발비 자산화": ["연구개발", "개발비", "자산화", "자본화", "경상연구개발비"],
    "무형자산 인식 및 상각": ["무형자산", "상각", "내용연수", "손상", "기술자산"],
    "수익인식 회계정책": ["수익인식", "매출", "계약", "수행의무", "거래가격"],
    "공정가치 평가": ["공정가치", "평가기법", "서열체계", "수준1", "수준2", "수준3"],
}

NOTE_PARENT_KEYWORDS = ["일반사항", "회사 개요", "중요한 회계정책", "주석", "사업의 내용", "연구개발활동"]
NOTEISH_TITLE_RE = re.compile(
    r"(주석|회계정책|유의사항|중요한\s*회계|재무제표|부문정보|추가정보|보충정보|사업의\s*내용|연구개발)"
)
ANNUAL_REPORT_YEAR_RE = re.compile(r"사업보고서\s*\((\d{4})\.\d{2}\)")

CORP_CODE_MAP: dict[str, str] = {}
CORP_NAME_MAP: dict[str, tuple[str, str, str]] = {}

TARGET_REPORT_YEARS = ("2025", "2024")
REPORT_LOOKBACK = len(TARGET_REPORT_YEARS)
MAX_SECTION_NODES = 12
SECTION_CHUNK_SIZE = 1000
SECTION_CHUNK_OVERLAP = 150
TOPIC_WINDOW_LEFT = 800
TOPIC_WINDOW_RIGHT = 1200
MERGED_BUNDLE_CHUNK_SIZE = 1200
MERGED_BUNDLE_OVERLAP = 150


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
        try:
            return int(self.length)
        except Exception:
            return 0


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://dart.fss.or.kr/",
        }
    )
    return session


SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    global SESSION
    if SESSION is None:
        SESSION = _build_session()
    return SESSION


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def _normalize_name(text: str) -> str:
    return re.sub(r"\s+", "", _normalize(text))


def _load_corp_reference_map(api_key: str) -> tuple[dict[str, str], dict[str, tuple[str, str, str]]]:
    print("[info] 기업코드 정보 로드 시작 (최대 30초)")
    try:
        r = _get_session().get(f"{DART_API_BASE}/corpCode.xml", params={"crtfc_key": api_key}, timeout=30)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        print("[error] 기업코드 API timeout - 네트워크 확인 후 재시도하세요")
        raise
    except Exception as e:
        print(f"[error] 기업코드 API 호출 실패: {e}")
        raise
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_bytes = zf.read("CORPCODE.xml")

    root = ET.fromstring(xml_bytes)
    by_stock: dict[str, str] = {}
    by_name: dict[str, tuple[str, str, str]] = {}

    for item in root.findall("list"):
        corp_name = _normalize(item.findtext("corp_name") or "")
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if corp_code and stock_code:
            by_stock[stock_code] = corp_code
            by_name[_normalize_name(corp_name)] = (corp_code, stock_code, corp_name)

    return by_stock, by_name


def _ensure_corp_reference_map(api_key: str) -> bool:
    global CORP_CODE_MAP, CORP_NAME_MAP
    if CORP_CODE_MAP and CORP_NAME_MAP:
        return True
    try:
        CORP_CODE_MAP, CORP_NAME_MAP = _load_corp_reference_map(api_key)
        return True
    except Exception as e:
        print(f"[fail] corpCode load failed: {e}")
        return False


def _find_company_reference(
    api_key: str,
    company_name: str,
    stock_code: str | None = None,
    aliases: tuple[str, ...] = (),
) -> tuple[str, str, str] | None:
    if not _ensure_corp_reference_map(api_key):
        return None

    if stock_code:
        corp_code = CORP_CODE_MAP.get(stock_code)
        if corp_code:
            return corp_code, stock_code, company_name

    candidates = [company_name, *aliases]
    for candidate in candidates:
        ref = CORP_NAME_MAP.get(_normalize_name(candidate))
        if ref:
            return ref

    normalized_candidates = {_normalize_name(name) for name in candidates}
    for corp_name_key, ref in CORP_NAME_MAP.items():
        if any(candidate in corp_name_key or corp_name_key in candidate for candidate in normalized_candidates):
            return ref

    return None


def _get_recent_reports(api_key: str, corp_code: str, limit: int = REPORT_LOOKBACK) -> list[dict[str, str]]:
    target_receipt_years = {str(int(year) + 1) for year in TARGET_REPORT_YEARS}
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "pblntf_ty": "A",
        "bgn_de": f"{min(target_receipt_years)}0101",
        "end_de": f"{max(target_receipt_years)}1231",
        "page_count": "50",
    }
    try:
        r = _get_session().get(f"{DART_API_BASE}/list.json", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "000":
            print(f"[fail] list.json status={data.get('status')} message={data.get('message')}")
            return []

        rows = data.get("list", []) or []
        annual: list[dict[str, str]] = []
        for row in rows:
            report_nm = row.get("report_nm") or ""
            if "사업보고서" not in report_nm:
                continue
            if "반기" in report_nm or "분기" in report_nm:
                continue
            match = ANNUAL_REPORT_YEAR_RE.search(report_nm)
            if not match:
                continue
            report_year = match.group(1)
            if report_year not in TARGET_REPORT_YEARS:
                continue
            if not row.get("rcept_no"):
                continue
            annual.append(
                {
                    "rcp_no": row.get("rcept_no", ""),
                    "report_nm": report_nm,
                    "rcept_dt": row.get("rcept_dt", ""),
                    "report_year": report_year,
                }
            )

        annual.sort(key=lambda item: TARGET_REPORT_YEARS.index(item["report_year"]))
        return annual[:limit]
    except Exception as e:
        print(f"[fail] report list fetch failed: {e}")
        return []


def _fetch_left_panel(rcp_no: str) -> str:
    time.sleep(1)  # Rate limiting 대비
    r = _get_session().get(
        f"{DART_VIEWER_BASE}/dsaf001/main.do",
        params={"rcpNo": rcp_no, "leftFrameDiv": "X"},
        timeout=20,
    )
    r.raise_for_status()
    return r.text


def _decode_js_value(value: str) -> str:
    value = value.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
    value = html.unescape(value)
    return _normalize(value)


def _parse_doc_nodes(left_html: str) -> list[DocNode]:
    field_pattern = re.compile(r"node(?P<idx>\d+)\['(?P<key>[^']+)'\]\s*=\s*\"(?P<value>.*?)\";", re.DOTALL)
    buckets: dict[str, dict[str, str]] = {}
    for match in field_pattern.finditer(left_html):
        idx = match.group("idx")
        key = match.group("key")
        val = _decode_js_value(match.group("value"))
        buckets.setdefault(idx, {})[key] = val

    nodes: list[DocNode] = []
    for bucket in buckets.values():
        required = ["dcmNo", "eleId", "offset", "length", "dtd", "text"]
        if not all(key in bucket for key in required):
            continue
        nodes.append(
            DocNode(
                dcm_no=bucket["dcmNo"],
                ele_id=bucket["eleId"],
                offset=bucket["offset"],
                length=bucket["length"],
                dtd=bucket["dtd"],
                title=bucket["text"],
            )
        )
    return nodes


def _find_notes_parent_node(nodes: list[DocNode]) -> DocNode | None:
    if not nodes:
        return None

    max_len = max(node.length_int for node in nodes) or 1
    keyword_re = re.compile("|".join(re.escape(keyword) for keyword in NOTE_PARENT_KEYWORDS))
    notes_anchor_re = re.compile(r"(재무제표\s*)?주석")

    anchor_idx: int | None = None
    for index, node in enumerate(nodes):
        if notes_anchor_re.search(_normalize(node.title)):
            anchor_idx = index
            break

    scored: list[tuple[float, DocNode]] = []
    for index, node in enumerate(nodes):
        title = _normalize(node.title)
        ratio = node.length_int / max_len
        score = ratio
        if keyword_re.search(title):
            score += 0.5
        if notes_anchor_re.search(title):
            score += 0.35
        if anchor_idx is not None and index >= anchor_idx:
            score += 0.15
        if ratio >= 0.08 or notes_anchor_re.search(title) or keyword_re.search(title):
            scored.append((score, node))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
    return max(nodes, key=lambda item: item.length_int)


def _select_note_nodes(nodes: list[DocNode]) -> list[DocNode]:
    if not nodes:
        return []

    max_len = max(node.length_int for node in nodes) or 1
    anchor_idx: int | None = None
    for index, node in enumerate(nodes):
        if NOTEISH_TITLE_RE.search(_normalize(node.title)):
            anchor_idx = index
            break

    candidates: list[tuple[float, DocNode]] = []
    for index, node in enumerate(nodes):
        title = _normalize(node.title)
        ratio = node.length_int / max_len
        score = ratio
        if any(re.search(re.escape(keyword), title) for keyword in NOTE_PARENT_KEYWORDS):
            score += 0.4
        if NOTEISH_TITLE_RE.search(title):
            score += 0.45
        if anchor_idx is not None and index >= anchor_idx:
            score += 0.12
        if ratio >= 0.06 or NOTEISH_TITLE_RE.search(title):
            candidates.append((score, node))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[DocNode] = []
    seen: set[str] = set()
    for _, node in candidates:
        key = _normalize(node.title)
        if key in seen:
            continue
        seen.add(key)
        selected.append(node)
        if len(selected) >= MAX_SECTION_NODES:
            break
    return selected


def _html_to_markdown(soup: BeautifulSoup) -> str:
    # Convert tables to markdown
    for table in soup.find_all("table"):
        markdown_table = []
        rows = table.find_all("tr")
        is_first_row = True
        for row in rows:
            cols = row.find_all(["th", "td"])
            if not cols:
                continue
            # Extract text, replace newlines and pipes to avoid breaking markdown table
            row_text = [col.get_text(separator=" ", strip=True).replace("\n", " ").replace("|", ",") for col in cols]
            markdown_table.append("| " + " | ".join(row_text) + " |")
            if is_first_row:
                markdown_table.append("|" + "|".join(["---"] * len(row_text)) + "|")
                is_first_row = False
        
        if markdown_table:
            # Insert markdown string before the table
            table.insert_before(soup.new_string("\n\n" + "\n".join(markdown_table) + "\n\n"))
        table.decompose()
        
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _fetch_section_text(rcp_no: str, node: DocNode) -> str:
    time.sleep(1)  # Rate limiting 대비
    params = {
        "rcpNo": rcp_no,
        "dcmNo": node.dcm_no,
        "eleId": node.ele_id,
        "offset": node.offset,
        "length": node.length,
        "dtd": node.dtd,
    }
    r = _get_session().get(f"{DART_VIEWER_BASE}/report/viewer.do", params=params, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return _html_to_markdown(soup)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_spans = sorted(spans, key=lambda item: item[0])
    if not sorted_spans:
        return []

    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _extract_topic_chunks(full_text: str, topic: str) -> list[str]:
    keywords = TOPIC_KEYWORDS.get(topic, [topic])
    spans: list[tuple[int, int]] = []
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), full_text):
            start = max(0, match.start() - TOPIC_WINDOW_LEFT)
            end = min(len(full_text), match.end() + TOPIC_WINDOW_RIGHT)
            spans.append((start, end))
    if not spans:
        return []

    chunks: list[str] = []
    for start, end in _merge_spans(spans):
        text = full_text[start:end].strip()
        if text:
            chunks.append(text)
    return chunks


def _build_doc_id(meta: dict[str, str], idx: int) -> str:
    company = re.sub(r"\s+", "", meta.get("company", "unknown"))
    year = meta.get("year", "unknown")
    section = re.sub(r"\s+", "_", meta.get("section", "unknown"))
    rcp_no = meta.get("rcpNo", "unknown")
    section_type = meta.get("section_type", "chunk")
    return f"{company}_{year}_{section}_{section_type}_{rcp_no}_{idx}"


def _save_to_chroma(embeddings: OpenAIEmbeddings, docs: list[Document]) -> None:
    if not docs:
        return

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    ids = [_build_doc_id(doc.metadata, index) for index, doc in enumerate(docs)]
    try:
        store.delete(ids=ids)
    except Exception:
        pass
    store.add_documents(docs, ids=ids)


def _build_documents_for_report(
    company_name: str,
    stock_code: str,
    corp_code: str,
    report: dict[str, str],
    sections: list[tuple[DocNode, str]],
    parent: DocNode | None,
) -> list[Document]:
    rcp_no = report["rcp_no"]
    report_nm = report.get("report_nm", "")
    report_dt = report.get("rcept_dt", "")
    year = report.get("report_year") or (report_dt[:4] if report_dt else "2025")
    docs: list[Document] = []

    merged_notes = "\n\n".join(text for _, text in sections)

    for node, text in sections:
        for idx, chunk in enumerate(_chunk_text(text, SECTION_CHUNK_SIZE, SECTION_CHUNK_OVERLAP)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": f"DART_{company_name}_{year}",
                        "company": company_name,
                        "ticker": stock_code,
                        "corp_code": corp_code,
                        "rcpNo": rcp_no,
                        "report_nm": report_nm,
                        "report_dt": report_dt,
                        "year": year,
                        "section": node.title,
                        "chunk_index": str(idx),
                        "section_type": "note_section",
                        "source_url": f"{DART_VIEWER_BASE}/dsaf001/main.do?rcpNo={rcp_no}",
                        "node_title": parent.title if parent else "",
                    },
                )
            )

    for idx, chunk in enumerate(_chunk_text(merged_notes, MERGED_BUNDLE_CHUNK_SIZE, MERGED_BUNDLE_OVERLAP)):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": f"DART_{company_name}_{year}",
                    "company": company_name,
                    "ticker": stock_code,
                    "corp_code": corp_code,
                    "rcpNo": rcp_no,
                    "report_nm": report_nm,
                    "report_dt": report_dt,
                    "year": year,
                    "section": "merged_notes",
                    "chunk_index": str(idx),
                    "section_type": "notes_bundle",
                    "source_url": f"{DART_VIEWER_BASE}/dsaf001/main.do?rcpNo={rcp_no}",
                    "node_title": parent.title if parent else "",
                },
            )
        )

    for topic in ACCOUNTING_TOPICS:
        for idx, chunk in enumerate(_extract_topic_chunks(merged_notes, topic)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": f"DART_{company_name}_{year}",
                        "company": company_name,
                        "ticker": stock_code,
                        "corp_code": corp_code,
                        "rcpNo": rcp_no,
                        "report_nm": report_nm,
                        "report_dt": report_dt,
                        "year": year,
                        "section": topic,
                        "chunk_index": str(idx),
                        "section_type": "topic_window",
                        "source_url": f"{DART_VIEWER_BASE}/dsaf001/main.do?rcpNo={rcp_no}",
                        "node_title": parent.title if parent else "",
                    },
                )
            )

    return docs


def _collect_sections_for_report(rcp_no: str) -> tuple[list[tuple[DocNode, str]], DocNode | None]:
    try:
        left_html = _fetch_left_panel(rcp_no)
    except Exception as e:
        print(f"[fail] left panel fetch failed: {e}")
        return [], None

    nodes = _parse_doc_nodes(left_html)
    if not nodes:
        print("[fail] node parsing failed: 0 nodes")
        return [], None

    parent = _find_notes_parent_node(nodes)
    selected_nodes = _select_note_nodes(nodes)
    if not selected_nodes:
        print("[fail] no candidate note nodes found")
        return [], parent

    sections: list[tuple[DocNode, str]] = []
    for node in selected_nodes:
        try:
            text = _fetch_section_text(rcp_no, node)
            if text.strip():
                sections.append((node, text))
        except Exception as e:
            print(f"[warn] section fetch failed ({node.title}): {e}")
    return sections, parent


def ingest_company(api_key: str, embeddings: OpenAIEmbeddings, target: TargetCompany) -> bool:
    ref = _find_company_reference(api_key, target.company_name, target.stock_code, target.aliases)
    if not ref:
        print(f"[skip] {target.company_name}: corp_code lookup failed")
        return False

    corp_code, stock_code, resolved_name = ref
    reports = _get_recent_reports(api_key, corp_code)
    if not reports:
        print(f"[skip] {target.company_name}: no target annual report found")
        return False

    docs: list[Document] = []
    total_sections = 0
    total_topic_windows = 0

    for report_idx, report in enumerate(reports):
        if report_idx > 0:
            time.sleep(2)  # 보고서 간 대기 대비
        sections, parent = _collect_sections_for_report(report["rcp_no"])
        if not sections:
            print(f"[skip] {target.company_name}: no sections for rcpNo={report['rcp_no']}")
            continue
        docs.extend(_build_documents_for_report(target.company_name, stock_code, corp_code, report, sections, parent))
        total_sections += len(sections)
        total_topic_windows += sum(
            1
            for doc in docs
            if doc.metadata.get("section_type") == "topic_window" and doc.metadata.get("rcpNo") == report["rcp_no"]
        )

    if not docs:
        print(f"[skip] {target.company_name}: no documents extracted")
        return False

    _save_to_chroma(embeddings, docs)
    print(
        f"[ok] {target.company_name}: saved {len(docs)} docs "
        f"(resolved={resolved_name}, reports={len(reports)}, sections={total_sections}, topic_windows={total_topic_windows})"
    )
    return True


def main() -> None:
    print("[시작] DART 데이터 수집 시작")
    dart_key = os.getenv("DART_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if not dart_key or dart_key == "your_dart_api_key_here":
        print("[error] DART_API_KEY is missing.")
        sys.exit(1)
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("[error] OPENAI_API_KEY is missing.")
        sys.exit(1)

    print("[info] 임베딩 모델 초기화 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("[info] 임베딩 모델 준비 완료")
    
    success = 0
    for idx, target in enumerate(TARGET_COMPANIES, 1):
        print(f"\n[진행] {idx}/{len(TARGET_COMPANIES)}: {target.company_name} 처리 중...", flush=True)
        try:
            if ingest_company(dart_key, embeddings, target):
                success += 1
            time.sleep(3)  # 회사 간 대기 대비 (rate limiting 방지)
        except Exception as e:
            print(f"[error] {target.company_name}: {e}")
            time.sleep(5)  # 에러 발생 시 더 대기

    print(f"\n[완료] ingest complete: {success}/{len(TARGET_COMPANIES)} companies")


if __name__ == "__main__":
    main()
