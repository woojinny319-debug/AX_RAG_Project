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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DART_API_BASE = "https://opendart.fss.or.kr/api"
DART_VIEWER_BASE = "https://dart.fss.or.kr"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "dart"

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

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "연구개발비 자산화": ["연구개발", "개발비", "자산화", "자본화", "경상연구개발비"],
    "무형자산 인식 및 상각": ["무형자산", "상각", "내용연수", "손상", "기술자산"],
    "수익인식 회계정책": ["수익인식", "매출", "계약", "수행의무", "거래가격"],
    "공정가치 평가": ["공정가치", "평가기법", "서열체계", "수준1", "수준2", "수준3"],
}

NOTE_PARENT_KEYWORDS = ["일반사항", "회사의 개요", "중요한 회계정책", "주석"]
NOTEISH_TITLE_RE = re.compile(
    r"(주석|회계정책|유의사항|중요한\s*회계|재무제표|부속명세|변동|추가정보|보충정보)"
)
CORP_CODE_MAP: dict[str, str] = {}

REPORT_LOOKBACK = 2
MAX_SECTION_NODES = 8
SECTION_CHUNK_SIZE = 2200
SECTION_CHUNK_OVERLAP = 250
TOPIC_WINDOW_LEFT = 1600
TOPIC_WINDOW_RIGHT = 4200
MERGED_BUNDLE_CHUNK_SIZE = 4000
MERGED_BUNDLE_OVERLAP = 350


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
        total=4,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://dart.fss.or.kr/",
        }
    )
    return session


SESSION = _build_session()


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def _load_corp_code_map(api_key: str) -> dict[str, str]:
    r = SESSION.get(f"{DART_API_BASE}/corpCode.xml", params={"crtfc_key": api_key}, timeout=45)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_bytes = zf.read("CORPCODE.xml")
    root = ET.fromstring(xml_bytes)
    result: dict[str, str] = {}
    for item in root.findall("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if stock_code and corp_code:
            result[stock_code] = corp_code
    return result


def _get_corp_code(api_key: str, stock_code: str) -> str | None:
    global CORP_CODE_MAP
    if not CORP_CODE_MAP:
        try:
            CORP_CODE_MAP = _load_corp_code_map(api_key)
        except Exception as e:
            print(f"[fail] corpCode load failed: {e}")
            return None
    return CORP_CODE_MAP.get(stock_code)


def _get_recent_reports(api_key: str, corp_code: str, limit: int = REPORT_LOOKBACK) -> list[dict[str, str]]:
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "pblntf_ty": "A",
        "bgn_de": "20220101",
        "end_de": "20261231",
        "page_count": "20",
    }
    try:
        r = SESSION.get(f"{DART_API_BASE}/list.json", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "000":
            print(f"[fail] list.json status={data.get('status')} message={data.get('message')}")
            return []

        rows = data.get("list", []) or []
        annual = [
            x for x in rows
            if "사업보고서" in (x.get("report_nm") or "")
            and "반기" not in (x.get("report_nm") or "")
            and "분기" not in (x.get("report_nm") or "")
        ]
        return [
            {
                "rcp_no": x.get("rcept_no", ""),
                "report_nm": x.get("report_nm", ""),
                "rcept_dt": x.get("rcept_dt", ""),
            }
            for x in annual[:limit]
            if x.get("rcept_no")
        ]
    except Exception as e:
        print(f"[fail] report list fetch failed: {e}")
        return []


def _fetch_left_panel(rcp_no: str) -> str:
    r = SESSION.get(
        f"{DART_VIEWER_BASE}/dsaf001/main.do",
        params={"rcpNo": rcp_no, "leftFrameDiv": "X"},
        timeout=40,
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
    for m in field_pattern.finditer(left_html):
        idx = m.group("idx")
        key = m.group("key")
        val = _decode_js_value(m.group("value"))
        buckets.setdefault(idx, {})[key] = val

    nodes: list[DocNode] = []
    for bucket in buckets.values():
        required = ["dcmNo", "eleId", "offset", "length", "dtd", "text"]
        if not all(k in bucket for k in required):
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

    max_len = max(n.length_int for n in nodes) or 1
    keyword_re = re.compile("|".join(re.escape(k) for k in NOTE_PARENT_KEYWORDS))
    notes_anchor_re = re.compile(r"(재무제표\s*)?주석")

    anchor_idx: int | None = None
    for i, n in enumerate(nodes):
        if notes_anchor_re.search(_normalize(n.title)):
            anchor_idx = i
            break

    scored: list[tuple[float, DocNode]] = []
    for i, n in enumerate(nodes):
        title = _normalize(n.title)
        ratio = n.length_int / max_len
        score = ratio
        if keyword_re.search(title):
            score += 0.5
        if notes_anchor_re.search(title):
            score += 0.35
        if anchor_idx is not None and i >= anchor_idx:
            score += 0.15
        if ratio >= 0.08 or notes_anchor_re.search(title) or keyword_re.search(title):
            scored.append((score, n))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    return max(nodes, key=lambda x: x.length_int)


def _select_note_nodes(nodes: list[DocNode]) -> list[DocNode]:
    if not nodes:
        return []

    max_len = max(n.length_int for n in nodes) or 1
    anchor_idx: int | None = None
    for i, n in enumerate(nodes):
        if NOTEISH_TITLE_RE.search(_normalize(n.title)):
            anchor_idx = i
            break

    candidates: list[tuple[float, DocNode]] = []
    for i, n in enumerate(nodes):
        title = _normalize(n.title)
        ratio = n.length_int / max_len
        score = ratio
        if any(re.search(re.escape(k), title) for k in NOTE_PARENT_KEYWORDS):
            score += 0.4
        if NOTEISH_TITLE_RE.search(title):
            score += 0.45
        if anchor_idx is not None and i >= anchor_idx:
            score += 0.12
        if ratio >= 0.06 or NOTEISH_TITLE_RE.search(title):
            candidates.append((score, n))

    candidates.sort(key=lambda x: x[0], reverse=True)
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


def _fetch_section_text(rcp_no: str, node: DocNode) -> str:
    params = {
        "rcpNo": rcp_no,
        "dcmNo": node.dcm_no,
        "eleId": node.ele_id,
        "offset": node.offset,
        "length": node.length,
        "dtd": node.dtd,
    }
    r = SESSION.get(f"{DART_VIEWER_BASE}/report/viewer.do", params=params, timeout=70)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_spans = sorted(spans, key=lambda x: x[0])
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
    for kw in keywords:
        for match in re.finditer(re.escape(kw), full_text):
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
    ids = [_build_doc_id(doc.metadata, i) for i, doc in enumerate(docs)]
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
    year = report_dt[:4] if report_dt else "2024"
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


def ingest_company(api_key: str, embeddings: OpenAIEmbeddings, stock_code: str, company_name: str) -> bool:
    corp_code = _get_corp_code(api_key, stock_code)
    if not corp_code:
        print(f"[skip] {company_name}: corp_code lookup failed")
        return False

    reports = _get_recent_reports(api_key, corp_code)
    if not reports:
        print(f"[skip] {company_name}: no recent annual report found")
        return False

    docs: list[Document] = []
    total_sections = 0
    total_topic_windows = 0

    for report in reports:
        sections, parent = _collect_sections_for_report(report["rcp_no"])
        if not sections:
            print(f"[skip] {company_name}: no sections for rcpNo={report['rcp_no']}")
            continue
        docs.extend(_build_documents_for_report(company_name, stock_code, corp_code, report, sections, parent))
        total_sections += len(sections)
        # Count topic windows in the generated docs for reporting only
        total_topic_windows += sum(1 for d in docs if d.metadata.get("section_type") == "topic_window" and d.metadata.get("rcpNo") == report["rcp_no"])

    if not docs:
        print(f"[skip] {company_name}: no documents extracted")
        return False

    _save_to_chroma(embeddings, docs)
    print(f"[ok] {company_name}: saved {len(docs)} docs (reports={len(reports)}, sections={total_sections}, topic_windows={total_topic_windows})")
    return True


def main() -> None:
    dart_key = os.getenv("DART_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if not dart_key or dart_key == "your_dart_api_key_here":
        print("[error] DART_API_KEY is missing.")
        sys.exit(1)
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("[error] OPENAI_API_KEY is missing.")
        sys.exit(1)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    success = 0
    for stock_code, company_name in TARGET_COMPANIES:
        try:
            if ingest_company(dart_key, embeddings, stock_code, company_name):
                success += 1
            time.sleep(0.4)
        except Exception as e:
            print(f"[error] {company_name}: {e}")

    print(f"ingest complete: {success}/{len(TARGET_COMPANIES)} companies")


if __name__ == "__main__":
    main()
