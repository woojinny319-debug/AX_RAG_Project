"""Prompt and citation helpers."""

from __future__ import annotations

from langchain_core.documents import Document


SYSTEM_PROMPT = """
당신은 제약/바이오 회계 및 공시 데이터(DART) 전문 AI 자문가입니다.
제공된 근거 문서를 바탕으로 질문에 답변하되, 질문의 의도를 폭넓게 해석하여 유연하고 풍부하게 답변하세요.

[현재 DART 검색이 가능한 32개 대상 기업]
삼성바이오로직스, 셀트리온, 한미약품, 유한양행, 종근당, SK바이오팜, SK바이오사이언스, 알테오젠, 리가켐바이오, 휴젤, GC녹십자, 대웅제약, 보령, HK이노엔, 동아에스티, 한올바이오파마, JW중외제약, 동국제약, 삼천당제약, 메디톡스, 에스티팜, 차바이오텍, 대원제약, 부광약품, 한국유나이티드제약, 한독, 안국약품, 삼진제약, 제일약품, 일동제약, 신풍제약, 오스코텍

[답변 규칙]
1) 핵심 주장, 수치, 판단, 또는 기업 공시 사례를 언급할 때 문장 끝에 반드시 [S#] 형식의 출처를 붙이세요. (예: [S1], [S3])
2) 근거 문서에 마크다운 형식의 표(Table)가 포함되어 있을 경우, 표의 행(Row)과 열(Column) 구조를 면밀히 분석하세요. "연구개발비", "자산화", "신약명(프로젝트명)" 등 구체적인 텍스트나 숫자(금액)가 표 안에 존재한다면 절대 누락하지 말고 반드시 답변에 구체적으로 인용하세요.
3) 만약 "아무거나 보여줘", "예시를 들어줘"와 같이 포괄적으로 질문한 경우, 제공된 근거 문서 내에 있는 어떤 기업의 사례라도 적극적으로 활용하여 답변하세요.
4) 제공된 근거 문서에 명시적인 정답이 없더라도, 회계 기준(K-IFRS)이나 다른 기업의 사례를 바탕으로 논리적인 추론이나 일반적인 원칙을 설명해주세요. 단, 추론인 경우 "근거 문서에 직접적인 언급은 없으나..."와 같이 밝히세요.
5) 대상 기업 리스트에 대해 묻는다면 위 32개 기업 목록을 활용하여 안내해주세요.
6) 한국어로, 실무자가 바로 이해할 수 있게 간결하고 정확하게 문단형으로 작성하세요.
7) 답변의 마지막에는 항상 "남은 불확실성"을 1~2줄로 적어 추가 확인이 필요한 부분을 명시하세요.
"""


def build_user_prompt(query: str, source_context: str) -> str:
    return f"""[질문]
{query}

[근거 문서]
{source_context}
"""


def format_cited_docs(docs: list[Document], start_index: int) -> tuple[str, list[dict[str, str]]]:
    """Return prompt context + source catalog with stable source ids."""
    lines: list[str] = []
    catalog: list[dict[str, str]] = []

    for i, doc in enumerate(docs, start=start_index):
        sid = f"S{i}"
        meta = doc.metadata or {}
        source_name = str(meta.get("source", "unknown"))
        page = str(meta.get("page", ""))
        company = str(meta.get("company", ""))
        section = str(meta.get("section", ""))
        url = str(meta.get("source_url", ""))

        header_parts = [f"[{sid}]"]
        if company:
            header_parts.append(f"{company}")
        if section:
            header_parts.append(f"섹션:{section}")
        if page:
            header_parts.append(f"p.{page}")
        header_parts.append(f"src:{source_name}")
        lines.append(" ".join(header_parts))
        lines.append(doc.page_content.strip())
        lines.append("")

        catalog.append(
            {
                "sid": sid,
                "source": source_name,
                "company": company,
                "section": section,
                "page": page,
                "url": url,
            }
        )

    return "\n".join(lines).strip(), catalog
