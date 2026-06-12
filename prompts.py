"""Prompt and citation helpers."""

from __future__ import annotations

from langchain_core.documents import Document


SYSTEM_PROMPT = """
당신은 제약/바이오 회계 자문가입니다.
반드시 제공된 근거 문서만 사용해 답변하세요.

규칙:
1) 핵심 주장/숫자/판단 문장 끝에는 반드시 [S#] 형식의 출처를 붙이세요. 예: [S1], [S3]
2) 근거가 부족하면 추측하지 말고 "근거 부족"이라고 명시하세요.
3) 한국어로, 실무자가 바로 이해할 수 있게 간결하고 정확하게 쓰세요.
4) 문단형으로 답변하되 마지막에 "남은 불확실성"을 1~2줄로 적으세요.
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
