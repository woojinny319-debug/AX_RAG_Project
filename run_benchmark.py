"""고정 벤치마크 5문항 — 모델 개선 전후 성능을 반복 측정하기 위한 회귀 하니스.

각 개선(프롬프트/검색/수집) 후 이 스크립트를 돌려 Notes/Benchmarks/ 에 타임스탬프 결과를
저장하고, 버전 간 답변·검색품질을 나란히 비교한다.

실행:  python run_benchmark.py [라벨]
   예:  python run_benchmark.py baseline
        python run_benchmark.py after_Q3_fix
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts import SYSTEM_PROMPT, build_user_prompt, format_cited_docs
from rag_engine import get_dart_retriever, get_kam_retriever, get_kifrs_retriever

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
OUT_DIR = BASE / "Notes" / "Benchmarks"

# 고정 벤치마크: 각 문항이 서로 다른 역량 차원을 측정한다. (순서/문구를 함부로 바꾸지 말 것)
BENCHMARK = [
    {
        "id": "Q1",
        "dim": "정량 추출·표 독해 (DART 주석)",
        "q": "삼천당제약의 최근 사업보고서 기준 연구개발비 총액과, 그중 무형자산으로 자산화한 금액·당기비용 처리 금액을 연도별로 구체적인 원화 금액과 함께 알려주세요.",
        "rubric": "정확한 표 수치 / 전체 원화 환산 표기 / 연도(2025·2024) 1:1 명시",
    },
    {
        "id": "Q2",
        "dim": "기준서↔공시 연결 (K-IFRS + DART)",
        "q": "K-IFRS 제1038호상 개발비를 무형자산으로 자산화하기 위한 요건을 설명하고, 이 요건을 충족했다고 실제로 공시한 제약·바이오 기업 사례를 DART에서 찾아 연결해주세요.",
        "rubric": "1038호 자산화 요건 설명 / 실제 기업명 / 그 기업의 공시 근거를 요건과 연결",
    },
    {
        "id": "Q3",
        "dim": "기업 간 비교 (다중 기업·현재 약점)",
        "q": "셀트리온과 한미약품의 연구개발비 자산화 정책(어느 단계부터 자산화하는지)을 비교하고, 두 회사의 회계처리 접근이 어떻게 다른지 설명해주세요.",
        "rubric": "두 기업 각각의 정책을 근거와 함께 / 차이점 대조 / 두 기업 DART 문서가 모두 검색됨",
    },
    {
        "id": "Q4",
        "dim": "감사 관점(KAM) → 실제 공시 연결",
        "q": "제약·바이오 기업의 연구개발비 자산화에서 감사인이 핵심감사사항(KAM)으로 주목하는 위험은 무엇이며, 그 위험이 왜 중요한지 삼일회계법인 자료를 근거로 설명해주세요. 그리고 해당 위험과 관련된 회계처리를 실제로 공시한 기업을 DART에서 찾아 실제 공시 내용을 함께 출력해주세요.",
        "rubric": "KAM 위험 설명 / 삼일 자료 인용 / 실제 기업 DART 공시 내용까지 연결",
    },
    {
        "id": "Q5",
        "dim": "정성 사업 분석 (사업의 내용)",
        "q": "유한양행의 주요 신약 파이프라인과 기술이전(라이선스 아웃) 계약 현황을 정리하고, 이것이 향후 매출·수익 인식에 어떤 의미가 있는지 설명해주세요.",
        "rubric": "실제 파이프라인·계약 / 수익인식(1115호) 관점 해석 / 유한양행 DART 문서 검색됨",
    },
]


def trim(docs: list[Document], max_docs: int, max_chars: int) -> list[Document]:
    out = []
    for d in docs[:max_docs]:
        t = (d.page_content or "").strip()
        if len(t) > max_chars:
            t = t[:max_chars] + "\n...(truncated)"
        out.append(Document(page_content=t, metadata=d.metadata))
    return out


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stamp}_{label}.md"

    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    print("리트리버 로딩...", flush=True)
    kifrs_r = get_kifrs_retriever(emb)
    kam_r = get_kam_retriever(emb)
    dart_r = get_dart_retriever(emb)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    out = io.open(out_path, "w", encoding="utf-8")
    out.write(f"# 벤치마크 결과 — {label} ({stamp})\n\n")
    out.write(f"- 생성시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    out.write("- 채점 기준: 루브릭 충족도 ✅(충족)/⚠️(부분)/❌(미충족)\n\n")
    # 채점 요약 placeholder — 실행 후 Claude가 각 문항 점수·핵심 관찰·직전 회차 대비 변화를 채운다.
    out.write("## 📊 채점 요약 (Claude 작성)\n\n")
    out.write("| Q | 차원 | 점수 | 한 줄 평 |\n|---|---|---|---|\n")
    for it in BENCHMARK:
        out.write(f"| {it['id']} | {it['dim']} | _(채점)_ | _(작성)_ |\n")
    out.write("\n**종합:** _(작성)_\n\n### 🔑 핵심 관찰\n- _(작성)_\n\n### 📈 직전 회차 대비 변화\n- _(작성)_\n\n---\n\n")

    for item in BENCHMARK:
        qid, q = item["id"], item["q"]
        print(f"[{qid}] 검색+생성 중...", flush=True)
        kifrs_docs = kifrs_r.invoke(q)
        kam_docs = kam_r.invoke(q)
        dart_docs = dart_r.invoke(q) if dart_r else []

        kifrs_t = trim(kifrs_docs, 4, 2000)
        dart_t = trim(dart_docs, 15, 3500)
        kam_t = trim(kam_docs, 3, 1500)
        p1, c1 = format_cited_docs(kifrs_t, 1)
        p2, c2 = format_cited_docs(dart_t, len(c1) + 1)
        p3, c3 = format_cited_docs(kam_t, len(c1) + len(c2) + 1)
        context = "\n\n".join([x for x in [p1, p2, p3] if x])
        msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=build_user_prompt(q, context))]
        ans = str(llm.invoke(msgs).content)

        dart_companies = sorted({d.metadata.get("company", "") for d in dart_docs if d.metadata.get("company")})
        out.write(f"## {qid} — {item['dim']}\n\n")
        out.write(f"**질문:** {q}\n\n")
        out.write(f"**루브릭:** {item['rubric']}\n\n")
        out.write(f"**검색 통계:** DART {len(dart_docs)}건(기업: {', '.join(dart_companies) or '-'}) · KIFRS {len(kifrs_docs)}건 · KAM {len(kam_docs)}건\n\n")
        out.write(f"**채점:** (직접 표시) \n\n")
        out.write(f"**답변:**\n\n{ans}\n\n---\n\n")
        print(f"[{qid}] 완료 (DART {len(dart_docs)}건: {', '.join(dart_companies) or '-'})", flush=True)

    out.close()
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
