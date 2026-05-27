---
tags: [concept, rag, langchain, advanced-rag]
aliases: [Self-Querying, Query Routing, 라우팅, 자동 기준서 라우팅]
created: 2026-05-25
---

# Query Routing (Self-Querying)

> **한 줄 정의**: 사용자의 질문(Query)을 분석하여, 가장 적합한 데이터소스나 필터 조건(메타데이터)을 LLM이 스스로 결정한 뒤 검색(Retrieval)을 수행하는 Advanced RAG 기법.

---

## 핵심 아이디어

기본적인 [[RAG (Retrieval-Augmented Generation)]]는 모든 문서를 획일적으로 검색하지만, 도메인이 복잡할 경우(예: 회계 기준서) 엉뚱한 문서가 검색될 수 있다.
이를 방지하기 위해 검색 전에 **분류(Classification) 단계**를 둔다.

```
[사용자 질문] 
   ↓
[Router LLM]: "이 질문은 무형자산(1038호)에 관한 것이군!"
   ↓
[Retriever (Filter 적용)]: metadata={"source": "1038호"} 에서만 검색
   ↓
[최종 답변 생성]
```

---

## 프로젝트 적용 방안 (v2 기획)

우리 프로젝트에서는 **K-IFRS 기준서 자동 매핑**에 이 기술을 사용한다.
사용자가 수동으로 체크박스를 누르는 방식 대신, 질문만 던지면 AI가 [[K-IFRS 회계기준]] 중 어떤 호수가 필요한지 추론하여 검색의 범위를 극도로 좁혀(Hallucination 방지) 신뢰도를 높인다.

- 관련 파일: `api/rag_core.py` (구현 예정)
- 기대 효과: 사용자의 질문 의도를 정확하게 파악하여 검색 범위를 최적화하고 답변의 신뢰도를 극대화함.

---

## 관련 노트

- [[Samil Accounting Insights]] — 벤치마킹 대상
- [[2026-05-25 — 프로젝트 기획 및 아키텍처 설계]] — 이 개념이 도입된 배경
