# RAG Chatbot — K-IFRS 회계처리 AI 어시스턴트

감사인을 위한 RAG 기반 회계처리 질의응답 챗봇입니다.  
K-IFRS 기준서, DART 기업 공시, 삼일회계법인 핵심감사사항(KAM) 보고서를 동시에 검색하여 답변합니다.

---

## 주요 기능

- **3-Source 병렬 RAG** — 질문 하나에 K-IFRS 기준 근거 + 실제 기업 공시 사례 + KAM 감사 관점을 동시에 제공
- **Hybrid Search + Re-ranking** — BM25(키워드) + Semantic(임베딩) 검색을 결합하고, Re-ranking으로 최종 관련도 순서를 재정렬하여 검색 정확도 향상
- **중복 안전 인제스트** — DART 데이터를 재수집해도 upsert 방식으로 중복 누적 없이 갱신
- **구조화 출력** — ① K-IFRS 기준 조항 ② 동종 기업 주석 사례 ③ KAM 감사 위험 관점 순서로 답변

---

## 아키텍처

```
사용자 질문
     │
     ▼
┌─────────────────────────────────────────────┐
│           Hybrid Search (BM25 + 임베딩)      │
├──────────────┬──────────────┬───────────────┤
│  K-IFRS 기준서│  DART 기업공시│  KAM 보고서    │
│  (ChromaDB)  │  (ChromaDB)  │ (ChromaDB)    │
└──────────────┴──────────────┴───────────────┘
     │
     ▼
  Re-ranking (관련도 재정렬)
     │
     ▼
  GPT-4o (LLM 합성)
     │
     ▼
  구조화 답변 + 출처
```

---

## 데이터 소스

| 소스 | 내용 |
|------|------|
| **K-IFRS 기준서** | 제1038호 무형자산, 제1115호 수익, 제2032호 웹사이트원가 등 |
| **DART 기업 공시** | 검색된 산업의 상위 규모 기업들의 사업보고서 주석 |
| **KAM 보고서** | 삼일회계법인 제약·바이오 핵심감사사항 및 대응방안 (2025) 등 |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| UI | Streamlit |
| RAG 프레임워크 | LangChain |
| 벡터 DB | ChromaDB |
| 임베딩 | OpenAI text-embedding-3-small |
| LLM | GPT-4o |
| DART 수집 | DART Open API + BeautifulSoup |
| 검색 | BM25 (rank-bm25) + Semantic (EnsembleRetriever) + Re-ranking |

---

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

`.env` 파일을 프로젝트 루트에 생성합니다.

```
OPENAI_API_KEY=sk-...
DART_API_KEY=...
```

### 2. DART 데이터 인제스트

DART 기업 공시 데이터를 ChromaDB에 적재합니다. 최초 1회 또는 데이터 갱신 시 실행합니다.

```bash
python dart_ingest.py
```

> DART 뷰어 HTML을 파싱하여 회계 주제별 청크를 추출합니다.  
> 재실행 시 upsert 방식으로 중복 없이 갱신됩니다.

### 3. 앱 실행

```bash
streamlit run app.py
```

---

## 프로젝트 구조

```
AX_RAG_Project/
├── app.py              # Streamlit 메인 앱
├── rag_engine.py       # ChromaDB 컬렉션 및 Retriever 초기화
├── dart_ingest.py      # DART 공시 수집 및 ChromaDB 적재
├── prompts.py          # 시스템 프롬프트 및 유저 프롬프트 빌더
├── requirements.txt
├── .env                # API 키
└── chroma_db/          # 벡터 DB 저장 경로 
```

---

## 예시 질문

- 제약사의 연구개발비 자산화 요건은?
- 기술이전 마일스톤 수익인식 시점은?
- 바이오시밀러 개발비의 손상 검토 기준은?
- CDMO 계약의 수익인식 방법은?
