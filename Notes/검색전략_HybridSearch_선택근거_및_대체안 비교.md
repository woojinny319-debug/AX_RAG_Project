# 검색 전략 — Hybrid Search 선택 근거 및 대체안 비교

> 관련 노트: [[프로젝트_설계계획서]] | [[RAG_아키텍처_개요]]  
> 작성일: 2026-05-26

---

## 1. 왜 Hybrid Search를 선택했는가

### 도메인 특성에서 나오는 요구사항

이 프로젝트의 질의 유형은 두 가지가 혼재한다.

| 질의 유형  | 예시                             | 검색에서 필요한 것        |
| ------ | ------------------------------ | ----------------- |
| 정확한 참조 | "K-IFRS 1038호 **문단 57**의 요건은?" | 키워드 정합 (BM25)     |
| 개념적 의미 | "연구개발비를 언제 자산으로 인식하는가?"        | 의미 유사도 (Semantic) |

법률·회계 텍스트는 두 유형이 섞여 있다. "문단 57", "손상차손 인식 요건", "수행의무 이전 시점" 같은 표현은 **특정 단어의 등장 자체**가 관련성의 핵심 신호다. 반면 "임상 3상 진입 기준이 자산화의 근거가 되는가?"는 단어 매칭이 아닌 의미 이해가 필요하다.

### OpenAI 임베딩의 한국어 전문 용어 한계

OpenAI `text-embedding-ada-002` 또는 `text-embedding-3` 계열은 범용 다국어 모델이다. "기술적 실현가능성", "손상차손 환입 제한" 같은 한국어 회계 용어는 임베딩 공간에서 의미론적 근접성이 낮거나 불안정할 수 있다. BM25는 어휘 기반이므로 임베딩 품질과 무관하게 정확한 용어 매칭을 보장한다.

### 설계 결정 요약

```
weights = [0.4(BM25), 0.6(Semantic)]
```

회계 텍스트는 의미 검색이 주도하되, 키워드 정확도 보완을 위해 BM25에 40% 가중치를 부여. 순수 Semantic 대비 특수 용어 회수율(Recall)을 높이는 것이 목적.

---

## 2. 검색 전략 대체안 비교

### 2-1. 순수 Semantic Search

**원리**: 질문과 청크를 각각 벡터로 변환 → 코사인 유사도 기준 Top-K 반환.

| 항목         | 평가                                                                                  |
| ---------- | ----------------------------------------------------------------------------------- |
| 장점         | 구현 가장 단순. ChromaDB `.as_retriever()` 한 줄로 완성. 의미 유사도 기반이므로 표현 방식이 달라도 관련 문서를 찾음     |
| 단점         | "문단 57", "손상차손" 같은 특수 용어를 쿼리에 그대로 써도 임베딩 거리가 멀 수 있음. 한국어 전문 용어에서 재현율(Recall) 저하 리스크 |
| 구현 복잡도     | ★☆☆☆☆ — 가장 낮음. 이미 ChromaDB 설정만으로 동작                                                 |
| 이 프로젝트 적합도 | 낮음. 조항 번호 인용이 필수인 회계 도메인에서 키워드 누락은 치명적                                              |

---

### 2-2. 순수 BM25 (TF-IDF 계열)

**원리**: 단어 빈도(TF)와 역문서 빈도(IDF)로 관련성을 점수화. 벡터 공간 없음.

| 항목         | 평가                                                                           |
| ---------- | ---------------------------------------------------------------------------- |
| 장점         | 특정 용어가 포함된 문서를 높은 확도로 회수. 설치 의존성 없음 (BM25s 라이브러리만 필요). 해석 가능성 높음             |
| 단점         | 동의어, 패러프레이즈에 완전히 무능. "자산 인식 요건"으로 검색 시 "자산화 기준"이 포함된 문서를 못 찾을 수 있음. 의미 이해 불가 |
| 구현 복잡도     | ★☆☆☆☆ — `BM25Retriever.from_documents(docs)` 한 줄                             |
| 이 프로젝트 적합도 | 낮음. 개념적 질의가 많아 의미 검색 없이는 답변 품질이 현저히 낮아짐                                      |

---

### 2-3. Hybrid Search — BM25 + Semantic (선택안) ✅

**원리**: BM25와 Semantic 검색 결과를 앙상블(가중합). LangChain `EnsembleRetriever` 사용.

| 항목         | 평가                                                                                    |
| ---------- | ------------------------------------------------------------------------------------- |
| 장점         | 키워드 정합과 의미 유사도를 동시에 커버. 각 방식의 단점을 상호 보완. 가중치 조정으로 도메인 특성 반영 가능                        |
| 단점         | BM25용 문서 객체와 ChromaDB 벡터스토어를 별도로 관리해야 함. 메모리에 BM25 인덱스를 상주시켜야 하므로 대규모 코퍼스에선 메모리 부담 증가 |
| 구현 복잡도     | ★★☆☆☆ — `EnsembleRetriever` 래핑으로 해결되나 BM25 인덱스 초기화 타이밍 주의 필요                          |
| 이 프로젝트 적합도 | 높음. 이 프로젝트의 코퍼스 크기(수백~수천 청크)에서 메모리 부담은 무시 가능                                          |
|            |                                                                                       |

---

### 2-4. Re-ranking (Cross-Encoder 2단계)

**원리**: 1단계에서 Semantic/BM25로 Top-20 후보를 넓게 회수 → 2단계에서 Cross-Encoder 모델이 질문-문서 쌍을 직접 평가해 Top-K로 재순위.

| 항목         | 평가                                                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| 장점         | 정밀도(Precision)가 가장 높음. Bi-Encoder의 정보 손실 없이 질문-문서 상호작용을 직접 계산. RAG 최고 품질을 원할 때 표준 패턴                                       |
| 단점         | Cross-Encoder 모델 별도 필요 (한국어 지원 모델: `ko-reranker`, `bge-reranker-v2-m3` 등). 레이턴시 추가 (Top-20 쌍을 순차 추론). API 키 또는 로컬 모델 배포 필요 |
| 구현 복잡도     | ★★★☆☆ — `CrossEncoderReranker` 또는 Cohere Rerank API 연동. 2단계 파이프라인 설계 필요                                                    |
| 이 프로젝트 적합도 | 중간. 품질 최우선이라면 고려할 만하나, 포트폴리오 데모 수준에서는 Hybrid Search로 충분                                                                    |

---

## 3. Hybrid Search 구현 상세

### 3-1. 기술 스택

| 라이브러리                 | 역할                                      | 비고                               |
| --------------------- | --------------------------------------- | -------------------------------- |
| `rank_bm25`           | BM25 알고리즘 구현체                           | `BM25Retriever`의 내부 의존성          |
| `langchain-community` | `BM25Retriever`, `EnsembleRetriever` 제공 |                                  |
| `chromadb`            | 벡터 저장소 (Semantic 검색 담당)                 | 로컬 파일 기반, 컬렉션 단위 분리              |
| `langchain-openai`    | `OpenAIEmbeddings` (임베딩 생성)             | `text-embedding-3-small` 사용      |
| `langchain`           | 문서 로딩, 청킹, 체인 조합                        | `RecursiveCharacterTextSplitter` |

```bash
pip install rank_bm25 langchain langchain-community langchain-openai chromadb
```

---

### 3-2. 구현 단계

#### Step 1 — 문서 로딩 및 청킹

PDF에서 텍스트를 추출하고 청크 단위로 분할한다. `chunk_size`와 `overlap`은 회계 조항 단위에 맞게 조정.

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyMuPDFLoader("K-IFRS_제1038호_무형자산.pdf")
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " "]  # 한국어 문장 경계 우선
)
docs = splitter.split_documents(raw_docs)
# → LangChain Document 리스트. 각 doc.metadata에 source, page 포함
```

---

#### Step 2 — ChromaDB 벡터스토어 생성 (Semantic용)

임베딩 모델로 각 청크를 벡터화해 ChromaDB에 저장한다. 컬렉션은 소스별로 분리 (`kifrs`, `dart`, `kam`).

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="kifrs",         # 소스별 컬렉션 분리
    persist_directory="./chroma_db"
)
```

이미 저장된 컬렉션을 재사용할 때는 `from_documents` 대신 `Chroma(collection_name=..., persist_directory=..., embedding_function=embeddings)`로 로드.

---

#### Step 3 — BM25 인덱스 생성

BM25는 벡터가 아닌 어휘(토큰) 기반이므로 ChromaDB와 별도로 인덱싱한다. `docs` 객체를 직접 받아 메모리에 인덱스를 구성한다.

```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3   # 반환할 문서 수
```

`BM25Retriever`는 메모리 내 인덱스다. 앱 재시작 시 `docs`로부터 매번 재생성하거나, pickle로 직렬화해 캐싱한다.

---

#### Step 4 — EnsembleRetriever 조합

두 Retriever를 가중치로 결합한다. 가중치의 합은 반드시 1.0이어야 한다.

```python
from langchain.retrievers import EnsembleRetriever

semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6]
    # BM25 0.4: 키워드 정합 보완
    # Semantic 0.6: 의미 유사도 주도
)
```

내부 동작: 각 Retriever가 Top-K를 독립 실행 → **Reciprocal Rank Fusion(RRF)** 방식으로 순위 점수를 합산 → 최종 순위 결정. RRF는 두 리스트의 절대 점수가 아닌 순위 자체를 기반으로 하므로, 점수 스케일이 다른 BM25와 코사인 유사도를 자연스럽게 통합한다.

---

#### Step 5 — rag_engine.py 통합 구조

프로젝트에서는 3개 컬렉션(kifrs, dart, kam)마다 동일한 패턴을 적용한다.

```python
# rag_engine.py

def build_retriever(collection_name: str, docs: list) -> EnsembleRetriever:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 3

    semantic = vectorstore.as_retriever(search_kwargs={"k": 3})

    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.4, 0.6]
    )

# 앱 초기화 시 3개 Retriever 빌드
kifrs_retriever = build_retriever("kifrs", kifrs_docs)
dart_retriever  = build_retriever("dart",  dart_docs)
kam_retriever   = build_retriever("kam",   kam_docs)
```

---

#### Step 6 — 검색 실행 및 컨텍스트 추출

```python
query = "제약사의 연구개발비 자산화 요건은?"

results = retriever.invoke(query)
# → List[Document], 각 doc.page_content + doc.metadata

# 프롬프트에 삽입할 컨텍스트 문자열로 변환
context = "\n\n".join([
    f"[출처: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
    for doc in results
])
```

---

### 3-3. 핵심 동작 흐름

```
질문 입력
    │
    ├─── BM25Retriever ──────────────────→ Top-3 (키워드 기반)
    │        rank_bm25, 메모리 인덱스
    │
    ├─── Semantic Retriever ─────────────→ Top-3 (임베딩 유사도)
    │        ChromaDB + OpenAI Embeddings
    │
    └─── EnsembleRetriever (RRF 합산) ───→ Top-6 최종 문서
                                                │
                                          GPT-4o 프롬프트 삽입
                                                │
                                     ① K-IFRS 기준  ② 기업 사례  ③ 감사 관점
```

---

## 5. 전체 비교 요약

| 전략              | 정밀도    | 재현율    | 한국어 특수용어 | 구현 복잡도    | 이 프로젝트          |
| --------------- | ------ | ------ | -------- | --------- | --------------- |
| 순수 Semantic     | 중      | 중      | 취약       | ★☆☆☆☆     | △               |
| 순수 BM25         | 중      | 중      | 강점       | ★☆☆☆☆     | △               |
| **Hybrid (선택)** | **중상** | **중상** | **보완됨**  | **★★☆☆☆** | **✅**           |
| Re-ranking      | 높음     | 중상     | 보완됨      | ★★★☆☆     | 고품질 버전으로 확장 시   |


---

## 6. 개선 로드맵 관점에서

현재 구조에서 품질을 높이는 가장 현실적인 다음 단계는 **Hybrid Search + Re-ranking** 조합이다.

```
[현재]  질문 → BM25 + Semantic 앙상블 → Top-6 → LLM

[개선]  질문 → BM25 + Semantic 앙상블 → Top-20
             → Cohere Rerank / bge-reranker → Top-6 → LLM
```

Cohere Rerank API는 한국어를 지원하고 LangChain `ContextualCompressionRetriever`로 연동이 간단하다. 포트폴리오 고도화 시 가장 ROI가 높은 업그레이드다.

---

## 관련 노트

- [[프로젝트_설계계획서]] — 전체 아키텍처 및 기술 스택 선택 근거
- [[RAG_아키텍처_개요]] — EnsembleRetriever 상세 구현
- [[K-IFRS_제1038호]] — 검색 대상 문서 특성
