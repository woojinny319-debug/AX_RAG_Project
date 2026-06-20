"""ChromaDB 최종 정상화: persist 디렉터리를 깨끗이 비우고 3개 컬렉션을 새로 구축.

배경: 'dart' 이름을 삭제→재생성 반복하면서 누적된 잔여 상태(orphan 세그먼트 + WAL
찌꺼기)가 HNSW 인덱스 영속화를 깨뜨렸다(2026-06-19 디버깅 노트 참고).
가장 확실한 수리는 persist 디렉터리 전체를 비우고 클린 상태에서 재구축하는 것.

- dart : dart_data.pkl(임베딩 포함)로 즉시 적재 → 재임베딩 불필요
- kifrs/kam : 원본 PDF에서 로드·청킹·임베딩 후 적재
- 적재는 모두 chromadb 직접 add() (HNSW 바이너리 영속화 보장)
"""
from __future__ import annotations

import os
import pickle
import shutil
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

import rag_engine as RE

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
# 기존 chroma_db는 디버깅 churn으로 컴팩터가 오염됨. 건드리지 않고 새 폴더에 클린 재구축.
CHROMA_DIR = str(BASE / "chroma_db_v2")
PKL = BASE / "dart_data.pkl"


def _add_in_batches(col, ids, vectors, texts, metas, step=2000) -> None:
    for i in range(0, len(ids), step):
        col.add(
            ids=ids[i : i + step],
            embeddings=vectors[i : i + step],
            documents=texts[i : i + step],
            metadatas=metas[i : i + step],
        )
        print(f"      add {min(i + step, len(ids))}/{len(ids)}", flush=True)


def build_pdf_collection(client, emb, name, pdf_specs) -> None:
    print(f"[빌드] {name} (PDF에서)...")
    raw = []
    for filename, source_id in pdf_specs:
        p = RE.PROJECT_ROOT / filename
        if p.exists():
            raw.extend(RE._load_pdf(p, source_id))
    chunks = RE._chunk(raw)
    texts = [d.page_content for d in chunks]
    metas = [d.metadata for d in chunks]
    ids = [f"{name}_{i}" for i in range(len(chunks))]
    print(f"      {len(chunks)}개 청크 임베딩 중...")
    vectors = []
    for i in range(0, len(texts), 256):
        vectors.extend(emb.embed_documents(texts[i : i + 256]))
    col = client.create_collection(name)
    _add_in_batches(col, ids, vectors, texts, metas)
    print(f"      {name} 완료: {col.count()}개")


def main() -> None:
    if not PKL.exists():
        print("[중단] dart_data.pkl 백업이 없습니다.")
        return

    print("[1/5] dart 백업(pkl) 로드 중...")
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    print(f"      dart 문서: {len(data['ids'])}개")

    print(f"[2/5] 클린 새 디렉터리 준비: {CHROMA_DIR} (기존 chroma_db는 보존)...")
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)  # v2 잔여물만 정리
        time.sleep(1)

    print("[3/5] 새 PersistentClient 생성...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    print("[4/5] dart 적재 (pkl, 재임베딩 없음)...")
    col = client.create_collection("dart")
    _add_in_batches(col, data["ids"], data["vectors"], data["texts"], data["metas"])
    print(f"      dart 완료: {col.count()}개")

    print("[5/5] kifrs / kam 적재 (PDF에서)...")
    build_pdf_collection(client, emb, "kifrs", RE.K_IFRS_PDFS)
    build_pdf_collection(client, emb, "kam", RE.KAM_PDFS)

    print("\n[완료] 3개 컬렉션 재구축 성공")


if __name__ == "__main__":
    main()
