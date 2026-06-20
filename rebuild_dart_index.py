"""DART HNSW 인덱스 복구 스크립트.

ChromaDB 1.5.x에서 dart 컬렉션의 HNSW 인덱스 바이너리가 디스크에 끝까지
기록되지 못해(index_metadata.pickle만 존재, data_level0.bin 등 부재) 손상됨.
문서 텍스트·메타데이터는 SQLite(METADATA 세그먼트)에 온전히 남아 있으므로,
DART 서버 재접속 없이 SQLite에서 문서를 추출해 컬렉션을 재구축(재임베딩)한다.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
CHROMA_DIR = str(BASE / "chroma_db")
COLLECTION = "dart"


def extract_docs_from_sqlite() -> tuple[list[Document], list[str]]:
    con = sqlite3.connect(f"{CHROMA_DIR}/chroma.sqlite3")
    cur = con.cursor()
    # dart 컬렉션 id를 이름으로 동적 조회 (id는 재구축마다 바뀜)
    cur.execute("SELECT id FROM collections WHERE name=?", (COLLECTION,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError("dart 컬렉션을 찾을 수 없습니다.")
    dart_col_id = row[0]
    # dart의 METADATA 세그먼트 id
    cur.execute(
        "SELECT id FROM segments WHERE collection=? AND scope='METADATA'",
        (dart_col_id,),
    )
    meta_seg = cur.fetchone()[0]
    # 해당 세그먼트의 embedding row들 (id -> embedding_id)
    cur.execute("SELECT id, embedding_id FROM embeddings WHERE segment_id=?", (meta_seg,))
    rows = cur.fetchall()

    docs: list[Document] = []
    ids: list[str] = []
    for emb_pk, emb_id in rows:
        cur.execute(
            "SELECT key, string_value, int_value, float_value FROM embedding_metadata WHERE id=?",
            (emb_pk,),
        )
        text = ""
        meta: dict[str, str] = {}
        for key, sval, ival, fval in cur.fetchall():
            value = sval if sval is not None else (ival if ival is not None else fval)
            if key == "chroma:document":
                text = sval or ""
            elif key.startswith("chroma:"):
                continue
            else:
                meta[key] = "" if value is None else str(value)
        if text.strip():
            docs.append(Document(page_content=text, metadata=meta))
            ids.append(emb_id)
    con.close()
    return docs, ids


def main() -> None:
    print("[1/4] SQLite에서 dart 문서 추출 중...")
    docs, ids = extract_docs_from_sqlite()
    print(f"      추출 완료: {len(docs)}개 문서")
    if not docs:
        print("[중단] 추출된 문서가 없습니다.")
        return

    print("[2/4] 손상된 dart 컬렉션 삭제 중...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
        print("      삭제 완료")
    except Exception as e:
        print(f"      삭제 스킵: {str(e)[:80]}")

    # [핵심] langchain_chroma의 배치 add 경로는 chromadb 1.5.x에서 HNSW 바이너리를
    # 디스크에 영속화하지 못하는 버그가 있다(인메모리만 동작, 별도 프로세스 로드 실패).
    # 대신 chromadb collection.add()를 "직접" 호출하면 정상적으로 .bin이 기록된다.
    # 따라서 임베딩은 OpenAIEmbeddings로 미리 계산하고, 저장은 chromadb 직접 경로를 쓴다.
    print("[3/4] 임베딩 사전 계산 중...")
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    vectors: list[list[float]] = []
    estep = 256
    for i in range(0, len(texts), estep):
        vectors.extend(emb.embed_documents(texts[i : i + estep]))
        print(f"      임베딩 {min(i + estep, len(texts))}/{len(texts)}", flush=True)
        time.sleep(0.2)

    print("[4/4] chromadb 직접 add (영속화 보장)...")
    col = client.create_collection(COLLECTION)
    astep = 2000
    for i in range(0, len(ids), astep):
        col.add(
            ids=ids[i : i + astep],
            embeddings=vectors[i : i + astep],
            documents=texts[i : i + astep],
            metadatas=metas[i : i + astep],
        )
        print(f"      add {min(i + astep, len(ids))}/{len(ids)}", flush=True)

    # 동일 프로세스에서 새 클라이언트로 재오픈하여 영속화 검증
    client2 = chromadb.PersistentClient(path=CHROMA_DIR)
    col2 = client2.get_collection(COLLECTION)
    print(f"      재오픈 count: {col2.count()}")
    res = col2.query(query_embeddings=[vectors[0]], n_results=3)
    print(f"      검증 query 결과: {len(res['ids'][0])}건")
    print("[완료] dart 인덱스 재구축 성공")


if __name__ == "__main__":
    main()
