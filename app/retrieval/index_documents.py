import json
from pathlib import Path
from qdrant_client import models
from app.retrieval.embedding import ZhipuEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNK_PATH = (
    PROJECT_ROOT / "data" / "processed" / "md380e" / "text_chunks.json"
)

INDEX_BATCH_SIZE = 32

def load_chunks(path):
    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

def build_retrieval_text(chunk):
    #补充必要的上下文
    parts = []
    section = chunk.get("section")
    caption = chunk.get("caption")
    if section:
        parts.append(
            f"章节：{section}"
        )
    if caption:
        parts.append(
            f"表格：{caption}"
        )
    parts.append(
        chunk["content"]
    )
    return "\n".join(parts)

def build_payload(chunk):
    return {
        "content": chunk.get("content"),
        "section": chunk.get("section"),
        "caption": chunk.get("caption"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "type": chunk.get("type"),
        "doc_id": chunk.get("doc_id"),
        "source": chunk.get("source"),
    }

def index_chunks():
    chunks = load_chunks(CHUNK_PATH)
    #先后引入向量化、Qdrant
    embedding_provider = ZhipuEmbeddingProvider()
    vector_store = QdrantVectorStore()
    vector_store.create_collection()
    for start in range(
        0,
        len(chunks),
        INDEX_BATCH_SIZE
    ):
        batch_chunks = chunks[
            start:start + INDEX_BATCH_SIZE
        ]
        texts = [
            build_retrieval_text(chunk)
            for chunk in batch_chunks
        ]
        vectors = (
            embedding_provider
            .embed_documents(texts)
        )
        points = []
        for offset, (chunk, vector) in enumerate(
            zip(batch_chunks, vectors)
        ):
            point_id = start + offset

            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=build_payload(chunk)
            )
            points.append(point)
        vector_store.upsert_points(points)
        print(
            f"已入库: "
            f"{min(start + len(batch_chunks), len(chunks))}"
            f"/{len(chunks)}"
        )
    print(
        "Qdrant 总 Point 数:",
        vector_store.count()
    )


if __name__ == "__main__":
    index_chunks()