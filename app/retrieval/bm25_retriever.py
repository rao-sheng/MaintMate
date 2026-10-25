import json
from pathlib import Path
import jieba

JIEBA_CACHE_DIR = Path.home() / ".cache" / "maintmate" / "jieba"
JIEBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

jieba.dt.tmp_dir = str(JIEBA_CACHE_DIR)#type:ignore
jieba.dt.cache_file = str(JIEBA_CACHE_DIR / "jieba.cache")#type:ignore
import numpy as np
from rank_bm25 import BM25Okapi
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK_PATH = PROJECT_ROOT / "data" / "processed" / "md380e" / "text_chunks.json"
#把一段中文切成词语
def tokenize(text):
    return [
        token.strip()
        for token in jieba.lcut(text.lower())
        if token.strip()
    ]

def build_retrieval_text(chunk):
    parts = []
    if chunk.get("section"):
        parts.append(
            f"章节：{chunk['section']}"
        )
    if chunk.get("caption"):
        parts.append(
            f"表格：{chunk['caption']}"
        )
    parts.append(chunk["content"])
    return "\n".join(parts)

class BM25Retriever:
    def __init__(self):
        with CHUNK_PATH.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.documents = [
            build_retrieval_text(chunk)
            for chunk in self.chunks
        ]#补充上下文
        tokenized_documents = [
            tokenize(text)
            for text in self.documents
        ]#分词
        self.bm25 = BM25Okapi(
            tokenized_documents
        )
    def retrieve(self, query, top_k=5):
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(
            query_tokens
        )
        top_indices = (
            np.argsort(scores)[::-1][:top_k]
        )#返回的是排序后的“索引”
        results = []
        for index in top_indices:
            results.append({
                "id": int(index),
                "score": float(scores[index]),
                "payload": self.chunks[index]
            })
        return results