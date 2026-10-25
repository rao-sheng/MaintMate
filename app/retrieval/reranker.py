from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
    MODEL_NAME,
    local_files_only=True
)
    def build_document_text(self, payload):
        parts = []
        section = payload.get("section")
        caption = payload.get("caption")
        content = payload.get("content")
        if section:
            parts.append(
                f"章节：{section}"
            )
        if caption:
            parts.append(
                f"表格：{caption}"
            )
        if content:
            parts.append(content)
        return "\n".join(parts)

    def rerank(self,query,candidates,top_k=5):
        if not candidates:
            return []
        pairs = []
        for candidate in candidates:
            document_text = self.build_document_text(
                candidate["payload"]
            )
            pairs.append(
                (query, document_text)
            )
        scores = self.model.predict(
            pairs
        )
        results = []
        for candidate, score in zip(
            candidates,
            scores
        ):
            results.append({
                "id": candidate["id"],
                "hybrid_score": candidate["score"],
                "rerank_score": float(score),
                "payload": candidate["payload"]
            })
        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )
        return results[:top_k]

from app.retrieval.embedding import ZhipuEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

if __name__ == "__main__":

    embedding_provider = ZhipuEmbeddingProvider()
    vector_store = QdrantVectorStore()
    try:
        dense_retriever = DenseRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store
        )
        bm25_retriever = BM25Retriever()
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever
        )
        reranker = Reranker()
        query = "变频器发生通讯故障应该怎么处理？"
        # Hybrid 先召回 10 个候选
        candidates = hybrid_retriever.retrieve(
            query=query,
            top_k=10,
            candidate_k=20
        )
        # Reranker 从这 10 个里选最终 Top 5
        results = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=5
        )
        for rank, result in enumerate(
            results,
            start=1
        ):
            payload = result["payload"]
            print("\n" + "=" * 60)
            print("排名:", rank)
            print(
                "原 Hybrid 分数:",
                result["hybrid_score"]
            )
            print(
                "Rerank 分数:",
                result["rerank_score"]
            )
            print(
                "section:",
                payload.get("section")
            )
            print(
                "page:",
                payload.get("page_start"),
                "-",
                payload.get("page_end")
            )
            print("content:")
            print(
                payload.get("content")
            )
    finally:
        vector_store.close()