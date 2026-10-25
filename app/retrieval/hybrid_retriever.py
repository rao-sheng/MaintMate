from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.embedding import ZhipuEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
RRF_K = 60

class HybridRetriever:
    def __init__(self,dense_retriever,bm25_retriever):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
    def retrieve(self,query,top_k=5,candidate_k=20):
        # 1. Dense 先召回一批
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k
        )
        # 2. BM25 也召回一批
        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k
        )
        # 保存每个 chunk 的 RRF 分数
        rrf_scores = {}
        # 保存每个 chunk 的 payload
        payloads = {}
        # 3. 计算 Dense 排名贡献
        for rank, result in enumerate(dense_results,start=1):
            point_id = int(result.id)
            rrf_scores[point_id] = (
                rrf_scores.get(point_id, 0)
                + 1 / (RRF_K + rank)
            )
            payloads[point_id] = result.payload
        # 4. 计算 BM25 排名贡献
        for rank, result in enumerate(bm25_results,start=1):
            point_id = result["id"]
            rrf_scores[point_id] = (
                rrf_scores.get(point_id, 0)
                + 1 / (RRF_K + rank)
            )
            payloads[point_id] = result["payload"]

        # 5. 按 RRF 分数重新排序
        ranked_ids = sorted(rrf_scores, key=rrf_scores.get,reverse=True)#type:ignore

        # 6. 只保留最终 Top-K
        results = []

        for point_id in ranked_ids[:top_k]:
            results.append({
                "id": point_id,
                "score": rrf_scores[point_id],
                "payload": payloads[point_id]
            })

        return results

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

        query = "变频器发生通讯故障应该怎么处理？"

        results = hybrid_retriever.retrieve(
            query=query,
            top_k=10
        )

        for rank, result in enumerate(
            results,
            start=1
        ):
            print("\n" + "=" * 60)
            print("排名:", rank)
            print("RRF分数:", result["score"])
            print(
                "section:",
                result["payload"].get("section")
            )
            print(
                "page:",
                result["payload"].get("page_start"),
                "-",
                result["payload"].get("page_end")
            )
            print("content:")
            print(
                result["payload"].get("content")
            )

    finally:
        vector_store.close()