from functools import lru_cache
from app.documents import document_directory
from app.retrieval.embedding import ZhipuEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever, CHUNK_PATH
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.context_builder import build_context
from app.retrieval.fault_rows import fault_codes, rank_evidence, row_fault_codes

from app.llm.llm_provider import ZhipuLLMProvider

@lru_cache(maxsize=8)
def document_bm25(document_id):
    if document_id == "md380e":
        return BM25Retriever() if CHUNK_PATH.is_file() else None
    return BM25Retriever(document_directory(document_id) / "chunks.json")


class RAGPipeline:
    def __init__(self):
        # Embedding
        self.embedding_provider = ZhipuEmbeddingProvider()
        # Vector Store
        self.vector_store = QdrantVectorStore()
        # Dense Retriever
        self.dense_retriever = DenseRetriever(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store
        )
        # BM25 Retriever
        # Reranker
        self.reranker = Reranker()
        # LLM
        self.llm_provider = ZhipuLLMProvider()
    
    def retrieve(
        self,
        query,
        final_top_k=5,
        hybrid_top_k=10,
        candidate_k=20,
        document_id="md380e"
    ):
        """
        只检索手册证据，不生成最终回答。
        """

        # 1. Hybrid Retrieval
        bm25 = document_bm25(document_id)
        if bm25 is None:
            return {"context": "", "sources": []}
        hybrid = HybridRetriever(self.dense_retriever, bm25)
        hybrid_results = hybrid.retrieve(
            query=query,
            top_k=hybrid_top_k,
            candidate_k=candidate_k,
            document_id=document_id
        )

        requested_codes = fault_codes(query)
        if requested_codes:
            candidates = {item["id"]: item for item in hybrid_results}
            for index, chunk in enumerate(bm25.chunks):
                if requested_codes & row_fault_codes(chunk):
                    chunk_id = chunk.get("chunk_id", index)
                    candidates[chunk_id] = {
                        "id": chunk_id,
                        "score": candidates.get(chunk_id, {}).get("score", 0.0),
                        "payload": chunk,
                    }
            hybrid_results = list(candidates.values())

        reranked_results = rank_evidence(self.reranker, query, hybrid_results, final_top_k)
        # 3. Build Context
        context = build_context(
            reranked_results
        )

        return {
            "context": context,
            "sources": reranked_results
        }

    def answer(
        self,
        query,
        final_top_k=5,
        hybrid_top_k=10,
        candidate_k=20
    ):
        """
        保留原接口：检索证据并生成回答。
        """

        result = self.retrieve(
            query=query,
            final_top_k=final_top_k,
            hybrid_top_k=hybrid_top_k,
            candidate_k=candidate_k
        )

        answer = self.llm_provider.generate_answer(
            query=query,
            context=result["context"]
        )

        return {
            "answer": answer,
            "sources": result["sources"]
        }

    def close(self):
        self.vector_store.close()


if __name__ == "__main__":

    rag = RAGPipeline()

    try:
        result = rag.retrieve(
            query="MD380E Err16 通讯故障怎么处理？"
        )

        print("检索到的证据数量：", len(result["sources"]))
        print("\n构建的上下文：")
        print(result["context"][:1500])

    finally:
        rag.close()
