from app.retrieval.embedding import ZhipuEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.context_builder import build_context

from app.llm.llm_provider import ZhipuLLMProvider

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
        self.bm25_retriever = BM25Retriever()
        # Hybrid Retriever
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever
        )
        # Reranker
        self.reranker = Reranker()
        # LLM
        self.llm_provider = ZhipuLLMProvider()
    
    def retrieve(
        self,
        query,
        final_top_k=5,
        hybrid_top_k=10,
        candidate_k=20
    ):
        """
        只检索手册证据，不生成最终回答。
        """

        # 1. Hybrid Retrieval
        hybrid_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=hybrid_top_k,
            candidate_k=candidate_k
        )

        # 2. Rerank
        reranked_results = self.reranker.rerank(
            query=query,
            candidates=hybrid_results,
            top_k=final_top_k
        )

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
