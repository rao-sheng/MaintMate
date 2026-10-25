from app.retrieval.embedding import ZhipuEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
class DenseRetriever:
    def __init__(self,embedding_provider,vector_store) :
        self.embedding_provider=embedding_provider
        self.vector_store=vector_store

    def retrieve(self,query,top_k=5):
        query_vector=(self.embedding_provider.embed_query(query))
        result=self.vector_store.search(query_vector=query_vector,limit=top_k)
        return result

if __name__ == "__main__":
    embedding_provider = ZhipuEmbeddingProvider()
    vector_store = QdrantVectorStore()

    try:
        retriever = DenseRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store
        )

        query = "变频器发生通讯故障应该怎么处理？"

        results = retriever.retrieve(
            query=query,
            top_k=10
        )

        for rank, result in enumerate(results, start=1):
            print("\n" + "=" * 60)
            print("排名:", rank)
            print("相似度:", result.score)
            print("section:", result.payload.get("section"))
            print(
                "page:",
                result.payload.get("page_start"),
                "-",
                result.payload.get("page_end")
            )
            print("content:")
            print(result.payload.get("content"))

    finally:
        vector_store.close()