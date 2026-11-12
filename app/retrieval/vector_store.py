import os
from pathlib import Path
from qdrant_client import QdrantClient,models
COLLECTION_NAME="maitemate_chunks"
VECTOR_SIZE=1024
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QDRANT_PATH = PROJECT_ROOT / "data" / "qdtant"
class QdrantVectorStore:
    #__inin__函数在初始化时自动执行
    def __init__(self) :
        #创建能操作Qdrant的客户端，并记住我要操作的collection名字
        url = os.getenv("QDRANT_URL")
        self.client = QdrantClient(url=url) if url else QdrantClient(
            path=os.getenv("QDRANT_PATH", str(QDRANT_PATH))
        )
        self.collection_name=COLLECTION_NAME
    def create_collection(self):
        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
            return 
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
    def upsert_points(self,points):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
    def search(self,query_vector,limit=5,document_id="md380e"):
        result=self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self.document_filter(document_id),
            limit=limit
        )
        return result.points
    @staticmethod
    def document_filter(document_id):
        return models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=document_id))
        ])

    def delete_document(self, document_id):
        if self.client.collection_exists(self.collection_name):
            self.client.delete(self.collection_name, self.document_filter(document_id), wait=True)

    def count(self):
        result=self.client.count(
            collection_name=self.collection_name,
            exact=True
        )
        return result.count
    def close(self):
        self.client.close()
