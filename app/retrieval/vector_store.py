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
        self.client=QdrantClient(
            path=str(QDRANT_PATH)
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
    def search(self,query_vector,limit=5):
        result=self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        return result.points
    def count(self):
        result=self.client.count(
            collection_name=self.collection_name,
            exact=True
        )
        return result.count
    def close(self):
        self.client.close()
