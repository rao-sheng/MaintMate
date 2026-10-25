import os
import time
import requests

from abc import ABC, abstractmethod
from dotenv import load_dotenv


MODEL_NAME = "embedding-3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 32
EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"


class EmbeddingProvider(ABC):

    @abstractmethod
    def embed_documents(self, texts):
        pass

    @abstractmethod
    def embed_query(self, text):
        pass


class ZhipuEmbeddingProvider(EmbeddingProvider):

    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv(
            "ZHIPU_API_KEY"
        )

        if not self.api_key:
            raise ValueError(
                "没有找到环境变量 ZHIPU_API_KEY"
            )

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        vectors = self._embed([text])
        return vectors[0]

    def _post_with_retry(
        self,
        payload,
        max_retries=3
    ):
        for attempt in range(max_retries):

            try:
                response = requests.post(
                    EMBEDDING_URL,
                    headers={
                        "Authorization": (
                            f"Bearer {self.api_key}"
                        ),
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )

                return response

            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout
            ) as e:

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Embedding API 网络请求失败：{e}"
                    ) from e

                wait_seconds = 2 ** attempt

                print(
                    f"Embedding 请求失败，"
                    f"{wait_seconds} 秒后重试..."
                )

                time.sleep(wait_seconds)

    def _embed(self, texts):

        all_vectors = []

        for start in range(
            0,
            len(texts),
            BATCH_SIZE
        ):
            batch = texts[
                start:start + BATCH_SIZE
            ]

            response = self._post_with_retry(
                payload={
                    "model": MODEL_NAME,
                    "input": batch,
                    "dimensions": EMBEDDING_DIM,
                }
            )

            if response.status_code != 200:#type:ignore
                raise RuntimeError(
                    "Embedding API 调用失败："
                    f"{response.status_code}\n"#type:ignore
                    f"{response.text}"#type:ignore
                )

            result = response.json()#type:ignore

            embeddings = sorted(
                result["data"],
                key=lambda item: item["index"]
            )

            for item in embeddings:
                all_vectors.append(
                    item["embedding"]
                )

        return all_vectors