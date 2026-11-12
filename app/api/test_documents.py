import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app import documents
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_store import QdrantVectorStore


def test_upload_scopes_dense_and_sparse_and_delete(tmp_path, monkeypatch):
    api = importlib.import_module("app.api.main")
    monkeypatch.setenv("MAINTMATE_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    vector_store = QdrantVectorStore()
    vector = [1.0] + [0.0] * 1023
    provider = SimpleNamespace(embed_documents=lambda texts: [vector for _ in texts])
    fake_agent = SimpleNamespace(
        rag=SimpleNamespace(vector_store=vector_store, embedding_provider=provider),
        process=Mock(return_value="answer"),
    )
    monkeypatch.setattr(api, "agent", fake_agent)

    def parse(pdf, output):
        text = pdf.read_bytes()[5:].decode()
        if text == "broken":
            raise ValueError("invalid PDF")
        output.write_text(json.dumps({
            "pages": [{"page_idx": 0, "blocks": [{"type": "text", "content": text}]}]
        }), encoding="utf-8")

    monkeypatch.setattr(documents, "parse_pdf", parse)
    try:
        client = TestClient(api.app)

        def upload(text):
            response = client.post("/documents/upload", files={
                "file": ("manual.pdf", b"%PDF-" + text.encode(), "application/pdf")
            })
            assert response.status_code == 202
            return response.json()["document_id"]

        first, second = upload("Err16 first"), upload("Err16 second")
        assert client.get(f"/documents/{first}").json()["status"] == "ready"
        for doc_id, text in [(first, "Err16 first"), (second, "Err16 second")]:
            dense = vector_store.search(vector, document_id=doc_id)
            sparse = BM25Retriever(documents.document_directory(doc_id) / "chunks.json").retrieve("Err16")
            assert len(dense) == len(sparse) == 1
            assert dense[0].id == sparse[0]["id"]
            assert dense[0].payload["content"] == text
        response = client.post("/chat", json={"message": "Err16", "document_id": first})
        assert response.status_code == 200
        assert client.post("/chat", json={"message": "Err16", "document_id": second,
                                          "session_id": response.json()["session_id"]}).status_code == 409
        assert client.delete(f"/documents/{first}").status_code == 204
        assert not vector_store.search(vector, document_id=first)
        assert vector_store.search(vector, document_id=second)
        failed = upload("broken")
        assert client.get(f"/documents/{failed}").json()["status"] == "failed"
        assert client.post("/chat", json={"message": "Err16", "document_id": failed}).status_code == 409
    finally:
        vector_store.close()
