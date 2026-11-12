"""Document lifecycle and conversation persistence for the single-instance API."""
import json
import logging
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore, Lock
from uuid import UUID, uuid5

from qdrant_client import models
from sqlalchemy import JSON, ForeignKey, URL, create_engine, delete, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

from app.ingestion.chunker import build_chunks
from app.retrieval.index_documents import build_payload, build_retrieval_text

logger = logging.getLogger(__name__)
job_lock = Lock()
job_slots = BoundedSemaphore(4)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    filename: Mapped[str]
    status: Mapped[str]
    error: Mapped[str | None]


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(primary_key=True)
    state: Mapped[dict] = mapped_column(JSON)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str]
    content: Mapped[str]


def data_directory() -> Path:
    return Path(os.getenv("MAINTMATE_DATA_DIR", "data/workspace")).resolve()


def document_directory(document_id: str) -> Path:
    return data_directory() / "documents" / str(UUID(document_id))


@lru_cache
def repository(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    engine = create_engine(URL.create("sqlite", database=str(root / "maintmate.db")), poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(engine)


def sessions():
    return repository(data_directory())


def document_status(document_id: str):
    with sessions()() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        return {"document_id": doc.id, "filename": doc.filename, "status": doc.status,
                "error": doc.error}


def change_status(document_id, status, error=None):
    with sessions().begin() as session:
        doc = session.get(Document, document_id)
        if doc is not None:
            doc.status = status
            doc.error = error


def recover_interrupted(vector_store):
    with sessions()() as session:
        ids = list(session.scalars(select(Document.id).where(
            Document.status.in_(("pending", "processing", "deleting"))
        )))
    for document_id in ids:
        change_status(document_id, "failed", "处理因服务重启而中断")
        try:
            vector_store.delete_document(document_id)
        except Exception:
            logger.exception("中断文档的向量清理失败")


def parse_pdf(pdf: Path, destination: Path):
    parser_python = os.getenv("MINERU_PYTHON") or sys.executable
    runner = Path(__file__).parent / "ingestion" / "parse_pdf.py"
    subprocess.run([parser_python, str(runner), str(pdf), str(destination)],
                   check=True, timeout=int(os.getenv("MINERU_TIMEOUT_SECONDS", "900")))


def run_document_job(document_id, vector_store, embedding_provider):
    try:
        process_document(document_id, vector_store, embedding_provider)
    finally:
        job_slots.release()


def process_document(document_id, vector_store, embedding_provider):
    with job_lock:
        doc = document_status(document_id)
        if not doc or doc["status"] != "pending":
            return
        root = document_directory(document_id)
        phase = "解析 PDF"
        try:
            change_status(document_id, "processing")
            parsed = root / "structured_content.json"
            parse_pdf(root / "manual.pdf", parsed)
            chunks = build_chunks(json.loads(parsed.read_text(encoding="utf-8")))
            if not chunks:
                raise ValueError("文档没有可检索内容")
            phase = "建立索引"
            for index, chunk in enumerate(chunks):
                chunk.update(doc_id=document_id, source=doc["filename"],
                             chunk_id=str(uuid5(UUID(document_id), str(index))))
            vector_store.create_collection()
            for start in range(0, len(chunks), 32):
                batch = chunks[start:start + 32]
                vectors = embedding_provider.embed_documents([build_retrieval_text(c) for c in batch])
                points = [
                    models.PointStruct(id=chunk["chunk_id"], vector=vector, payload=build_payload(chunk))
                    for chunk, vector in zip(batch, vectors, strict=True)
                ]
                vector_store.upsert_points(points)
            (root / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            change_status(document_id, "ready")
        except Exception:
            logger.exception("文档 %s 在%s阶段失败", document_id, phase)
            change_status(document_id, "failed", f"{phase}失败，请检查服务日志")
            try:
                vector_store.delete_document(document_id)
            except Exception:
                logger.exception("失败文档的向量清理失败")


def delete_document(document_id, vector_store):
    change_status(document_id, "deleting")
    try:
        vector_store.delete_document(document_id)
        root = document_directory(document_id)
        if root.exists():
            shutil.rmtree(root)
        with sessions().begin() as session:
            scoped = select(Conversation.id).where(
                Conversation.state["document_id"].as_string() == document_id
            )
            session.execute(delete(Message).where(Message.conversation_id.in_(scoped)))
            session.execute(delete(Conversation).where(Conversation.id.in_(scoped)))
            doc = session.get(Document, document_id)
            if doc is not None:
                session.delete(doc)
    except Exception:
        change_status(document_id, "failed", "清理失败，可以重试删除")
        raise
