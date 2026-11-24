import asyncio
import os
from contextlib import ExitStack, asynccontextmanager
import logging
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import documents
from app.agent.agent import MaintMateAgent
from app.agent.evidence import EvidenceUnavailable
from app.agent.state import AgentState

logger = logging.getLogger(__name__)
sessions: dict[str, AgentState] = {}
agent: MaintMateAgent | None = None
session_locks = [Lock() for _ in range(64)]
document_locks = [Lock() for _ in range(64)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = MaintMateAgent()
    try:
        for attempt in range(10 if os.getenv("QDRANT_URL") else 1):
            try:
                await asyncio.to_thread(agent.rag.vector_store.create_collection)
                break
            except Exception:
                if attempt == 9 or not os.getenv("QDRANT_URL"):
                    raise
                await asyncio.sleep(2)
        documents.recover_interrupted(agent.rag.vector_store)
        app.state.persist_sessions = True
        yield
    finally:
        app.state.persist_sessions = False
        agent.close()
        agent = None


app = FastAPI(lifespan=lifespan)


@app.post("/documents/upload", status_code=202)
def upload_document(file: UploadFile, background: BackgroundTasks):
    current_agent = agent
    if current_agent is None:
        raise HTTPException(503, "服务尚未启动")
    filename = Path((file.filename or "").replace("\\", "/")).name
    if not filename.lower().endswith(".pdf") or file.file.read(5) != b"%PDF-":
        raise HTTPException(415, "只接受 PDF 文件")
    document_id = str(uuid4())
    directory = documents.document_directory(document_id)
    if not documents.job_slots.acquire(blocking=False):
        raise HTTPException(429, "文档处理队列已满，请稍后上传")
    pdf = directory / "manual.pdf"
    try:
        directory.mkdir(parents=True)
        with pdf.open("wb") as output:
            output.write(b"%PDF-")
            size = 5
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > documents.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "PDF 超过 25 MiB")
                output.write(block)
        with documents.sessions().begin() as session:
            session.add(documents.Document(id=document_id, filename=filename[:255], status="pending"))
    except Exception:
        documents.job_slots.release()
        pdf.unlink(missing_ok=True)
        if directory.exists():
            directory.rmdir()
        raise
    background.add_task(
        documents.run_document_job, document_id, current_agent.rag.vector_store,
        current_agent.rag.embedding_provider,
    )
    return {"document_id": document_id, "status": "pending"}


@app.get("/documents/{document_id}")
def get_document(document_id: UUID):
    result = documents.document_status(str(document_id))
    if result is None:
        raise HTTPException(404, "文档不存在")
    return result


@app.post("/documents/{document_id}/retry", status_code=202)
def retry_document(document_id: UUID, background: BackgroundTasks):
    current_agent = agent
    if current_agent is None:
        raise HTTPException(503, "服务尚未启动")
    if not documents.job_slots.acquire(blocking=False):
        raise HTTPException(429, "文档处理队列已满，请稍后重试")
    try:
        with documents.job_lock:
            doc = get_document(document_id)
            if doc["status"] != "failed" or not (documents.document_directory(str(document_id)) / "manual.pdf").is_file():
                raise HTTPException(409, "当前文档无法重试")
            current_agent.rag.vector_store.delete_document(str(document_id))
            documents.change_status(str(document_id), "pending")
    except Exception:
        documents.job_slots.release()
        raise
    background.add_task(documents.run_document_job, str(document_id), current_agent.rag.vector_store,
                        current_agent.rag.embedding_provider)
    return {"document_id": str(document_id), "status": "pending"}


@app.delete("/documents/{document_id}", status_code=204)
def remove_document(document_id: UUID):
    if agent is None:
        raise HTTPException(503, "服务尚未启动")
    if not documents.job_lock.acquire(blocking=False):
        raise HTTPException(409, "正在处理文档，请稍后删除")
    try:
        doc_id = str(document_id)
        lock = document_locks[hash(doc_id) % len(document_locks)]
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "文档正在用于问答，请稍后删除")
        try:
            get_document(document_id)
            documents.delete_document(doc_id, agent.rag.vector_store)
        finally:
            lock.release()
    finally:
        documents.job_lock.release()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None
    document_id: UUID | None = None


class Source(BaseModel):
    reference: int
    document_id: str | None
    manual: str | None
    page_start: int | None
    page_end: int | None
    excerpt: str | None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    retrieval_attempts: int = 0


def load_state(session_id, persist):
    if persist:
        with documents.sessions()() as session:
            conversation = session.get(documents.Conversation, session_id)
            return AgentState.model_validate(conversation.state) if conversation else None
    state = sessions.get(session_id)
    return state.model_copy(deep=True) if state else None


def save_state(session_id, state, message, answer, persist):
    if not persist:
        sessions[session_id] = state
        return
    with documents.sessions().begin() as session:
        session.merge(documents.Conversation(id=session_id, state=state.model_dump(mode="json")))
        session.flush()
        session.add_all([
            documents.Message(conversation_id=session_id, role="user", content=message),
            documents.Message(conversation_id=session_id, role="assistant", content=answer),
        ])


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if agent is None:
        raise RuntimeError("Agent未初始化")
    session_id = request.session_id or str(uuid4())
    selected = str(request.document_id) if request.document_id else None
    persist = getattr(app.state, "persist_sessions", False)
    with session_locks[hash(session_id) % len(session_locks)], ExitStack() as resources:
        if request.session_id is None:
            state = AgentState(user_query=request.message, document_id=selected)
            if selected:
                state.diagnostic_context.doc_ids = [selected]
        else:
            state = load_state(session_id, persist)
            if state is None:
                raise HTTPException(404, "会话不存在，请重新创建会话")
            if selected is not None and selected != state.document_id:
                raise HTTPException(409, "切换手册请新建会话")
        if state.document_id:
            resources.enter_context(document_locks[hash(state.document_id) % len(document_locks)])
            document = documents.document_status(state.document_id)
            if document is None:
                raise HTTPException(404, "文档不存在")
            if document["status"] != "ready":
                raise HTTPException(409, "文档尚未就绪")
            state.diagnostic_context.doc_ids = [state.document_id]
        state.retrieval_attempts = 0
        try:
            answer = agent.process(state, user_input=request.message)
            save_state(session_id, state, request.message, answer, persist)
        except EvidenceUnavailable:
            logger.exception("证据判断服务请求失败")
            raise HTTPException(503, "证据判断服务暂不可用，请稍后重试")
        except Exception:
            logger.exception("Agent 处理请求失败")
            raise HTTPException(500, "系统处理请求失败，请稍后再试")
        sources = [
            Source(reference=index, document_id=payload.get("doc_id"),
                   manual=payload.get("source"), page_start=payload.get("page_start"),
                   page_end=payload.get("page_end"),
                   excerpt=(payload.get("content") or "")[:600] if state.required_role == "user" else None)
            for index, result in enumerate(state.retrieved_sources, 1)
            if (payload := result.get("payload"))
        ]
        return ChatResponse(session_id=session_id, answer=answer, sources=sources,
                            retrieval_attempts=state.retrieval_attempts)


@app.get("/conversations/{session_id}/report")
def conversation_report(session_id: str):
    """Generate a report from the saved conversation without advancing the agent."""
    from fastapi.responses import Response
    from app.agent.report import generate_troubleshooting_report

    persist = getattr(app.state, "persist_sessions", False)
    try:
        with session_locks[hash(session_id) % len(session_locks)]:
            state = load_state(session_id, persist)
        if state is None:
            raise HTTPException(404, "会话不存在")
        report = generate_troubleshooting_report(state)
    except (ValueError, TypeError):
        logger.exception("会话报告数据格式异常")
        raise HTTPException(422, "会话数据格式异常，无法生成报告")
    except HTTPException:
        raise
    except Exception:
        logger.exception("生成排查报告失败")
        raise HTTPException(500, "生成排查报告失败")
    return Response(content=report, media_type="text/markdown")
