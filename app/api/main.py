from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from app.agent.agent import MaintMateAgent
from app.agent.state import AgentState
import logging

logger=logging.getLogger(__name__)
#1.保存不同会话的AgentState
sessions:dict[str,AgentState]={}

#保存全局Agent实例
agent:MaintMateAgent|None=None

#管理Agent的启动和关闭
@asynccontextmanager
async def lifespan(app:FastAPI):
    global agent
    agent=MaintMateAgent()
    try:
        yield
    finally:
        agent.close()
        agent=None
app=FastAPI(lifespan=lifespan)

#请求模型
class ChatRequest(BaseModel):
    message:str=Field(min_length=1)
    session_id:str|None=None

#响应模型
class ChatResponse(BaseModel):
    session_id:str
    answer:str

#对外提供的接口
@app.post("/chat",response_model=ChatResponse)
def chat(request:ChatRequest):
    if agent is None:
        raise RuntimeError("Agent未初始化")
    if request.session_id is None:
        session_id=str(uuid4())
        state=AgentState(user_query=request.message)
        is_new_session=True
    else:
        session_id=request.session_id
        if session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail="会话不存在，请重新创建会话"
            )
        state=sessions[session_id].model_copy(deep=True)
        is_new_session=False
    try:
        answer=agent.process(
            state,
            user_input=request.message
        )
    except Exception:
        logger.exception("Agent 处理请求失败")
        raise HTTPException(
            status_code=500,
            detail="系统处理请求失败，请稍后再试"
        )
    
    sessions[session_id]=state
    return ChatResponse(
        session_id=session_id,
        answer=answer
    )
