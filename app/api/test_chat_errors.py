import importlib
from unittest.mock import Mock,patch
from fastapi.testclient import TestClient

api=importlib.import_module("app.api.main")

def test_success():
    """测试正常请求返回200，并保存会话"""
    mock_agent=Mock()
    mock_agent.process.return_value="模拟成功回答"
    with(
        patch.object(api,"agent",mock_agent),
        patch.dict(api.sessions,{},clear=True)
    ):
        client=TestClient(api.app)
        response=client.post(
            "/chat",
            json={
                "session_id":None,
                "message":"MD380E出现ERR16"
            }
        )
        assert response.status_code==200
        assert response.json()["answer"]=="模拟成功回答"

        session_id=response.json()["session_id"]
        assert session_id in api.sessions
        mock_agent.process.assert_called_once()

def test_session_not_found():
    """测试不存在的会话返回404"""
    mock_agent=Mock()
    with(
        patch.object(api,"agent",mock_agent),
        patch.dict(api.sessions,{},clear=True)
    ):
        client=TestClient(api.app)
        response=client.post(
            "/chat",
            json={
                "session_id":"wrong123",
                "message":"我是维修人员"
            }
        )
        assert response.status_code==404
        #找不到会话是不应该调用Agent
        mock_agent.process.assert_not_called()

def test_agent_failure():
    mock_agent=Mock()
    #模拟LLM调用失败
    mock_agent.process.side_effect=RuntimeError(
        "模拟LLM接口故障"
    )
    with(
        patch.object(api,"agent",mock_agent),
        patch.dict(api.sessions,{},clear=True)
    ):
        client=TestClient(api.app)
        response=client.post(
            "/chat",
            json={
                "session_id":None,
                "message":"MD380E出现ERR16"
            }
        )
        assert response.status_code==500
        assert response.json()["detail"]==(
            "系统处理请求失败，请稍后再试"
        )
        assert len(api.sessions)==0
        mock_agent.process.assert_called_once()

def test_existing_session_failure():
    """已有会话处理失败 原有状态不能被修改"""
    mock_agent=Mock()
    with(
        patch.object(api,"agent",mock_agent),
        patch.dict(api.sessions,{},clear=True)
    ):
        client=TestClient(api.app)
        #准备已有会话
        session_id="session_001"
        state=api.AgentState(
            user_query="MD380E出现ERR16"
        )
        state.diagnostic_context.fault_code="ERR16"
        api.sessions[session_id]=state
        #模拟先修改state 再发生异常
        def fake_process(state,user_input):
            state.diagnostic_context.fault_code="ERR22"
            raise RuntimeError("模拟Agent中途失败")
        mock_agent.process.side_effect=fake_process
        #发送请求
        response = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "另一台设备出现 Err22"
            }
        )
        assert response.status_code==500
        assert(
            api.sessions[session_id].diagnostic_context.fault_code
            =="ERR16"
        )
        assert api.sessions[session_id] is state
    print("已有会话异常保护测试通过")

if __name__=="__main__":
    test_success()
    test_session_not_found()
    test_agent_failure()
    test_existing_session_failure()
    print("四个测试通过")