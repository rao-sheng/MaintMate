import importlib
from unittest.mock import Mock,patch
from fastapi.testclient import TestClient

def test_session_isolation():
    api=importlib.import_module("app.api.main")
    received_states=[]
    def fake_process(state,user_input):
        received_states.append(state)
        if "ERR16" in user_input:
            state.diagnostic_context.fault_code="ERR16"
        elif "ERR22" in user_input:
            state.diagnostic_context.fault_code="ERR22"
        return "模拟回答"
    mock_agent=Mock()
    mock_agent.process.side_effect=fake_process
    with (
        patch.object(api,"agent",mock_agent),
        patch.dict(api.sessions,{},clear=True)
    ):
        client=TestClient(api.app)
        response_a=client.post(
            "/chat",
            json={
                "session_id":None,
                "message":"设备A出现ERR16"
            }
        )
        assert response_a.status_code==200
        id_a=response_a.json()["session_id"]
        response_b=client.post(
            "/chat",
            json={
                "session_id":None,
                "message":"设备B出现ERR22"
            }
        )
        assert response_b.status_code==200
        id_b=response_b.json()["session_id"]
        assert id_a!=id_b
        assert api.sessions[id_a] is not api.sessions[id_b]
        assert (
            api.sessions[id_a].diagnostic_context.fault_code
            =="ERR16"
        )
        assert (
            api.sessions[id_b].diagnostic_context.fault_code
        )
        response_a2=client.post(
            "/chat",
            json={
                "session_id":id_a,
                "message":"继续之前的故障"
            }
        )
        assert response_a2.status_code==200
        assert response_a2.json()["session_id"]==id_a
        assert received_states[0] is not received_states[2]

        assert (
            received_states[2].diagnostic_context.fault_code
            == "ERR16"
        )
        assert (
            api.sessions[id_b].diagnostic_context.fault_code
            =="ERR22"
        )
        assert mock_agent.process.call_count==3
    print("测试通过")
if __name__=="__main__":
    test_session_isolation()