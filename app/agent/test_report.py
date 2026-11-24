import pytest
from app.agent.report import generate_troubleshooting_report
from app.agent.state import AgentState, DiagnosticContext, TroubleshootingRecord, TroubleshootingStep


def complete_state():
    return AgentState(
        user_query="Err16",
        diagnostic_context=DiagnosticContext(equipment_model="MD380E", fault_code="Err16"),
        troubleshooting_steps=[TroubleshootingStep(step_id="pending", action="待检查项目")],
        troubleshooting_history=[TroubleshootingRecord(action="已检查项目", result="resolved")],
        retrieved_sources=[{"payload": {"source": "设备手册", "page_start": 249,
                                       "page_end": 249, "content": "通讯故障"}}],
    )


def test_complete_report_and_unchanged_state():
    state = complete_state()
    before = state.model_dump()
    report = generate_troubleshooting_report(state)
    for expected in ("MD380E", "Err16", "已检查项目", "已解决", "设备手册", "249", "通讯故障", "后续建议"):
        assert expected in report
    assert "待检查项目" not in report.split("## 排查结果")[0]
    assert state.model_dump() == before


@pytest.mark.parametrize("field,label", [("equipment_model", "设备型号"), ("fault_code", "故障代码")])
def test_missing_fields(field, label):
    state = complete_state()
    setattr(state.diagnostic_context, field, None)
    assert f"{label}：未提供" in generate_troubleshooting_report(state)


def test_no_evidence_or_feedback():
    report = generate_troubleshooting_report(AgentState(user_query="问题"))
    assert "未生成手册引用" in report
    assert "尚未确认解决" in report
    assert "249" not in report
    assert "未提供已执行步骤记录" in report


def test_unresolved_and_conflicting_results():
    state = complete_state()
    state.troubleshooting_history[0].result = "not_resolved"
    assert "故障状态：尚未解决" in generate_troubleshooting_report(state)
    state.troubleshooting_history.append(TroubleshootingRecord(action="已检查项目", result="resolved"))
    assert "相互冲突" in generate_troubleshooting_report(state)


@pytest.mark.parametrize("value", [None, {}, "", []])
def test_invalid_input(value):
    with pytest.raises(ValueError):
        generate_troubleshooting_report(value)


def test_malformed_evidence():
    state = complete_state()
    state.retrieved_sources = [{"payload": "invalid"}]
    with pytest.raises(ValueError):
        generate_troubleshooting_report(state)


def test_report_api(monkeypatch):
    import app.api.main as api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "load_state", lambda session_id, persist: complete_state() if session_id == "known" else None)
    client = TestClient(api.app)
    response = client.get("/conversations/known/report")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "# 故障排查报告" in response.text
    assert client.get("/conversations/missing/report").status_code == 404


def test_failed_step_then_successful_step_is_resolved():
    state = complete_state()
    state.troubleshooting_history.insert(0, TroubleshootingRecord(action="先前检查", result="not_resolved"))
    assert "故障状态：已解决" in generate_troubleshooting_report(state)


def test_api_invalid_state(monkeypatch):
    import app.api.main as api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "load_state", lambda *args: {})
    assert TestClient(api.app).get("/conversations/bad/report").status_code == 422
