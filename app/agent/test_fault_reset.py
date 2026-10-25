from app.agent.agent import MaintMateAgent
from app.agent.state import (
    AgentState,
    TroubleshootingRecord
)

def main():
    # 不调用 RAG，只验证状态清理方法
    agent = MaintMateAgent.__new__(MaintMateAgent)
    state = AgentState(
        user_query="MD380E 报 Err16"
    )
    state.intent = "fault_diagnosis"
    state.diagnostic_context.equipment_model = "MD380E"
    state.diagnostic_context.fault_code = "Err16"
    state.diagnostic_context.user_role = "maintenance"
    state.pending_query = "MD380E Err16 怎么处理？"
    state.rag_context = "旧故障的手册证据"
    state.troubleshooting_history.append(
        TroubleshootingRecord(
            action="检查通讯连接线",
            result="not_resolved"
        )
    )
    # 模拟切换新故障
    agent._reset_fault_context(state)
    # 验证用户身份保留
    assert state.diagnostic_context.user_role == "maintenance"
    # 验证旧故障信息清理
    assert state.diagnostic_context.fault_code is None
    assert state.diagnostic_context.equipment_model is None
    assert state.pending_query is None
    assert state.rag_context is None
    assert state.troubleshooting_history == []
    assert state.safety_checked is False
    print("故障状态重置测试通过！")


if __name__ == "__main__":
    main()
