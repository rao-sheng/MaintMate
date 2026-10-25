from app.agent.agent import MaintMateAgent
from app.agent.state import AgentState

def main():
    # 不初始化 RAG，直接测试已有证据下的权限分支
    agent = MaintMateAgent.__new__(MaintMateAgent)
    state = AgentState(
        user_query="我想执行一项厂家专属维修操作"
    )
    state.diagnostic_context.equipment_model = "MD380E"
    state.diagnostic_context.user_role = "maintenance"

    # 模拟已经完成 RAG 和 Safety 分析
    state.pending_query = state.user_query
    state.rag_context = "模拟的手册证据"
    state.required_role = "vendor_technician"
    state.safety_checked = True
    state.safety_reason = "该操作明确要求厂家技术人员执行"

    answer = agent._finish_from_evidence(state)

    print("\n========== 权限升级测试 ==========")
    print(answer)

    print("\n最终动作：", state.next_action)
    print("升级原因：", state.escalation_reason)
    print("升级目标：", state.escalation_target)

    assert state.next_action == "escalate"
    assert state.escalation_reason == "permission_required"
    assert state.escalation_target == "vendor"
    assert state.escalated is True

    print("\n权限升级测试通过！")


if __name__ == "__main__":
    main()
