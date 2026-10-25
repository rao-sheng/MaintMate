from app.agent.agent import MaintMateAgent
from app.agent.state import (
    AgentState,
    TroubleshootingRecord,
)

def test_continue_old_fault(agent: MaintMateAgent):
    """测试用户确认继续之前的故障。"""

    print("\n========== 测试1：继续旧故障 ==========")

    state = AgentState(
        user_query="MD380E 出现 Err16"
    )
    # 模拟之前已经处理过 Err16
    state.intent = "fault_diagnosis"
    state.pending_query = "MD380E Err16 故障"
    state.diagnostic_context.equipment_model = "MD380E"
    state.diagnostic_context.fault_code = "Err16"

    # 模拟上一轮正在等待故障关系确认
    state.next_action = "ask_user"
    state.missing_fields = ["fault_relation"]

    # 本轮用户确认是原来的故障
    answer = agent.process(
        state,
        user_input="是之前的故障"
    )

    print("回答：", answer)
    print("故障代码：", state.diagnostic_context.fault_code)
    print("下一步动作：", state.next_action)
    print("缺失字段：", state.missing_fields)

    # 验证旧故障信息没有被清除
    assert state.diagnostic_context.fault_code == "Err16"
    assert state.diagnostic_context.equipment_model == "MD380E"
    assert state.pending_query == "MD380E Err16 故障"

    # 验证等待状态已经结束
    assert "fault_relation" not in state.missing_fields
    assert state.next_action == "answer"

    # 验证回复内容
    assert "继续处理之前的设备故障" in answer

    print("测试1通过！")


def test_start_new_fault(agent: MaintMateAgent):
    """测试用户确认开始一个新故障。"""

    print("\n========== 测试2：开始新故障 ==========")

    # 使用独立 State，避免受到测试1影响
    state = AgentState(
        user_query="MD380E 出现 Err16"
    )

    # 模拟旧故障上下文
    state.intent = "fault_diagnosis"
    state.pending_query = "MD380E Err16 故障"
    state.diagnostic_context.equipment_model = "MD380E"
    state.diagnostic_context.fault_code = "Err16"

    state.rag_context = "之前检索到的 Err16 手册证据"

    # 模拟之前已有排障记录
    state.troubleshooting_history.append(
        TroubleshootingRecord(
            action="检查通讯连接线",
            result="not_resolved",
            step_id="step_2"
        )
    )

    # 模拟正在等待确认新旧故障关系
    state.next_action = "ask_user"
    state.missing_fields = ["fault_relation"]

    # 本轮用户确认是新故障
    answer = agent.process(
        state,
        user_input="这是新故障"
    )

    print("回答：", answer)
    print("设备型号：", state.diagnostic_context.equipment_model)
    print("故障代码：", state.diagnostic_context.fault_code)
    print("排障历史：", state.troubleshooting_history)
    print("RAG证据：", state.rag_context)
    print("下一步动作：", state.next_action)

    # 验证旧故障数据已经清除
    assert state.diagnostic_context.fault_code is None
    assert state.diagnostic_context.equipment_model is None
    assert state.pending_query is None
    assert state.rag_context is None
    assert len(state.troubleshooting_history) == 0

    # 验证开始询问新故障信息
    assert state.next_action == "ask_user"
    assert "fault_relation" not in state.missing_fields
    assert "equipment_model" in state.missing_fields

    assert "开始处理新的设备故障" in answer

    print("测试2通过！")


def main():
    agent = MaintMateAgent()

    try:
        test_continue_old_fault(agent)
        test_start_new_fault(agent)

        print("\n========== 测试结果 ==========")
        print("故障关系确认测试全部通过！")

    finally:
        # 关闭 RAG 等资源
        agent.close()


if __name__ == "__main__":
    main()
