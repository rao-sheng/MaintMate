
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.agent import MaintMateAgent
from app.agent.state import (
    AgentState,
    TroubleshootingRecord,
)


def create_agent():
    """跳过初始化，不加载模型和 RAG。"""
    return object.__new__(MaintMateAgent)


def create_state():
    """模拟正在处理 MD380E Err16 故障。"""

    state = AgentState(
        user_query="MD380E 出现 Err16"
    )

    state.intent = "fault_diagnosis"
    state.pending_query = "MD380E Err16 故障"

    state.diagnostic_context.equipment_model = "MD380E"
    state.diagnostic_context.fault_code = "Err16"

    state.rag_context = "Err16 手册证据"

    state.troubleshooting_history.append(
        TroubleshootingRecord(
            action="检查通讯连接线",
            result="not_resolved",
            step_id="step_2"
        )
    )

    return state


def create_analysis(
    turn_relation,
    intent="fault_diagnosis",
    user_role="unknown",
    escalation_requested=False
):
    """模拟 InputAnalyzer 的分析结果。"""

    return SimpleNamespace(
        turn_relation=turn_relation,
        intent=intent,
        user_role=user_role,
        escalation_requested=escalation_requested
    )


def test_new_fault():
    """测试1：新故障应该清空旧故障状态。"""

    agent = create_agent()
    state = create_state()
    analysis = create_analysis("new_fault")

    answer, waiting = agent._handle_turn_relation(
        state=state,
        analysis=analysis,
        has_active_fault=True,
        waiting_for_user_role=True
    )

    assert answer is None
    assert waiting is False

    # 原故障上下文应该清空
    assert state.diagnostic_context.fault_code is None
    assert state.pending_query is None
    assert state.rag_context is None
    assert state.troubleshooting_history == []

    print("测试1通过：新故障正确重置状态")


def test_uncertain():
    """测试2：关系不明确时，追问用户。"""

    agent = create_agent()
    state = create_state()
    analysis = create_analysis("uncertain")

    answer, waiting = agent._handle_turn_relation(
        state=state,
        analysis=analysis,
        has_active_fault=True,
        waiting_for_user_role=False
    )

    assert answer is not None
    assert "之前那个故障" in answer

    assert waiting is False
    assert state.next_action == "ask_user"
    assert "fault_relation" in state.missing_fields

    # 不允许擅自清空或合并旧故障
    assert state.diagnostic_context.fault_code == "Err16"
    assert state.pending_query == "MD380E Err16 故障"

    print("测试2通过：不确定关系正确追问")


def test_unrelated():
    """测试3：普通聊天不应该破坏故障状态。"""

    agent = create_agent()
    state = create_state()

    analysis = create_analysis(
        turn_relation="unrelated",
        intent="general_chat"
    )

    # 保存处理前的故障状态
    old_pending_query = state.pending_query
    old_history = [
        record.model_dump()
        for record in state.troubleshooting_history
    ]

    with patch.object(
        agent,
        "_answer_without_rag",
        return_value="你好，有什么可以帮你？"
    ) as mock_answer:

        answer, waiting = agent._handle_turn_relation(
            state=state,
            analysis=analysis,
            has_active_fault=True,
            waiting_for_user_role=False
        )

        mock_answer.assert_called_once_with(state)

    assert answer == "你好，有什么可以帮你？"
    assert waiting is False
    assert state.final_answer == answer

    # 旧故障信息不能丢失
    assert state.diagnostic_context.fault_code == "Err16"
    assert state.pending_query == old_pending_query
    assert [
        record.model_dump()
        for record in state.troubleshooting_history
    ] == old_history

    print("测试3通过：普通聊天保留故障状态")


def test_continue_fault():
    """测试4：继续原故障时不进行特殊处理。"""

    agent = create_agent()
    state = create_state()
    analysis = create_analysis("continue_fault")

    answer, waiting = agent._handle_turn_relation(
        state=state,
        analysis=analysis,
        has_active_fault=True,
        waiting_for_user_role=False
    )

    assert answer is None
    assert waiting is False

    assert state.diagnostic_context.fault_code == "Err16"
    assert state.pending_query == "MD380E Err16 故障"

    print("测试4通过：继续旧故障进入后续流程")


def test_waiting_for_user_role():
    """测试5：用户补充身份时，不应误触发故障关系追问。"""

    agent = create_agent()
    state = create_state()

    analysis = create_analysis(
        turn_relation="uncertain",
        user_role="maintenance"
    )

    answer, waiting = agent._handle_turn_relation(
        state=state,
        analysis=analysis,
        has_active_fault=True,
        waiting_for_user_role=True
    )

    assert answer is None
    assert waiting is True

    assert state.diagnostic_context.fault_code == "Err16"
    assert "fault_relation" not in state.missing_fields

    print("测试5通过：身份回复不被错误拦截")


def test_escalation_request():
    """测试6：明确请求升级时，不应被 uncertain 分支拦截。"""

    agent = create_agent()
    state = create_state()

    analysis = create_analysis(
        turn_relation="uncertain",
        escalation_requested=True
    )

    answer, waiting = agent._handle_turn_relation(
        state=state,
        analysis=analysis,
        has_active_fault=True,
        waiting_for_user_role=False
    )

    # 让 process 后续的升级分支继续执行
    assert answer is None
    assert waiting is False

    # 不应该进入故障关系澄清
    assert "fault_relation" not in state.missing_fields

    print("测试6通过：升级请求未被关系澄清拦截")


def main():
    test_new_fault()
    test_uncertain()
    test_unrelated()
    test_continue_fault()
    test_waiting_for_user_role()
    test_escalation_request()

    print("\n========== 测试结果 ==========")
    print("对话关系分流 Mock 测试全部通过！")


if __name__ == "__main__":
    main()

