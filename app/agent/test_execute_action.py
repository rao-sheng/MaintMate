
from unittest.mock import Mock, patch

from app.agent.agent import MaintMateAgent
from app.agent.state import AgentState


def create_agent():
    """跳过 __init__，避免加载模型。"""
    return object.__new__(MaintMateAgent)


def create_state(action):
    """创建指定 next_action 的测试状态。"""
    state = AgentState(
        user_query="MD380E Err16 怎么处理？"
    )
    state.next_action = action
    return state


def test_ask_user():
    """测试1：执行追问动作。"""

    agent = create_agent()
    state = create_state("ask_user")

    agent.followup_builder = Mock()
    agent.followup_builder.build.return_value = (
        "请提供设备型号和故障代码。"
    )

    answer = agent._execute_action(state)

    assert answer == "请提供设备型号和故障代码。"
    assert state.final_answer == answer

    agent.followup_builder.build.assert_called_once_with(
        state
    )

    print("测试1通过：ask_user")


def test_answer():
    """测试2：执行普通回答动作。"""

    agent = create_agent()
    state = create_state("answer")

    with patch.object(
        agent,
        "_answer_without_rag",
        return_value="你好，我是 MaintMate。"
    ) as mock_answer:

        answer = agent._execute_action(state)

        mock_answer.assert_called_once_with(state)

    assert answer == "你好，我是 MaintMate。"
    assert state.final_answer == answer

    print("测试2通过：answer")


def test_retrieve():
    """测试3：正确调用检索处理函数。"""

    agent = create_agent()
    state = create_state("retrieve")

    with patch.object(
        agent,
        "_handle_retrieval",
        return_value="Err16 表示通讯故障。"
    ) as mock_retrieve:

        answer = agent._execute_action(state)

        mock_retrieve.assert_called_once_with(state)

    assert answer == "Err16 表示通讯故障。"

    print("测试3通过：retrieve")


def test_escalate():
    """测试4：正确调用统一升级函数。"""

    agent = create_agent()
    state = create_state("escalate")

    state.escalation_reason = "troubleshooting_failed"
    state.escalation_target = "vendor"

    with patch.object(
        agent,
        "_handle_escalation",
        return_value="已整理厂家技术支持交接信息。"
    ) as mock_escalation:

        answer = agent._execute_action(state)

        mock_escalation.assert_called_once_with(
            state=state,
            reason="troubleshooting_failed",
            target="vendor"
        )

    assert "交接信息" in answer

    print("测试4通过：escalate")


def test_escalate_missing_fields():
    """测试5：升级信息不完整时拒绝执行。"""

    agent = create_agent()
    state = create_state("escalate")

    state.escalation_reason = None
    state.escalation_target = None

    with patch.object(
        agent,
        "_handle_escalation"
    ) as mock_escalation:

        try:
            agent._execute_action(state)

        except RuntimeError as error:
            assert "升级操作缺少" in str(error)

        else:
            raise AssertionError(
                "升级信息缺失时应该抛出 RuntimeError"
            )

        mock_escalation.assert_not_called()

    print("测试5通过：升级缺少字段时拒绝执行")


def test_unknown_action():
    """测试6：未知动作应该抛出异常。"""

    agent = create_agent()
    state = create_state("answer")

    # 仅为测试未知值，绕过 Pydantic 字段赋值校验
    object.__setattr__(
        state,
        "next_action",
        "invalid_action"
    )

    try:
        agent._execute_action(state)

    except RuntimeError as error:
        assert "未知 Agent 动作" in str(error)

    else:
        raise AssertionError(
            "未知动作应该抛出 RuntimeError"
        )

    print("测试6通过：未知动作抛出异常")


def main():
    test_ask_user()
    test_answer()
    test_retrieve()
    test_escalate()
    test_escalate_missing_fields()
    test_unknown_action()

    print("\n========== 测试结果 ==========")
    print("execute_action Mock 测试全部通过！")


if __name__ == "__main__":
    main()
