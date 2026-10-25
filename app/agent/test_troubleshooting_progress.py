
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.agent import MaintMateAgent
from app.agent.state import (
    AgentState,
    TroubleshootingRecord,
    TroubleshootingStep,
)


def create_agent():
    """跳过 __init__，避免加载模型和 RAG。"""
    return object.__new__(MaintMateAgent)


def create_state():
    """创建满足继续排障条件的测试状态。"""

    state = AgentState(
        user_query="MD380E Err16 怎么处理？"
    )

    state.intent = "fault_diagnosis"
    state.pending_query = "MD380E Err16"
    state.rag_context = "模拟的 Err16 手册证据"
    state.safety_checked = True
    state.required_role = "maintenance"

    state.diagnostic_context.equipment_model = "MD380E"
    state.diagnostic_context.fault_code = "Err16"
    state.diagnostic_context.user_role = "maintenance"

    return state


def create_analysis():
    """模拟用户继续反馈故障未解决。"""

    return SimpleNamespace(
        turn_relation="continue_fault",
        troubleshooting_feedback=[
            TroubleshootingRecord(
                action="检查通讯连接线",
                result="not_resolved",
                step_id="step_2"
            )
        ]
    )


def test_permission_check():
    """测试1：没有维修权限时，不进入排障处理。"""

    agent = create_agent()
    state = create_state()
    analysis = create_analysis()

    state.diagnostic_context.user_role = "user"

    with patch(
        "app.agent.agent.evaluate_troubleshooting"
    ) as mock_evaluate:

        answer = agent._handle_troubleshooting_progress(
            state=state,
            analysis=analysis,
            has_active_fault=True
        )

        assert answer is None
        mock_evaluate.assert_not_called()

    print("测试1通过：权限不足时不进入排障分支")


def test_need_confirmation():
    """测试2：部分步骤未确认，返回追问。"""

    agent = create_agent()
    state = create_state()
    analysis = create_analysis()

    pending_steps = [
        TroubleshootingStep(
            step_id="step_3",
            action="正确设置通讯扩展卡F0-28类型",
            source_page=249
        ),
        TroubleshootingStep(
            step_id="step_4",
            action="正确设置通讯参数FD组",
            source_page=249
        )
    ]

    with patch(
        "app.agent.agent.evaluate_troubleshooting",
        return_value=("need_confirmation", pending_steps)
    ) as mock_evaluate:

        answer = agent._handle_troubleshooting_progress(
            state=state,
            analysis=analysis,
            has_active_fault=True
        )

        mock_evaluate.assert_called_once_with(
            state.troubleshooting_steps,
            state.troubleshooting_history
        )

    assert answer is not None
    assert "F0-28" in answer
    assert "FD组" in answer
    assert state.next_action == "answer"
    assert state.final_answer == answer

    print("测试2通过：正确追问未确认步骤")


def test_suggest_escalation():
    """测试3：已提取步骤全部失败，建议升级。"""

    agent = create_agent()
    state = create_state()
    analysis = create_analysis()

    with patch(
        "app.agent.agent.evaluate_troubleshooting",
        return_value=("suggest_escalation", [])
    ):

        answer = agent._handle_troubleshooting_progress(
            state=state,
            analysis=analysis,
            has_active_fault=True
        )

    assert answer is not None
    assert "厂家" in answer
    assert state.next_action == "answer"
    assert state.final_answer == answer

    # 记录升级原因，但尚未真正执行升级交接
    assert state.escalation_reason == "troubleshooting_failed"
    assert state.escalated is False

    print("测试3通过：正确建议升级，但未直接升级")


def test_unrelated_feedback():
    """测试4：不属于当前故障延续，不进入排障处理。"""

    agent = create_agent()
    state = create_state()
    analysis = create_analysis()

    analysis.turn_relation = "unrelated"

    with patch(
        "app.agent.agent.evaluate_troubleshooting"
    ) as mock_evaluate:

        answer = agent._handle_troubleshooting_progress(
            state=state,
            analysis=analysis,
            has_active_fault=True
        )

        assert answer is None
        mock_evaluate.assert_not_called()

    print("测试4通过：无关输入不进入排障分支")


def test_clear_pending_escalation():
    """测试5：用户继续排障时，清除旧升级建议并重新评估。"""

    agent = create_agent()
    state = create_state()
    analysis = create_analysis()

    # 模拟之前建议过升级
    state.escalation_reason = "troubleshooting_failed"
    state.escalation_target = "vendor"
    state.escalated = False

    pending_steps = [
        TroubleshootingStep(
            step_id="step_4",
            action="正确设置通讯参数FD组",
            source_page=249
        )
    ]

    with patch(
        "app.agent.agent.evaluate_troubleshooting",
        return_value=("need_confirmation", pending_steps)
    ):

        answer = agent._handle_troubleshooting_progress(
            state=state,
            analysis=analysis,
            has_active_fault=True
        )

    assert answer is not None
    assert "FD组" in answer

    # 旧的待升级原因被清除
    assert state.escalation_reason is None
    assert state.escalation_target is None

    print("测试5通过：重新排障时清除过期升级建议")


def main():

    test_permission_check()
    test_need_confirmation()
    test_suggest_escalation()
    test_unrelated_feedback()
    test_clear_pending_escalation()

    print("\n========== 测试结果 ==========")
    print("排障进度处理 Mock 测试全部通过！")


if __name__ == "__main__":
    main()
