from types import SimpleNamespace
from unittest.mock import Mock, patch
from app.agent.agent import MaintMateAgent
from app.agent.state import (
    AgentState,
    TroubleshootingRecord,
    TroubleshootingStep,
)

def create_agent():
    """绕过初始化，不加载 RAG 和模型。"""
    agent = object.__new__(MaintMateAgent)
    agent.input_analyzer = Mock()
    agent.step_matcher = Mock()
    return agent

def create_old_fault_state():
    """模拟一个正在处理的 Err16 故障。"""
    state = AgentState(
        user_query="MD380E 出现 Err16"
    )

    state.intent = "fault_diagnosis"
    state.pending_query = "MD380E Err16"

    state.diagnostic_context.equipment_model = "MD380E"
    state.diagnostic_context.fault_code = "Err16"

    state.troubleshooting_steps = [
        TroubleshootingStep(
            step_id="step_1",
            action="检查上位机接线",
            source_page=249
        )
    ]

    return state

def test_resolved_skips_step_matcher():
    """测试1：故障解决后不调用 StepMatcher。"""

    agent = create_agent()
    state = create_old_fault_state()

    analysis = SimpleNamespace(
        intent="unknown",
        turn_relation="continue_fault",
        user_role="unknown",
        escalation_requested=False,
        troubleshooting_feedback=[
            TroubleshootingRecord(
                action="确认 Err16 故障已消失",
                result="resolved"
            )
        ]
    )

    agent.input_analyzer.analyze.return_value = analysis#type:ignore

    def fake_update_state(state, analysis):
        state.intent = analysis.intent

    agent.input_analyzer.update_state.side_effect = (#type:ignore
        fake_update_state
    )

    answer = agent.process(
        state,
        user_input="Err16 已经解决，设备恢复正常。"
    )

    assert "故障已经解决" in answer
    assert state.intent == "fault_diagnosis"

    # 故障解决记录必须保留
    assert len(state.troubleshooting_history) == 1
    assert state.troubleshooting_history[0].result == "resolved"
    assert state.troubleshooting_history[0].step_id is None

    # 不应该进行无意义的步骤匹配
    agent.step_matcher.match.assert_not_called()#type:ignore

    print("测试1通过：故障解决后跳过 StepMatcher")


def test_new_fault_resets_old_state():
    """测试2：新故障不会继承旧故障信息。"""

    agent = create_agent()
    state = create_old_fault_state()

    state.rag_context = "旧故障的手册证据"

    state.troubleshooting_history.append(
        TroubleshootingRecord(
            action="检查上位机接线",
            result="not_resolved",
            step_id="step_1"
        )
    )

    analysis = SimpleNamespace(
        intent="fault_diagnosis",
        turn_relation="new_fault",
        user_role="unknown",
        escalation_requested=False,
        troubleshooting_feedback=[]
    )

    agent.input_analyzer.analyze.return_value = analysis#type:ignore

    def fake_update_state(state, analysis):
        state.intent = "fault_diagnosis"
        state.diagnostic_context.fault_code = "Err22"

    agent.input_analyzer.update_state.side_effect = (#type:ignore
        fake_update_state
    )

    agent.sufficiency_policy = Mock()
    agent.sufficiency_policy.evaluate.return_value = (
        SimpleNamespace(
            missing_fields=[],
            next_action="answer"
        )
    )

    with patch.object(
        agent,
        "_execute_action",
        return_value="已开始处理 Err22"
    ):
        answer = agent.process(
            state,
            user_input="另一台设备出现 Err22"
        )

    assert answer == "已开始处理 Err22"

    # 当前信息应属于新故障
    assert state.diagnostic_context.fault_code == "Err22"

    # 旧故障数据已清除
    assert state.pending_query is None
    assert state.rag_context is None
    assert state.troubleshooting_history == []

    print("测试2通过：新故障正确重置旧状态")


def test_escalation_uses_latest_information():
    """测试3：升级前先保存本轮最新信息。"""

    agent = create_agent()

    state = AgentState(
        user_query="设备故障咨询"
    )

    analysis = SimpleNamespace(
        intent="fault_diagnosis",
        turn_relation="new_fault",
        user_role="unknown",
        escalation_requested=True,
        escalation_target="vendor",
        troubleshooting_feedback=[]
    )

    agent.input_analyzer.analyze.return_value = analysis#type:ignore

    def fake_update_state(state, analysis):
        state.intent = "fault_diagnosis"
        state.diagnostic_context.equipment_model = "MD380E"
        state.diagnostic_context.fault_code = "Err16"

    agent.input_analyzer.update_state.side_effect = (#type:ignore
        fake_update_state
    )

    def fake_escalation(state, reason, target):
        # 升级时应该已经获得最新故障信息
        assert state.diagnostic_context.equipment_model == "MD380E"
        assert state.diagnostic_context.fault_code == "Err16"
        assert reason == "user_requested"
        assert target == "vendor"
        return "已生成厂家技术支持交接信息"

    with patch.object(
        agent,
        "_handle_escalation",
        side_effect=fake_escalation
    ) as mock_escalation:
        answer = agent.process(
            state,
            user_input="MD380E 报 Err16，请帮我整理厂家支持信息"
        )

    assert "交接信息" in answer
    mock_escalation.assert_called_once()

    print("测试3通过：升级时已保存本轮故障信息")


def main():
    test_resolved_skips_step_matcher()
    test_new_fault_resets_old_state()
    test_escalation_uses_latest_information()

    print("\n========== 测试结果 ==========")
    print("process Mock 工作流测试全部通过！")


if __name__ == "__main__":
    main()
