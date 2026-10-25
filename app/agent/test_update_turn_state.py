from types import SimpleNamespace
from unittest.mock import Mock
from app.agent.agent import MaintMateAgent
from app.agent.state import AgentState

def main():
    # 不调用 __init__，避免加载 RAG 和本地模型
    agent = object.__new__(MaintMateAgent)
    # 用 Mock 替代真实 InputAnalyzer
    agent.input_analyzer = Mock()
    def fake_update_state(state, analysis):
        # 模拟 InputAnalyzer 将意图更新为 unknown
        state.intent = analysis.intent
    agent.input_analyzer.update_state.side_effect = (
        fake_update_state
    )
    # 模拟当前存在正在处理的故障
    state = AgentState(
        user_query="MD380E Err16 怎么处理？"
    )
    state.intent = "fault_diagnosis"
    # 模拟本轮用户补充的信息
    analysis = SimpleNamespace(
        intent="unknown",
        turn_relation="continue_fault",
        troubleshooting_feedback=[]
    )
    # 执行被测试的方法
    agent._update_turn_state(
        state=state,
        analysis=analysis,
        previous_intent="fault_diagnosis",
        has_active_fault=True,
        waiting_for_user_role=False
    )
    print("更新后的意图：", state.intent)
    # 验证原故障意图被保留
    assert state.intent == "fault_diagnosis"
    # 验证 InputAnalyzer 确实执行过状态更新
    agent.input_analyzer.update_state.assert_called_once()
    print("状态更新 Mock 测试通过！")


if __name__ == "__main__":
    main()
