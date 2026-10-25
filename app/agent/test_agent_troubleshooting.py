from unittest.mock import patch
from app.agent.agent import MaintMateAgent
from app.agent.state import AgentState
from app.agent.troubleshooting import evaluate_troubleshooting
from app.agent.troubleshooting import (
    all_steps_failed,
    build_step_status
)
def main():
    agent = MaintMateAgent()

    try:
        state = AgentState(
            user_query="MD380E 出现 Err16，应该怎么处理？"
        )

        with (
            patch.object(
                agent.rag,
                "retrieve",
                wraps=agent.rag.retrieve
            ) as mock_retrieve,
            patch.object(
                agent.safety_gate,
                "_analyze_evidence",
                wraps=agent.safety_gate._analyze_evidence
            ) as mock_safety
        ):

            # 第一轮：故障咨询
            answer1 = agent.process(state)
            print("\n========== 标准排障步骤 ==========")

            for step in state.troubleshooting_steps:
                print(step.model_dump())

            assert len(state.troubleshooting_steps) > 0
            print("\n========== 第一轮 ==========")
            print(answer1)
            print("动作：", state.next_action)
            
            assert state.safety_checked is True

            if state.next_action == "answer":
                assert state.required_role == "unknown"
                print("安全回退测试通过：采用保守回答。")
                print("本次未覆盖后续五轮排障流程。")
                return

            assert state.next_action == "ask_user"
            assert state.required_role == "maintenance"
            # 第二轮：确认维修人员身份
            answer2 = agent.process(
                state,
                user_input="我是维修人员"
            )

            print("\n========== 第二轮 ==========")
            print(answer2)
            print("动作：", state.next_action)

            assert state.next_action == "answer"

            # 保存原有证据
            first_context = state.rag_context

            # 第三轮：反馈部分排障失败
            answer3 = agent.process(
                state,
                user_input=(
                    "我已经检查过上位机接线和通讯连接线，"
                    "但是 Err16 还是存在。"
                )
            )

            print("\n========== 第三轮 ==========")
            print(answer3)
            print("动作：", state.next_action)

            
            
            # 第四轮：剩余两项已确认，故障仍未解决
            answer4 = agent.process(
                state,
                user_input=(
                    "我已经对照手册确认通讯扩展卡F0-28类型设置正确，"
                    "FD组通讯参数也设置正确，"
                    "但是Err16故障仍然存在。"
                )
            )

            print("\n========== 第四轮：排障失败 ==========")
            print(answer4)

            # 验证 Python 排障状态
            decision, pending_steps = evaluate_troubleshooting(
                state.troubleshooting_steps,
                state.troubleshooting_history
            )

            print("排障决策：", decision)
            print("待确认步骤：", pending_steps)

            for record in state.troubleshooting_history:
                print(record.model_dump())

            assert decision == "suggest_escalation"
            assert state.escalation_reason == "troubleshooting_failed"
            assert state.escalated is False

            # 检查没有重新检索手册或重新做安全分析
            assert state.rag_context == first_context
            assert mock_retrieve.call_count == 1
            assert mock_safety.call_count == 1

            # 第五轮：用户确认需要厂家支持
            answer5 = agent.process(
                state,
                user_input="需要厂家支持，请帮我整理故障信息"
            )

            print("\n========== 第五轮：确认升级 ==========")
            print(answer5)

            print("是否升级：", state.escalated)
            print("升级原因：", state.escalation_reason)
            print("升级目标：", state.escalation_target)

            assert state.escalated is True
            assert state.escalation_reason == "troubleshooting_failed"
            assert state.escalation_target == "vendor"

            # 仅验证升级交接，不代表已经真实联系厂家
            assert "Err16" in answer5

            print("\n完整排障升级流程测试通过！")


    finally:
        agent.close()


if __name__ == "__main__":
    main()
