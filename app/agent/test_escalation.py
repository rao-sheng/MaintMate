from unittest.mock import patch
from app.agent.agent import MaintMateAgent
from app.agent.state import AgentState


def main():
    agent = MaintMateAgent()

    try:
        state = AgentState(
            user_query=(
                "我的 MD380E 报 Err16，"
                "我不想自己维修，想直接联系厂家处理。"
            )
        )

        with patch.object(
            agent.rag,
            "retrieve",
            wraps=agent.rag.retrieve
        ) as mock_retrieve:

            answer = agent.process(state)

            print("\n========== 升级结果 ==========")
            print(answer)

            print("\n========== Agent 状态 ==========")
            print("最终动作：", state.next_action)
            print("升级原因：", state.escalation_reason)
            print("升级目标：", state.escalation_target)
            print("是否进入升级流程：", state.escalated)
            print("RAG 检索次数：", mock_retrieve.call_count)

            assert state.next_action == "escalate"
            assert state.escalation_reason == "user_requested"
            assert state.escalation_target == "vendor"
            assert state.escalated is True

            # 主动升级不应强制先做 RAG
            assert mock_retrieve.call_count == 0

            print("\n主动升级测试通过！")

    finally:
        agent.close()


if __name__ == "__main__":
    main()
