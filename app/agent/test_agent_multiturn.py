from unittest.mock import patch
from app.agent.agent import MaintMateAgent
from app.agent.state import AgentState


def main():
    agent = MaintMateAgent()

    try:
        # 同一次会话复用同一个 AgentState
        state = AgentState(
            user_query="我的 MD380E 报 Err16，应该怎么办？"
        )

        # 记录 RAG 和 Safety LLM 的调用次数
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

            # =========================
            # 第一轮：不知道用户身份
            # =========================
            answer1 = agent.process(state)

            print("\n" + "=" * 60)
            print("第一轮：故障咨询")
            print("=" * 60)

            print("回答：\n", answer1)
            print("\n当前身份：", state.diagnostic_context.user_role)
            print("最低要求身份：", state.required_role)
            print("最终动作：", state.next_action)
            print("缺失字段：", state.missing_fields)

            print("\nRAG 调用次数：", mock_retrieve.call_count)
            print("Safety 分析次数：", mock_safety.call_count)

            # 保存第一轮检索结果
            first_context = state.rag_context
            first_query = state.pending_query

            # 验证第一轮是否进入身份追问
            assert state.next_action == "ask_user", (
                "第一轮没有进入身份追问"
            )
            assert "user_role" in state.missing_fields
            assert first_context is not None
            assert first_query is not None
            assert mock_retrieve.call_count == 1
            assert mock_safety.call_count == 1

            # =========================
            # 第二轮：用户补充身份
            # =========================
            answer2 = agent.process(
                state,
                user_input="我是维修人员"
            )

            print("\n" + "=" * 60)
            print("第二轮：补充身份")
            print("=" * 60)

            print("回答：\n", answer2)
            print("\n当前身份：", state.diagnostic_context.user_role)
            print("最低要求身份：", state.required_role)
            print("最终动作：", state.next_action)

            print(
                "是否复用原始问题：",
                state.pending_query == first_query
            )
            print(
                "是否复用 RAG 证据：",
                state.rag_context == first_context
            )

            print("\nRAG 总调用次数：", mock_retrieve.call_count)
            print("Safety 总分析次数：", mock_safety.call_count)

            # =========================
            # 自动验证
            # =========================
            assert state.diagnostic_context.user_role == "maintenance"
            assert state.required_role == "maintenance"
            assert state.next_action == "answer"

            assert state.pending_query == first_query
            assert state.rag_context == first_context

            # 重点：第二轮没有重复调用
            assert mock_retrieve.call_count == 1, (
                "第二轮重复执行了 RAG 检索"
            )
            assert mock_safety.call_count == 1, (
                "第二轮重复执行了 Safety LLM 分析"
            )

            assert answer1 != answer2, (
                "两轮回答相同，请检查回答范围"
            )

            print("\n" + "=" * 60)
            print("测试结果：全部通过！")
            print("=" * 60)

    finally:
        agent.close()


if __name__ == "__main__":
    main()
