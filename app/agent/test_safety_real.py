from app.agent.state import (
    AgentState,
    DiagnosticContext,
)
from app.agent.safety import SafetyGate
from app.llm.llm_provider import ZhipuLLMProvider
from app.rag_pipeline import RAGPipeline

def main():

    # 1. 初始化真实 LLM 和 RAG
    llm_provider = ZhipuLLMProvider()

    rag = RAGPipeline()

    safety_gate = SafetyGate(
        llm_provider=llm_provider
    )

    # 2. 一个真实维修类问题
    query = "MD380E 的 Err16 是什么意思？"

    # 先假设当前用户是普通用户
    state = AgentState(
        user_query=query,
        diagnostic_context=DiagnosticContext(
            equipment_model="MD380E",
            fault_code="Err16",
            user_role="user"
        )
    )

    # 3. 真实 RAG
    rag_result = rag.answer(
        query=query
    )

    state.rag_answer = rag_result["answer"]
    state.retrieved_sources = rag_result["sources"]

    # 4. 真实 Safety LLM
    decision = safety_gate.evaluate(
        state
    )

    # 5. 查看结果
    print("=" * 60)

    print("用户问题：")
    print(query)

    print("\nRAG 回答：")
    print(state.rag_answer)

    print("\nSafety 判断：")
    print(f"当前身份：{state.diagnostic_context.user_role}")
    print(f"最低要求身份：{decision.required_role}")
    print(f"最终动作：{decision.action}")
    print(f"原因：{decision.reason}")

    print("=" * 60)

    rag.close()


if __name__ == "__main__":
    main()