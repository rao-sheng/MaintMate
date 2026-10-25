
from app.agent.state import AgentState
from app.agent.input_analyzer import InputAnalyzer
from app.llm.llm_provider import ZhipuLLMProvider


def main():
    analyzer = InputAnalyzer(
        llm_provider=ZhipuLLMProvider()
    )

    # 模拟已经存在一个正在处理的故障
    state = AgentState(
        user_query="MD380E 电机运行时一直抖动"
    )

    state.pending_query = (
        "MD380E 电机运行时一直抖动"
    )
    state.diagnostic_context.equipment_model = "MD380E"
    state.diagnostic_context.symptom_description = (
        "电机运行时一直抖动"
    )

    test_cases = [
        (
            "我检查了电机接线，但还是抖动。",
            "continue_fault"
        ),
        (
            "我还发现启动的时候振动特别明显。",
            "continue_fault"
        ),
        (
            "另外一台变频器现在报 Err16 了。",
            "new_fault"
        ),
        (
            "现在设备又出现了别的问题。",
            "uncertain"
        ),
        (
            "你好。",
            "unrelated"
        )
    ]

    for user_input, expected in test_cases:
        analysis = analyzer.analyze(
            user_input=user_input,
            state=state
        )

        print(f"\n用户：{user_input}")
        print(f"预期：{expected}")
        print(f"实际：{analysis.turn_relation}")

    print("\n测试完成，请检查以上分类结果。")


if __name__ == "__main__":
    main()
