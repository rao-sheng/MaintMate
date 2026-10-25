from app.agent.state import AgentState
from app.agent.input_analyzer import InputAnalyzer
from app.llm.llm_provider import ZhipuLLMProvider

def main():
    llm = ZhipuLLMProvider()
    analyzer = InputAnalyzer(
        llm_provider=llm
    )
    state = AgentState(
        user_query="MD380E 出现 Err16，怎么办？"
    )
    test_inputs = [
        "我检查过上位机接线和通讯连接线了，但是 Err16 还是存在。",
        "我准备检查 FD 组参数。",
        "我检查了通讯扩展卡设置，但是还没确认故障是否解决。"
    ]
    for index, user_input in enumerate(
        test_inputs,
        start=1
    ):
        print(f"\n========== 第 {index} 轮 ==========")
        print("用户：", user_input)

        analysis = analyzer.analyze(user_input)

        print("\n本轮提取的排障反馈：")
        for record in analysis.troubleshooting_feedback:
            print(record.model_dump())

        analyzer.update_state(
            state=state,
            analysis=analysis
        )

    print("\n========== 最终排障历史 ==========")

    for index, record in enumerate(
        state.troubleshooting_history,
        start=1
    ):
        print(
            f"{index}. {record.action} | "
            f"{record.result} | {record.note}"
        )

    # 第一轮：两项检查都已执行且未解决
    # 第二轮：只是准备检查，不应新增记录
    # 第三轮：已检查，但解决情况不明确
    assert len(state.troubleshooting_history) == 3

    assert sum(
        record.result == "not_resolved"
        for record in state.troubleshooting_history
    ) == 2

    assert sum(
        record.result == "unknown"
        for record in state.troubleshooting_history
    ) == 1

    print("\n测试通过！")


if __name__ == "__main__":
    main()
