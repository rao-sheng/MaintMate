from app.agent.input_analyzer import InputAnalyzer
from app.llm.llm_provider import ZhipuLLMProvider

def main():
    analyzer = InputAnalyzer(
        llm_provider=ZhipuLLMProvider()
    )
    cases = [
        (
            "我的 MD380E 报 Err16，我想直接联系厂家处理。",
            True,
            "vendor"
        ),
        (
            "我不想自己维修，请转交维修人员。",
            True,
            "maintenance"
        ),
        (
            "MD380E 出现 Err16，应该怎么解决？",
            False,
            None
        ),
        (
            "如果以后排查无效，再联系厂家。",
            False,
            None
        )
    ]

    for user_input, expected_requested, expected_target in cases:
        analysis = analyzer.analyze(user_input)

        print("\n用户：", user_input)
        print("是否请求升级：", analysis.escalation_requested)
        print("升级目标：", analysis.escalation_target)

        assert analysis.escalation_requested == expected_requested
        assert analysis.escalation_target == expected_target

    print("\n主动升级意图提取测试通过！")


if __name__ == "__main__":
    main()
