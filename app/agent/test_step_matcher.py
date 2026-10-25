from app.agent.state import (
    TroubleshootingStep,
    TroubleshootingRecord
)
from app.agent.step_matcher import StepMatcher
from app.llm.llm_provider import ZhipuLLMProvider


def main():

    matcher = StepMatcher(
        llm_provider=ZhipuLLMProvider()
    )

    steps = [
        TroubleshootingStep(
            step_id="step_1",
            action="检查上位机接线"
        ),
        TroubleshootingStep(
            step_id="step_2",
            action="检查通讯连接线"
        ),
        TroubleshootingStep(
            step_id="step_3",
            action="检查通讯扩展卡设置"
        )
    ]

    feedback = [
        TroubleshootingRecord(
            action="我检查过通讯线",
            result="not_resolved"
        ),
        TroubleshootingRecord(
            action="上位机的线也检查过了",
            result="not_resolved"
        ),
        TroubleshootingRecord(
            action="我更换了电机轴承",
            result="unknown"
        )
    ]

    result = matcher.match(
        steps=steps,
        feedback=feedback
    )

    print("匹配结果：", result)

    assert result[1] == "step_2"
    assert result[2] == "step_1"
    assert result[3] is None

    print("StepMatcher 测试通过！")


if __name__ == "__main__":
    main()
