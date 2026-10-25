from app.agent.state import (
    AgentState,
    TroubleshootingRecord
)

def main():
    state = AgentState(
        user_query="MD380E 出现 Err16，怎么办？"
    )

    # 模拟用户的第一次排障反馈
    state.troubleshooting_history.append(
        TroubleshootingRecord(
            action="检查上位机接线",
            result="not_resolved",
            note="已检查，Err16 仍存在"
        )
    )

    # 模拟第二次排障反馈
    state.troubleshooting_history.append(
        TroubleshootingRecord(
            action="检查通讯连接线",
            result="not_resolved",
            note="已检查，故障仍未排除"
        )
    )

    print("当前问题：", state.user_query)

    print("\n排障历史：")

    for index, record in enumerate(
        state.troubleshooting_history,
        start=1
    ):
        print(
            f"{index}. {record.action}"
            f" | 结果：{record.result}"
            f" | 备注：{record.note}"
        )

    assert len(state.troubleshooting_history) == 2
    assert all(
        record.result == "not_resolved"
        for record in state.troubleshooting_history
    )

    print("\n测试通过！")


if __name__ == "__main__":
    main()
