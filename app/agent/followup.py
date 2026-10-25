from app.agent.state import AgentState

class FollowUpQuestionBuilder:
    def build(self,state: AgentState) -> str:
        missing_fields = state.missing_fields
        questions = []
        if "equipment_model" in missing_fields:
            questions.append(
                "请提供设备型号"
            )
        if "fault_information" in missing_fields:
            questions.append(
                "如果面板有故障代码、报警信息或具体异常现象，也请一并告诉我"
            )
        if "parameter_code" in missing_fields:
            questions.append(
                "请提供你想查询的参数或功能码，例如 F0-28"
            )
        if "operation_request" in missing_fields:
            questions.append(
                "请说明你想进行什么操作或设置"
            )
        if "user_intent" in missing_fields:
            questions.append(
                "请说明你是想排查故障、查询参数，还是了解某项操作方法"
            )
        if "user_role" in missing_fields:
            questions.append(
                "请确认你的身份：普通用户、设备维修人员，还是厂家维修人员"
            )
        if not questions:
                return "请补充与当前设备问题相关的信息。"
                
        return "；".join(questions) + "。"


if __name__ == "__main__":
    from app.agent.state import AgentState
    builder = FollowUpQuestionBuilder()
    state = AgentState(
        user_query="设备坏了",
        missing_fields=[
            "equipment_model",
            "fault_information"
        ]
    )
    question = builder.build(state)
    print(question)