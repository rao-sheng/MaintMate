from app.agent.safety import (
    SafetyGate,
    SafetyAnalysis,
)
from app.agent.state import (
    AgentState,
    DiagnosticContext,
)

class MockSafetyGate(SafetyGate):
    """
    测试专用 SafetyGate：
    不调用真实 LLM，
    直接规定当前操作需要 maintenance 权限。
    """
    def __init__(self):
        pass

    def _analyze_evidence(
        self,
        state: AgentState
    ) -> SafetyAnalysis:

        return SafetyAnalysis(
    required_role="maintenance",
    reason="测试：当前操作需要维修人员权限"
)


def run_test(user_role: str):

    state = AgentState(
        user_query="Err16 应该怎么处理？",
        diagnostic_context=DiagnosticContext(
            user_role=user_role#type:ignore
        )
    )

    safety_gate = MockSafetyGate()

    decision = safety_gate.evaluate(state)

    print("=" * 50)
    print(f"当前用户身份：{user_role}")
    print(f"最低要求身份：{decision.required_role}")
    print(f"最终动作：{decision.action}")
    print(f"原因：{decision.reason}")


def main():

    run_test("user")

    run_test("maintenance")

    run_test("vendor_technician")
    run_test("unknown")

if __name__ == "__main__":
    main()