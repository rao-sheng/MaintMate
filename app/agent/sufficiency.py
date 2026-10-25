from pydantic import BaseModel, Field
from app.agent.state import AgentState

class SufficiencyResult(BaseModel):
    """
    信息完整性判断结果。
    """
    is_sufficient: bool

    missing_fields: list[str] = Field(
        default_factory=list
    )#当没传default_factory时，用一个函数来创建【】
    next_action: str

class InformationSufficiencyPolicy:
    def evaluate(self, state: AgentState) -> SufficiencyResult:
        intent = state.intent
        context = state.diagnostic_context
        # -------------------------
        # 1. 故障诊断
        # -------------------------
        if intent == "fault_diagnosis":
            return self._check_fault_diagnosis(
                state
            )
        # -------------------------
        # 2. 参数查询
        # -------------------------
        if intent == "parameter_lookup":
            return self._check_parameter_lookup(
                state
            )
        # -------------------------
        # 3. 操作指导
        # -------------------------
        if intent == "operation_guidance":
            return self._check_operation_guidance(
                state
            )
        # -------------------------
        # 4. 普通手册问答
        # -------------------------
        if intent == "manual_qa":
            return self._check_manual_qa(
                state
            )
        # -------------------------
        # 5. 普通聊天
        # -------------------------
        if intent == "general_chat":
            return SufficiencyResult(
                is_sufficient=True,
                missing_fields=[],
                next_action="answer"
            )
        # -------------------------
        # 6. 无法判断意图
        # -------------------------
        return SufficiencyResult(
            is_sufficient=False,
            missing_fields=[
                "user_intent"
            ],
            next_action="ask_user"
        )
    def _has_document_scope(self,state: AgentState) -> bool:
        """
        判断我们是否知道应该查哪台设备/哪些文档。
        未来用户上传文档后：
        doc_ids 有值，也算已经确定知识范围。

        当前：
        equipment_model 有值，也可以作为检索范围依据。
        """
        context = state.diagnostic_context
        return bool(
            context.equipment_model
            or context.doc_ids
        )
    def _check_fault_diagnosis(
        self,
        state: AgentState
    ) -> SufficiencyResult:
        context = state.diagnostic_context
        missing_fields = []
        # 必须先知道查哪台设备 / 哪份手册
        if not self._has_document_scope(state):
            missing_fields.append(
                "equipment_model"
            )
        # 故障诊断至少需要一种有效故障线索
        has_fault_information = any([
            context.fault_code,
            context.panel_display,
            context.symptom_description,
        ])
        if not has_fault_information:
            missing_fields.append(
                "fault_information"
            )
        if missing_fields:
            return SufficiencyResult(
                is_sufficient=False,
                missing_fields=missing_fields,
                next_action="ask_user"
            )
        return SufficiencyResult(
            is_sufficient=True,
            missing_fields=[],
            next_action="retrieve"
        )
    def _check_parameter_lookup(
        self,
        state: AgentState
    ) -> SufficiencyResult:
        context = state.diagnostic_context
        missing_fields = []
        if not self._has_document_scope(state):
            missing_fields.append(
                "equipment_model"
            )
        if not context.parameter_code:
            missing_fields.append(
                "parameter_code"
            )
        if missing_fields:
            return SufficiencyResult(
                is_sufficient=False,
                missing_fields=missing_fields,
                next_action="ask_user"
            )
        return SufficiencyResult(
            is_sufficient=True,
            missing_fields=[],
            next_action="retrieve"
        )
    def _check_operation_guidance(
        self,
        state: AgentState
    ) -> SufficiencyResult:
        context = state.diagnostic_context
        missing_fields = []
        if not self._has_document_scope(state):
            missing_fields.append(
                "equipment_model"
            )
        if not context.operation_request:
            missing_fields.append(
                "operation_request"
            )
        if missing_fields:
            return SufficiencyResult(
                is_sufficient=False,
                missing_fields=missing_fields,
                next_action="ask_user"
            )
        return SufficiencyResult(
            is_sufficient=True,
            missing_fields=[],
            next_action="retrieve"
        )
    def _check_manual_qa(
        self,
        state: AgentState
    ) -> SufficiencyResult:
        if not self._has_document_scope(state):
            return SufficiencyResult(
                is_sufficient=False,
                missing_fields=[
                    "equipment_model"
                ],
                next_action="ask_user"
            )
        return SufficiencyResult(
            is_sufficient=True,
            missing_fields=[],
            next_action="retrieve"
        )


if __name__ == "__main__":
    from app.agent.state import (
        AgentState,
        DiagnosticContext,
    )
    policy = InformationSufficiencyPolicy()
    state = AgentState(
        user_query="我的 MD380E 报 Err16 怎么处理？",
        intent="fault_diagnosis",
        diagnostic_context=DiagnosticContext(
            
            fault_code="Err16"
        )
    )
    result = policy.evaluate(state)
    print(
        result.model_dump_json(
            indent=2
        )
    )