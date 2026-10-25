from typing import Literal
from pydantic import BaseModel, Field

IntentType = Literal[
    "fault_diagnosis",
    "parameter_lookup",
    "operation_guidance",
    "manual_qa",
    "general_chat",
    "unknown",
]

class TroubleshootingStep(BaseModel):
    step_id: str
    action: str
    source_page: int | None = None


ActionType = Literal[
    "ask_user",
    "retrieve",
    "answer",
    "escalate",
]

UserRole = Literal[
    "user",
    "maintenance",
    "vendor_technician",
    "unknown",
]


class TroubleshootingRecord(BaseModel):
    """记录用户反馈的一项排障操作及其结果。"""

    action: str
    result: Literal[
        "resolved",
        "not_resolved",
        "unknown"
    ] = "unknown"
    note: str | None = None
    step_id: str | None = None


class DiagnosticContext(BaseModel):
    """
    当前设备 / 故障诊断过程中已经获得的信息。
    """
    # 设备型号，例如 MD380E
    equipment_model: str | None = None
    # 用户身份
    user_role: UserRole = "unknown"
    # 故障代码，例如 Err16
    fault_code: str | None = None
    # 用户描述的故障现象
    symptom_description: str | None = None
    # 面板当前显示内容
    panel_display: str | None = None
    # 设备当前状态，例如：
    # 运行中 / 停机 / 上电后 / 启动时
    operating_state: str | None = None
    # 故障出现之前是否有参数修改、
    # 接线变化、维修等
    recent_changes: str | None = None
    #参数查询 操作请求
    parameter_code: str | None = None
    operation_request: str | None = None
    # 当前允许检索的文档
    doc_ids: list[str] = Field(
        default_factory=list
    )

 
EscalationReason = Literal[
    "user_requested",
    "troubleshooting_failed",
    "permission_required",
    "safety_risk"
]

EscalationTarget = Literal[
    "maintenance",
    "vendor"
]
   

class AgentState(BaseModel):
    """
    MaintMate Agent 在一次会话中的核心状态。
    """
    troubleshooting_steps: list[TroubleshootingStep] = Field(
            default_factory=list
        )
    escalation_reason: EscalationReason | None = None
    escalation_target: EscalationTarget | None = None

    troubleshooting_history: list[TroubleshootingRecord] = Field(
        default_factory=list
    )
    rag_context: str | None = None

    pending_query: str | None = None
    # 用户这一轮最新输入
    user_query: str
    # Agent 判断出的任务类型
    intent: IntentType = "unknown"
    # 当前已经收集到的诊断信息
    diagnostic_context: DiagnosticContext = Field(
        default_factory=DiagnosticContext
    )
    # 还缺少哪些必要信息
    missing_fields: list[str] = Field(
        default_factory=list
    )
    # Agent 下一步准备执行什么动作
    next_action: ActionType | None = None
    # RAG 返回的答案
    rag_answer: str | None = None
    # 最终回答
    final_answer: str | None = None
    # 是否需要升级给专业人员
    escalated: bool = False
    retrieved_sources: list[dict] = Field(
        default_factory=list
    )
    # 手册要求的最低用户角色
    required_role: UserRole = "unknown"

    # 当前安全判断原因
    safety_reason: str | None = None

    # 是否已经完成安全检查
    safety_checked: bool = False