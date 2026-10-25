import json
from pydantic import BaseModel, Field
from app.agent.state import EscalationTarget
from app.agent.state import (
    AgentState,
    IntentType,
    UserRole,
    TroubleshootingRecord
)
from app.llm.llm_provider import (
    ZhipuLLMProvider,
)

from typing import Literal

TurnRelation = Literal[
    "continue_fault",
    "new_fault",
    "uncertain",
    "unrelated"
]

class InputAnalysis(BaseModel):
    """
    LLM 对当前用户输入提取出的结构化信息。
    """
    escalation_requested: bool = False
    escalation_target: EscalationTarget | None = None
    turn_relation: TurnRelation = "uncertain"
    intent: IntentType = "unknown"
    equipment_model: str | None = None
    user_role: UserRole = "unknown"
    fault_code: str | None = None
    symptom_description: str | None = None
    panel_display: str | None = None
    operating_state: str | None = None
    recent_changes: str | None = None
    parameter_code: str | None = None
    operation_request: str | None = None
    troubleshooting_feedback: list[TroubleshootingRecord] = Field(
        default_factory=list
    )

class InputAnalyzer:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
    def analyze(self,user_input: str,state: AgentState | None = None) -> InputAnalysis:
        
        # 构建当前正在处理的故障上下文
        if state is not None:
            context = state.diagnostic_context

            current_fault = f"""
        当前问题：{state.pending_query or state.user_query}
        设备型号：{context.equipment_model or "未知"}
        故障代码：{context.fault_code or "未知"}
        故障现象：{context.symptom_description or "未知"}
        """.strip()

            has_active_fault = bool(
                state.pending_query
                or context.fault_code
                or context.symptom_description
            )
        else:
            current_fault = "无"
            has_active_fault = False

        system_prompt = """
你是工业设备运维 Agent 的信息提取模块。
你的任务不是回答用户问题，而是从用户输入中提取结构化信息。
任务类型 intent 只能选择：
fault_diagnosis
- 用户描述设备故障、报警、异常、停机等问题，希望诊断或解决。
parameter_lookup
- 用户询问某个参数、功能码、故障码等具体含义。
operation_guidance
- 用户询问如何设置、配置、操作设备。
manual_qa
- 用户询问设备手册中的其他知识，但不属于故障诊断、参数查询或操作指导。
general_chat
- 与设备和手册无关的普通聊天。
unknown
- 无法判断。
字段规则：
equipment_model：
设备型号，例如 MD380E。
用户没有明确提供时必须返回 null，不允许猜测。
user_role：
只能是 user、maintenance、vendor_technician、unknown。

user：普通用户。
maintenance：设备维修人员。
vendor_technician：厂家技术人员。

只有用户明确提供身份时才提取。
未说明身份时返回 unknown。
fault_code：
例如 Err16、Err23。
没有明确故障代码返回 null。
symptom_description：
用户描述的实际故障或异常现象。
不要自己补充不存在的信息。
panel_display：
用户明确说明面板显示内容时填写。
否则返回 null。
operating_state：
例如“运行中”“启动时”“上电后”“停机状态”。
没有明确说明返回 null。
recent_changes：
例如最近修改过参数、重新接线、维修、更换设备等。
没有明确说明返回 null。
parameter_code：
例如 F0-28、Fd-04。
如果用户明确询问某个参数或功能码，则填写。
否则返回 null。
operation_request：
如果用户明确询问如何进行某项操作，例如
“怎么设置通讯超时时间”，填写用户希望执行的操作。
否则返回 null。

troubleshooting_feedback：
提取用户明确反馈已经尝试的排障操作及其结果。
返回数组，每个元素包含：
- action：用户已经尝试的具体排障操作。
- result：只能是 resolved、not_resolved、unknown。
- note：用户对检查结果的补充说明，没有则为 null。

turn_relation：
判断当前用户输入与正在处理的故障之间的关系。
只能选择：
continue_fault、new_fault、uncertain、unrelated。





escalation_requested：
判断用户是否明确要求将当前问题交给人工维修人员
或厂家技术支持处理。

只有用户明确提出当前升级请求时，返回 true。

例如：
"我不想自己修了，直接联系厂家吧"
→ true

"请帮我转交维修人员处理"
→ true

"Err16 应该怎么解决？"
→ false

"如果以后解决不了，再联系厂家"
→ false

"手册上说要联系厂家是什么意思？"
→ false

注意：
1. 不得仅因为故障严重或手册提到联系厂家就返回 true。
2. 用户询问维修方法，不代表要求升级。
3. 不得把未来可能需要升级误判为当前升级请求。

escalation_target：
只能是 "maintenance"、"vendor" 或 null。

明确要求厂家、厂商、产品代理商提供支持：
返回 "vendor"。

明确要求设备维修人员、维修部门接手：
返回 "maintenance"。

用户要求人工处理，但没有明确指定对象：
返回 null。

没有主动升级请求时也返回 null。


判断规则：

判断规则：

1. 如果没有正在处理的故障：
   - 用户提出设备故障，返回 new_fault。
   - 普通聊天，返回 unrelated。

2. 如果存在正在处理的故障，优先判断用户
   是否明确切换了故障：
   - 明确表示是另一台设备，返回 new_fault。
   - 明确表示是另一个独立故障，返回 new_fault。

3. 如果用户正在补充当前故障的信息、排障结果、
   设备状态，或者回答 Agent 的追问，
   返回 continue_fault。

4. 特别注意排障反馈：
   用户明确描述已经检查、维修或调整某个项目，
   并表示原故障仍然存在，应返回 continue_fault。

   例如：
   当前故障：MD380E Err16
   用户："我检查过上位机接线和通讯连接线了，
          但是 Err16 还是存在。"
   → continue_fault

   当前故障：MD380E Err16
   用户："通讯线检查过了，还是不行。"
   → continue_fault

5. 用户没有重复设备型号或故障代码，
   不代表发生了新故障。
   相同故障的描述措辞发生变化，
   也不代表发生了新故障。

6. 如果用户提出新的异常现象，
   但无法确定是否属于当前故障，
   返回 uncertain，不得擅自合并。

7. 如果用户输入与当前故障无关，
   例如问候、感谢、普通闲聊，
   返回 unrelated。

8. 只有无法确定用户是在继续当前故障，
   还是提出新的设备问题时，
   才返回 uncertain。

9. 判断优先级：
   明确的新故障 > 明确的当前故障延续 >
   明确的无关聊天 > 无法确定。
   不得因为缺少设备型号等非必要信息，
   将明确的排障反馈判断为 uncertain。


1. 用户已经执行某项检查或操作，并明确表示故障仍存在：
   result = not_resolved。
2. 用户明确表示执行某项操作后故障已经解决：
   result = resolved。
3. 用户明确表示已经执行操作，但没有交代故障是否解决：
   result = unknown。
4. 用户只是询问如何操作，或者表示准备检查：
   不得生成排障记录。
5. 一句话包含多个不同的已完成排障操作时，
   必须将每项操作拆分为独立记录。
   每条记录的 action 只表示一项具体操作。
6. 如果多个操作共享同一个结果描述，
   应将该结果分别应用于每一项操作。
例如用户说：
"我检查过上位机接线和通讯连接线了，但是Err16还是存在。"
必须提取为：
[
  {
    "action": "检查上位机接线",
    "result": "not_resolved",
    "note": null
  },
  {
    "action": "检查通讯连接线",
    "result": "not_resolved",
    "note": null
  }
]
禁止合并为：
[
  {
    "action": "检查上位机接线和通讯连接线",
    "result": "not_resolved",
    "note": null
  }
]
7. 如果一句话中不同操作的处理结果不同，
   必须分别记录实际结果，不得统一设置。
8. 不能仅根据标点、"和"、"以及"机械拆分。
   只有明确属于不同排障操作的内容才单独记录。
   不得拆分一个不可分割的技术操作，也不得编造操作。

注意：
“检查结果没有发现异常”和“故障已经解决”
是不同概念，不能混淆。

特别注意 uncertain 和 unrelated 的区别：
uncertain：
用户确实提出了设备故障或异常信息，
但无法确定是否属于当前故障。
例如："现在设备又出现了别的问题。"
unrelated：
用户输入与当前故障处理无关，
包括问候、闲聊、感谢等。
例如："你好"、"谢谢"、"你是谁"。
这些情况必须返回 unrelated，
不能仅因为存在历史故障就返回 uncertain。

重要要求：
1. 不允许根据常识猜测用户没有提供的信息。
2. 不允许回答用户问题。
3. 只返回 JSON。
4. 不要使用 Markdown。
""".strip()

        user_prompt = f"""
当前是否存在正在处理的故障：
{has_active_fault}

当前故障上下文：
{current_fault}

请分析以下用户输入：
{user_input}

返回下面结构的 JSON：

{{
  "intent": "unknown",
  "equipment_model": null,
  "user_role": "unknown",
  "fault_code": null,
  "symptom_description": null,
  "panel_display": null,
  "operating_state": null,
  "recent_changes": null,
  "parameter_code": null,
  "operation_request": null,
  "troubleshooting_feedback": [],
  "turn_relation": "uncertain",
  "escalation_requested": false,
  "escalation_target": null
}}
""".strip()

        raw_result = self.llm_provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.0
        )

        json_text = self._clean_json(
            raw_result
        )

        try:
            data = json.loads(json_text)

        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Agent 信息提取结果不是合法 JSON：\n"
                f"{raw_result}"
            ) from e

        return InputAnalysis.model_validate(
            data
        )

    def update_state(self,state: AgentState,analysis: InputAnalysis
    ):
        """
        把新提取出的信息合并进现有 AgentState。
        只更新用户本轮真正提供的新信息，
        不把 None 覆盖掉之前已经获得的信息。
        """
        state.intent = analysis.intent
        context = state.diagnostic_context
        if analysis.equipment_model:
            context.equipment_model = (
                analysis.equipment_model
            )
        if analysis.user_role != "unknown":
            context.user_role = (
                analysis.user_role
            )
        if analysis.fault_code:
            context.fault_code = (
                analysis.fault_code
            )
        if analysis.symptom_description:
            context.symptom_description = (
                analysis.symptom_description
            )
        if analysis.panel_display:
            context.panel_display = (
                analysis.panel_display
            )
        if analysis.operating_state:
            context.operating_state = (
                analysis.operating_state
            )
        if analysis.recent_changes:
            context.recent_changes = (
                analysis.recent_changes
            )
        if analysis.parameter_code:
            context.parameter_code = (
                analysis.parameter_code
            )
        if analysis.operation_request:
            context.operation_request = (
                analysis.operation_request
            )
        
        
    def _clean_json(self,text: str) -> str:
        """
        防止模型偶尔返回：
        ```json
        {...}
        ```
        把 Markdown 外壳去掉。
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

if __name__ == "__main__":

    llm_provider = ZhipuLLMProvider()

    analyzer = InputAnalyzer(
        llm_provider=llm_provider
    )

    state = AgentState(
        user_query=(
            "我的 MD380E 报 Err16，"
            "通讯突然断了，我是现场操作员。"
        )
    )

    analysis = analyzer.analyze(
        state.user_query
    )

    analyzer.update_state(
        state,
        analysis
    )

    print("\n提取结果：")
    print(
        analysis.model_dump_json(
            indent=2
        )
    )

    print("\nAgentState：")
    print(
        state.model_dump_json(
            indent=2
        )
    )