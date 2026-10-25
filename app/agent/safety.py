from typing import Literal

from pydantic import BaseModel

from app.agent.state import (
    AgentState,
    UserRole,
)


SafetyAction = Literal[
    "answer",
    "ask_user",
    "escalate",
]


class SafetyAnalysis(BaseModel):
    required_role: UserRole = "unknown"
    reason: str | None = None


class SafetyDecision(BaseModel):
    action: SafetyAction
    required_role: UserRole = "unknown"
    reason: str | None = None


class SafetyGate:

    ROLE_LEVEL = {
        "unknown": 0,
        "user": 1,
        "maintenance": 2,
        "vendor_technician": 3,
    }

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def evaluate(
        self,
        state: AgentState
    ) -> SafetyDecision:

        # 1. 第一次：让 LLM 分析手册证据
        analysis = self._analyze_evidence(
            state
        )

        # 2. 保存安全分析结果
        state.required_role = (
            analysis.required_role
        )

        state.safety_reason = (
            analysis.reason
        )

        state.safety_checked = True

        # 3. Python 做权限比较
        return self.check_role(
            user_role=state.diagnostic_context.user_role,
            required_role=analysis.required_role,
            reason=analysis.reason
        )
    
    def _analyze_evidence(
        self,
        state: AgentState
    ) -> SafetyAnalysis:

        evidence = self._build_evidence_text(
            state
        )

        system_prompt = """
你是工业设备运维系统中的安全权限分析模块。

你的任务不是回答用户问题，而是严格依据设备手册证据，
判断当前问题对应的最低用户角色要求。

角色只能选择：

user
- 普通用户。
- 只能查询和阅读信息。
- 例如查询故障含义、参数含义、设备状态等。

maintenance
- 设备维修人员。
- 可以执行手册明确说明的普通检查、维修、
  参数调整、接线检查、部件更换等操作。


vendor_technician
- 厂家技术人员。
- 只有当用户当前请求执行的具体维修操作，
  根据手册证据必须由厂家技术人员完成时使用。
- 例如手册明确要求由厂家专业人员实施的维修、
  调试或其他专属技术操作。
- 注意：“联系厂家”“寻求厂家服务”
  本身只是升级处理建议，
  不代表当前所有排查步骤都需要厂家权限。


unknown
- 根据当前证据无法判断。

权限判断规则：

1. 判断对象是用户当前请求的操作，而不是检索证据中所有
   解决方案的最高权限要求。

2. 如果用户只是询问故障代码、参数含义、设备状态等知识，
   没有请求执行维修操作，required_role 应为 user。

3. 如果用户要求故障处理，手册提供了维修人员可执行的
   排查方法，同时还包含“无效时联系厂家”的建议，
   不得仅因为存在这条建议就把 required_role
   判定为 vendor_technician。

4. “联系厂家”“寻求厂家技术支持”属于升级处理建议，
   不等于只有厂家技术人员才能执行的维修操作。

5. 只有当用户当前请求的具体操作明确要求厂家技术人员
   执行时，才将 required_role 判定为 vendor_technician。

6. 如果手册明确禁止当前角色执行某项操作，
   不得因为其他步骤允许执行就降低该操作的权限要求。

7. 不得推断用户已经尝试过某种解决方法。
   是否已经排查失败，应由用户反馈或可信记录确认。

8. 如果证据无法确定某项维修操作的权限要求，
   不得猜测其安全性或直接授权执行，应明确说明不确定。

9. 只返回 JSON，不要使用 Markdown。
""".strip()

        user_prompt = f"""
用户问题：
{state.user_query}

手册证据：
{evidence}

返回：

{{
  "required_role": "unknown",
  "reason": null
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

        return SafetyAnalysis.model_validate_json(
            json_text
        )

    def _build_evidence_text(
        self,
        state: AgentState
    ) -> str:

        parts = []

        for index, source in enumerate(
            state.retrieved_sources,
            start=1
        ):
            payload = source["payload"]

            section = payload.get("section")
            page_start = payload.get("page_start")
            page_end = payload.get("page_end")
            content = payload.get("content")

            evidence_part = [
                f"[证据 {index}]"
            ]

            if section:
                evidence_part.append(
                    f"章节：{section}"
                )

            if page_start is not None:
                if (
                    page_end is not None
                    and page_end != page_start
                ):
                    evidence_part.append(
                        f"页码：{page_start}-{page_end}"
                    )
                else:
                    evidence_part.append(
                        f"页码：{page_start}"
                    )

            if content:
                evidence_part.append(
                    f"内容：\n{content}"
                )

            parts.append(
                "\n".join(evidence_part)
            )

        return "\n\n".join(parts)

    def _clean_json(
        self,
        text: str
    ) -> str:

        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def check_role(
        self,
        user_role: UserRole,
        required_role: UserRole,
        reason: str | None = None
    ) -> SafetyDecision:
        """
        不重新分析手册证据，
        只比较当前用户身份和已经确定的最低权限要求。
        """

        # 普通查询
        if required_role == "user":
            return SafetyDecision(
                action="answer",
                required_role=required_role,
                reason=reason
            )

        # 当前版本：无法判断时允许回答
        if required_role == "unknown":
            return SafetyDecision(
                action="answer",
                required_role=required_role,
                reason=reason
            )

        # 操作需要专业权限，但还不知道用户身份
        if user_role == "unknown":
            return SafetyDecision(
                action="ask_user",
                required_role=required_role,
                reason=reason
            )

        user_level = self.ROLE_LEVEL[
            user_role
        ]

        required_level = self.ROLE_LEVEL[
            required_role
        ]

        if user_level >= required_level:
            return SafetyDecision(
                action="answer",
                required_role=required_role,
                reason=reason
            )

        return SafetyDecision(
            action="escalate",
            required_role=required_role,
            reason=reason
        )