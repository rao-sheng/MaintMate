import json
from pydantic import BaseModel, Field
from app.agent.state import (
    TroubleshootingStep,
    TroubleshootingRecord
)

class StepMatch(BaseModel):
    feedback_index: int
    step_id: str | None = None

class StepMatchResult(BaseModel):
    matches: list[StepMatch] = Field(default_factory=list)

class StepMatcher:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
    def match(
        self,
        steps: list[TroubleshootingStep],
        feedback: list[TroubleshootingRecord]
    ) -> dict[int, str | None]:
        """
        将用户排障反馈匹配到已有标准步骤。

        返回：
        {
            1: "step_2",
            2: None
        }

        字典的 key 是反馈编号，从 1 开始。
        """
        if not steps or not feedback:
            return {}
        # 1. 整理标准步骤
        steps_text = "\n".join(
            f"{step.step_id}: {step.action}"
            for step in steps
        )

        # 2. 整理用户反馈
        feedback_text = "\n".join(
            f"{index}. {record.action}"
            for index, record in enumerate(
                feedback,
                start=1
            )
        )

        system_prompt = """
你是工业设备运维系统中的排障步骤匹配模块。

任务：
将用户已完成的排障操作，匹配到已有标准步骤。

规则：
1. 只能匹配提供的 step_id，不得创造新 ID。
2. 用户表达与标准步骤含义一致时可以匹配，
   不要求字符串完全相同。
3. 仅仅涉及同一设备部件，不代表属于同一操作。
4. 无法确定、存在多个合理候选时返回 null。
5. 每条反馈独立匹配。
6. 不要判断故障是否解决。
7. 不要判断是否应该升级。
8. 只返回 JSON，不要使用 Markdown。

返回格式：
{
  "matches": [
    {
      "feedback_index": 1,
      "step_id": "step_2"
    }
  ]
}
""".strip()

        user_prompt = f"""
标准步骤：
{steps_text}

用户排障反馈：
{feedback_text}

请为每条反馈返回一个匹配结果。
不能确定时 step_id 为 null。
""".strip()

        raw_result = self.llm_provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.0
        )

        # 3. 清理 JSON
        json_text = raw_result.strip()

        if json_text.startswith("```json"):
            json_text = json_text[7:]
        elif json_text.startswith("```"):
            json_text = json_text[3:]

        if json_text.endswith("```"):
            json_text = json_text[:-3]

        parsed = StepMatchResult.model_validate(
            json.loads(json_text.strip())
        )

        # 4. Python 校验匹配结果
        valid_ids = {
            step.step_id for step in steps
        }

        result = {
            index: None
            for index in range(1, len(feedback) + 1)
        }

        seen_indices = set()

        for item in parsed.matches:

            if not 1 <= item.feedback_index <= len(feedback):
                raise ValueError("反馈编号不存在")

            if item.feedback_index in seen_indices:
                raise ValueError("重复的反馈编号")

            seen_indices.add(item.feedback_index)

            if (
                item.step_id is not None
                and item.step_id not in valid_ids
            ):
                raise ValueError(
                    f"不存在的步骤 ID：{item.step_id}"
                )

            result[item.feedback_index] = item.step_id#type:ignore

        return result#type:ignore
