import json
from pydantic import BaseModel, Field
from app.agent.state import TroubleshootingStep

class ExtractedStep(BaseModel):
    action: str
    source_index: int

class StepExtraction(BaseModel):
    steps: list[ExtractedStep] = Field(
        default_factory=list
    )

class StepExtractor:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
    def extract(
        self,
        query: str,
        sources: list[dict]
    ) -> list[TroubleshootingStep]:
        """从当前问题的检索证据中提取排障步骤。"""

        if not sources:
            return []

        evidence_parts = []

        for index, source in enumerate(sources, start=1):
            payload = source["payload"]

            evidence_parts.append(
                f"[证据 {index}]\n"
                f"页码：{payload.get('page_start')}\n"
                f"内容：{payload.get('content', '')}"
            )

        evidence = "\n\n".join(evidence_parts)

        system_prompt = """
你是工业设备手册的排障步骤提取模块。

根据用户当前故障和手册证据，
提取明确适用于该故障的排障措施。

规则：
1. 只提取手册证据明确支持的排障措施。
2. 一项独立排障操作对应一条记录。
3. 不要编造手册不存在的操作。
4. 不要提取无关故障的排障步骤。
5. 不要将“联系厂家”“寻求技术支持”
   等升级建议作为排障步骤。
6. 不要重复提取相同的操作。
7. 每条操作必须填写对应证据编号 source_index。
8. 如果证据中没有明确的排障措施，返回空列表。
9. 不要判断用户是否有权限执行这些操作。
10. 只返回 JSON，不要使用 Markdown。
11.提取排障步骤时，尽量保留手册中明确出现的
故障代码、参数组、功能码和设备部件名称。

例如 FD 组、F0-28 等关键标识不能随意省略。

不得将“检查某参数”和“正确修改某参数”
当作完全相同的操作。

返回格式：
{
  "steps": [
    {
      "action": "检查通讯连接线",
      "source_index": 1
    }
  ]
}
""".strip()

        user_prompt = f"""
当前问题：
{query}

检索证据：
{evidence}
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

        # 清理可能出现的 Markdown JSON 代码块
        json_text = raw_result.strip()

        if json_text.startswith("```json"):
            json_text = json_text[7:]
        elif json_text.startswith("```"):
            json_text = json_text[3:]

        if json_text.endswith("```"):
            json_text = json_text[:-3]

        parsed = StepExtraction.model_validate(
            json.loads(json_text.strip())
        )

        steps = []
        seen_actions = set()

        for item in parsed.steps:

            # 验证引用的证据编号确实存在
            if not 1 <= item.source_index <= len(sources):
                raise ValueError(
                    f"无效的证据编号：{item.source_index}"
                )

            action = item.action.strip()

            if not action or action in seen_actions:
                continue

            seen_actions.add(action)

            # 页码从原始检索结果取得，不依赖 LLM 编造
            source = sources[item.source_index - 1]
            page = source["payload"].get("page_start")

            steps.append(
                TroubleshootingStep(
                    step_id=f"step_{len(steps) + 1}",
                    action=action,
                    source_page=page
                )
            )

        return steps
