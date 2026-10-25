import os
import time
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv


LLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class LLMProvider(ABC):

    @abstractmethod
    def generate_answer(self, query, context):
        pass


class ZhipuLLMProvider(LLMProvider):

    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("ZHIPU_API_KEY")
        self.model = os.getenv("ZHIPU_LLM_MODEL")

        if not self.api_key:
            raise ValueError(
                "没有找到环境变量 ZHIPU_API_KEY"
            )

        if not self.model:
            raise ValueError(
                "没有找到环境变量 ZHIPU_LLM_MODEL"
            )

    def _post_with_retry(
        self,
        payload,
        max_retries=3
    ):
        for attempt in range(max_retries):

            try:
                response = requests.post(
                    LLM_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )

                return response

            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout
            ) as e:

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"LLM API 网络请求失败：{e}"
                    ) from e

                wait_seconds = 2 ** attempt

                print(
                    f"LLM 请求失败，"
                    f"{wait_seconds} 秒后重试..."
                )

                time.sleep(wait_seconds)
        raise RuntimeError("LLM API 网络请求失败：已达到最大重试次数")

    def chat(
    self,
    messages,
    temperature=0.0
):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        response = self._post_with_retry(
            payload=payload
        )

        if response.status_code != 200:
            raise RuntimeError(
                "LLM API 调用失败："
                f"{response.status_code}\n"
                f"{response.text}"
            )

        result = response.json()

        return (
            result["choices"][0]
            ["message"]["content"]
        )

    def generate_answer(self,query,context):
        system_prompt = """
你是工业设备运维助手。

请严格依据提供的设备手册证据回答用户问题。

回答规则：

1. 只能依据提供的证据回答，不允许编造手册中不存在的信息。

2. 必须严格围绕用户实际提出的问题回答，
   不要因为证据中包含更多内容，就主动扩展到用户没有询问的内容。

3. 如果用户只是询问：
   - 故障代码是什么意思
   - 参数是什么意思
   - 某个功能是什么
   - 某个状态代表什么

   则只回答相关含义、必要的背景或原因。

   不要主动提供：
   - 维修步骤
   - 接线操作
   - 参数修改步骤
   - 拆装或部件更换方法

4. 只有当用户明确询问：
   - 怎么处理
   - 怎么维修
   - 怎么检查
   - 怎么设置
   - 怎么操作

   时，才提供相应操作步骤。

5. 回答应尽量直接，不要机械地把所有检索证据全部复述出来。

6. 重要结论需要标明来源和页码。

7. 如果现有证据不足以回答，应明确说明证据不足，
   不要自行补充答案。
""".strip()

        user_prompt = f"""
用户问题：
{query}

以下是从设备手册中检索到的证据：

{context}

请仅根据以上证据回答用户问题。
""".strip()

        return self.chat(messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ],
    temperature=0.2
)