def build_context(results):
    """
    把 Reranker 的最终 Top-K 结果整理成给 LLM 的上下文文本。

    results 中每个元素格式类似：
    {
        "id": ...,
        "hybrid_score": ...,
        "rerank_score": ...,
        "payload": {
            "section": ...,
            "caption": ...,
            "page_start": ...,
            "page_end": ...,
            "content": ...,
            "source": ...,
            ...
        }
    }
    """

    context_parts = []

    for rank, result in enumerate(results, start=1):
        payload = result["payload"]

        section = payload.get("section")
        caption = payload.get("caption")
        page_start = payload.get("page_start")
        page_end = payload.get("page_end")
        content = payload.get("content")
        source = payload.get("source")

        evidence_parts = [
            f"[证据 {rank}]"
        ]

        if source:
            evidence_parts.append(
                f"来源：{source}"
            )

        if section:
            evidence_parts.append(
                f"章节：{section}"
            )

        if caption:
            evidence_parts.append(
                f"表格：{caption}"
            )

        if page_start is not None:
            if (
                page_end is not None
                and page_end != page_start
            ):
                evidence_parts.append(
                    f"页码：{page_start}-{page_end}"
                )
            else:
                evidence_parts.append(
                    f"页码：{page_start}"
                )

        if content:
            evidence_parts.append(
                f"内容：\n{content}"
            )

        context_parts.append(
            "\n".join(evidence_parts)
        )

    return "\n\n".join(context_parts)


if __name__ == "__main__":
    # 这里只是测试 context_builder 是否正常
    test_results = [
        {
            "id": 100,
            "hybrid_score": 0.03,
            "rerank_score": 0.95,
            "payload": {
                "section": "9.3 故障报警及对策",
                "caption": None,
                "page_start": 249,
                "page_end": 249,
                "content": (
                    "故障名称：通讯故障\n"
                    "操作面板显示：Err16\n"
                    "故障原因排查：1、上位机工作不正常 "
                    "2、通讯线不正常\n"
                    "故障处理对策：1、检查上位机接线 "
                    "2、检查通讯连接线"
                ),
                "source": "MD380E系列专用变频器用户手册"
            }
        },
        {
            "id": 200,
            "hybrid_score": 0.02,
            "rerank_score": 0.88,
            "payload": {
                "section": "4.15 变频器串行通讯的使用方法",
                "caption": None,
                "page_start": 70,
                "page_end": 70,
                "content": (
                    "通讯端口的硬件通讯参数配置见 Fd 组功能，"
                    "通讯速率、数据格式需要与上位机保持一致。"
                ),
                "source": "MD380E系列专用变频器用户手册"
            }
        }
    ]

    context = build_context(test_results)

    print(context)