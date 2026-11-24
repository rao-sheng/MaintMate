"""Deterministic Markdown reports from a snapshot of AgentState."""
import html

from pydantic import ValidationError

from app.agent.state import AgentState


def text(value: str | None) -> str:
    """Keep stored text literal inside Markdown."""
    if not value or not value.strip():
        return "未提供"
    escaped = html.escape(value)
    for character in ("\\", "`", "*", "_", "[", "]", "#", "|"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped.replace("\r", "").replace("\n", " / ")


def generate_troubleshooting_report(state: AgentState) -> str:
    """Return Markdown without modifying state or contacting external services.

    Raises ValueError when the input is not a valid AgentState snapshot.
    Only feedback history is treated as executed work.
    """
    if not isinstance(state, AgentState):
        raise ValueError("报告输入必须是 AgentState")
    try:
        snapshot = AgentState.model_validate(state.model_dump(), strict=True)
    except (ValidationError, TypeError, ValueError) as error:
        raise ValueError("AgentState 数据格式异常") from error
    context = snapshot.diagnostic_context
    history = snapshot.troubleshooting_history
    # Compare feedback for the same step; failed earlier steps can precede a solution.
    grouped = {}
    for record in history:
        key = record.step_id or record.action
        grouped.setdefault(key, set()).add(record.result)
    confirmed_solution = any(results == {"resolved"} for results in grouped.values())
    conflicting = any("resolved" in results and len(results) > 1 for results in grouped.values())
    if confirmed_solution:
        status = "已解决（依据已记录反馈）"
    elif conflicting:
        status = "尚未确认解决（存在相互冲突的反馈）"
    elif any(record.result == "not_resolved" for record in history):
        status = "尚未解决"
    else:
        status = "尚未确认解决（未提供明确结果）"
    lines = [
        "# 故障排查报告", "", "## 设备基本信息", "",
        f"- 设备型号：{text(context.equipment_model)}",
        f"- 运行状态：{text(context.operating_state)}",
        f"- 最近变化：{text(context.recent_changes)}",
        "", "## 故障信息", "",
        f"- 故障代码：{text(context.fault_code)}",
        f"- 故障现象：{text(context.symptom_description)}",
        f"- 面板显示：{text(context.panel_display)}",
        "", "## 已执行的排查步骤", "",
    ]
    result_labels = {"resolved": "已解决", "not_resolved": "未解决", "unknown": "结果未确认"}
    if not history:
        lines.append("未提供已执行步骤记录。")
    for index, record in enumerate(history, 1):
        lines.append(f"{index}. {text(record.action)}")
        lines.append(f"   - 结果：{result_labels[record.result]}；备注：{text(record.note)}")

    lines.extend(["", "## 排查结果", "", f"故障状态：{status}", "",
                  "## 手册检索依据及来源页码", ""])
    if not snapshot.retrieved_sources:
        lines.append("未提供检索证据，未生成手册引用。")
    for index, source in enumerate(snapshot.retrieved_sources, 1):
        payload = source.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("检索证据 payload 必须是对象")
        for field in ("source", "content", "section"):
            if payload.get(field) is not None and not isinstance(payload[field], str):
                raise ValueError(f"检索证据 {field} 必须是字符串")
        pages = []
        for field in ("page_start", "page_end"):
            page = payload.get(field)
            if page is not None and (type(page) is not int or page < 1):
                raise ValueError("检索证据页码必须是正整数")
            pages.append(page)
        start, end = pages
        if start is not None and end is not None and end < start:
            raise ValueError("检索证据结束页不得小于起始页")
        page_text = str(start) if start is not None else "未提供"
        if end is not None and end != start:
            page_text += f" 至 {end}"
        lines.extend([
            f"{index}. 手册：{text(payload.get('source'))}；页码：{page_text}",
            f"   - 章节：{text(payload.get('section'))}",
            f"   - 检索原文：{text(payload.get('content'))}",
        ])

    lines.extend(["", "## 后续建议", ""])
    if status.startswith("已解决"):
        lines.append("已记录故障解决反馈，建议持续观察并在异常复现时补充记录。")
    else:
        lines.append("建议补充未确认的排查结果，由具备资质和现场授权的人员继续核实。")
    if snapshot.escalated:
        target = {"vendor": "厂家技术支持", "maintenance": "设备维修人员"}.get(
            snapshot.escalation_target, "专业人员"
        )
        lines.append(f"当前会话已建议升级至{target}，实际联系情况未提供。")
    lines.extend(["", "### 待确认的手册排查步骤", ""])
    executed = {record.step_id for record in history if record.step_id is not None}
    remaining = [step for step in snapshot.troubleshooting_steps if step.step_id not in executed]
    if remaining:
        lines.extend(f"- {text(step.action)}（执行情况未提供）" for step in remaining)
    else:
        lines.append("未提供额外待确认步骤。")
    return "\n".join(lines) + "\n"
