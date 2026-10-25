def all_steps_failed(
    steps: list[str],
    history: dict[str, str]
) -> bool:
    """判断所有给定排障步骤是否均已尝试且故障未解决。"""

    if not steps:
        return False

    for step_id in steps:
        if history.get(step_id) != "not_resolved":
            return False

    return True


from app.agent.state import AgentState, TroubleshootingRecord
from app.agent.step_matcher import StepMatcher

def save_troubleshooting_feedback(
    state: AgentState,
    feedback: list[TroubleshootingRecord],
    matcher: StepMatcher
) -> None:
    """
    保存本轮排障反馈，并将尚未匹配的记录
    关联到当前故障的标准排障步骤。
    """

    # 1. 保存用户新反馈
    for record in feedback:

        # 避免完全相同的记录重复添加
        exists = any(
            old.action == record.action
            and old.result == record.result
            and old.note == record.note
            for old in state.troubleshooting_history
        )

        if not exists:
            state.troubleshooting_history.append(
                record.model_copy()
            )

    # 2. 没有标准步骤，先保留原始反馈
    if not state.troubleshooting_steps:
        return

    # 3. 找出还没有匹配 step_id 的记录
    unmatched = [
        record
        for record in state.troubleshooting_history
        if record.step_id is None
        and record.result != "resolved"
    ]

    if not unmatched:
        return

    # 4. 调用 LLM 进行语义匹配
    matches = matcher.match(
        steps=state.troubleshooting_steps,
        feedback=unmatched
    )

    # 5. 将匹配结果写回记录
    for index, record in enumerate(
        unmatched,
        start=1
    ):
        record.step_id = matches.get(index)

def build_step_status(history):
    """根据排障历史汇总每个步骤的状态。"""

    grouped = {}

    for record in history:
        if record.step_id is None:
            continue

        grouped.setdefault(
            record.step_id, set()
        ).add(record.result)

    status = {}

    for step_id, results in grouped.items():
        if len(results) == 1:
            status[step_id] = next(iter(results))
        else:
            # 存在相互矛盾的反馈，暂不认定已完成
            status[step_id] = "unknown"

    return status

def evaluate_troubleshooting(steps, history):
    """
    评估当前排障状态。
    返回：
    - insufficient_evidence：没有标准步骤
    - resolved：存在已解决反馈
    - need_confirmation：仍有步骤未确认
    - suggest_escalation：已提取步骤均未解决故障
    """

    status = build_step_status(history)

    if not steps:
        return "insufficient_evidence", []

    # 用户确认某一步已解决故障
    if any(
        status.get(step.step_id) == "resolved"
        for step in steps
    ):
        return "resolved", []

    # 找出尚未确认排查结果的标准步骤
    pending_steps = [
        step
        for step in steps
        if status.get(step.step_id) != "not_resolved"
    ]

    if pending_steps:
        return "need_confirmation", pending_steps

    # 当前提取的全部步骤均有故障未解决的反馈
    return "suggest_escalation", []
