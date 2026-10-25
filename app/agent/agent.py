from app.agent.state import AgentState, DiagnosticContext
from app.agent.input_analyzer import InputAnalyzer
from app.agent.sufficiency import InformationSufficiencyPolicy
from app.agent.followup import FollowUpQuestionBuilder
from app.llm.llm_provider import ZhipuLLMProvider
from app.rag_pipeline import RAGPipeline
from app.agent.safety import SafetyGate
from app.agent.answer_builder import AnswerBuilder
from app.agent.step_extractor import StepExtractor
from app.agent.step_matcher import StepMatcher
from app.agent.troubleshooting import save_troubleshooting_feedback,evaluate_troubleshooting

class MaintMateAgent:
    def __init__(self):
        self.llm_provider = ZhipuLLMProvider()
        self.input_analyzer = InputAnalyzer(
            llm_provider=self.llm_provider
        )
        self.sufficiency_policy = (
            InformationSufficiencyPolicy()
        )
        self.followup_builder = (
            FollowUpQuestionBuilder()
        )
        self.rag = RAGPipeline()
        self.safety_gate = SafetyGate(
            llm_provider=self.llm_provider
        )
        self.answer_builder = AnswerBuilder(
            llm_provider=self.llm_provider
        )
        self.step_extractor = StepExtractor(
            llm_provider=self.llm_provider
        )
        self.step_matcher = StepMatcher(
            llm_provider=self.llm_provider
        )
   
    def _continue_troubleshooting(
        self,
        state: AgentState
    ) -> str:
        """
        结合当前故障的排障历史，提供后续建议。
        不重新检索 RAG，不重新执行 Safety LLM。
        """

        # 1. 整理已经记录的排障历史
        history_parts = []

        for record in state.troubleshooting_history:
            history_parts.append(
                f"- 操作：{record.action}\n"
                f"  结果：{record.result}\n"
                f"  备注：{record.note or '无'}"
            )

        history_text = "\n".join(history_parts)

        # 2. 组织新的回答问题
        query = f"""
    当前正在处理的故障：
    {state.pending_query}

    用户最新反馈：
    {state.user_query}

    已经记录的排障历史：
    {history_text}

    请根据手册证据和排障历史继续处理当前故障。

    要求：
    1. 首先简要确认用户已经尝试的措施。
    2. 不要把已尝试且未解决的措施重复作为新的排查步骤。
    3. 优先说明手册中尚未确认的排查项目。
    4. unknown 表示结果未确认，不代表检查成功或失败。
    5. 不得编造新的排查操作。
    6. 不得提供超出维修人员权限的专属操作。
    7. 如果无法确定剩余的排查方法，应说明情况。
    8. 不要仅因为某些检查失败，就断言全部方法都已用尽。
    9. 如果手册建议在常规排查无效后联系厂家，
    可以提示后续需要厂家技术支持。
    10.用户反馈已经检查某项参数，不等于确认该参数
已经按照设备手册正确设置。

对于无法对应标准步骤的排障反馈，
应指出尚需确认的信息，不要直接认定该步骤完成。

不得自行宣称手册中的全部排障方法已经穷尽。
是否满足升级条件由程序的排障状态判断负责。
    """.strip()

        # 3. 复用已有证据生成后续回答
        answer = self.answer_builder.build(
            query=query,
            context=state.rag_context, #type:ignore
            scope="maintenance"
        )

        # 4. 保存结果
        state.next_action = "answer"
        state.final_answer = answer

        return answer

    def _reset_fault_context(self, state: AgentState):
        """
        开始一个新故障时，清理上一故障的数据。
        保留当前用户身份。
        """

        # 保存用户身份
        current_role = state.diagnostic_context.user_role

        # 重建故障上下文，但保留身份
        state.diagnostic_context = DiagnosticContext(
            user_role=current_role
        )

        # 清理旧故障的 RAG 缓存
        state.pending_query = None
        state.rag_context = None
        state.rag_answer = None
        state.retrieved_sources = []

        # 清理安全分析结果
        state.required_role = "unknown"
        state.safety_reason = None
        state.safety_checked = False

        # 清理排障历史
        state.troubleshooting_history = []

        # 重置流程状态
        state.intent = "unknown"
        state.missing_fields = []
        state.escalated = False
        state.final_answer = None
        state.escalation_reason = None
        state.escalation_target = None       
        state.troubleshooting_steps = []
    
    def _handle_fault_relation_confirmation(
        self,
        state: AgentState,
        user_input: str
    ) -> str | None:
        """
        处理用户对新旧故障关系的确认。

        返回：
        str  -> 已处理，本轮直接结束
        None -> 无法直接识别，交给 InputAnalyzer
        """

        # 1. 清理空格和句末标点
        text = "".join(user_input.split()).rstrip("。！？.!?")

        # 2. 用户确认是之前的故障
        old_fault_answers = {
            "是之前的故障",
            "之前的故障",
            "还是之前的故障",
            "是原来的故障",
            "还是原来的问题",
            "是同一个故障",
            "继续之前的故障",
            "不是新故障",
        }

        if text in old_fault_answers:

            # 清除等待确认的标记
            state.missing_fields = [
                field
                for field in state.missing_fields
                if field != "fault_relation"
            ]

            state.next_action = "answer"
            state.final_answer = (
                "好的，我们继续处理之前的设备故障。"
                "请补充你刚才提到的新异常现象，"
                "我会结合已有的故障信息继续分析。"
            )

            return state.final_answer

        # 3. 用户确认是一个新的故障
        new_fault_answers = {
            "是新故障",
            "这是新故障",
            "一个新故障",
            "是另一个故障",
            "不是之前的故障",
            "是另一台设备",
            "换了一个问题",
        }

        if text in new_fault_answers:

            # 清除之前的故障上下文
            self._reset_fault_context(state)

            state.next_action = "ask_user"
            state.missing_fields = [
                "equipment_model",
                "fault_code",
            ]

            state.final_answer = (
                "好的，我们开始处理新的设备故障。"
                "请提供设备型号、故障代码（如果有），"
                "以及具体的异常现象。"
            )

            return state.final_answer

        # 4. 无法直接识别，交给正常分析流程
        return None

    def process(
        self,
        state: AgentState,
        user_input: str | None = None
    ) -> str:#type:ignore
        """  
        处理用户的一轮输入。
        多轮对话时继续复用同一个 AgentState，
        新信息会不断合并到 State 中。
        """
        waiting_for_user_role = (
            state.next_action == "ask_user"
            and "user_role" in state.missing_fields
            and state.safety_checked
            and state.rag_context is not None
            and state.pending_query is not None
        )#进入这一轮之前 上一轮Agent是不是在等待user_role
        
        #先保留之前的意图
        previous_intent = state.intent
        # 是否存在正在处理的故障
        has_active_fault = bool(
            state.pending_query
            and (
                state.intent == "fault_diagnosis"
                or state.diagnostic_context.fault_code
                or state.diagnostic_context.symptom_description
            )
        )
        # -------------------------
        # 1. 更新这一轮用户输入
        # -------------------------
        if user_input is not None:
            state.user_query = user_input

        # 是否正在等待新旧故障确认
        waiting_for_fault_relation = (
            state.next_action == "ask_user"
            and "fault_relation" in state.missing_fields
        )

        if waiting_for_fault_relation:
            answer = self._handle_fault_relation_confirmation(
                state,
                state.user_query
            )

            if answer is not None:
                return answer

        # 正常执行输入分析
        analysis = self.input_analyzer.analyze(
            user_input=state.user_query,
            state=state
        )
        # -------------------------
        # 3. 根据对话关系分流
        # -------------------------
        # 情况 A：明确开始新故障
        # 处理本轮与当前故障的关系
        relation_answer, waiting_for_user_role = (
            self._handle_turn_relation(
                state=state,
                analysis=analysis,
                has_active_fault=has_active_fault,
                waiting_for_user_role=waiting_for_user_role
            )
        )

        if relation_answer is not None:
            return relation_answer
        # -------------------------
        # 5. 合并本轮信息
        # -------------------------
        self._update_turn_state(
            state=state,
            analysis=analysis,
            previous_intent=previous_intent,
            has_active_fault=has_active_fault,
            waiting_for_user_role=waiting_for_user_role
        )
        # -------------------------
        # 用户主动请求升级
        # -------------------------
        if analysis.escalation_requested:
            reason = (
                "troubleshooting_failed"
                if state.escalation_reason == "troubleshooting_failed"
                else "user_requested"
            )

            return self._handle_escalation(
                state=state,
                reason=reason,
                target=analysis.escalation_target or "vendor"
            )
        # 当前故障明确解决：任何身份都可以报告结果
        if (
            has_active_fault
            and analysis.turn_relation == "continue_fault"
            and any(
                record.result == "resolved"
                for record in analysis.troubleshooting_feedback
            )
        ):
            state.escalation_reason = None
            state.escalation_target = None
            state.next_action = "answer"

            state.final_answer = (
                "根据你的反馈，当前故障已经解决。"
                "本次排障记录已保存。"
            )

            return state.final_answer
        # -------------------------
        # 6. 处理用户补充身份
        # -------------------------
        if waiting_for_user_role:
            return self._finish_from_evidence(state)

        # -------------------------
        # 继续当前故障的排障过程
        # -------------------------
        # 优先处理当前故障的排障进度
        troubleshooting_answer = (
            self._handle_troubleshooting_progress(
                state=state,
                analysis=analysis,
                has_active_fault=has_active_fault
            )
        )

        if troubleshooting_answer is not None:
            return troubleshooting_answer
        # -------------------------
        # 判断当前信息是否足够
        # -------------------------
        # 根据当前信息决定下一步动作
        sufficiency = self.sufficiency_policy.evaluate(state)
        state.missing_fields = sufficiency.missing_fields
        state.next_action = sufficiency.next_action#type:ignore
        # 执行决定好的动作
        return self._execute_action(state)
            
    def _build_rag_query(
        self,
        state: AgentState
    ) -> str:
        """
        把多轮对话中已经累计的信息重新组成
        一个完整的 Retrieval Query。
        不能只使用用户最后一句话。
        """
        context = state.diagnostic_context
        parts = []
        if context.equipment_model:
            parts.append(
                f"设备型号：{context.equipment_model}"
            )
        if context.fault_code:
            parts.append(
                f"故障代码：{context.fault_code}"
            )
        if context.panel_display:
            parts.append(
                f"面板显示：{context.panel_display}"
            )
        if context.symptom_description:
            parts.append(
                f"故障现象：{context.symptom_description}"
            )
        if context.operating_state:
            parts.append(
                f"设备状态：{context.operating_state}"
            )
        if context.recent_changes:
            parts.append(
                f"最近变化：{context.recent_changes}"
            )
        if context.parameter_code:
            parts.append(
                f"参数代码：{context.parameter_code}"
            )
        if context.operation_request:
            parts.append(
                f"用户希望进行的操作："
                f"{context.operation_request}"
            )
        parts.append(
            f"用户当前问题：{state.user_query}"
        )
        return "\n".join(parts)
    def _answer_without_rag(
        self,
        state: AgentState
    ) -> str:
        """
        非设备知识问题不需要调用 RAG。
        """
        return self.llm_provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 MaintMate 工业运维助手。"
                        "当前问题不需要查询设备手册，"
                        "请简洁回答用户。"
                    )
                },
                {
                    "role": "user",
                    "content": state.user_query
                }
            ],
            temperature=0.2
        )
    def close(self):
        self.rag.close()

    def _finish_from_evidence(
        self,
        state: AgentState
    ) -> str:
        """
        根据已缓存的 RAG 证据和当前用户身份生成回答。
        不重新检索，不重新分析证据。
        """

        if not state.rag_context or not state.pending_query:
            raise RuntimeError("缺少待处理问题或 RAG 证据")

        role = state.diagnostic_context.user_role
        required_role = state.required_role

        # 厂家专属操作：当前版本不生成厂家级操作指导
        if required_role == "vendor_technician":
            return self._handle_escalation(
                state=state,
                reason="permission_required",
                target="vendor"
            )
        # 使用 SafetyGate 已有的 Python 权限比较
        decision = self.safety_gate.check_role(
            user_role=role,
            required_role=required_role,
            reason=state.safety_reason
        )

        # 默认只提供信息性回答
        scope = "overview"

        # 只有确定是维修操作，并且权限满足，
        # 才允许生成维修指导
        if (
            required_role == "maintenance"
            and decision.action == "answer"
        ):
            scope = "maintenance"

        answer = self.answer_builder.build(
            query=state.pending_query,
            context=state.rag_context,
            scope=scope
        )

        # 无法确定权限时，只给信息性回答
        if required_role == "unknown":
            answer += (
                "\n\n当前手册证据不足以确认相关操作的"
                "权限要求，请勿据此直接开展维修操作，"
                "建议由具备相应授权的专业人员核实。"
            )

        # 需要补充身份：先给信息性回答，再追问
        if decision.action == "ask_user":
            state.next_action = "ask_user"
            state.missing_fields = ["user_role"]

            question = self.followup_builder.build(state)

            state.final_answer = (
                answer + "\n\n" + question
            )
            return state.final_answer

        # 其他情况均只返回对应范围的回答
        state.next_action = "answer"
        state.missing_fields = []
        state.final_answer = answer

        return state.final_answer
 
    def _handle_escalation(
        self,
        state: AgentState,
        reason: str,
        target: str | None = None
    ) -> str:
        """
        统一处理升级请求。
        当前版本只整理升级信息，不实际提交工单。
       """
        state.next_action = "escalate"
        state.escalated = True
        state.escalation_reason = reason#type:ignore
        state.escalation_target = target#type:ignore

        context = state.diagnostic_context

        # 1. 整理故障信息
        equipment = context.equipment_model or "未提供"
        fault_code = context.fault_code or "未提供"
        symptom = context.symptom_description or "未提供"

        # 2. 整理排障记录
        history = []

        for record in state.troubleshooting_history:
            history.append(
                f"- {record.action}：{record.result}"
            )

        history_text = (
            "\n".join(history)
            if history
            else "暂无排障记录"
        )

        # 3. 根据升级目标选择提示
        if target == "vendor":
            target_name = "厂家技术支持"
        elif target == "maintenance":
            target_name = "设备维修人员"
        else:
            target_name = "人工技术支持"

        # 4. 根据升级原因生成说明
        reason_messages = {
            "user_requested": "你已主动请求人工升级处理。",
            "troubleshooting_failed": "现有适用排障措施未能解决故障。",
            "permission_required": (
                "当前请求涉及受限维修操作，"
                "需要由具备相应授权的技术人员进一步处理或确认。"
            ),
            "safety_risk": "当前操作存在需要人工评估的安全风险。"
        }

        reason_text = reason_messages.get(
            reason,
            "当前问题需要人工进一步处理。"
        )

        # 5. 生成升级摘要
        answer = f"""
    {reason_text}

    建议联系：{target_name}

    故障信息：
    - 设备型号：{equipment}
    - 故障代码：{fault_code}
    - 故障现象：{symptom}

    已记录的排障过程：
    {history_text}

    以上信息可以作为联系技术支持时的故障说明。

    当前 MaintMate 尚未接入实际工单提交或厂家联系工具，
    因此尚未自动提交支持请求。
    """.strip()

        state.final_answer = answer

        return answer

    def _handle_retrieval(self, state: AgentState) -> str:
        """执行证据检索、排障步骤提取、安全判断和回答。"""

        # 1. 构造查询
        rag_query = self._build_rag_query(state)
        state.pending_query = rag_query

        # 2. 检索并保存证据
        result = self.rag.retrieve(query=rag_query)

        state.rag_context = result["context"]
        state.retrieved_sources = result["sources"]

        # 3. 故障诊断：提取步骤并匹配已有反馈
        if state.intent == "fault_diagnosis":

            state.troubleshooting_steps = (
                self.step_extractor.extract(
                    query=rag_query,
                    sources=state.retrieved_sources
                )
            )

            save_troubleshooting_feedback(
                state=state,
                feedback=[],
                matcher=self.step_matcher
            )

        # 4. 安全判断
        self.safety_gate.evaluate(state)

        # 5. 根据证据生成回答
        return self._finish_from_evidence(state)

    def _update_turn_state(
        self,
        state: AgentState,
        analysis,
        previous_intent: str,
        has_active_fault: bool,
        waiting_for_user_role: bool
    ) -> None:
        """更新本轮对话状态，不负责生成回答。"""

        # 1. 合并用户提供的设备、故障、身份等信息
        self.input_analyzer.update_state(
            state=state,
            analysis=analysis
        )

        # 2. 继续旧故障时保留原来的意图
        if (
            has_active_fault
            and (
                waiting_for_user_role
                or analysis.turn_relation == "continue_fault"
            )
        ):
            state.intent = previous_intent#type:ignore

        # 3. 保存并匹配排障反馈
        if analysis.troubleshooting_feedback:
            save_troubleshooting_feedback(
                state=state,
                feedback=analysis.troubleshooting_feedback,
                matcher=self.step_matcher
            )

    def _handle_troubleshooting_progress(
        self,
        state: AgentState,
        analysis,
        has_active_fault: bool
    ) -> str | None:
        """处理当前故障的排障反馈和进度决策。"""

        # 1. 判断是否满足继续排障的条件
        can_continue = (
            has_active_fault
            and analysis.turn_relation == "continue_fault"
            and bool(analysis.troubleshooting_feedback)
            and state.rag_context is not None
            and state.pending_query is not None
            and state.safety_checked
            and state.required_role == "maintenance"
            and state.diagnostic_context.user_role in (
                "maintenance",
                "vendor_technician"
            )
        )

        if not can_continue:
            return None

        # 2. 本轮有新反馈，之前的待升级建议需要重新评估
        if (
            state.escalation_reason == "troubleshooting_failed"
            and not state.escalated
        ):
            state.escalation_reason = None
            state.escalation_target = None

        # 3. 交给 Python 评估排障进度
        decision, pending_steps = evaluate_troubleshooting(
            state.troubleshooting_steps,
            state.troubleshooting_history
        )

        # 4. 仍有未确认的排障步骤
        if decision == "need_confirmation":

            pending_text = "\n".join(
                f"- {step.action}"
                for step in pending_steps
            )

            state.final_answer = (
                "根据当前排障记录，以下项目还没有"
                "得到明确的完成情况确认：\n"
                f"{pending_text}\n\n"
                "请确认这些项目是否已按照设备手册要求"
                "完成检查或设置，以及故障是否仍然存在。"
            )

            state.next_action = "answer"
            return state.final_answer

        # 5. 当前提取的步骤均未解决故障
        if decision == "suggest_escalation":

            state.escalation_reason = "troubleshooting_failed"

            state.final_answer = (
                "根据当前排障记录，已提取的排障步骤"
                "均已反馈故障仍未解决。\n\n"
                "这些步骤不一定覆盖所有适用的处理方法。"
                "结合手册建议，可以联系厂家或产品代理商"
                "寻求进一步技术支持。\n\n"
                "如果需要我帮你整理故障信息和排障记录，"
                "请回复“需要厂家支持”。"
            )

            state.next_action = "answer"
            return state.final_answer

        # 6. 其他情况沿用原来的排障处理
        return self._continue_troubleshooting(state)

    def _handle_turn_relation(
        self,
        state: AgentState,
        analysis,
        has_active_fault: bool,
        waiting_for_user_role: bool
    ) -> tuple[str | None, bool]:
        """
        处理本轮输入与当前故障的关系。
        返回：
        (answer, waiting_for_user_role)
        answer:
            str  -> 本轮已经处理，可以直接返回
            None -> 继续 process 后续流程

        waiting_for_user_role:
            返回更新后的身份等待状态
        """

        # A. 用户明确开始新的故障
        if (
            has_active_fault
            and analysis.turn_relation == "new_fault"
        ):
            self._reset_fault_context(state)

            # 新故障不再等待旧故障的身份回复
            waiting_for_user_role = False

            return None, waiting_for_user_role

        # B. 故障关系不明确
        if (
            has_active_fault
            and analysis.turn_relation == "uncertain"
            and not analysis.escalation_requested
            and not (
                waiting_for_user_role
                and analysis.user_role != "unknown"
            )
        ):
            state.next_action = "ask_user"
            state.missing_fields = ["fault_relation"]

            state.final_answer = (
                "请确认一下："
                "你现在描述的是之前那个故障的延续，"
                "还是一个新的设备故障？"
            )

            return state.final_answer, waiting_for_user_role

        # C. 无关的普通聊天
        if (
            has_active_fault
            and analysis.turn_relation == "unrelated"
            and analysis.intent == "general_chat"
            and not analysis.escalation_requested
        ):
            answer = self._answer_without_rag(state)
            state.final_answer = answer

            return answer, waiting_for_user_role

        # D. 不需要特殊处理，继续主流程
        return None, waiting_for_user_role

    def _execute_action(self, state: AgentState) -> str:
        """
        根据 next_action 执行对应操作。

        不负责判断信息是否充足，
        只执行已经确定的动作。
        """

        # 1. 信息不足，追问用户
        if state.next_action == "ask_user":
            question = self.followup_builder.build(state)

            state.final_answer = question
            return question

        # 2. 普通聊天，直接回答
        if state.next_action == "answer":
            answer = self._answer_without_rag(state)

            state.final_answer = answer
            return answer

        # 3. 需要手册知识，执行 RAG
        if state.next_action == "retrieve":
            return self._handle_retrieval(state)

        # 4. 需要升级处理
        if state.next_action == "escalate":

            if (
                state.escalation_reason is None
                or state.escalation_target is None
            ):
                raise RuntimeError(
                    "升级操作缺少 escalation_reason "
                    "或 escalation_target"
                )

            return self._handle_escalation(
                state=state,
                reason=state.escalation_reason,
                target=state.escalation_target
            )

        # 5. 未知动作
        raise RuntimeError(
            f"未知 Agent 动作：{state.next_action}"
        )


