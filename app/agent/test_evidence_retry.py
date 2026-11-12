from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent.agent import MaintMateAgent
from app.agent.evidence import EvidenceDecision
from app.agent.state import AgentState


@pytest.mark.parametrize("decisions, expected_attempts, enough", [
    ([True], 1, True),
    ([False, True], 2, True),
    ([False, False], 2, False),
])
def test_evidence_retry_preserves_document_scope(decisions, expected_attempts, enough):
    agent = object.__new__(MaintMateAgent)
    agent._build_rag_query = Mock(return_value="MD380E Err16")
    agent.evidence_policy = Mock()
    agent.evidence_policy.evaluate.side_effect = [
        EvidenceDecision(sufficient=value, rewrite="故障处理")
        for value in decisions
    ]
    source = {"id": "evidence", "hybrid_score": 0.1, "payload": {"content": "手册证据"}}
    agent.rag = SimpleNamespace(
        retrieve=Mock(return_value={"context": "手册证据", "sources": [source]}),
        reranker=SimpleNamespace(rerank=Mock(return_value=[source])),
    )
    agent.safety_gate = Mock()
    agent._finish_from_evidence = Mock(return_value="有依据的回答")
    state = AgentState(user_query="怎么处理？", document_id="selected")
    answer = agent._handle_retrieval(state)
    assert state.retrieval_attempts == expected_attempts
    assert agent.rag.retrieve.call_count == expected_attempts
    for call in agent.rag.retrieve.call_args_list:
        assert call.kwargs["document_id"] == "selected"
        assert "MD380E Err16" in call.kwargs["query"]
    assert agent.safety_gate.evaluate.called is enough
    assert ("未找到足够证据" in answer) is (not enough)


def test_evidence_service_failure_is_distinct_from_insufficient_evidence():
    from app.agent.evidence import EvidencePolicy, EvidenceUnavailable

    llm = Mock()
    llm.chat.side_effect = RuntimeError("HTTP 429")
    with pytest.raises(EvidenceUnavailable):
        EvidencePolicy(llm).evaluate("Err16", "手册证据")
