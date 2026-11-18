import json
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

def test_real_md380e_row_survives_retrieval_and_reaches_evidence_policy(monkeypatch):
    from app.agent.evidence import EvidencePolicy
    from app.rag_pipeline import RAGPipeline, document_bm25
    from app.retrieval.bm25_retriever import CHUNK_PATH
    from app.retrieval.hybrid_retriever import HybridRetriever

    if not CHUNK_PATH.is_file():
        pytest.skip("需要本地 MD380E 手册切片")
    corpus = document_bm25("md380e").chunks
    row_id, row = next(
        (index, chunk) for index, chunk in enumerate(corpus)
        if "操作面板显示：Err16\n" in chunk["content"]
    )
    wrong_id = next(
        index for index, chunk in enumerate(corpus)
        if chunk["page_start"] == 249 and "操作面板显示：Err09" in chunk["content"]
    )
    monkeypatch.setattr(HybridRetriever, "retrieve", lambda *args, **kwargs: [
        {"id": wrong_id, "payload": corpus[wrong_id], "score": 1.0}
    ])
    rag = object.__new__(RAGPipeline)
    rag.dense_retriever = None

    def rerank(query, candidates, top_k):
        # Give the unrelated row the best model rank.
        return [dict(item, hybrid_score=item["score"]) for item in candidates[:top_k]]

    rag.reranker = SimpleNamespace(rerank=rerank)
    agent = object.__new__(MaintMateAgent)
    agent.rag = rag
    agent.safety_gate = Mock()
    agent._finish_from_evidence = Mock(return_value="通讯故障")

    def decide(messages):
        assert row["content"] in json.loads(messages[-1]["content"])["evidence"]
        return "```json\n{\"sufficient\":true}\n```"

    agent.evidence_policy = EvidencePolicy(SimpleNamespace(chat=decide))
    state = AgentState(user_query="MD380E 出现 Err16，应该怎么处理？")
    assert agent._handle_retrieval(state) == "通讯故障"
    assert state.retrieval_attempts == 1
    assert state.retrieved_sources[0]["id"] == row.get("chunk_id", row_id)
    missing = rag.retrieve("MD380E Err160000 怎么处理？")
    assert [item["id"] for item in missing["sources"]] == [wrong_id]
    agent.evidence_policy = EvidencePolicy(Mock(chat=Mock(return_value='{"sufficient":false}')))
    missing_state = AgentState(user_query="MD380E Err160000 怎么处理？")
    assert "未找到足够证据" in agent._handle_retrieval(missing_state)
    assert missing_state.retrieval_attempts == 2


def test_invalid_evidence_format_is_unavailable():
    from app.agent.evidence import EvidencePolicy, EvidenceUnavailable

    with pytest.raises(EvidenceUnavailable, match="无效格式"):
        EvidencePolicy(Mock(chat=Mock(return_value="invalid"))).evaluate("Err16", "手册证据")
