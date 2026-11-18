"""Exact fault-code matching for fault table rows."""
import logging
import re

logger = logging.getLogger(__name__)
FAULT_CODE = re.compile(r"(?<![a-z0-9_])err\s*\d+(?![a-z0-9_])", re.IGNORECASE)
PANEL_LINE = re.compile(r"^操作面板显示[^\S\r\n]*[:：][^\S\r\n]*(.+)$", re.MULTILINE)


def fault_codes(text):
    return {re.sub(r"\s+", "", match.group()).lower() for match in FAULT_CODE.finditer(text)}


def row_fault_codes(chunk):
    if chunk.get("type") != "table":
        return set()
    panel = PANEL_LINE.search(chunk.get("content") or "")
    return fault_codes(panel.group(1)) if panel else set()


def rank_evidence(reranker, query, candidates, top_k):
    requested = fault_codes(query)
    if not requested:
        return reranker.rerank(query, candidates, top_k=top_k)
    exact = {
        item["id"] for item in candidates
        if requested & row_fault_codes(item["payload"])
    }
    ranked = reranker.rerank(query, candidates, top_k=len(candidates) if exact else top_k)
    ranked.sort(key=lambda item: item["id"] in exact, reverse=True)
    return ranked[:top_k]


def log_retrieval(attempt, document_id, sources):
    logger.info("检索轮次=%s 文档=%s 候选(id, 页码, 故障码，按排名)=%s",
                attempt, document_id or "md380e", [
                    (item["id"], item["payload"].get("page_start"),
                     sorted(row_fault_codes(item["payload"])))
                    for item in sources
                ])
