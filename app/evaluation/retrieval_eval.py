import json
from pathlib import Path
from app.retrieval.embedding import ZhipuEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker

EVAL_PATH = Path(
    "data/evaluation/retrieval_cases.json"
)

CANDIDATE_K = 20
HYBRID_TOP_K = 10
FINAL_TOP_K = 5

def load_cases():
    with EVAL_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

def build_result_text(result):
    """
    把一个检索结果中可用于判断的文本拼起来。
    """

    payload = result["payload"]
    parts = []
    section = payload.get("section")
    caption = payload.get("caption")
    content = payload.get("content")
    if section:
        parts.append(section)
    if caption:
        parts.append(caption)
    if content:
        parts.append(content)
    return "\n".join(parts)

def is_relevant(result, case):
    text = build_result_text(result).lower()
    acceptable_evidence = case.get(
        "acceptable_evidence",
        []
    )
    for keyword_group in acceptable_evidence:
        all_matched = True
        for keyword in keyword_group:
            if keyword.lower() not in text:
                all_matched = False
                break
        if all_matched:
            return True
    return False

def find_first_relevant_rank(
    results,
    case
):
    """
    找到第一个正确证据所在的排名。

    如果 Top-K 中完全没有正确证据，
    返回 None。
    """
    for rank, result in enumerate(
        results,
        start=1
    ):
        if is_relevant(result, case):
            return rank
    return None

def evaluate():
    cases = load_cases()
    embedding_provider = (
        ZhipuEmbeddingProvider()
    )
    vector_store = QdrantVectorStore()
    dense_retriever = DenseRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store
    )
    bm25_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever
    )
    reranker = Reranker()
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_rank_sum = 0.0
    try:
        for index, case in enumerate(
            cases,
            start=1
        ):
            query = case["query"]
            print("\n" + "=" * 70)
            print(
                f"[{index}/{len(cases)}]"
            )
            print("问题:", query)
            # 1. Hybrid Recall
            hybrid_results = (
                hybrid_retriever.retrieve(
                    query=query,
                    top_k=HYBRID_TOP_K,
                    candidate_k=CANDIDATE_K
                )
            )
            # 2. Rerank
            final_results = reranker.rerank(
                query=query,
                candidates=hybrid_results,
                top_k=FINAL_TOP_K
            )
            # 3. 找正确证据第一次出现的位置
            relevant_rank = (
                find_first_relevant_rank(
                    final_results,
                    case
                )
            )
            if relevant_rank is None:
                print(
                    "结果: ❌ Top-5 未命中正确证据"
                )
            else:
                print(
                    "结果: ✅ 正确证据排名:",
                    relevant_rank
                )

                reciprocal_rank_sum += (
                    1 / relevant_rank
                )

                if relevant_rank <= 1:
                    hit_at_1 += 1

                if relevant_rank <= 3:
                    hit_at_3 += 1

                if relevant_rank <= 5:
                    hit_at_5 += 1

            # 打印 Top-5，方便人工检查
            for rank, result in enumerate(
                final_results,
                start=1
            ):
                payload = result["payload"]

                marker = ""

                if is_relevant(result, case):
                    marker = " <-- 正确证据"

                print(
                    f"\nTop {rank}{marker}"
                )
                print(
                    "section:",
                    payload.get("section")
                )
                print(
                    "page:",
                    payload.get("page_start"),
                    "-",
                    payload.get("page_end")
                )

                content = (
                    payload.get("content")
                    or ""
                )

                # 避免输出太长
                print(
                    "content:",
                    content[:300]
                )

        # -------------------------
        # 汇总指标
        # -------------------------

        total = len(cases)

        hit1 = hit_at_1 / total
        hit3 = hit_at_3 / total
        hit5 = hit_at_5 / total

        mrr = (
            reciprocal_rank_sum
            / total
        )

        print("\n" + "=" * 70)
        print("Retrieval Evaluation")
        print("=" * 70)

        print(
            f"测试问题数: {total}"
        )

        print(
            f"Hit@1: {hit1:.2%}"
        )

        print(
            f"Hit@3: {hit3:.2%}"
        )

        print(
            f"Hit@5: {hit5:.2%}"
        )

        print(
            f"MRR@5: {mrr:.4f}"
        )
    finally:
        vector_store.close()

if __name__ == "__main__":
    evaluate()