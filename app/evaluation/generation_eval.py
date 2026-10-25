import json
import re
from pathlib import Path
from app.rag_pipeline import RAGPipeline

EVAL_PATH = Path(
    "data/evaluation/generation_cases.json"
)

RESULT_PATH = Path(
    "data/evaluation/generation_results.json"
)

KEYPOINT_PASS_THRESHOLD = 0.8

def load_cases():
    with EVAL_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

def normalize_text(text):
    return text.lower().replace(" ", "")

def check_keypoints(
    answer,
    must_include
):
    """
    检查标准关键点有多少出现在回答里。
    """
    normalized_answer = normalize_text(answer)
    matched = []
    missing = []
    for keyword in must_include:
        normalized_keyword = normalize_text(
            keyword
        )
        if normalized_keyword in normalized_answer:
            matched.append(keyword)
        else:
            missing.append(keyword)

    if not must_include:
        coverage = 1.0
    else:
        coverage = (
            len(matched)
            / len(must_include)
        )

    return coverage, matched, missing

def extract_cited_pages(answer):
    """
    从回答中提取类似：
    第249页
    第249-250页

    返回一个页码集合。
    """

    pages = set()

    pattern = r"第\s*(\d+)(?:\s*-\s*(\d+))?\s*页"

    matches = re.findall(
        pattern,
        answer
    )

    for start, end in matches:
        start_page = int(start)

        if end:
            end_page = int(end)

            for page in range(
                start_page,
                end_page + 1
            ):
                pages.add(page)

        else:
            pages.add(start_page)

    return pages


def get_retrieved_pages(sources):
    """
    收集最终 Top-K evidence 实际包含的页码。
    """

    pages = set()

    for source in sources:
        payload = source["payload"]

        page_start = payload.get(
            "page_start"
        )

        page_end = payload.get(
            "page_end"
        )

        if page_start is None:
            continue

        if page_end is None:
            pages.add(page_start)
            continue

        for page in range(
            page_start,
            page_end + 1
        ):
            pages.add(page)

    return pages


def evaluate():
    cases = load_cases()

    rag = RAGPipeline()

    results = []

    try:
        for index, case in enumerate(
            cases,
            start=1
        ):
            print("\n" + "=" * 70)
            print(
                f"[{index}/{len(cases)}]"
            )
            print(
                "问题:",
                case["query"]
            )

            try:
                rag_result = rag.answer(
                    query=case["query"]
                )

            except Exception as e:
                print(
                    "❌ RAG运行失败:",
                    str(e)
                )

                results.append({
                    "id": case["id"],
                    "query": case["query"],
                    "error": str(e)
                })

                continue

            answer = rag_result["answer"]
            sources = rag_result["sources"]

            # -------------------------
            # 1. 关键点覆盖
            # -------------------------

            (
                coverage,
                matched,
                missing
            ) = check_keypoints(
                answer=answer,
                must_include=case.get(
                    "must_include",
                    []
                )
            )

            # -------------------------
            # 2. Citation
            # -------------------------

            cited_pages = extract_cited_pages(
                answer
            )

            retrieved_pages = (
                get_retrieved_pages(
                    sources
                )
            )

            expected_pages = set(
                case.get(
                    "expected_pages",
                    []
                )
            )

            expected_citation_hit = (
                bool(
                    cited_pages
                    & expected_pages
                )
                if expected_pages
                else True
            )

            # 引用了 Retrieval 根本没有提供的页码
            hallucinated_pages = (
                cited_pages
                - retrieved_pages
            )

            citation_grounded = (
                len(hallucinated_pages) == 0
            )

            # -------------------------
            # 3. 自动 Pass
            # -------------------------

            keypoint_pass = (
                coverage
                >= KEYPOINT_PASS_THRESHOLD
            )

            auto_pass = (
                keypoint_pass
                and expected_citation_hit
                and citation_grounded
            )

            print("\n回答：")
            print(answer)

            print("\n--- 自动评测 ---")

            print(
                f"关键点覆盖率: "
                f"{coverage:.0%}"
            )

            print(
                "命中关键点:",
                matched
            )

            print(
                "缺失关键点:",
                missing
            )

            print(
                "回答引用页码:",
                sorted(cited_pages)
            )

            print(
                "期望页码:",
                sorted(expected_pages)
            )

            print(
                "检索证据页码:",
                sorted(retrieved_pages)
            )

            if hallucinated_pages:
                print(
                    "⚠️ 疑似编造引用页码:",
                    sorted(
                        hallucinated_pages
                    )
                )

            print(
                "自动结果:",
                "✅ PASS"
                if auto_pass
                else "❌ FAIL"
            )

            print(
                "\n人工还需要检查："
                "回答是否真的正确、"
                "是否存在证据外推断。"
            )

            # -------------------------
            # 保存结果
            # -------------------------

            result_item = {
                "id": case["id"],
                "query": case["query"],
                "answer": answer,

                "keypoint_coverage": coverage,
                "matched_keypoints": matched,
                "missing_keypoints": missing,

                "cited_pages": sorted(
                    cited_pages
                ),

                "expected_pages": sorted(
                    expected_pages
                ),

                "retrieved_pages": sorted(
                    retrieved_pages
                ),

                "hallucinated_pages": sorted(
                    hallucinated_pages
                ),

                "auto_pass": auto_pass,

                "sources": []
            }

            for source in sources:
                payload = source["payload"]

                result_item[
                    "sources"
                ].append({
                    "section": payload.get(
                        "section"
                    ),
                    "page_start": payload.get(
                        "page_start"
                    ),
                    "page_end": payload.get(
                        "page_end"
                    ),
                    "content": payload.get(
                        "content"
                    )
                })

            results.append(
                result_item
            )

    finally:
        rag.close()

    # -------------------------
    # 保存完整报告
    # -------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with RESULT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # -------------------------
    # 汇总
    # -------------------------

    successful_results = [
        item
        for item in results
        if "error" not in item
    ]

    passed = sum(
        1
        for item in successful_results
        if item["auto_pass"]
    )

    total = len(
        successful_results
    )

    print("\n" + "=" * 70)
    print("Generation Evaluation")
    print("=" * 70)

    print(
        "成功执行问题数:",
        total
    )

    print(
        "自动通过数:",
        passed
    )

    if total:
        print(
            "自动通过率:",
            f"{passed / total:.2%}"
        )

    print(
        "\n结果已保存到:",
        RESULT_PATH
    )


if __name__ == "__main__":
    evaluate()