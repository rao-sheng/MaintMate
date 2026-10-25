from app.rag_pipeline import RAGPipeline
from app.agent.answer_builder import AnswerBuilder

def main():
    rag = RAGPipeline()

    try:
        query = "MD380E 出现 Err16，应该怎么处理？"
        # 1. 只检索一次手册证据
        result = rag.retrieve(query=query)
        # 2. 初始化 AnswerBuilder
        builder = AnswerBuilder(
            llm_provider=rag.llm_provider
        )
        # 3. 生成普通用户可阅读的回答
        overview_answer = builder.build(
            query=query,
            context=result["context"],
            scope="overview"
        )
        print("\n========== 信息性回答 ==========")
        print(overview_answer)
        # 4. 使用同一份证据生成维修指导
        maintenance_answer = builder.build(
            query=query,
            context=result["context"],
            scope="maintenance"
        )
        print("\n========== 维修指导 ==========")
        print(maintenance_answer)

    finally:
        rag.close()


if __name__ == "__main__":
    main()
