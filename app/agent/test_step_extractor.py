from app.rag_pipeline import RAGPipeline
from app.agent.step_extractor import StepExtractor

def main():
    rag = RAGPipeline()

    try:
        query = "MD380E Err16 通讯故障怎么处理？"

        # 检索手册证据
        result = rag.retrieve(query=query)

        extractor = StepExtractor(
            llm_provider=rag.llm_provider
        )

        # 提取标准排障步骤
        steps = extractor.extract(
            query=query,
            sources=result["sources"]
        )

        print("\n========== 提取结果 ==========")

        for step in steps:
            print(step.model_dump())

        # 基础检查
        assert len(steps) > 0
        assert len({
            step.step_id for step in steps
        }) == len(steps)

        print("\nStepExtractor 基础测试通过！")

    finally:
        rag.close()


if __name__ == "__main__":
    main()
