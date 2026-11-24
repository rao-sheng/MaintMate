# MaintMate 🔧

**基于 RAG 与 Agent 的工业设备运维辅助系统**

面向工业设备技术手册的故障知识检索、多轮排障记录与技术支持交接原型。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/RAG-%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA-orange?style=for-the-badge" alt="RAG">
  <img src="https://img.shields.io/badge/Agent-%E5%A4%9A%E8%BD%AE%E6%8E%92%E9%9A%9C-blue?style=for-the-badge" alt="Agent">
  <img src="https://img.shields.io/badge/Status-%E5%8E%9F%E5%9E%8B%E9%98%B6%E6%AE%B5-lightgrey?style=for-the-badge" alt="Status">
</p>

> ⚠️ **当前范围：** 主要使用《MD380E 系列专用变频器用户手册》构建和验证。这是一个辅助诊断原型，**不直接控制设备、不执行维修操作，也不会自动联系厂家**。

---

## 📑 目录

- [项目背景](#1-项目背景)
- [已实现功能](#2-已实现功能)
- [系统架构](#3-系统架构)
- [技术栈](#4-技术栈)
- [项目结构](#5-项目结构)
- [本地运行](#6-本地运行)
- [测试与评估](#7-测试与评估)
- [当前限制与安全说明](#8-当前限制与安全说明)
- [上传手册与 Agentic RAG](#上传手册与-agentic-rag)

---

## 1. 项目背景

工业设备故障处理通常需要反复查阅用户手册、核对故障代码，并根据已经完成的检查逐步排查。单轮 RAG 可以帮助定位手册知识，但难以持续追踪「当前是什么故障、检查过哪些项目、下一步还需要确认什么」。

MaintMate 将可上传的设备手册知识库与有状态的 Agent 工作流结合。

- **RAG** 提供可追溯的文档证据；
- **Agent** 负责分析新旧故障关系、维护排障历史、检查操作权限要求，并根据确定性规则决定是否继续追问或建议升级。

---

## 2. 已实现功能

| 功能                 | 说明                                                                                                                                                |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔍 **设备手册检索**   | 基于 MinerU 解析后的结构化数据切分文本与表格，使用智谱 `embedding-3`、Qdrant 向量检索、BM25 关键词检索、RRF 融合和 CrossEncoder 重排。              |
| **手册上传** | 上传 PDF 后调用 MinerU 解析并按手册索引，支持查询处理状态、重试和删除。 |
| **Agentic RAG** | 判断检索证据是否支持问题，证据不足时限定在同一手册内改写并重检一次。 |
| 📎 **证据溯源**       | 检索结果保留文档、章节及页码信息，回答基于手册证据组织；在证据不足时要求谨慎回答。                                                                  |
| 🔄 **多轮故障上下文** | 通过 `AgentState` 保存故障码、设备信息、用户角色、检索证据、待确认项和排障历史；识别继续故障与新故障。                                              |
| 🧩 **排障步骤管理**   | `StepExtractor` 从手册证据提取标准步骤，`StepMatcher` 尝试将用户反馈关联到步骤 ID，再由 Python 规则评估剩余待确认步骤。                             |
| 🛡️ **安全约束**       | `SafetyGate` 根据手册分析操作所需角色，再由 Python 执行角色分支；证据不确定时采用保守回答。当前角色来自用户声明，不是正式身份认证。                 |
| 🤝 **升级交接**       | 对无法解决的故障建议联系维修人员或厂家，并可在用户请求后生成技术支持交接信息；不执行实际工单提交或外部联系。                                        |
| 🌐 **HTTP 接口**      | FastAPI `POST /chat` 支持新建会话、复用会话、状态隔离和异常响应。成功后在 SQLite 中提交会话状态和本轮消息。 |

---

## 3. 系统架构

```mermaid
flowchart TB
    A["用户上传 PDF"] --> B["MinerU 解析"]
    B --> C["按章节和表格切分"]
    C --> D["Qdrant 向量索引"]
    C --> E["BM25 文档切片"]
    A --> S["SQLite 文档状态"]

    I["POST /chat"] --> J["Agent 加载会话与手册范围"]
    J --> L{"用户信息充足？"}
    L -->|否| P["追问"]
    L -->|是| F["按手册混合检索、RRF 与重排"]
    D --> F
    E --> F
    F --> H{"证据充足？"}
    H -->|首次不足| R["改写查询，仅重检一次"]
    R --> F
    H -->|仍不足| X["说明证据不足"]
    H -->|充足| N["安全判断与回答"]
    N --> Q["返回回答、来源和页码"]
    Q --> T["SQLite 保存会话和消息"]
```

### 技术取舍

| 设计                   | 原因                                                           |
| ---------------------- | -------------------------------------------------------------- |
| 向量检索 + BM25        | 结合语义相似性与故障码、功能码等精确关键词匹配                 |
| RRF + Reranker         | 先合并两路召回结果，再精排候选证据                             |
| 结构化 `AgentState`    | 除聊天输入外，还保存故障、证据和排障进度                       |
| LLM 提取 + Python 决策 | LLM 理解自然语言；明确的完成条件、权限分支和升级规则由代码执行 |
| 本地 Qdrant            | 简化原型部署，不需要单独启动 Qdrant 服务进程                   |
| 深拷贝后提交           | 防止现有会话因 Agent 中途抛错而直接污染已保存状态              |

---

## 4. 技术栈

| 类别            | 技术                                                          |
| --------------- | ------------------------------------------------------------- |
| **语言 / 框架** | Python 3.12、FastAPI、Pydantic、Uvicorn                       |
| **LLM**         | 智谱大模型 API（模型通过 `ZHIPU_LLM_MODEL` 配置）             |
| **向量模型**    | 智谱 `embedding-3`（代码中向量维度为 1024）                   |
| **向量存储**    | Qdrant 本地嵌入模式（`QdrantClient(path=...)`）               |
| **检索 / 分词** | jieba、rank-bm25、NumPy                                       |
| **重排**        | sentence-transformers CrossEncoder：`BAAI/bge-reranker-v2-m3` |
| **数据处理**    | MinerU 生成的手册结构化数据、pandas（用于 HTML 表格处理）     |
| **测试**        | unittest.mock、FastAPI TestClient                             |

---

## 5. 项目结构

以下为主要目录（省略缓存及部分调试测试脚本）：

```
app/
├── agent/                 # 多轮工作流、角色约束、步骤提取和匹配
│   ├── agent.py           # MaintMateAgent / process()
│   ├── state.py           # AgentState、诊断上下文、排障记录
│   ├── input_analyzer.py
│   ├── safety.py
│   ├── step_extractor.py
│   ├── step_matcher.py
│   ├── troubleshooting.py
│   └── test_process_flow_mock.py
├── api/
│   ├── main.py            # POST /chat、会话管理
│   ├── test_chat_sessions.py
│   └── test_chat_errors.py
├── ingestion/
│   ├── chunker.py         # 文本切分、章节与页码保留
│   └── table_chunk.py     # 表格切分；故障表格逐行成块
├── retrieval/
│   ├── index_documents.py # 构建本地向量索引
│   ├── embedding.py
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── vector_store.py
├── evaluation/
│   ├── retrieval_eval.py
│   └── generation_eval.py
├── llm/
│   └── llm_provider.py
└── rag_pipeline.py

data/
├── processed/md380e/
│   ├── structured_content.json # 已解析的手册结构化数据
│   └── text_chunks.json        # 文本/表格 chunks
├── evaluation/                 # 评估样例与部分结果
└── qdtant/                     # 当前源码中的本地 Qdrant 路径（保留原拼写）
```

> 📊 本次检查的已处理数据中共有 **1,059 个 chunks**：其中 **467 个正文块**、**592 个表格块**。这些数字基于当前数据文件，不表示对其他手册的通用结果。

---

## 6. 本地运行

以下命令默认在项目根目录执行（即同时包含 `app/` 和 `data/` 的目录）。

项目依赖记录在 requirements.txt。上传接口通过单独的 MinerU 环境调用解析器，配置方式见下文。

### 6.1 创建环境、安装依赖（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

jieba 默认使用用户缓存目录，可用 XDG_CACHE_HOME 指定其他位置。
>
> ⚠️ 重排器初始化使用 `local_files_only=True`，因此需要提前将 `BAAI/bge-reranker-v2-m3` 模型下载到本机 Hugging Face 模型缓存；否则即使安装了 Python 依赖也可能无法启动。首次下载需要网络连接及相应的模型访问条件。

### 6.2 配置模型 API

在项目根目录创建 `.env`：

```dotenv
ZHIPU_API_KEY=替换为你自己的密钥
ZHIPU_LLM_MODEL=替换为你实际使用的智谱模型名称
```

源码中使用 `python-dotenv` 读取上述环境变量。**不要将真实 `.env`、API Key 上传到公开仓库。**

### 6.3 准备检索数据

使用自带的 MD380E 样例检索时，需要以下数据。只使用新上传手册时，样例文件可以省略。

- `data/processed/md380e/text_chunks.json`（供 BM25 使用）；
- 本地 Qdrant 索引：代码路径是 `data/qdtant/`，集合名是 `maitemate_chunks`（保留源码拼写）；
- 已缓存的 CrossEncoder 重排模型。

如果已有 `data/processed/md380e/structured_content.json`，可以从项目根目录执行：

```bash
python -m app.ingestion.chunker
```

这会生成或更新 `data/processed/md380e/text_chunks.json`。之后可在未同时运行 API 服务的情况下构建向量索引：

```bash
python -m app.retrieval.index_documents
```

> ⚠️ 索引脚本会调用在线 Embedding API，并产生调用成本。为避免本地 Qdrant 数据库被多个进程同时打开，索引前请先停止 API 服务。

MD380E 样例手册及其解析产物随本地工作区使用。新手册可通过上传接口解析与入库。

### 6.4 启动 API

```bash
python -m uvicorn app.api.main:app --reload
```

打开 http://127.0.0.1:8000/docs ，通过 Swagger UI 测试 `POST /chat`。

`--reload` 用于本地开发。服务重启后，SQLite 中已提交的会话可以继续使用。

### 6.5 示例请求

**首次请求：**

```json
{
  "session_id": null,
  "message": "MD380E 出现 Err16，应该怎么处理？"
}
```

成功后会收到服务器生成的会话 ID 和回答：

```json
{
  "session_id": "服务器生成的 UUID",
  "answer": "根据手册证据生成的回答（内容可能随模型结果变化）",
  "retrieval_attempts": 1,
  "sources": []
}
```

**第二轮**提交服务器返回的相同 `session_id`：

```json
{
  "session_id": "上一轮返回的 UUID",
  "message": "我是维修人员"
}
```

> 💡 新对话使用 `session_id: null`；已传入但不存在的会话 ID 返回 `404`。实际用户端应自动保存和携带 ID，不要求用户手工管理。

---

## 7. 测试与评估

### 快速 Mock 回归

在项目根目录执行：

```bash
python -m app.agent.test_process_flow_mock
python -m app.api.test_chat_sessions
python -m app.api.test_chat_errors
```

已在开发环境中运行通过的覆盖场景包括：故障解决记录、新故障状态重置、升级前状态更新、不同会话隔离、同一会话状态延续、HTTP 200/404/500，以及新旧会话失败后的状态保护。

> ⚠️ `test_chat_errors.py` 会故意模拟异常；终端输出包含 `RuntimeError` 堆栈是预期日志，是否通过应以最终测试输出和退出状态判断。

### RAG 评估入口

```bash
python -m app.evaluation.retrieval_eval
python -m app.evaluation.generation_eval
```

源码包含 **6 条**检索评估样例、**5 条**生成评估样例。检索评估计算 Hit@1/3/5 和 MRR@5；生成评估检查关键点覆盖与引用页码是否能由检索来源支持。

### 检索效果评估

针对 MD380E 变频器用户手册知识库，构建检索测试集，对 **BM25 + Dense Retrieval + RRF 融合 + BGE Reranker** 检索流程进行初步验证。

| 评估指标   | 测试结果 |
| ---------- | -------: |
| 测试问题数 |        6 |
| Hit@1      |   83.33% |
| Hit@3      |  100.00% |
| Hit@5      |  100.00% |
| MRR@5      |   0.9167 |

**评估方法：**

- 使用故障代码、自然语言故障描述和设备参数查询作为测试问题。
- 每条测试问题配置 `acceptable_evidence` 关键词组，用于自动判断检索结果是否包含预期证据。
- 混合检索获取 Top 10 候选文档，经过 BGE Reranker 重排序后，评估最终 Top 5 检索结果。
- 评估代码位于 `app/evaluation/retrieval_eval.py`，测试集位于 `data/evaluation/retrieval_cases.json`。

**局限性：** 当前为 6 条样本的小规模初步评估，相关性采用关键词规则自动判定，尚未覆盖复杂、多跳或模糊故障问题，也未开展不同检索方法之间的消融对照实验。上述结果不代表真实工业故障诊断的整体准确率。

> ⚠️ 这些小规模启发式检查不能替代人工评估或生产级安全验证。

### 真实集成测试

`app/agent/` 下还有真实 LLM/RAG 的测试脚本。真实模型的步骤提取、语义匹配和 `SafetyGate` 判断可能波动；一次 Mock 回归通过不能证明完整真实排障链路在每次运行时都产生相同结果。

---

## 8. 当前限制与安全说明

| 限制                             | 说明                                                                             |
| -------------------------------- | -------------------------------------------------------------------------------- |
| 🧯 **辅助工具，不是设备控制系统** | 不会自动执行参数修改、电气操作或厂家联系。                                       |
| 🧠 **安全判定仍依赖 LLM**         | 相同证据可能出现不同角色判断；证据不足时需保守处理，不能据此授权实际维修。       |
| 🔐 **身份未认证**                 | 角色来源于对话中的用户声明，不能作为生产环境的权限凭证。                         |
| 🎯 **模型提取存在误差**           | `StepExtractor` / `StepMatcher` 可能遗漏、误解或错误关联检查步骤。               |
| 💾 **单实例会话存储** | SQLite 持久化会话，服务按会话串行更新。文档 ID 用于限定检索范围。 |
| 📋 **设备覆盖有限**               | 当前数据仅验证 MD380E 手册，不代表支持其他设备型号。                             |
| 🖥️ **部署依赖本机缓存**           | 本地 Qdrant、jieba 缓存和 CrossEncoder 模型缓存需要按实际环境准备。              |

> ⚠️ 实际检修工作必须遵守设备手册、现场安全规程及授权流程；模型回答不应替代有资质人员的判断。



> 📌 **项目状态：** 核心 Agent 与 FastAPI 的主要 Mock 回归已通过；当前面向本地演示和技术学习，尚非生产部署版本。

## 上传手册与 Agentic RAG

用 Python 3.12 安装 `requirements.txt`，配置 `.env` 中的智谱密钥与模型名。
Windows 本地运行可以复用已有 MinerU 4.0.12 环境。将 `MINERU_PYTHON` 设为该环境中
可启动的 `Scripts/python.exe`，并准备 MinerU 所需模型。重排模型默认读取本地缓存，
设置 `RERANKER_ALLOW_DOWNLOAD=1` 可下载权重。

```powershell
python -m uvicorn app.api.main:app
curl.exe -F "file=@manual.pdf" http://localhost:8000/documents/upload
```

- `GET /documents/{document_id}` 查询状态，成功时返回 `ready`。
- `POST /chat` 接受 `{"message":"Err16 是什么？","document_id":"<上传返回的 UUID>"}`。
  继续对话时传 `session_id`。会话绑定手册，换手册需新建会话。省略文档 ID 使用 MD380E 样例范围。
- `POST /documents/{document_id}/retry` 重试失败任务。
- `DELETE /documents/{document_id}` 清除文件与向量。已删除的手册无法继续问答。

上传限 25 MiB。服务以单实例运行，串行解析，最多接受 4 个处理或排队中的任务，重启时把中断任务标为失败。SQLite
保存文档状态、会话快照与消息，文档目录保存结构化 JSON 和 BM25 的输入，Qdrant 保存向量。
Dense 和 BM25 使用同一手册范围与 Chunk ID。混合检索、RRF 和重排结束后，Agent
判断证据。依据不足时补充查询关键词并重检一次，合并候选后重新排序。第二次判断仍不足时
返回证据不足。响应返回候选来源、页码和检索次数，原文片段按现有权限判断提供。

```powershell
docker compose up --build
```

Compose 的 API 端口为 8001，配置独立 Qdrant、模型缓存卷和文档数据卷。
初次启动需要可下载的模型与有效的智谱配置。若要在 Compose 使用旧手册，先备好
`data/processed/md380e/text_chunks.json`，启动后执行
`docker compose exec api python -m app.retrieval.index_documents` 建立容器的样例向量。
单机部署使用单个 API worker。

```powershell
python -m pip install pytest
python -m pytest -q app/api/test_documents.py app/agent/test_evidence_retry.py
python -m app.evaluation.retrieval_eval
```

上传回归使用模拟的 MinerU 输出及 Embedding 和真实本地 Qdrant，覆盖双文档隔离、失败与删除。
检索评测分别报告重排前后 MRR@5、按现有关键词标注计算的 Recall@5。

## 故障排查报告

使用现有聊天返回的 session_id 请求报告。

```powershell
curl.exe "http://localhost:8001/conversations/你的会话ID/report" -o troubleshooting-report.md
```

接口即时读取已保存的 AgentState，返回 UTF-8 Markdown（text/markdown）。
报告在内存生成，通过响应下载到上述命令所在目录，服务器与数据库不额外保存报告。
生成过程不调用 LLM，不推进对话。报告包含设备和故障信息、已执行步骤、
反馈结果、检索原文和页码、待确认步骤及后续建议。只有反馈历史算作已执行记录。
缺失信息标记为“未提供”，无检索证据时不添加引用。
没有明确解决反馈时标记尚未解决或尚未确认解决。
会话不存在返回 404，状态数据格式异常返回 422。

本地开发可以启动 python -m uvicorn app.api.main:app 后在 8000 端口访问。
容器部署修改代码后执行 docker compose up -d --build api，在 8001/docs
调用 GET /conversations/{session_id}/report。

```powershell
python -m pytest -q app/agent/test_report.py app/api/test_chat_sessions.py app/api/test_chat_errors.py
```

这些测试使用本地构造的 AgentState 和会话读取替身，不依赖在线模型。
