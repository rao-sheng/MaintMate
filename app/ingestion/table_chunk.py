from io import StringIO 
import pandas as pd

TABLE_MAX_CHARS = 800

def is_fault_table(headers):
    header_text = " ".join(headers)

    keywords = [
        "故障名称",
        "故障原因",
        "故障处理",
        "操作面板"
    ]

    return any(
        keyword in header_text
        for keyword in keywords
    )
# 1. Markdown 表格解析
# 这部分逻辑比较简单，你以后应该能自己写

def parse_markdown_table(content):
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return [], []

    def split_row(line):
        return [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

    # 第一行：表头
    headers = split_row(lines[0])

    # 第二行通常是：
    # | --- | --- |
    # 所以从第三行开始读取数据
    rows = []

    for line in lines[2:]:
        if not line.startswith("|"):
            continue

        row = split_row(line)

        if row:
            rows.append(row)

    return headers, rows

# 2. HTML 表格解析
# 这里主要让 pandas 帮我做

def parse_html_table(content):
    #StringIO:把一串字符串包装成一个“像文件一样可以读取的东西”
    tables = pd.read_html(StringIO(content))

    if not tables:
        return [], []

    df = tables[0]

    # NaN → 空字符串
    df = df.fillna("")

    headers = []

    for column in df.columns:

        # HTML 有可能产生多级表头
        if isinstance(column, tuple):
            parts = []

            for item in column:
                text = str(item).strip()

                if not text.startswith("Unnamed"):
                    parts.append(text)

            header = " / ".join(parts)

        else:
            header = str(column).strip()

        headers.append(header)

    rows = []

    for _, row in df.iterrows():
        values = []

        for value in row.tolist():
            values.append(str(value).strip())

        rows.append(values)

    return headers, rows

# 3. 统一入口
# 这个设计你必须理解

def normalize_table(content):
    content = content.strip()

    if not content:
        return [], []

    # HTML 表格
    if "<table" in content.lower():
        try:
            return parse_html_table(content)
        except Exception:
            return [], []

    # Markdown 表格
    if content.startswith("|"):
        return parse_markdown_table(content)

    return [], []

# 4. 把一行数据转成适合 RAG 的自然语言

def row_to_text(headers, row):
    parts = []

    for header, value in zip(headers, row):

        header = str(header).strip()
        value = str(value).strip()

        if not value:
            continue

        parts.append(
            f"{header}：{value}"
        )

    return "\n".join(parts)

# 5. 真正切 table chunk
# 这是项目核心逻辑，你应该重点理解

def build_table_chunks(
    block,
    page_num,
    current_section,
    source="MD380E系列专用变频器用户手册",
    doc_id="md380e",
):
    content = block.get("content", "").strip()
    if not content:
        return []
    # 表题
    caption = None
    captions = block.get("captions")
    if captions:
        caption = captions[0].get("content")

    # 统一解析
    headers, rows = normalize_table(content)

    # 无法解析时：
    # 不丢数据，整张表原样保存
    
    if not headers or not rows:
        return [
            {
                "type": "table",
                "section": current_section,
                "caption": caption,
                "content": content,
                "page_start": page_num,
                "page_end": page_num,
                "source": source,
                "doc_id": doc_id,
            }
        ]
    # 单列说明类表格
    """例如：
    | 注意事项 |
    | --- |
    | 很长的一段说明 |"""
    # 这种本质接近普通文本
    
    if len(headers) == 1:
        text_parts = []

        if caption:
            text_parts.append(
                f"表：{caption}"
            )

        text_parts.append(
            f"{headers[0]}"
        )

        for row in rows:
            if row and row[0]:
                text_parts.append(
                    str(row[0]).strip()
                )

        return [
            {
                "type": "table",
                "section": current_section,
                "caption": caption,
                "content": "\n".join(text_parts),
                "page_start": page_num,
                "page_end": page_num,
                "source": source,
                "doc_id": doc_id,
            }
        ]

     # 普通多列表
    # 多行一起组成 chunk
    # 达到约 800 字再切

    if is_fault_table(headers):
        chunks = []

        for row in rows:
            row_text = row_to_text(headers, row)

            if not row_text:
                continue

            chunks.append({
                "type": "table",
                "section": current_section,
                "caption": caption,
                "content": build_table_chunk_text(
                    caption,
                    [row_text]
                ),
                "page_start": page_num,
                "page_end": page_num,
                "source": source,
                "doc_id": doc_id,
            })

        return chunks
    chunks = []
    buffer = []

    current_length = 0

    for row in rows:

        row_text = row_to_text(
            headers,
            row
        )

        if not row_text:
            continue

        # 如果加入这一行后超长
        if (
            buffer
            and current_length + len(row_text) > TABLE_MAX_CHARS
        ):
            chunk_content = build_table_chunk_text(
                caption,
                buffer
            )

            chunks.append(
                {
                    "type": "table",
                    "section": current_section,
                    "caption": caption,
                    "content": chunk_content,
                    "page_start": page_num,
                    "page_end": page_num,
                    "source": source,
                    "doc_id": doc_id,
                }
            )

            buffer = []
            current_length = 0

        buffer.append(row_text)
        current_length += len(row_text)

    # 最后一批数据
    if buffer:
        chunk_content = build_table_chunk_text(
            caption,
            buffer
        )

        chunks.append(
            {
                "type": "table",
                "section": current_section,
                "caption": caption,
                "content": chunk_content,
                "page_start": page_num,
                "page_end": page_num,
                "source": source,
                "doc_id": doc_id,
            }
        )

    return chunks

def build_table_chunk_text(caption, rows):
    parts = []

    if caption:
        parts.append(
            f"表：{caption}"
        )

    parts.extend(rows)

    return "\n\n".join(parts)