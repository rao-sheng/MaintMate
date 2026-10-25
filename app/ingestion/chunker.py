import json
from pathlib import Path
from table_chunk import build_table_chunks
json_path = Path("data/processed/md380e/structured_content.json")
output_path = Path("data/processed/md380e/text_chunks.json")

MAX_CHARS = 1000
OVERLAP_CHARS = 150

def load_document(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_chunk(buffer, section):
    return {
        "type": "text",
        "section": section,
        "content": "\n".join(item["text"] for item in buffer),
        "page_start": buffer[0]["page"],
        "page_end": buffer[-1]["page"],
        "source": "MD380E系列专用变频器用户手册",
        "doc_id": "md380e"
    }

def is_noise_text(content):
    noise_texts = {
        "Date / /",
        "专用变频器用户手册",
    }

    return content in noise_texts

def get_overlap(buffer):
    overlap = []
    length = 0

    # 从上一块末尾往前拿完整段落
    for item in reversed(buffer):
        overlap.insert(0, item)
        length += len(item["text"])

        if length >= OVERLAP_CHARS:
            break

    return overlap


def build_chunks(data):
    current_section = None
    buffer = []
    chunks = []

    for page in data["pages"]:
        page_num = page["page_idx"] + 1

        for block in page["blocks"]:
            block_type = block["type"]

            # 1.遇到新章节：先结束上一章节
            if block_type == "paragraph_title":
                if buffer:
                    chunks.append(
                        make_chunk(buffer, current_section)
                    )
                    buffer = []

                current_section = block["content"]

            # 2.普通正文
            elif block_type == "text":
                content = block["content"].strip()

                if not content:
                    continue
                if is_noise_text(content):
                    continue

                current_length = sum(
                    len(item["text"]) for item in buffer
                )

                # 加入新段落后会太长
                if buffer and current_length + len(content) > MAX_CHARS:
                    chunks.append(
                        make_chunk(buffer, current_section)
                    )

                    # 保留上一块结尾作为 overlap
                    buffer = get_overlap(buffer)

                buffer.append({
                    "text": content,
                    "page": page_num
                })
            #3.遇到表格
            elif block_type == "table":

                # 先结束表格前面的正文
                if buffer:
                    chunks.append(
                        make_chunk(buffer, current_section)
                    )
                    buffer = []

                # 再处理表格
                table_chunks = build_table_chunks(
                    block=block,
                    page_num=page_num,
                    current_section=current_section
                )

                chunks.extend(table_chunks)
    # 最后一块别漏掉
    if buffer:
        chunks.append(
            make_chunk(buffer, current_section)
        )

    return chunks


def save_chunks(chunks, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


data = load_document(json_path)
chunks = build_chunks(data)
save_chunks(chunks, output_path)

text_count = 0
table_count = 0

for chunk in chunks:
    if chunk["type"] == "text":
        text_count += 1
    elif chunk["type"] == "table":
        table_count += 1

print("总 chunks:", len(chunks))
print("text chunks:", text_count)
print("table chunks:", table_count)