import json
import random
from pathlib import Path
chunk_path = Path(
    "data/processed/md380e/text_chunks.json"
)

with chunk_path.open("r", encoding="utf-8") as f:
    chunks = json.load(f)

# 1. 基本统计


text_chunks = [
    chunk for chunk in chunks
    if chunk["type"] == "text"
]

table_chunks = [
    chunk for chunk in chunks
    if chunk["type"] == "table"
]


print("总 chunk 数量:", len(chunks))
print("text chunk 数量:", len(text_chunks))
print("table chunk 数量:", len(table_chunks))
# 2. 长度统计

lengths = [
    len(chunk["content"])
    for chunk in chunks
]

print()
print("最短 chunk:", min(lengths))
print("最长 chunk:", max(lengths))
print("平均长度:", sum(lengths) // len(lengths))


# 3. 找异常 chunk

short_chunks = [
    chunk for chunk in chunks
    if len(chunk["content"]) < 50
]

long_chunks = [
    chunk for chunk in chunks
    if len(chunk["content"]) > 1500
]


print()
print("小于50字:", len(short_chunks))
print("大于1500字:", len(long_chunks))


# 4. 随机抽样

print("\n===== TEXT SAMPLE =====")

for chunk in random.sample(
    text_chunks,
    min(3, len(text_chunks))
):
    print("\n------------------")
    print("section:", chunk.get("section"))
    print(
        "page:",
        chunk.get("page_start"),
        "-",
        chunk.get("page_end")
    )
    print("length:", len(chunk["content"]))
    print(chunk["content"][:500])


print("\n===== TABLE SAMPLE =====")

#for chunk in random.sample(
    # table_chunks,
    # min(3, len(table_chunks))
#):
    # print("\n------------------")
    # print("section:", chunk.get("section"))
    # print("caption:", chunk.get("caption"))
    # print("page:", chunk.get("page_start"))
    # print("length:", len(chunk["content"]))
    # print(chunk["content"][:700])

short_text_chunks = [
    chunk for chunk in text_chunks
    if len(chunk["content"]) < 50
]

short_table_chunks = [
    chunk for chunk in table_chunks
    if len(chunk["content"]) < 50
]

print()
print("短 text chunk:", len(short_text_chunks))
print("短 table chunk:", len(short_table_chunks))

sorted_chunks = sorted(
    chunks,
    key=lambda chunk: len(chunk["content"])
)

print("\n===== SHORTEST 10 =====")

for chunk in sorted_chunks[:10]:
    print("\n------------------")
    print("type:", chunk["type"])
    print("section:", chunk.get("section"))
    print("caption:", chunk.get("caption"))
    print("length:", len(chunk["content"]))
    print("content:", chunk["content"])