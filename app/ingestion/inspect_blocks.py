#为了知道真实JSON里是什么类型的
import json
from pathlib import Path
json_path=Path("data/processed/md380e/structured_content.json")
with json_path.open("r",encoding="utf-8") as f:
    data=json.load(f)
counts={}
for page in data["pages"]:
    for block in page["blocks"]:
        block_type=block["type"]
        counts[block_type]=counts.get(block_type,0)+1
print(counts)
print("\n--- aside_text samples ---")

count = 0

for page in data["pages"]:
    for block in page["blocks"]:
        if block["type"] == "aside_text":
            print("page:", page["page_idx"])
            print("content:", block.get("content"))
            print()

            count += 1

            if count >= 5:
                break

    if count >= 5:
        break