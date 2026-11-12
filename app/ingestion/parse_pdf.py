"""Run inside the existing MinerU environment."""
import json
import sys
from pathlib import Path


def main():
    from mineru.parser import parse

    result = parse(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(
        json.dumps(result.structured_content(), ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
