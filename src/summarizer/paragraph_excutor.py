#!/usr/bin/env python3
"""Split a Repliqa document into numbered paragraphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .keyword_reader import KeywordReader
except ImportError:
    from keyword_reader import KeywordReader


class ParagraphExcutor:
    """Read document_extracted and split it into numbered paragraphs."""

    def __init__(self, json_file: str | Path) -> None:
        self.json_file = Path(json_file)
        self.reader = KeywordReader(self.json_file)

    def execute(self) -> list[dict[str, Any]]:
        """Return paragraphs in source order with 1-based paragraph ids."""
        document_text = self.reader.read("document_extracted")
        paragraphs = [
            paragraph.strip()
            for paragraph in document_text.split("\n\n")
            if paragraph.strip()
        ]
        return [
            {
                "paragraph_id": index,
                "paragraph": paragraph,
            }
            for index, paragraph in enumerate(paragraphs, start=1)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read document_extracted and split it into numbered paragraphs."
    )
    parser.add_argument("json_file", type=Path, help="Path to one <document_id>.json file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excutor = ParagraphExcutor(args.json_file)
    print(json.dumps(excutor.execute(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
