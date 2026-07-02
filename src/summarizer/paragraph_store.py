#!/usr/bin/env python3
"""Build a structured paragraph store with document questions and answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .paragraph_excutor import ParagraphExcutor
except ImportError:
    from paragraph_excutor import ParagraphExcutor


class ParagraphStore:
    """Store structured paragraph and QA records for one document JSON file."""

    def __init__(self, json_file: str | Path) -> None:
        self.json_file = Path(json_file)
        self.excutor = ParagraphExcutor(self.json_file)
        self.document = self.excutor.reader.document

    def questions(self) -> list[dict[str, str]]:
        """Return the document questions with their answers."""
        results: list[dict[str, str]] = []
        for question in self.document.get("questions", []):
            results.append(
                {
                    "question_id": str(question.get("question_id", "")),
                    "question": str(question.get("question", "")),
                    "answer": str(question.get("answer", "")),
                    "long_answer": str(question.get("long_answer", "")),
                }
            )
        return results

    def document_record(self) -> dict[str, Any]:
        """Return one clear document-level JSON structure."""
        paragraphs = self.excutor.execute()
        questions = self.questions()
        return {
            "document": {
                "document_id": self.document.get("document_id"),
                "document_topic": self.document.get("document_topic"),
                "source_json": str(self.json_file),
                "source_pdf": self.document.get("document_path"),
            },
            "paragraph_count": len(paragraphs),
            "question_count": len(questions),
            "paragraphs": paragraphs,
            "questions": questions,
        }

    def save_json(self, output_path: str | Path) -> Path:
        """Save one clear document-level JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.document_record(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paragraph records with document questions and answers."
    )
    parser.add_argument("json_file", type=Path, help="Path to one <document_id>.json file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path. Defaults to printing JSON to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ParagraphStore(args.json_file)

    if args.output:
        store.save_json(args.output)
        return

    print(json.dumps(store.document_record(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
