#!/usr/bin/env python3
"""Small smoke test for keyword_reader.py against one exported Repliqa JSON file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from keyword_reader import KeywordReader


DEFAULT_JSON = Path("/tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    reader = KeywordReader(json_path)
    doc = reader.document

    document_text = reader.read("document_extracted")
    assert_true(isinstance(document_text, str), "document_extracted should be text")
    assert_true(len(document_text) > 100, "document_extracted is unexpectedly short")

    document_id = reader.read("document_id")
    assert_true(document_id == doc["document_id"], "document_id mismatch")

    first_question = reader.read("questions.0.question")
    assert_true(isinstance(first_question, str), "questions.0.question should be text")
    assert_true(first_question, "questions.0.question should not be empty")

    first_answer = reader.read("questions.0.answer")
    assert_true(isinstance(first_answer, str), "questions.0.answer should be text")
    assert_true(first_answer, "questions.0.answer should not be empty")

    try:
        reader.read("missing_field")
    except KeyError:
        pass
    else:
        raise AssertionError("missing_field should raise KeyError")

    result = {
        "status": "ok",
        "json_path": str(json_path),
        "document_id": document_id,
        "document_text_chars": len(document_text),
        "first_question": first_question,
        "first_answer": first_answer,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
