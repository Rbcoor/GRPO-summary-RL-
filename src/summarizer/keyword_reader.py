#!/usr/bin/env python3
"""Read one field from one Repliqa document JSON file and print plain text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class KeywordReader:
    """Read field content from one exported Repliqa document JSON file."""

    REQUIRED_KEYS = {"document_id", "document_extracted", "questions"}

    def __init__(self, json_file: str | Path) -> None:
        self.json_file = Path(json_file)
        self.document = self._load_document()

    def _load_document(self) -> dict[str, Any]:
        with self.json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        missing = sorted(self.REQUIRED_KEYS - set(data))
        if missing:
            raise ValueError(
                f"{self.json_file} is missing required keys: {', '.join(missing)}"
            )

        if not isinstance(data["document_extracted"], str):
            raise ValueError("document_extracted must be a string")
        if not isinstance(data["questions"], list):
            raise ValueError("questions must be a list")

        return data

    def read(self, field_path: str) -> str:
        """Read a field path and return it as plain text.

        Examples:
          document_extracted
          document_topic
          questions.0.question
          questions.0.answer
        """
        value = self._read_field(field_path)
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, indent=2)

    def _read_field(self, field_path: str) -> Any:
        current: Any = self.document
        for part in field_path.split("."):
            if isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"field not found: {field_path}")
                current = current[part]
            elif isinstance(current, list):
                try:
                    index = int(part)
                except ValueError as exc:
                    raise KeyError(
                        f"field path {field_path!r} needs a numeric list index at {part!r}"
                    ) from exc
                try:
                    current = current[index]
                except IndexError as exc:
                    raise KeyError(
                        f"list index out of range in field path: {field_path}"
                    ) from exc
            else:
                raise KeyError(f"cannot descend into non-container value at {part!r}")
        return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read selected field(s) from one Repliqa document JSON file."
    )
    parser.add_argument("json_file", type=Path, help="Path to one <document_id>.json file.")
    parser.add_argument(
        "field",
        help="Field name/path to read, such as document_extracted or questions.0.answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = KeywordReader(args.json_file)
    print(reader.read(args.field))


if __name__ == "__main__":
    main()
