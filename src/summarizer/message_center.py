#!/usr/bin/env python3
"""Message center for multi-round summary generation workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MessageCenter:
    """Maintain document, retrieval, summary, and submit-decision state."""

    document_id: str
    document_text: str
    questions: list[dict[str, str]]
    document_topic: str | None = None
    source_json: str | None = None
    source_pdf: str | None = None
    current_round: int = 0
    recall_sets: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_document_json(cls, json_file: str | Path) -> "MessageCenter":
        path = Path(json_file)
        with path.open("r", encoding="utf-8") as f:
            document = json.load(f)

        questions = [
            {
                "question_id": str(question.get("question_id", "")),
                "question": str(question.get("question", "")),
                "answer": str(question.get("answer", "")),
                "long_answer": str(question.get("long_answer", "")),
            }
            for question in document.get("questions", [])
        ]

        center = cls(
            document_id=str(document.get("document_id", "")),
            document_topic=document.get("document_topic"),
            source_json=str(path),
            source_pdf=document.get("document_path"),
            document_text=str(document.get("document_extracted", "")),
            questions=questions,
        )
        center.add_event(
            "document_loaded",
            {
                "current_round": center.current_round,
                "source_json": str(path),
                "question_count": len(questions),
                "document_text_chars": len(center.document_text),
            },
        )
        return center

    def add_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "created_at": utc_now(),
                "current_round": self.current_round,
                "payload": payload or {},
            }
        )

    def set_round(self, round_id: int) -> None:
        if round_id < 0:
            raise ValueError("round_id must be greater than or equal to 0")
        self.current_round = round_id
        self.add_event("round_set", {"round_id": round_id})

    def next_round(self) -> int:
        self.current_round += 1
        self.add_event("round_advanced", {"round_id": self.current_round})
        return self.current_round

    def add_recall_set(
        self,
        *,
        round_id: int,
        keywords: list[str],
        paragraphs: list[dict[str, Any]],
        source: str,
    ) -> None:
        record = {
            "round_id": round_id,
            "source": source,
            "keywords": keywords,
            "paragraph_count": len(paragraphs),
            "paragraphs": paragraphs,
            "created_at": utc_now(),
        }
        self.recall_sets.append(record)
        self.add_event(
            "recall_added",
            {
                "round_id": round_id,
                "source": source,
                "keywords": keywords,
                "paragraph_count": len(paragraphs),
            },
        )

    def add_summary(
        self,
        *,
        round_id: int,
        summary: str,
        source: str,
    ) -> None:
        record = {
            "round_id": round_id,
            "source": source,
            "summary": summary,
            "summary_chars": len(summary),
            "created_at": utc_now(),
        }
        self.summaries.append(record)
        self.add_event(
            "summary_added",
            {
                "round_id": round_id,
                "source": source,
                "summary_chars": len(summary),
            },
        )

    def add_decision(
        self,
        *,
        round_id: int,
        should_submit: bool,
        reason: str,
        additional_keywords: list[str] | None = None,
        raw_output: str | None = None,
    ) -> None:
        record = {
            "round_id": round_id,
            "should_submit": should_submit,
            "reason": reason,
            "additional_keywords": additional_keywords or [],
            "raw_output": raw_output,
            "created_at": utc_now(),
        }
        self.decisions.append(record)
        self.add_event(
            "decision_added",
            {
                "round_id": round_id,
                "should_submit": should_submit,
                "additional_keyword_count": len(additional_keywords or []),
            },
        )

    def latest_summary(self) -> str | None:
        if not self.summaries:
            return None
        return self.summaries[-1]["summary"]

    def latest_decision(self) -> dict[str, Any] | None:
        if not self.decisions:
            return None
        return self.decisions[-1]

    def all_recalled_paragraphs(self) -> list[dict[str, Any]]:
        """Return deduplicated recalled paragraphs across all recall sets."""
        by_id: dict[int, dict[str, Any]] = {}
        for recall_set in self.recall_sets:
            for paragraph in recall_set["paragraphs"]:
                paragraph_id = int(paragraph["paragraph_id"])
                if paragraph_id not in by_id:
                    by_id[paragraph_id] = dict(paragraph)
                    by_id[paragraph_id]["sources"] = []
                by_id[paragraph_id]["sources"].append(
                    {
                        "round_id": recall_set["round_id"],
                        "source": recall_set["source"],
                        "keywords": recall_set["keywords"],
                    }
                )
        return [by_id[key] for key in sorted(by_id)]

    def model_visible_state(self) -> dict[str, Any]:
        """Return the full model-visible state.

        Prefer the task-specific visible-state methods below when building prompts.
        """
        return {
            "current_round": self.current_round,
            "document_id": self.document_id,
            "document_topic": self.document_topic,
            "document_text": self.document_text,
            "recalled_paragraphs": self.all_recalled_paragraphs(),
            "summaries": self.summaries,
            "latest_summary": self.latest_summary(),
            "latest_decision": self.latest_decision(),
        }

    def model_visible_json(self) -> str:
        """Return the full model-visible state serialized as JSON text."""
        return json.dumps(self.model_visible_state(), ensure_ascii=False, indent=2)

    def round_message(self) -> dict[str, int]:
        return {"current_round": self.current_round}

    def round_json(self) -> str:
        return json.dumps(self.round_message(), ensure_ascii=False, indent=2)

    def document_id_message(self) -> dict[str, str]:
        return {"document_id": self.document_id}

    def document_id_json(self) -> str:
        return json.dumps(self.document_id_message(), ensure_ascii=False, indent=2)

    def document_topic_message(self) -> dict[str, str | None]:
        return {"document_topic": self.document_topic}

    def document_topic_json(self) -> str:
        return json.dumps(self.document_topic_message(), ensure_ascii=False, indent=2)

    def document_text_message(self) -> dict[str, str]:
        return {"document_text": self.document_text}

    def document_text_json(self) -> str:
        return json.dumps(self.document_text_message(), ensure_ascii=False, indent=2)

    def questions_message(self) -> dict[str, Any]:
        return {"questions": self.questions}

    def questions_json(self) -> str:
        return json.dumps(self.questions_message(), ensure_ascii=False, indent=2)

    def recall_sets_message(self) -> dict[str, Any]:
        return {"recall_sets": self.recall_sets}

    def recall_sets_json(self) -> str:
        return json.dumps(self.recall_sets_message(), ensure_ascii=False, indent=2)

    def recalled_paragraphs_message(self, latest_only: bool = False) -> dict[str, Any]:
        if latest_only and self.recall_sets:
            paragraphs = self.recall_sets[-1]["paragraphs"]
        else:
            paragraphs = self.all_recalled_paragraphs()
        return {"recalled_paragraphs": paragraphs}

    def recalled_paragraphs_json(self, latest_only: bool = False) -> str:
        return json.dumps(
            self.recalled_paragraphs_message(latest_only=latest_only),
            ensure_ascii=False,
            indent=2,
        )

    def summaries_message(self) -> dict[str, Any]:
        return {"summaries": self.summaries}

    def summaries_json(self) -> str:
        return json.dumps(self.summaries_message(), ensure_ascii=False, indent=2)

    def latest_summary_message(self) -> dict[str, str | None]:
        return {"latest_summary": self.latest_summary()}

    def latest_summary_json(self) -> str:
        return json.dumps(self.latest_summary_message(), ensure_ascii=False, indent=2)

    def decisions_message(self) -> dict[str, Any]:
        return {"decisions": self.decisions}

    def decisions_json(self) -> str:
        return json.dumps(self.decisions_message(), ensure_ascii=False, indent=2)

    def latest_decision_message(self) -> dict[str, Any]:
        return {"latest_decision": self.latest_decision()}

    def latest_decision_json(self) -> str:
        return json.dumps(self.latest_decision_message(), ensure_ascii=False, indent=2)

    def system_state(self) -> dict[str, Any]:
        """Return full state, including audit events and source paths."""
        return {
            "document": {
                "current_round": self.current_round,
                "document_id": self.document_id,
                "document_topic": self.document_topic,
                "source_json": self.source_json,
                "source_pdf": self.source_pdf,
                "document_text_chars": len(self.document_text),
                "question_count": len(self.questions),
            },
            "questions": self.questions,
            "recall_sets": self.recall_sets,
            "summaries": self.summaries,
            "decisions": self.decisions,
            "events": self.events,
        }

    def save(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.system_state(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a message center state file.")
    parser.add_argument("json_file", type=Path, help="Path to one exported document JSON file.")
    parser.add_argument("--output", type=Path, help="Optional output state JSON path.")
    parser.add_argument(
        "--model-visible",
        action="store_true",
        help="Print only the model-visible state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    center = MessageCenter.from_document_json(args.json_file)
    payload = center.model_visible_state() if args.model_visible else center.system_state()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
