#!/usr/bin/env python3
"""Trajectory storage for local multi-round summary workflows."""

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
class TrajectoryStore:
    """Record model-visible messages, model outputs, and system steps."""

    document_id: str
    task_name: str = "multi_round_summary"
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    created_at: str = field(default_factory=utc_now)

    def add_model_call(
        self,
        *,
        step_name: str,
        round_id: int,
        messages: list[dict[str, str]],
        output: str,
        parsed_output: Any | None = None,
        model_name: str | None = None,
    ) -> None:
        """Record one model call and its output."""
        self.steps.append(
            {
                "type": "model_call",
                "step_name": step_name,
                "round_id": round_id,
                "model_name": model_name,
                "messages": messages,
                "output": output,
                "parsed_output": parsed_output,
                "created_at": utc_now(),
            }
        )

    def add_system_step(
        self,
        *,
        step_name: str,
        round_id: int,
        payload: dict[str, Any],
    ) -> None:
        """Record one non-model system step, such as retrieval."""
        self.steps.append(
            {
                "type": "system_step",
                "step_name": step_name,
                "round_id": round_id,
                "payload": payload,
                "created_at": utc_now(),
            }
        )

    def set_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "document_id": self.document_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "steps": self.steps,
            "metrics": self.metrics,
            "reward": self.reward,
        }

    def save(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an empty trajectory file.")
    parser.add_argument("document_id")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = TrajectoryStore(document_id=args.document_id)
    store.save(args.output)


if __name__ == "__main__":
    main()
