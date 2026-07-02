#!/usr/bin/env python3
"""Adapt saved multi-round summaries into ART trainable trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import art
from openai.types.chat.chat_completion import Choice


DEFAULT_TRAINABLE_STEPS = ("initial_summary", "revision_summary")


class ARTTrajectoryAdapter:
    """Convert saved model calls into independent ART trajectories.

    Each saved model call is an independent prompt/response interaction. It becomes
    one ART trajectory so ART only trains on that call's assistant response.
    """

    def __init__(self, trajectory_file: str | Path) -> None:
        self.trajectory_file = Path(trajectory_file)
        with self.trajectory_file.open("r", encoding="utf-8") as f:
            self.payload: dict[str, Any] = json.load(f)

    def build(
        self,
        *,
        reward: float | None = None,
        trainable_steps: Iterable[str] = DEFAULT_TRAINABLE_STEPS,
    ) -> list[art.Trajectory]:
        """Build ART trajectories for the selected saved model-call steps.

        ``reward`` must be supplied unless the saved trajectory already contains a
        numeric reward. Question/answer data is deliberately not loaded here; it is
        only used by the external reward evaluator.
        """
        resolved_reward = self._resolve_reward(reward)
        selected_steps = set(trainable_steps)
        trajectories: list[art.Trajectory] = []

        for step in self.payload.get("steps", []):
            if step.get("type") != "model_call":
                continue
            if step.get("step_name") not in selected_steps:
                continue

            messages = self._validated_messages(step.get("messages", []))
            output = str(step.get("output", "")).strip()
            if not messages:
                raise ValueError(f"model call has no messages: {step.get('step_name')}")
            if not output:
                raise ValueError(f"model call has no output: {step.get('step_name')}")

            choice = Choice.model_validate(
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": output},
                }
            )
            round_id = int(step.get("round_id", 0))
            trajectory = art.Trajectory(
                messages_and_choices=[*messages, choice],
                reward=resolved_reward,
                metrics=self._metrics_for_step(step, round_id),
                metadata={
                    "document_id": str(self.payload.get("document_id", "")),
                    "task_name": str(self.payload.get("task_name", "")),
                    "step_name": str(step.get("step_name", "")),
                    "round_id": round_id,
                    "source_trajectory": str(self.trajectory_file),
                },
            )
            trajectory.log(
                f"Adapted {step.get('step_name')} from {self.trajectory_file.name}"
            )
            trajectories.append(trajectory.finish())

        if not trajectories:
            selected = ", ".join(sorted(selected_steps))
            raise ValueError(f"no trainable model calls found for: {selected}")
        return trajectories

    def build_group(
        self,
        *,
        reward: float | None = None,
        trainable_steps: Iterable[str] = DEFAULT_TRAINABLE_STEPS,
    ) -> art.TrajectoryGroup:
        """Build one ART group from the selected calls of this saved workflow."""
        return art.TrajectoryGroup(
            self.build(reward=reward, trainable_steps=trainable_steps)
        )

    def _resolve_reward(self, override: float | None) -> float:
        if override is not None:
            return float(override)
        saved_reward = self.payload.get("reward")
        if isinstance(saved_reward, int | float):
            return float(saved_reward)
        raise ValueError(
            "a numeric reward is required; evaluate the final summary first and "
            "pass it with reward=..."
        )

    @staticmethod
    def _validated_messages(raw_messages: Any) -> list[dict[str, str]]:
        if not isinstance(raw_messages, list):
            raise ValueError("saved messages must be a list")

        messages: list[dict[str, str]] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise ValueError("saved message must be an object")
            role = raw_message.get("role")
            content = raw_message.get("content")
            if role not in {"system", "developer", "user"}:
                raise ValueError(f"unsupported saved message role: {role!r}")
            if not isinstance(content, str):
                raise ValueError("saved message content must be text")
            messages.append({"role": role, "content": content})
        return messages

    def _metrics_for_step(self, step: dict[str, Any], round_id: int) -> dict[str, int | float | bool]:
        saved_metrics = self.payload.get("metrics", {})
        metrics: dict[str, int | float | bool] = {
            "round_id": round_id,
            "output_chars": len(str(step.get("output", ""))),
        }
        if isinstance(saved_metrics, dict):
            for name, value in saved_metrics.items():
                if isinstance(value, int | float | bool):
                    metrics[f"workflow_{name}"] = value
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and preview ART trajectories from a saved workflow."
    )
    parser.add_argument("trajectory_file", type=Path)
    parser.add_argument("--reward", type=float, required=True)
    parser.add_argument(
        "--step",
        action="append",
        dest="steps",
        help="Trainable model-call step; repeat to include several steps. "
        "Default: initial_summary and revision_summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = ARTTrajectoryAdapter(args.trajectory_file)
    trajectories = adapter.build(
        reward=args.reward,
        trainable_steps=args.steps or DEFAULT_TRAINABLE_STEPS,
    )
    preview = [trajectory.for_logging() for trajectory in trajectories]
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
