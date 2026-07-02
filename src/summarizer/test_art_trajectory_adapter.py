#!/usr/bin/env python3
"""Minimal no-GPU verification for ARTTrajectoryAdapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from art_trajectory_adapter import ARTTrajectoryAdapter


def main() -> None:
    payload = {
        "task_name": "multi_round_summary",
        "document_id": "doc-001",
        "metrics": {"submitted": True, "summary_rounds_used": 2},
        "reward": None,
        "steps": [
            {
                "type": "model_call",
                "step_name": "initial_keyword_extraction",
                "round_id": 0,
                "messages": [{"role": "user", "content": "document"}],
                "output": '["specific topic"]',
            },
            {
                "type": "model_call",
                "step_name": "initial_summary",
                "round_id": 1,
                "messages": [
                    {"role": "system", "content": "summarize"},
                    {"role": "user", "content": "paragraphs"},
                ],
                "output": "A supported summary.",
            },
            {
                "type": "model_call",
                "step_name": "revision_summary",
                "round_id": 2,
                "messages": [{"role": "user", "content": "new paragraphs"}],
                "output": "A revised supported summary.",
            },
        ],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "workflow.trajectory.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        trajectories = ARTTrajectoryAdapter(source).build(reward=0.8)

    assert len(trajectories) == 2
    assert all(trajectory.reward == 0.8 for trajectory in trajectories)
    assert [item.metadata["step_name"] for item in trajectories] == [
        "initial_summary",
        "revision_summary",
    ]
    assert trajectories[0].for_logging()["messages"][-1]["trainable"] is True
    assert trajectories[0].messages()[-1] == {
        "role": "assistant",
        "content": "A supported summary.",
    }
    print("ART trajectory adapter test passed")


if __name__ == "__main__":
    main()
