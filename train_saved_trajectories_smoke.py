#!/usr/bin/env python3
"""Smoke-test ART local training from saved Summary-RL trajectories."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import art
from art.local import LocalBackend

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "summarizer"))
from art_trajectory_adapter import ARTTrajectoryAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start one ART local training call from saved trajectories."
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=Path("/tmp/summary_validation_benchmark_91_vllm_r5_batch4"),
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--base-model",
        default="/root/yaojiaxin/RL/models/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--art-path", type=Path, default=Path("/tmp/summary_art_smoke"))
    parser.add_argument("--model-name", default="summary-rl-smoke")
    parser.add_argument("--project", default="summary-rl-local")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length for the ART/Unsloth local backend.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM GPU memory fraction used by Unsloth fast inference.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--server-port",
        type=int,
        default=18000,
        help="Port for the ART local OpenAI-compatible vLLM server.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the ART smoke directory before running.",
    )
    return parser.parse_args()


def build_training_group(root: Path, limit: int) -> tuple[art.TrajectoryGroup, list[dict[str, object]]]:
    files = sorted(root.glob("*/*.trajectory.json"))[:limit]
    if len(files) < limit:
        raise ValueError(f"found only {len(files)} trajectory files under {root}")

    trajectories: list[art.Trajectory] = []
    manifest: list[dict[str, object]] = []
    for path in files:
        adapter = ARTTrajectoryAdapter(path)
        built = adapter.build(trainable_steps=("initial_summary",))
        if not built:
            raise ValueError(f"no initial_summary trajectory built from {path}")
        trajectory = built[0]
        trajectories.append(trajectory)
        manifest.append(
            {
                "trajectory_file": str(path),
                "document_id": trajectory.metadata.get("document_id"),
                "reward": trajectory.reward,
                "step_name": trajectory.metadata.get("step_name"),
            }
        )

    rewards = {trajectory.reward for trajectory in trajectories}
    if len(rewards) < 2:
        raise ValueError("training smoke needs reward variance inside the group")
    return art.TrajectoryGroup(trajectories), manifest


async def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/tmp/repliqa_hf_home")

    if args.clean and args.art_path.exists():
        shutil.rmtree(args.art_path)
    args.art_path.mkdir(parents=True, exist_ok=True)

    group, manifest = build_training_group(args.trajectory_root, args.limit)
    manifest_path = args.art_path / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "trajectories_loaded",
                "trajectory_count": len(group),
                "reward_min": min(item["reward"] for item in manifest),
                "reward_max": max(item["reward"] for item in manifest),
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    backend = LocalBackend(path=str(args.art_path))
    internal_config = {
        "init_args": {
            "max_seq_length": args.max_seq_length,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "load_in_4bit": True,
        },
        "engine_args": {
            "enable_sleep_mode": False,
        },
        "trainer_args": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
    }
    model = art.TrainableModel(
        name=args.model_name,
        project=args.project,
        base_model=args.base_model,
        _internal_config=internal_config,
    )
    try:
        print(json.dumps({"event": "register_start", "server_port": args.server_port}, ensure_ascii=False), flush=True)
        await model.register(
            backend,
            _openai_client_config={
                "server_args": {
                    "host": "0.0.0.0",
                    "port": args.server_port,
                }
            },
        )
        print(
            json.dumps(
                {
                    "event": "register_complete",
                    "inference_base_url": model.inference_base_url,
                    "inference_model_name": model.inference_model_name,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        print(json.dumps({"event": "train_start"}, ensure_ascii=False), flush=True)
        await model.train(
            [group],
            config=art.TrainConfig(learning_rate=args.learning_rate),
            _config={
                "allow_training_without_logprobs": True,
                "plot_tensors": False,
                "scale_rewards": True,
            },
            verbose=True,
        )
        print(json.dumps({"event": "train_complete"}, ensure_ascii=False), flush=True)
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
