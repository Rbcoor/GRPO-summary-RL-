#!/usr/bin/env python3
"""Multi-round summary generation pipeline with keyword recall."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_sampler import DatasetSampler
from message_center import MessageCenter
from paragraph_retriever import ParagraphRetriever
from prompt_manager import PromptManager
from trajectory_store import TrajectoryStore


class LocalModelRunner:
    """Small local chat wrapper for Qwen-style chat models."""

    def __init__(self, model_path: str | Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def chat(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


class MultiRoundSummaryPipeline:
    """Run up to max_summary_rounds; each generated summary counts as one round."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        model_path: str | Path,
        output_dir: str | Path,
        splits: list[str] | None = None,
        seed: int | None = None,
        max_summary_rounds: int = 5,
        fuzzy_threshold: float = 0.88,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.splits = splits
        self.seed = seed
        self.max_summary_rounds = max_summary_rounds
        self.fuzzy_threshold = fuzzy_threshold
        self.prompt_manager = PromptManager()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        sampled = self._sample_document()
        center = MessageCenter.from_document_json(sampled["document_path"])
        runner = LocalModelRunner(self.model_path)
        retriever = ParagraphRetriever(
            sampled["document_path"],
            fuzzy_threshold=self.fuzzy_threshold,
        )

        trace: dict[str, Any] = {
            "sampled_document": sampled,
            "max_summary_rounds": self.max_summary_rounds,
            "rounds": [],
        }
        trajectory = TrajectoryStore(
            document_id=sampled["document_id"],
            metadata={
                "document_path": sampled["document_path"],
                "document_topic": sampled["document_topic"],
                "source_pdf": sampled["source_pdf"],
                "paragraph_count": sampled["paragraph_count"],
                "max_summary_rounds": self.max_summary_rounds,
            },
        )

        keyword_messages = self.prompt_manager.keyword_extraction_messages(
            center.document_text_message()["document_text"]
        )
        raw_keywords = runner.chat(keyword_messages, max_new_tokens=128)
        keywords = self._parse_keyword_list(raw_keywords)
        trajectory.add_model_call(
            step_name="initial_keyword_extraction",
            round_id=0,
            messages=keyword_messages,
            output=raw_keywords,
            parsed_output=keywords,
            model_name=str(self.model_path),
        )
        center.add_event(
            "initial_keywords_extracted",
            {"keywords": keywords, "raw_output": raw_keywords},
        )

        pending_keywords = keywords
        final_summary: str | None = None
        submitted = False

        for summary_round in range(1, self.max_summary_rounds + 1):
            center.set_round(summary_round)

            recall = retriever.recall(pending_keywords)
            trajectory.add_system_step(
                step_name="paragraph_recall",
                round_id=summary_round,
                payload={
                    "keywords": pending_keywords,
                    "top_k": recall["query"]["top_k"],
                    "retrieved_count": recall["retrieved_count"],
                    "paragraph_ids": [p["paragraph_id"] for p in recall["paragraphs"]],
                },
            )
            center.add_recall_set(
                round_id=summary_round,
                keywords=pending_keywords,
                paragraphs=recall["paragraphs"],
                source="paragraph_retriever",
            )

            if summary_round == 1:
                summary_messages = self.prompt_manager.initial_summary_messages(
                    round_json=center.round_json(),
                    document_id_json=center.document_id_json(),
                    recalled_paragraphs_json=center.recalled_paragraphs_json(),
                )
            else:
                summary_messages = self.prompt_manager.revision_summary_messages(
                    round_json=center.round_json(),
                    document_id_json=center.document_id_json(),
                    latest_summary_json=center.latest_summary_json(),
                    recalled_paragraphs_json=center.recalled_paragraphs_json(latest_only=True),
                    latest_decision_json=center.latest_decision_json(),
                )

            summary = runner.chat(summary_messages, max_new_tokens=512)
            trajectory.add_model_call(
                step_name="initial_summary" if summary_round == 1 else "revision_summary",
                round_id=summary_round,
                messages=summary_messages,
                output=summary,
                parsed_output={"summary": summary},
                model_name=str(self.model_path),
            )
            center.add_summary(
                round_id=summary_round,
                summary=summary,
                source="initial_summary" if summary_round == 1 else "revision_summary",
            )
            final_summary = summary

            decision_messages = self.prompt_manager.submit_decision_messages(
                round_json=center.round_json(),
                document_id_json=center.document_id_json(),
                latest_summary_json=center.latest_summary_json(),
                recalled_paragraphs_json=center.recalled_paragraphs_json(),
            )
            raw_decision = runner.chat(decision_messages, max_new_tokens=192)
            decision = self._parse_decision(raw_decision)
            trajectory.add_model_call(
                step_name="submit_decision",
                round_id=summary_round,
                messages=decision_messages,
                output=raw_decision,
                parsed_output=decision,
                model_name=str(self.model_path),
            )
            center.add_decision(
                round_id=summary_round,
                should_submit=decision["should_submit"],
                reason=decision["reason"],
                additional_keywords=decision["additional_keywords"],
                raw_output=raw_decision,
            )

            trace["rounds"].append(
                {
                    "summary_round": summary_round,
                    "keywords": pending_keywords,
                    "recall_top_k": recall["query"]["top_k"],
                    "retrieved_count": recall["retrieved_count"],
                    "paragraph_ids": [p["paragraph_id"] for p in recall["paragraphs"]],
                    "summary": summary,
                    "summary_chars": len(summary),
                    "decision": decision,
                    "raw_decision": raw_decision,
                }
            )

            if decision["should_submit"]:
                submitted = True
                break

            pending_keywords = decision["additional_keywords"]
            if not pending_keywords:
                center.add_event(
                    "stopped_no_additional_keywords",
                    {"round_id": summary_round},
                )
                break

        state_path = self.output_dir / f"{sampled['document_id']}.multi_round_state.json"
        result_path = self.output_dir / f"{sampled['document_id']}.multi_round_result.json"
        trajectory_path = self.output_dir / f"{sampled['document_id']}.trajectory.json"
        trajectory.set_metrics(
            {
                "submitted": submitted,
                "summary_rounds_used": center.current_round,
                "summary_count": len(center.summaries),
                "decision_count": len(center.decisions),
                "recall_set_count": len(center.recall_sets),
                "final_summary_chars": len(final_summary or ""),
            }
        )
        center.save(state_path)
        trajectory.save(trajectory_path)
        result = {
            "sampled_document": sampled,
            "submitted": submitted,
            "summary_rounds_used": center.current_round,
            "max_summary_rounds": self.max_summary_rounds,
            "final_summary": final_summary,
            "message_center_state": str(state_path),
            "trajectory_path": str(trajectory_path),
            "trace": trace,
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["result_path"] = str(result_path)
        return result

    def _sample_document(self) -> dict[str, Any]:
        seed = self.seed
        if seed is None:
            sampler = DatasetSampler(self.dataset_root, splits=self.splits, shuffle=True)
        else:
            sampler = DatasetSampler(self.dataset_root, splits=self.splits, seed=seed)
        sampled = sampler.sample(1)[0]
        return {
            "document_path": sampled["document_path"],
            "document_id": sampled["document_id"],
            "document_topic": sampled["document_topic"],
            "source_pdf": sampled["source_pdf"],
            "paragraph_count": len(sampled["paragraphs"]),
        }

    def _parse_keyword_list(self, raw_output: str) -> list[str]:
        parsed = self._parse_json_from_output(raw_output)
        if not isinstance(parsed, list):
            raise ValueError(f"keyword output must be a JSON list: {raw_output}")
        keywords = [str(item).strip() for item in parsed if str(item).strip()]
        if not keywords:
            raise ValueError(f"keyword output contains no keywords: {raw_output}")
        return keywords

    def _parse_decision(self, raw_output: str) -> dict[str, Any]:
        parsed = self._parse_json_from_output(raw_output)
        if not isinstance(parsed, dict):
            raise ValueError(f"decision output must be a JSON object: {raw_output}")

        should_submit = bool(parsed.get("should_submit", False))
        reason = str(parsed.get("reason", "")).strip()
        additional_keywords = [
            str(item).strip()
            for item in parsed.get("additional_keywords", [])
            if str(item).strip()
        ]
        if not should_submit and not additional_keywords:
            raise ValueError(
                "decision says not submit but did not provide additional_keywords"
            )
        return {
            "should_submit": should_submit,
            "reason": reason,
            "additional_keywords": additional_keywords,
        }

    @staticmethod
    def _parse_json_from_output(raw_output: str) -> Any:
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", raw_output)
            if not match:
                raise
            return json.loads(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-round summary generation.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/tmp/repliqa_documents_by_file"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/yaojiaxin/RL/models/Qwen2.5-3B-Instruct"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/multi_round_summary_pipeline"))
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--max-summary-rounds", type=int, default=5)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HOME", "/root/yaojiaxin/RL/hf_home")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    pipeline = MultiRoundSummaryPipeline(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        splits=args.splits,
        seed=args.seed,
        max_summary_rounds=args.max_summary_rounds,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    result = pipeline.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
