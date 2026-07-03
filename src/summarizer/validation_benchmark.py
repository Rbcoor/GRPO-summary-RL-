#!/usr/bin/env python3
"""Run the fixed Repliqa validation split through the multi-round workflow."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import queue
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from message_center import MessageCenter
from paragraph_excutor import ParagraphExcutor
from paragraph_retriever import ParagraphRetriever
from prompt_manager import PromptManager
from summary_judge import SummaryJudge, compute_training_reward, update_trajectory_reward
from trajectory_store import TrajectoryStore


def vllm_judge_worker(
    benchmark: "ValidationBenchmark",
    input_queue: Any,
    output_queue: Any,
) -> None:
    """Run judge-side vLLM in a process with its own CUDA visibility."""
    if benchmark.judge_cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = benchmark.judge_cuda_visible_devices
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/tmp/repliqa_hf_home")

    try:
        judge_model = VLLMChatModel(
            benchmark.judge_model_path,
            gpu_memory_utilization=benchmark.judge_gpu_memory_utilization,
            max_model_len=benchmark.judge_max_model_len,
        )
        output_queue.put(
            {
                "event": "judge_worker_ready",
                "judge_runner": "vllm",
                "judge_cuda_visible_devices": benchmark.judge_cuda_visible_devices,
                "judge_model_path": str(benchmark.judge_model_path),
                "judge_gpu_memory_utilization": benchmark.judge_gpu_memory_utilization,
                "judge_max_model_len": benchmark.judge_max_model_len,
            }
        )
        while True:
            states = input_queue.get()
            if states is None:
                break
            output_queue.put({"event": "judge_batch_result", "items": benchmark._judge_states(states, judge_model)})
    except BaseException as exc:
        output_queue.put(
            {
                "event": "judge_worker_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise


class DeviceChatModel:
    """Minimal local chat model pinned to one CUDA device."""

    def __init__(self, model_path: str | Path, device: str) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device} if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model.to(device)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        enable_thinking: bool | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)

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

    def chat_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        max_new_tokens: int,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        return [
            self.chat(
                messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
            for messages in messages_batch
        ]


class VLLMChatModel:
    """vLLM-backed chat model for faster summary-side inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int | None = None,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.model_path = Path(model_path)
        self.SamplingParams = SamplingParams
        kwargs: dict[str, Any] = {
            "model": str(self.model_path),
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        enable_thinking: bool | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)

        sampling_params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def chat_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        max_new_tokens: int,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        prompts: list[str] = []
        for messages in messages_batch:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            try:
                prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
            prompts.append(prompt)

        sampling_params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]


@dataclass
class DocumentRunState:
    """Per-document state used by batched summary generation."""

    validation_index: int
    document_path: Path
    started_at: float
    center: MessageCenter
    retriever: ParagraphRetriever
    paragraph_count: int
    doc_output_dir: Path
    trajectory: TrajectoryStore
    pending_keywords: list[str] = field(default_factory=list)
    final_summary: str | None = None
    submitted: bool = False
    submit_round: int | None = None
    rounds: list[dict[str, Any]] = field(default_factory=list)
    summary_done: bool = False
    error: str | None = None


class ValidationBenchmark:
    """Evaluate the current multi-round summarizer on a fixed validation split."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        summary_model_path: str | Path,
        judge_model_path: str | Path,
        output_dir: str | Path,
        validation_size: int = 91,
        split: str = "repliqa_0",
        split_seed: int = 80,
        max_summary_rounds: int = 5,
        summary_device: str = "cuda:0",
        judge_device: str = "cuda:1",
        summary_runner: str = "vllm",
        summary_gpu_memory_utilization: float = 0.80,
        summary_max_model_len: int | None = None,
        judge_runner: str = "vllm",
        judge_gpu_memory_utilization: float = 0.70,
        judge_max_model_len: int | None = 4096,
        judge_max_new_tokens: int = 1024,
        judge_cuda_visible_devices: str | None = None,
        judge_batch_size: int | None = None,
        fuzzy_threshold: float = 0.88,
        limit: int | None = None,
        start_index: int = 0,
        summary_batch_size: int = 4,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.summary_model_path = Path(summary_model_path)
        self.judge_model_path = Path(judge_model_path)
        self.output_dir = Path(output_dir)
        self.validation_size = validation_size
        self.split = split
        self.split_seed = split_seed
        self.max_summary_rounds = max_summary_rounds
        self.summary_device = summary_device
        self.judge_device = judge_device
        self.summary_runner = summary_runner
        self.summary_gpu_memory_utilization = summary_gpu_memory_utilization
        self.summary_max_model_len = summary_max_model_len
        self.judge_runner = judge_runner
        self.judge_gpu_memory_utilization = judge_gpu_memory_utilization
        self.judge_max_model_len = judge_max_model_len
        self.judge_max_new_tokens = judge_max_new_tokens
        self.judge_cuda_visible_devices = judge_cuda_visible_devices
        self.judge_batch_size = max(1, judge_batch_size or summary_batch_size)
        self.fuzzy_threshold = fuzzy_threshold
        self.limit = limit
        self.start_index = start_index
        self.summary_batch_size = max(1, summary_batch_size)
        self.prompt_manager = PromptManager()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        files = self.validation_files()
        if self.start_index:
            files = files[self.start_index :]
        if self.limit is not None:
            files = files[: self.limit]

        if self.summary_runner == "vllm":
            summary_model = VLLMChatModel(
                self.summary_model_path,
                gpu_memory_utilization=self.summary_gpu_memory_utilization,
                max_model_len=self.summary_max_model_len,
            )
        elif self.summary_runner == "transformers":
            summary_model = DeviceChatModel(self.summary_model_path, self.summary_device)
        else:
            raise ValueError(f"unsupported summary_runner: {self.summary_runner}")
        judge_model = None
        if self.judge_runner == "transformers":
            judge_model = DeviceChatModel(self.judge_model_path, self.judge_device)
        elif self.judge_runner != "vllm":
            raise ValueError(f"unsupported judge_runner: {self.judge_runner}")

        started_at = time.time()
        if self.summary_batch_size > 1 or self.judge_runner == "vllm":
            items = self.run_batched_documents(
                files=files,
                summary_model=summary_model,
                judge_model=judge_model,
            )
        else:
            items = self.run_sequential_documents(
                files=files,
                summary_model=summary_model,
                judge_model=judge_model,
            )

        report = {
            "config": self.config(len(files)),
            "aggregate": self.aggregate(items, time.time() - started_at),
            "items": items,
        }
        report_path = self.output_dir / "validation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"report_path": str(report_path), **report}, ensure_ascii=False, indent=2))
        return report

    def run_sequential_documents(
        self,
        *,
        files: list[Path],
        summary_model: Any,
        judge_model: DeviceChatModel,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        progress_path = self.output_dir / "validation_items.jsonl"
        for offset, document_path in enumerate(files, start=self.start_index):
            item_started_at = time.time()
            try:
                item = self.run_document(
                    document_path=document_path,
                    summary_model=summary_model,
                    judge_model=judge_model,
                    validation_index=offset,
                )
            except Exception as exc:  # Keep long validation runs inspectable.
                item = {
                    "validation_index": offset,
                    "document_path": str(document_path),
                    "error": self._stage_error("document", exc),
                    "total_seconds": round(time.time() - item_started_at, 3),
                }
            self._append_progress_item(progress_path, items, item)
        return items

    def run_batched_documents(
        self,
        *,
        files: list[Path],
        summary_model: Any,
        judge_model: Any,
    ) -> list[dict[str, Any]]:
        if self.judge_runner == "vllm":
            return self._run_batched_documents_with_vllm_judge(
                files=files,
                summary_model=summary_model,
            )

        items: list[dict[str, Any]] = []
        progress_path = self.output_dir / "validation_items.jsonl"
        pending_judge_batches: list[concurrent.futures.Future[list[dict[str, Any]]]] = []

        def collect_finished_judge_batches(*, wait_for_all: bool = False) -> None:
            nonlocal pending_judge_batches
            if wait_for_all:
                concurrent.futures.wait(pending_judge_batches)
            remaining: list[concurrent.futures.Future[list[dict[str, Any]]]] = []
            for future in pending_judge_batches:
                if not future.done():
                    remaining.append(future)
                    continue
                for item in future.result():
                    self._append_progress_item(progress_path, items, item)
            pending_judge_batches = remaining

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as judge_executor:
            for start in range(0, len(files), self.summary_batch_size):
                batch_files = files[start : start + self.summary_batch_size]
                states: list[DocumentRunState] = []
                for local_offset, document_path in enumerate(batch_files, start=start):
                    validation_index = self.start_index + local_offset
                    try:
                        states.append(self._init_document_state(document_path, validation_index))
                    except Exception as exc:
                        item = {
                            "validation_index": validation_index,
                            "document_path": str(document_path),
                            "error": self._stage_error("document_init", exc),
                            "total_seconds": 0.0,
                        }
                        self._append_progress_item(progress_path, items, item)

                self._run_summary_batch(states, summary_model)
                pending_judge_batches.append(
                    judge_executor.submit(self._judge_states, states, judge_model)
                )
                collect_finished_judge_batches()

            collect_finished_judge_batches(wait_for_all=True)
        return items

    def _run_batched_documents_with_vllm_judge(
        self,
        *,
        files: list[Path],
        summary_model: Any,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        progress_path = self.output_dir / "validation_items.jsonl"
        ctx = mp.get_context("spawn")
        input_queue: Any = ctx.Queue(maxsize=2)
        output_queue: Any = ctx.Queue()
        judge_buffer: list[DocumentRunState] = []
        worker = ctx.Process(
            target=vllm_judge_worker,
            args=(self, input_queue, output_queue),
            daemon=False,
        )
        worker.start()
        pending_batches = 0
        worker_ready = False

        def handle_message(message: dict[str, Any]) -> None:
            nonlocal pending_batches, worker_ready
            event = message.get("event")
            if event == "judge_worker_ready":
                worker_ready = True
                print(json.dumps(message, ensure_ascii=False), flush=True)
                return
            if event == "judge_batch_result":
                pending_batches -= 1
                for item in message["items"]:
                    self._append_progress_item(progress_path, items, item)
                return
            if event == "judge_worker_error":
                raise RuntimeError(message.get("error", "judge worker failed"))
            raise RuntimeError(f"unknown judge worker message: {message}")

        def drain_available_messages() -> None:
            while True:
                try:
                    message = output_queue.get_nowait()
                except queue.Empty:
                    break
                handle_message(message)

        def submit_judge_batch(batch_states: list[DocumentRunState]) -> None:
            nonlocal pending_batches
            input_queue.put(batch_states)
            pending_batches += 1

        try:
            for start in range(0, len(files), self.summary_batch_size):
                drain_available_messages()
                batch_files = files[start : start + self.summary_batch_size]
                states: list[DocumentRunState] = []
                for local_offset, document_path in enumerate(batch_files, start=start):
                    validation_index = self.start_index + local_offset
                    try:
                        states.append(self._init_document_state(document_path, validation_index))
                    except Exception as exc:
                        item = {
                            "validation_index": validation_index,
                            "document_path": str(document_path),
                            "error": self._stage_error("document_init", exc),
                            "total_seconds": 0.0,
                        }
                        self._append_progress_item(progress_path, items, item)

                self._run_summary_batch(states, summary_model)
                judge_buffer.extend(states)
                while len(judge_buffer) >= self.judge_batch_size:
                    submit_judge_batch(judge_buffer[: self.judge_batch_size])
                    del judge_buffer[: self.judge_batch_size]
                drain_available_messages()

            if judge_buffer:
                submit_judge_batch(list(judge_buffer))
                judge_buffer.clear()
            input_queue.put(None)
            while pending_batches > 0 or worker.is_alive():
                try:
                    message = output_queue.get(timeout=1.0)
                except queue.Empty:
                    if not worker.is_alive() and pending_batches > 0:
                        raise RuntimeError("judge worker exited before returning all batches")
                    continue
                handle_message(message)
                if pending_batches == 0 and not worker.is_alive():
                    break
            worker.join(timeout=5.0)
            if worker.exitcode not in (0, None):
                raise RuntimeError(f"judge worker exited with code {worker.exitcode}")
            if not worker_ready:
                raise RuntimeError("judge worker did not report ready")
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)
        return items

    def _judge_states(
        self,
        states: list[DocumentRunState],
        judge_model: Any,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        prepared: list[dict[str, Any]] = []

        for state in states:
            if state.error is not None:
                items.append(
                    {
                        "validation_index": state.validation_index,
                        "document_path": str(state.document_path),
                        "document_id": state.center.document_id,
                        "error": state.error,
                        "total_seconds": round(time.time() - state.started_at, 3),
                    }
                )
                continue
            try:
                saved = self._save_workflow_outputs(state)
                answer_messages = SummaryJudge.answer_messages(
                    document_id=state.center.document_id,
                    summary=state.final_summary or "",
                    questions=state.center.questions,
                )
                prepared.append({"state": state, "saved": saved, "answer_messages": answer_messages})
            except Exception as exc:
                items.append(
                    {
                        "validation_index": state.validation_index,
                        "document_path": str(state.document_path),
                        "document_id": state.center.document_id,
                        "error": self._stage_error("judge_prepare", exc),
                        "total_seconds": round(time.time() - state.started_at, 3),
                    }
                )

        if not prepared:
            return items

        answer_raw_outputs = judge_model.chat_batch(
            [item["answer_messages"] for item in prepared],
            max_new_tokens=self.judge_max_new_tokens,
            enable_thinking=False,
        )

        score_requests: list[dict[str, Any]] = []
        for item, answer_raw_output in zip(prepared, answer_raw_outputs):
            state = item["state"]
            try:
                parsed_answers = SummaryJudge._parse_json(answer_raw_output)
                generated_answers = SummaryJudge._normalize_generated_answers(parsed_answers)
            except Exception as exc:
                self._save_judge_parse_error(
                    doc_output_dir=state.doc_output_dir,
                    document_id=state.center.document_id,
                    stage="answer_questions",
                    raw_output=answer_raw_output,
                    error=exc,
                )
                items.append(
                    {
                        "validation_index": state.validation_index,
                        "document_path": str(state.document_path),
                        "document_id": state.center.document_id,
                        "error": self._stage_error("answer_questions", exc),
                        "total_seconds": round(time.time() - state.started_at, 3),
                    }
                )
                continue

            score_messages = SummaryJudge.score_messages(
                document_id=state.center.document_id,
                generated_answers=generated_answers,
                questions=state.center.questions,
            )
            score_requests.append(
                {
                    **item,
                    "answer_raw_output": answer_raw_output,
                    "generated_answers": generated_answers,
                    "score_messages": score_messages,
                }
            )

        if not score_requests:
            return items

        score_raw_outputs = judge_model.chat_batch(
            [item["score_messages"] for item in score_requests],
            max_new_tokens=self.judge_max_new_tokens,
            enable_thinking=False,
        )

        for item, score_raw_output in zip(score_requests, score_raw_outputs):
            state = item["state"]
            saved = item["saved"]
            try:
                parsed_judge = SummaryJudge._parse_json(score_raw_output)
                judge_result = SummaryJudge._normalize_result(parsed_judge)
                reward = compute_training_reward(
                    judge_result=judge_result,
                    total_questions=len(state.center.questions),
                    submitted=state.submitted,
                    used_rounds=state.center.current_round,
                    max_rounds=self.max_summary_rounds,
                    summary_chars=len(state.final_summary or ""),
                )
                judge_result["answer_raw_output"] = item["answer_raw_output"]
                judge_result["score_raw_output"] = score_raw_output
                judge_result["generated_answers"] = item["generated_answers"]
                judge_result["judge_model"] = str(self.judge_model_path)
                judge_result["judge_model_final_score"] = judge_result["final_score"]
                judge_result["final_score"] = reward["final_score"]
                judge_result["reward_components"] = reward["components"]
                judge_result["reward_weights"] = reward["weights"]
                judge_result["document_id"] = state.center.document_id
                judge_result["result_json"] = str(saved["result_path"])
                judge_result["trajectory_path"] = str(saved["trajectory_path"])
                judge_result["trajectory_reward_updated"] = True
                saved["judge_path"].write_text(
                    json.dumps(judge_result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                update_trajectory_reward(saved["trajectory_path"], judge_result)
                components = judge_result["reward_components"]
                items.append(
                    {
                        "validation_index": state.validation_index,
                        "document_id": state.center.document_id,
                        "answered_questions": components["answered_questions"],
                        "total_questions": components["total_questions"],
                        "answer_score": components["answer_score"],
                        "final_score": judge_result["final_score"],
                        "submitted": state.submitted,
                        "submit_round": state.submit_round,
                        "early_submit": saved["result"]["early_submit"],
                        "summary_rounds_used": state.center.current_round,
                        "summary_chars": len(state.final_summary or ""),
                        "total_seconds": round(time.time() - state.started_at, 3),
                        "result_path": str(saved["result_path"]),
                        "state_path": str(saved["state_path"]),
                        "trajectory_path": str(saved["trajectory_path"]),
                        "judge_result_path": str(saved["judge_path"]),
                    }
                )
            except Exception as exc:
                self._save_judge_parse_error(
                    doc_output_dir=state.doc_output_dir,
                    document_id=state.center.document_id,
                    stage="score_answers",
                    raw_output=score_raw_output,
                    error=exc,
                    generated_answers=item["generated_answers"],
                )
                items.append(
                    {
                        "validation_index": state.validation_index,
                        "document_path": str(state.document_path),
                        "document_id": state.center.document_id,
                        "error": self._stage_error("score_answers", exc),
                        "total_seconds": round(time.time() - state.started_at, 3),
                    }
                )
        return items

    @staticmethod
    def _append_progress_item(
        progress_path: Path,
        items: list[dict[str, Any]],
        item: dict[str, Any],
    ) -> None:
        items.append(item)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)

    def validation_files(self) -> list[Path]:
        split_dir = self.dataset_root / self.split
        files = sorted(split_dir.glob("*.json"))
        if not files:
            raise ValueError(f"no validation files found in {split_dir}")
        rng = random.Random(self.split_seed)
        rng.shuffle(files)
        return files[: self.validation_size]

    def _init_document_state(
        self,
        document_path: Path,
        validation_index: int,
    ) -> DocumentRunState:
        center = MessageCenter.from_document_json(document_path)
        retriever = ParagraphRetriever(
            document_path,
            fuzzy_threshold=self.fuzzy_threshold,
        )
        excutor = ParagraphExcutor(document_path)
        paragraph_count = len(excutor.execute())
        doc_output_dir = self.output_dir / f"{validation_index:03d}_{center.document_id}"
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        trajectory = TrajectoryStore(
            document_id=center.document_id,
            metadata={
                "validation_index": validation_index,
                "document_path": str(document_path),
                "document_topic": center.document_topic,
                "source_pdf": center.source_pdf,
                "paragraph_count": paragraph_count,
                "max_summary_rounds": self.max_summary_rounds,
                "summary_batch_size": self.summary_batch_size,
            },
        )
        return DocumentRunState(
            validation_index=validation_index,
            document_path=document_path,
            started_at=time.time(),
            center=center,
            retriever=retriever,
            paragraph_count=paragraph_count,
            doc_output_dir=doc_output_dir,
            trajectory=trajectory,
        )

    @staticmethod
    def _stage_error(stage: str, exc: Exception) -> str:
        return f"{stage}: {type(exc).__name__}: {exc}"

    def _chat_batch_with_fallback(
        self,
        *,
        summary_model: Any,
        requests: list[dict[str, Any]],
        max_new_tokens: int,
        stage: str,
    ) -> list[str | None]:
        """Run chat_batch, then fall back to per-document calls if the batch fails."""
        if not requests:
            return []
        messages_batch = [request["messages"] for request in requests]
        try:
            return summary_model.chat_batch(messages_batch, max_new_tokens=max_new_tokens)
        except Exception as batch_exc:
            print(
                json.dumps(
                    {
                        "event": "batch_generation_failed",
                        "stage": stage,
                        "error": repr(batch_exc),
                        "fallback": "per_document",
                        "count": len(requests),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        outputs: list[str | None] = []
        for request in requests:
            state = request["state"]
            try:
                outputs.append(
                    summary_model.chat(
                        request["messages"],
                        max_new_tokens=max_new_tokens,
                    )
                )
            except Exception as exc:
                state.error = self._stage_error(stage, exc)
                outputs.append(None)
        return outputs

    def _run_summary_batch(
        self,
        states: list[DocumentRunState],
        summary_model: Any,
    ) -> None:
        active = [state for state in states if state.error is None]
        if not active:
            return

        keyword_requests = [
            {
                "state": state,
                "messages": self.prompt_manager.keyword_extraction_messages(
                    state.center.document_text_message()["document_text"]
                ),
            }
            for state in active
        ]
        raw_keyword_outputs = self._chat_batch_with_fallback(
            summary_model=summary_model,
            requests=keyword_requests,
            max_new_tokens=256,
            stage="initial_keyword_extraction",
        )

        for request, raw_keywords in zip(keyword_requests, raw_keyword_outputs):
            state = request["state"]
            messages = request["messages"]
            if state.error is not None or raw_keywords is None:
                continue
            try:
                keywords = self._parse_keyword_list(raw_keywords)
            except Exception as exc:
                state.error = self._stage_error("initial_keyword_extraction_parse", exc)
                continue
            state.pending_keywords = keywords
            state.trajectory.add_model_call(
                step_name="initial_keyword_extraction",
                round_id=0,
                messages=messages,
                output=raw_keywords,
                parsed_output=keywords,
                model_name=str(self.summary_model_path),
            )
            state.center.add_event(
                "initial_keywords_extracted",
                {"keywords": keywords, "raw_output": raw_keywords},
            )

        for summary_round in range(1, self.max_summary_rounds + 1):
            active = [
                state
                for state in states
                if state.error is None and not state.summary_done and state.pending_keywords
            ]
            if not active:
                break

            summary_requests: list[dict[str, Any]] = []
            for state in active:
                try:
                    state.center.set_round(summary_round)
                    recall = state.retriever.recall(state.pending_keywords)
                    state.trajectory.add_system_step(
                        step_name="paragraph_recall",
                        round_id=summary_round,
                        payload={
                            "keywords": state.pending_keywords,
                            "top_k": recall["query"]["top_k"],
                            "retrieved_count": recall["retrieved_count"],
                            "paragraph_ids": [p["paragraph_id"] for p in recall["paragraphs"]],
                        },
                    )
                    state.center.add_recall_set(
                        round_id=summary_round,
                        keywords=state.pending_keywords,
                        paragraphs=recall["paragraphs"],
                        source="paragraph_retriever",
                    )
                except Exception as exc:
                    state.error = self._stage_error("paragraph_recall", exc)
                    continue
                if summary_round == 1:
                    messages = self.prompt_manager.initial_summary_messages(
                        round_json=state.center.round_json(),
                        document_id_json=state.center.document_id_json(),
                        recalled_paragraphs_json=state.center.recalled_paragraphs_json(),
                    )
                    step_name = "initial_summary"
                else:
                    messages = self.prompt_manager.revision_summary_messages(
                        round_json=state.center.round_json(),
                        document_id_json=state.center.document_id_json(),
                        latest_summary_json=state.center.latest_summary_json(),
                        recalled_paragraphs_json=state.center.recalled_paragraphs_json(latest_only=True),
                        latest_decision_json=state.center.latest_decision_json(),
                    )
                    step_name = "revision_summary"
                summary_requests.append(
                    {
                        "state": state,
                        "messages": messages,
                        "step_name": step_name,
                        "recall": recall,
                        "keywords": list(state.pending_keywords),
                    }
                )

            summaries = self._chat_batch_with_fallback(
                summary_model=summary_model,
                requests=summary_requests,
                max_new_tokens=768,
                stage="summary_generation",
            )

            decision_requests: list[dict[str, Any]] = []
            for request, summary in zip(summary_requests, summaries):
                state = request["state"]
                if state.error is not None or summary is None:
                    continue
                state.trajectory.add_model_call(
                    step_name=request["step_name"],
                    round_id=summary_round,
                    messages=request["messages"],
                    output=summary,
                    parsed_output={"summary": summary},
                    model_name=str(self.summary_model_path),
                )
                state.center.add_summary(
                    round_id=summary_round,
                    summary=summary,
                    source=request["step_name"],
                )
                state.final_summary = summary
                decision_messages = self.prompt_manager.submit_decision_messages(
                    round_json=state.center.round_json(),
                    document_id_json=state.center.document_id_json(),
                    latest_summary_json=state.center.latest_summary_json(),
                    recalled_paragraphs_json=state.center.recalled_paragraphs_json(),
                )
                decision_requests.append(
                    {
                        "state": state,
                        "messages": decision_messages,
                        "recall": request["recall"],
                        "keywords": request["keywords"],
                        "summary": summary,
                    }
                )

            raw_decisions = self._chat_batch_with_fallback(
                summary_model=summary_model,
                requests=decision_requests,
                max_new_tokens=192,
                stage="submit_decision",
            )

            for request, raw_decision in zip(decision_requests, raw_decisions):
                state = request["state"]
                if state.error is not None or raw_decision is None:
                    continue
                try:
                    decision = self._parse_decision(raw_decision)
                except Exception as exc:
                    state.error = self._stage_error("submit_decision_parse", exc)
                    continue
                state.trajectory.add_model_call(
                    step_name="submit_decision",
                    round_id=summary_round,
                    messages=request["messages"],
                    output=raw_decision,
                    parsed_output=decision,
                    model_name=str(self.summary_model_path),
                )
                state.center.add_decision(
                    round_id=summary_round,
                    should_submit=decision["should_submit"],
                    reason=decision["reason"],
                    additional_keywords=decision["additional_keywords"],
                    raw_output=raw_decision,
                )
                recall = request["recall"]
                state.rounds.append(
                    {
                        "summary_round": summary_round,
                        "keywords": request["keywords"],
                        "recall_top_k": recall["query"]["top_k"],
                        "retrieved_count": recall["retrieved_count"],
                        "paragraph_ids": [p["paragraph_id"] for p in recall["paragraphs"]],
                        "summary_chars": len(request["summary"]),
                        "decision": decision,
                    }
                )
                if decision["should_submit"]:
                    state.submitted = True
                    state.submit_round = summary_round
                    state.summary_done = True
                    continue
                state.pending_keywords = decision["additional_keywords"]
                if not state.pending_keywords:
                    state.center.add_event(
                        "stopped_no_additional_keywords",
                        {"round_id": summary_round},
                    )
                    state.summary_done = True

        for state in states:
            if state.error is None and state.final_summary is None:
                state.error = f"summary_generation: ValueError: no summary generated for {state.document_path}"

    def _save_workflow_outputs(self, state: DocumentRunState) -> dict[str, Any]:
        if state.final_summary is None:
            raise ValueError(f"no summary generated for {state.document_path}")

        center = state.center
        final_summary = state.final_summary
        state_path = state.doc_output_dir / f"{center.document_id}.multi_round_state.json"
        result_path = state.doc_output_dir / f"{center.document_id}.multi_round_result.json"
        trajectory_path = state.doc_output_dir / f"{center.document_id}.trajectory.json"
        judge_path = state.doc_output_dir / f"{center.document_id}.judge_result.json"

        state.trajectory.set_metrics(
            {
                "submitted": state.submitted,
                "summary_rounds_used": center.current_round,
                "summary_count": len(center.summaries),
                "decision_count": len(center.decisions),
                "recall_set_count": len(center.recall_sets),
                "final_summary_chars": len(final_summary),
                "summary_batch_size": self.summary_batch_size,
            }
        )
        center.save(state_path)
        state.trajectory.save(trajectory_path)

        result = {
            "sampled_document": {
                "document_path": str(state.document_path),
                "document_id": center.document_id,
                "document_topic": center.document_topic,
                "source_pdf": center.source_pdf,
                "paragraph_count": state.paragraph_count,
            },
            "validation_index": state.validation_index,
            "submitted": state.submitted,
            "submit_round": state.submit_round,
            "early_submit": bool(
                state.submitted
                and state.submit_round is not None
                and state.submit_round < self.max_summary_rounds
            ),
            "summary_rounds_used": center.current_round,
            "max_summary_rounds": self.max_summary_rounds,
            "final_summary": final_summary,
            "message_center_state": str(state_path),
            "trajectory_path": str(trajectory_path),
            "trace": {"rounds": state.rounds},
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "state_path": state_path,
            "result_path": result_path,
            "trajectory_path": trajectory_path,
            "judge_path": judge_path,
            "result": result,
        }

    def _judge_and_save_state(
        self,
        state: DocumentRunState,
        judge_model: DeviceChatModel,
    ) -> dict[str, Any]:
        if state.final_summary is None:
            raise ValueError(f"no summary generated for {state.document_path}")

        center = state.center
        final_summary = state.final_summary
        state_path = state.doc_output_dir / f"{center.document_id}.multi_round_state.json"
        result_path = state.doc_output_dir / f"{center.document_id}.multi_round_result.json"
        trajectory_path = state.doc_output_dir / f"{center.document_id}.trajectory.json"
        judge_path = state.doc_output_dir / f"{center.document_id}.judge_result.json"

        state.trajectory.set_metrics(
            {
                "submitted": state.submitted,
                "summary_rounds_used": center.current_round,
                "summary_count": len(center.summaries),
                "decision_count": len(center.decisions),
                "recall_set_count": len(center.recall_sets),
                "final_summary_chars": len(final_summary),
                "summary_batch_size": self.summary_batch_size,
            }
        )
        center.save(state_path)
        state.trajectory.save(trajectory_path)

        result = {
            "sampled_document": {
                "document_path": str(state.document_path),
                "document_id": center.document_id,
                "document_topic": center.document_topic,
                "source_pdf": center.source_pdf,
                "paragraph_count": state.paragraph_count,
            },
            "validation_index": state.validation_index,
            "submitted": state.submitted,
            "submit_round": state.submit_round,
            "early_submit": bool(
                state.submitted
                and state.submit_round is not None
                and state.submit_round < self.max_summary_rounds
            ),
            "summary_rounds_used": center.current_round,
            "max_summary_rounds": self.max_summary_rounds,
            "final_summary": final_summary,
            "message_center_state": str(state_path),
            "trajectory_path": str(trajectory_path),
            "trace": {"rounds": state.rounds},
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        answer_messages = SummaryJudge.answer_messages(
            document_id=center.document_id,
            summary=final_summary,
            questions=center.questions,
        )
        answer_raw_output = judge_model.chat(
            answer_messages,
            max_new_tokens=1024,
            enable_thinking=False,
        )
        try:
            parsed_answers = SummaryJudge._parse_json(answer_raw_output)
        except Exception as exc:
            self._save_judge_parse_error(
                doc_output_dir=state.doc_output_dir,
                document_id=center.document_id,
                stage="answer_questions",
                raw_output=answer_raw_output,
                error=exc,
            )
            raise
        generated_answers = SummaryJudge._normalize_generated_answers(parsed_answers)

        score_messages = SummaryJudge.score_messages(
            document_id=center.document_id,
            generated_answers=generated_answers,
            questions=center.questions,
        )
        score_raw_output = judge_model.chat(
            score_messages,
            max_new_tokens=1024,
            enable_thinking=False,
        )
        try:
            parsed_judge = SummaryJudge._parse_json(score_raw_output)
        except Exception as exc:
            self._save_judge_parse_error(
                doc_output_dir=state.doc_output_dir,
                document_id=center.document_id,
                stage="score_answers",
                raw_output=score_raw_output,
                error=exc,
                generated_answers=generated_answers,
            )
            raise
        judge_result = SummaryJudge._normalize_result(parsed_judge)
        reward = compute_training_reward(
            judge_result=judge_result,
            total_questions=len(center.questions),
            submitted=state.submitted,
            used_rounds=center.current_round,
            max_rounds=self.max_summary_rounds,
            summary_chars=len(final_summary),
        )
        judge_result["answer_raw_output"] = answer_raw_output
        judge_result["score_raw_output"] = score_raw_output
        judge_result["generated_answers"] = generated_answers
        judge_result["judge_model"] = str(self.judge_model_path)
        judge_result["judge_model_final_score"] = judge_result["final_score"]
        judge_result["final_score"] = reward["final_score"]
        judge_result["reward_components"] = reward["components"]
        judge_result["reward_weights"] = reward["weights"]
        judge_result["document_id"] = center.document_id
        judge_result["result_json"] = str(result_path)
        judge_result["trajectory_path"] = str(trajectory_path)
        judge_result["trajectory_reward_updated"] = True
        judge_path.write_text(
            json.dumps(judge_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        update_trajectory_reward(trajectory_path, judge_result)

        components = judge_result["reward_components"]
        return {
            "validation_index": state.validation_index,
            "document_id": center.document_id,
            "answered_questions": components["answered_questions"],
            "total_questions": components["total_questions"],
            "answer_score": components["answer_score"],
            "final_score": judge_result["final_score"],
            "submitted": state.submitted,
            "submit_round": state.submit_round,
            "early_submit": result["early_submit"],
            "summary_rounds_used": center.current_round,
            "summary_chars": len(final_summary),
            "total_seconds": round(time.time() - state.started_at, 3),
            "result_path": str(result_path),
            "state_path": str(state_path),
            "trajectory_path": str(trajectory_path),
            "judge_result_path": str(judge_path),
        }

    def run_document(
        self,
        *,
        document_path: Path,
        summary_model: Any,
        judge_model: DeviceChatModel,
        validation_index: int,
    ) -> dict[str, Any]:
        document_started_at = time.time()
        center = MessageCenter.from_document_json(document_path)
        retriever = ParagraphRetriever(
            document_path,
            fuzzy_threshold=self.fuzzy_threshold,
        )
        excutor = ParagraphExcutor(document_path)
        paragraph_count = len(excutor.execute())

        doc_output_dir = self.output_dir / f"{validation_index:03d}_{center.document_id}"
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        trajectory = TrajectoryStore(
            document_id=center.document_id,
            metadata={
                "validation_index": validation_index,
                "document_path": str(document_path),
                "document_topic": center.document_topic,
                "source_pdf": center.source_pdf,
                "paragraph_count": paragraph_count,
                "max_summary_rounds": self.max_summary_rounds,
            },
        )

        keyword_messages = self.prompt_manager.keyword_extraction_messages(
            center.document_text_message()["document_text"]
        )
        raw_keywords = summary_model.chat(keyword_messages, max_new_tokens=256)
        keywords = self._parse_keyword_list(raw_keywords)
        trajectory.add_model_call(
            step_name="initial_keyword_extraction",
            round_id=0,
            messages=keyword_messages,
            output=raw_keywords,
            parsed_output=keywords,
            model_name=str(self.summary_model_path),
        )
        center.add_event(
            "initial_keywords_extracted",
            {"keywords": keywords, "raw_output": raw_keywords},
        )

        pending_keywords = keywords
        final_summary: str | None = None
        submitted = False
        submit_round: int | None = None
        rounds: list[dict[str, Any]] = []

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
                step_name = "initial_summary"
            else:
                summary_messages = self.prompt_manager.revision_summary_messages(
                    round_json=center.round_json(),
                    document_id_json=center.document_id_json(),
                    latest_summary_json=center.latest_summary_json(),
                    recalled_paragraphs_json=center.recalled_paragraphs_json(latest_only=True),
                    latest_decision_json=center.latest_decision_json(),
                )
                step_name = "revision_summary"

            summary = summary_model.chat(summary_messages, max_new_tokens=768)
            trajectory.add_model_call(
                step_name=step_name,
                round_id=summary_round,
                messages=summary_messages,
                output=summary,
                parsed_output={"summary": summary},
                model_name=str(self.summary_model_path),
            )
            center.add_summary(round_id=summary_round, summary=summary, source=step_name)
            final_summary = summary

            decision_messages = self.prompt_manager.submit_decision_messages(
                round_json=center.round_json(),
                document_id_json=center.document_id_json(),
                latest_summary_json=center.latest_summary_json(),
                recalled_paragraphs_json=center.recalled_paragraphs_json(),
            )
            raw_decision = summary_model.chat(decision_messages, max_new_tokens=192)
            decision = self._parse_decision(raw_decision)
            trajectory.add_model_call(
                step_name="submit_decision",
                round_id=summary_round,
                messages=decision_messages,
                output=raw_decision,
                parsed_output=decision,
                model_name=str(self.summary_model_path),
            )
            center.add_decision(
                round_id=summary_round,
                should_submit=decision["should_submit"],
                reason=decision["reason"],
                additional_keywords=decision["additional_keywords"],
                raw_output=raw_decision,
            )

            rounds.append(
                {
                    "summary_round": summary_round,
                    "keywords": pending_keywords,
                    "recall_top_k": recall["query"]["top_k"],
                    "retrieved_count": recall["retrieved_count"],
                    "paragraph_ids": [p["paragraph_id"] for p in recall["paragraphs"]],
                    "summary_chars": len(summary),
                    "decision": decision,
                }
            )
            if decision["should_submit"]:
                submitted = True
                submit_round = summary_round
                break

            pending_keywords = decision["additional_keywords"]
            if not pending_keywords:
                center.add_event(
                    "stopped_no_additional_keywords",
                    {"round_id": summary_round},
                )
                break

        if final_summary is None:
            raise ValueError(f"no summary generated for {document_path}")

        state_path = doc_output_dir / f"{center.document_id}.multi_round_state.json"
        result_path = doc_output_dir / f"{center.document_id}.multi_round_result.json"
        trajectory_path = doc_output_dir / f"{center.document_id}.trajectory.json"
        judge_path = doc_output_dir / f"{center.document_id}.judge_result.json"

        trajectory.set_metrics(
            {
                "submitted": submitted,
                "summary_rounds_used": center.current_round,
                "summary_count": len(center.summaries),
                "decision_count": len(center.decisions),
                "recall_set_count": len(center.recall_sets),
                "final_summary_chars": len(final_summary),
            }
        )
        center.save(state_path)
        trajectory.save(trajectory_path)

        result = {
            "sampled_document": {
                "document_path": str(document_path),
                "document_id": center.document_id,
                "document_topic": center.document_topic,
                "source_pdf": center.source_pdf,
                "paragraph_count": paragraph_count,
            },
            "validation_index": validation_index,
            "submitted": submitted,
            "submit_round": submit_round,
            "early_submit": bool(submitted and submit_round is not None and submit_round < self.max_summary_rounds),
            "summary_rounds_used": center.current_round,
            "max_summary_rounds": self.max_summary_rounds,
            "final_summary": final_summary,
            "message_center_state": str(state_path),
            "trajectory_path": str(trajectory_path),
            "trace": {"rounds": rounds},
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        answer_messages = SummaryJudge.answer_messages(
            document_id=center.document_id,
            summary=final_summary,
            questions=center.questions,
        )
        answer_raw_output = judge_model.chat(
            answer_messages,
            max_new_tokens=1024,
            enable_thinking=False,
        )
        try:
            parsed_answers = SummaryJudge._parse_json(answer_raw_output)
        except Exception as exc:
            self._save_judge_parse_error(
                doc_output_dir=doc_output_dir,
                document_id=center.document_id,
                stage="answer_questions",
                raw_output=answer_raw_output,
                error=exc,
            )
            raise
        generated_answers = SummaryJudge._normalize_generated_answers(parsed_answers)

        score_messages = SummaryJudge.score_messages(
            document_id=center.document_id,
            generated_answers=generated_answers,
            questions=center.questions,
        )
        score_raw_output = judge_model.chat(
            score_messages,
            max_new_tokens=1024,
            enable_thinking=False,
        )
        try:
            parsed_judge = SummaryJudge._parse_json(score_raw_output)
        except Exception as exc:
            self._save_judge_parse_error(
                doc_output_dir=doc_output_dir,
                document_id=center.document_id,
                stage="score_answers",
                raw_output=score_raw_output,
                error=exc,
                generated_answers=generated_answers,
            )
            raise
        judge_result = SummaryJudge._normalize_result(parsed_judge)
        reward = compute_training_reward(
            judge_result=judge_result,
            total_questions=len(center.questions),
            submitted=submitted,
            used_rounds=center.current_round,
            max_rounds=self.max_summary_rounds,
            summary_chars=len(final_summary),
        )
        judge_result["answer_raw_output"] = answer_raw_output
        judge_result["score_raw_output"] = score_raw_output
        judge_result["generated_answers"] = generated_answers
        judge_result["judge_model"] = str(self.judge_model_path)
        judge_result["judge_model_final_score"] = judge_result["final_score"]
        judge_result["final_score"] = reward["final_score"]
        judge_result["reward_components"] = reward["components"]
        judge_result["reward_weights"] = reward["weights"]
        judge_result["document_id"] = center.document_id
        judge_result["result_json"] = str(result_path)
        judge_result["trajectory_path"] = str(trajectory_path)
        judge_result["trajectory_reward_updated"] = True
        judge_path.write_text(
            json.dumps(judge_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        update_trajectory_reward(trajectory_path, judge_result)

        components = judge_result["reward_components"]
        return {
            "validation_index": validation_index,
            "document_id": center.document_id,
            "answered_questions": components["answered_questions"],
            "total_questions": components["total_questions"],
            "answer_score": components["answer_score"],
            "final_score": judge_result["final_score"],
            "submitted": submitted,
            "submit_round": submit_round,
            "early_submit": result["early_submit"],
            "summary_rounds_used": center.current_round,
            "summary_chars": len(final_summary),
            "total_seconds": round(time.time() - document_started_at, 3),
            "result_path": str(result_path),
            "state_path": str(state_path),
            "trajectory_path": str(trajectory_path),
            "judge_result_path": str(judge_path),
        }

    def config(self, processed_count: int) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "split": self.split,
            "split_seed": self.split_seed,
            "validation_size": self.validation_size,
            "processed_count": processed_count,
            "start_index": self.start_index,
            "limit": self.limit,
            "max_summary_rounds": self.max_summary_rounds,
            "summary_batch_size": self.summary_batch_size,
            "summary_model_path": str(self.summary_model_path),
            "judge_model_path": str(self.judge_model_path),
            "summary_runner": self.summary_runner,
            "summary_gpu_memory_utilization": self.summary_gpu_memory_utilization,
            "summary_max_model_len": self.summary_max_model_len,
            "judge_runner": self.judge_runner,
            "judge_gpu_memory_utilization": self.judge_gpu_memory_utilization,
            "judge_max_model_len": self.judge_max_model_len,
            "judge_max_new_tokens": self.judge_max_new_tokens,
            "judge_cuda_visible_devices": self.judge_cuda_visible_devices,
            "judge_batch_size": self.judge_batch_size,
            "summary_device": self.summary_device,
            "judge_device": self.judge_device,
        }

    @staticmethod
    def aggregate(items: list[dict[str, Any]], total_seconds: float) -> dict[str, Any]:
        valid_items = [item for item in items if "error" not in item]
        error_count = len(items) - len(valid_items)

        def mean(name: str) -> float:
            values = [float(item[name]) for item in valid_items if item.get(name) is not None]
            return round(sum(values) / len(values), 6) if values else 0.0

        submit_rounds = [
            float(item["submit_round"])
            for item in valid_items
            if item.get("submit_round") is not None
        ]
        submitted_count = sum(1 for item in valid_items if item.get("submitted"))
        early_submit_count = sum(1 for item in valid_items if item.get("early_submit"))
        processed = len(valid_items)
        return {
            "processed_count": processed,
            "error_count": error_count,
            "total_seconds": round(total_seconds, 3),
            "mean_total_seconds_per_doc": mean("total_seconds"),
            "total_answered_questions": sum(int(item["answered_questions"]) for item in valid_items),
            "mean_answered_questions": mean("answered_questions"),
            "mean_answer_score": mean("answer_score"),
            "mean_final_score": mean("final_score"),
            "submit_count": submitted_count,
            "submit_rate": round(submitted_count / processed, 6) if processed else 0.0,
            "mean_submit_round": round(sum(submit_rounds) / len(submit_rounds), 6) if submit_rounds else 0.0,
            "early_submit_count": early_submit_count,
            "early_submit_rate": round(early_submit_count / processed, 6) if processed else 0.0,
            "mean_summary_rounds_used": mean("summary_rounds_used"),
            "mean_summary_chars": mean("summary_chars"),
        }

    def _parse_keyword_list(self, raw_output: str) -> list[str]:
        try:
            parsed = self._parse_json_from_output(raw_output)
        except json.JSONDecodeError:
            parsed = self._extract_quoted_strings(raw_output)
        if not isinstance(parsed, list):
            raise ValueError(f"keyword output must be a JSON list: {raw_output}")
        keywords = self._normalize_keywords(parsed)
        if not keywords:
            raise ValueError(f"keyword output contains no keywords: {raw_output}")
        return keywords

    def _normalize_keywords(self, raw_keywords: list[Any]) -> list[str]:
        keywords: list[str] = []
        seen: set[str] = set()
        for item in raw_keywords:
            keyword = str(item).strip()
            if not keyword:
                continue
            normalized = keyword.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            keywords.append(keyword)
            if len(keywords) >= self.prompt_manager.keyword_max_count:
                break
        return keywords

    @staticmethod
    def _extract_quoted_strings(raw_output: str) -> list[str]:
        values: list[str] = []
        for match in re.finditer(r'"((?:[^"\\]|\\.)*)"', raw_output):
            try:
                values.append(json.loads(f'"{match.group(1)}"'))
            except json.JSONDecodeError:
                continue
        return values

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
        return {
            "should_submit": should_submit,
            "reason": reason,
            "additional_keywords": additional_keywords,
        }

    @staticmethod
    def _save_judge_parse_error(
        *,
        doc_output_dir: Path,
        document_id: str,
        stage: str,
        raw_output: str,
        error: Exception,
        generated_answers: list[dict[str, str]] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "document_id": document_id,
            "stage": stage,
            "error": repr(error),
            "raw_output": raw_output,
        }
        if generated_answers is not None:
            payload["generated_answers"] = generated_answers
        error_path = doc_output_dir / f"{document_id}.judge_parse_error.json"
        error_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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
    parser = argparse.ArgumentParser(description="Run fixed validation benchmark.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/tmp/repliqa_documents_by_file"))
    parser.add_argument("--summary-model-path", type=Path, default=Path("/tmp/models/Qwen2.5-3B-Instruct"))
    parser.add_argument("--judge-model-path", type=Path, default=Path("/tmp/models/Qwen3-14B"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/summary_validation_benchmark"))
    parser.add_argument("--split", default="repliqa_0")
    parser.add_argument("--split-seed", type=int, default=80)
    parser.add_argument("--validation-size", type=int, default=91)
    parser.add_argument("--max-summary-rounds", type=int, default=5)
    parser.add_argument("--gpus", default="1,4")
    parser.add_argument("--summary-runner", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--summary-gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--summary-max-model-len", type=int)
    parser.add_argument("--judge-runner", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--judge-max-model-len", type=int, default=4096)
    parser.add_argument("--judge-max-new-tokens", type=int, default=1024)
    parser.add_argument("--judge-cuda-visible-devices")
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        help="Number of completed summaries to batch together for judge-side calls. Defaults to summary batch size.",
    )
    parser.add_argument("--summary-device", default="cuda:0")
    parser.add_argument("--judge-device", default="cuda:1")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--summary-batch-size",
        type=int,
        default=4,
        help="Number of documents to batch together for summary-side vLLM calls. Use 1 for sequential mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/tmp/repliqa_hf_home")
    judge_cuda_visible_devices = args.judge_cuda_visible_devices
    if judge_cuda_visible_devices is None:
        visible_devices = [device.strip() for device in args.gpus.split(",") if device.strip()]
        if len(visible_devices) >= 2:
            judge_cuda_visible_devices = visible_devices[1]
        elif visible_devices:
            judge_cuda_visible_devices = visible_devices[0]

    benchmark = ValidationBenchmark(
        dataset_root=args.dataset_root,
        summary_model_path=args.summary_model_path,
        judge_model_path=args.judge_model_path,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        split=args.split,
        split_seed=args.split_seed,
        max_summary_rounds=args.max_summary_rounds,
        summary_device=args.summary_device,
        judge_device=args.judge_device,
        summary_runner=args.summary_runner,
        summary_gpu_memory_utilization=args.summary_gpu_memory_utilization,
        summary_max_model_len=args.summary_max_model_len,
        judge_runner=args.judge_runner,
        judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
        judge_max_model_len=args.judge_max_model_len,
        judge_max_new_tokens=args.judge_max_new_tokens,
        judge_cuda_visible_devices=judge_cuda_visible_devices,
        judge_batch_size=args.judge_batch_size,
        fuzzy_threshold=args.fuzzy_threshold,
        limit=args.limit,
        start_index=args.start_index,
        summary_batch_size=args.summary_batch_size,
    )
    benchmark.run()


if __name__ == "__main__":
    main()
