#!/usr/bin/env python3
"""Smoke test for message-center driven keyword recall and summary generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_sampler import DatasetSampler
from message_center import MessageCenter
from paragraph_retriever import ParagraphRetriever
from prompt_manager import PromptManager


MODEL_PATH = "/root/yaojiaxin/RL/models/Qwen2.5-3B-Instruct"
DATASET_ROOT = "/tmp/repliqa_documents_by_file"
OUTPUT_DIR = Path("/tmp/message_center_flow_test")


class LocalQwen:
    def __init__(self, model_path: str) -> None:
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
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
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


def parse_keywords(raw_output: str) -> list[str]:
    parsed = json.loads(raw_output)
    if not isinstance(parsed, list):
        raise ValueError(f"keyword output is not a list: {raw_output}")
    return [str(item).strip() for item in parsed if str(item).strip()]


def build_summary_messages(center: MessageCenter) -> list[dict[str, str]]:
    context = "\n\n".join(
        [
            "CURRENT ROUND:\n" + center.round_json(),
            "DOCUMENT ID:\n" + center.document_id_json(),
            "RECALLED PARAGRAPHS:\n" + center.recalled_paragraphs_json(),
        ]
    )
    return [
        {
                "role": "system",
                "content": (
                    "You are a careful document summarization assistant. "
                    "Use only the provided recalled paragraphs and preserve concrete, "
                    "document-specific information."
                ),
            },
        {
            "role": "user",
            "content": (
                "Create a concise summary of 350 words or less based on the message "
                "center content below. Return only the summary text.\n\n"
                f"{context}"
            ),
        },
    ]


def main() -> None:
    os.environ.setdefault("HF_HOME", "/root/yaojiaxin/RL/hf_home")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sampler = DatasetSampler(DATASET_ROOT, seed=8675309)
    sampled = sampler.sample(1)[0]
    center = MessageCenter.from_document_json(sampled["document_path"])
    qwen = LocalQwen(MODEL_PATH)

    trace: dict[str, object] = {
        "sampled_document": {
            "document_path": sampled["document_path"],
            "document_id": sampled["document_id"],
            "document_topic": sampled["document_topic"],
            "paragraph_count": len(sampled["paragraphs"]),
        },
        "steps": [],
    }

    # Step 1: keyword extraction from message-center document text.
    center.set_round(1)
    document_text = center.document_text_message()["document_text"]
    keyword_messages = PromptManager().keyword_extraction_messages(document_text)
    raw_keywords = qwen.chat(keyword_messages, max_new_tokens=128)
    keywords = parse_keywords(raw_keywords)
    center.add_event(
        "keywords_extracted",
        {
            "round_id": center.current_round,
            "keywords": keywords,
            "raw_output": raw_keywords,
            "model_read_messages": ["document_text_message"],
        },
    )
    trace["steps"].append(
        {
            "step": "keyword_extraction",
            "round": center.current_round,
            "model_read_messages": ["document_text_message"],
            "keywords": keywords,
            "raw_output": raw_keywords,
        }
    )

    # Step 2: recall paragraphs and update message center.
    retriever = ParagraphRetriever(sampled["document_path"])
    recall = retriever.recall(keywords)
    center.add_recall_set(
        round_id=center.current_round,
        keywords=keywords,
        paragraphs=recall["paragraphs"],
        source="paragraph_retriever",
    )
    trace["steps"].append(
        {
            "step": "paragraph_recall",
            "round": center.current_round,
            "keywords": keywords,
            "top_k": recall["query"]["top_k"],
            "retrieved_count": recall["retrieved_count"],
            "paragraph_ids": [p["paragraph_id"] for p in recall["paragraphs"]],
        }
    )

    # Step 3: generate summary from message-center single-message interfaces.
    center.set_round(2)
    summary_messages = build_summary_messages(center)
    summary = qwen.chat(summary_messages, max_new_tokens=512)
    center.add_summary(
        round_id=center.current_round,
        summary=summary,
        source="qwen_summary_from_recalled_paragraphs",
    )
    trace["steps"].append(
        {
            "step": "summary_generation",
            "round": center.current_round,
            "model_read_messages": [
                "round_json",
                "document_id_json",
                "recalled_paragraphs_json",
            ],
            "summary": summary,
            "summary_chars": len(summary),
        }
    )

    state_path = OUTPUT_DIR / f"{sampled['document_id']}.message_center_state.json"
    trace_path = OUTPUT_DIR / f"{sampled['document_id']}.flow_trace.json"
    center.save(state_path)
    trace["message_center_checks"] = {
        "current_round": center.current_round,
        "recall_set_count": len(center.recall_sets),
        "summary_count": len(center.summaries),
        "latest_summary_matches": center.latest_summary() == summary,
        "model_visible_single_messages": {
            "round": center.round_message(),
            "document_id": center.document_id_message(),
            "questions_count": len(center.questions_message()["questions"]),
            "recalled_paragraph_count": len(
                center.recalled_paragraphs_message()["recalled_paragraphs"]
            ),
            "latest_summary_chars": len(center.latest_summary_message()["latest_summary"] or ""),
        },
    }
    trace["outputs"] = {
        "message_center_state": str(state_path),
        "flow_trace": str(trace_path),
    }
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
