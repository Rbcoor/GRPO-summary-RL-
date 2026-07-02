#!/usr/bin/env python3
"""End-to-end keyword extraction and paragraph recall pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from .dataset_sampler import DatasetSampler
    from .keyword_reader import KeywordReader
    from .paragraph_retriever import ParagraphRetriever
    from .paragraph_store import ParagraphStore
    from .prompt_manager import PromptManager
except ImportError:
    from dataset_sampler import DatasetSampler
    from keyword_reader import KeywordReader
    from paragraph_retriever import ParagraphRetriever
    from paragraph_store import ParagraphStore
    from prompt_manager import PromptManager


class KeywordRecallPipeline:
    """Run document sampling, paragraph storage, keyword extraction, and recall."""

    def __init__(
        self,
        dataset_root: str | Path,
        model_path: str | Path,
        output_dir: str | Path,
        *,
        splits: list[str] | None = None,
        seed: int | None = None,
        fuzzy_threshold: float = 0.88,
        max_new_tokens: int = 128,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.splits = splits
        self.seed = seed
        self.fuzzy_threshold = fuzzy_threshold
        self.max_new_tokens = max_new_tokens
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        sampled_document = self._sample_document()
        json_file = Path(sampled_document["document_path"])

        paragraph_store_path = self._save_paragraph_store(json_file)
        keywords, raw_keyword_output = self._extract_keywords(json_file)
        recall_result = self._recall_paragraphs(json_file, keywords)

        result = {
            "sampled_document": sampled_document,
            "paragraph_store_path": str(paragraph_store_path),
            "keyword_extraction": {
                "keywords": keywords,
                "raw_model_output": raw_keyword_output,
            },
            "recall": recall_result,
        }

        final_path = self.output_dir / f"{sampled_document['document_id']}.keyword_recall_result.json"
        final_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["result_path"] = str(final_path)
        return result

    def _sample_document(self) -> dict[str, Any]:
        sampler = DatasetSampler(
            dataset_root=self.dataset_root,
            splits=self.splits,
            seed=self.seed,
        )
        sampled = sampler.sample(1)[0]
        return {
            "document_path": sampled["document_path"],
            "document_id": sampled["document_id"],
            "document_topic": sampled["document_topic"],
            "source_pdf": sampled["source_pdf"],
            "paragraph_count": len(sampled["paragraphs"]),
            "sampler_seed": sampler.seed,
            "sampler_seed_was_auto_generated": sampler.seed_was_auto_generated,
        }

    def _save_paragraph_store(self, json_file: Path) -> Path:
        store = ParagraphStore(json_file)
        document_id = store.document.get("document_id", json_file.stem)
        output_path = self.output_dir / f"{document_id}.paragraph_store.json"
        return store.save_json(output_path)

    def _extract_keywords(self, json_file: Path) -> tuple[list[str], str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        document_text = KeywordReader(json_file).read("document_extracted")
        messages = PromptManager().keyword_extraction_messages(document_text)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        raw_output = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return self._parse_keywords(raw_output), raw_output

    def _parse_keywords(self, raw_output: str) -> list[str]:
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", raw_output)
            if not match:
                raise ValueError(f"model output is not a JSON array: {raw_output}")
            parsed = json.loads(match.group(0))

        if not isinstance(parsed, list):
            raise ValueError(f"model output must be a JSON array: {raw_output}")

        keywords = [str(item).strip() for item in parsed if str(item).strip()]
        if not keywords:
            raise ValueError(f"model returned no keywords: {raw_output}")
        return keywords

    def _recall_paragraphs(self, json_file: Path, keywords: list[str]) -> dict[str, Any]:
        retriever = ParagraphRetriever(
            json_file,
            fuzzy_threshold=self.fuzzy_threshold,
        )
        return retriever.recall(keywords)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one document, store paragraphs, extract keywords, and recall paragraphs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/tmp/repliqa_documents_by_file"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/yaojiaxin/RL/models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/keyword_recall_pipeline"),
    )
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--seed", type=int, help="Random seed. Omit for different random samples each run.")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HOME", "/root/yaojiaxin/RL/hf_home")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    pipeline = KeywordRecallPipeline(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        splits=args.splits,
        seed=args.seed,
        fuzzy_threshold=args.fuzzy_threshold,
        max_new_tokens=args.max_new_tokens,
    )
    result = pipeline.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
