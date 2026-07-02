#!/usr/bin/env python3
"""Prompt templates for Summary-RL local workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptManager:
    """Build prompts used by local training and retrieval workflows."""

    keyword_min_count: int = 3
    keyword_max_count: int = 7
    summary_word_limit: int = 350

    def keyword_extraction_messages(self, document_text: str) -> list[dict[str, str]]:
        """Build the first-round prompt for extracting document-specific keywords."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a precise keyword extraction assistant. "
                    "Your job is to identify document-specific keywords that capture "
                    "the core entities, themes, events, methods, products, places, "
                    "or distinctive facts in the document."
                ),
            },
            {
                "role": "user",
                "content": self.keyword_extraction_prompt(document_text),
            },
        ]

    def keyword_extraction_prompt(self, document_text: str) -> str:
        """Return the user prompt for keyword extraction."""
        return f"""Read the document below and extract {self.keyword_min_count}-{self.keyword_max_count} high-quality keywords.

Requirements:
- Keywords must be specific to the document's core content.
- Avoid generic words such as "business", "technology", "development", "people", "challenge", "success", or other broad topic labels unless they are part of a distinctive named concept.
- Prefer named entities, distinctive products, locations, methods, events, domain-specific terms, and central concepts.
- Cover the document comprehensively. Choose keywords from multiple important aspects when present, such as main topic, causes or drivers, impacts or consequences, solutions or recommendations, key actors, places, and distinctive events.
- Do not only extract the most frequent terms. Include less frequent keywords when they represent an important document-specific point.
- Do not include duplicate or near-duplicate keywords.
- Each keyword should be concise, usually 1-4 words.
- Use natural readable phrases with spaces when appropriate. Do not concatenate words into camelCase or PascalCase.
- Prefer {self.keyword_max_count} keywords when the document contains several distinct important aspects; use fewer only when the document is genuinely narrow.
- Return only a JSON array of strings. Do not include explanations.

Document:
{document_text}
"""

    def initial_summary_messages(
        self,
        *,
        round_json: str,
        document_id_json: str,
        recalled_paragraphs_json: str,
    ) -> list[dict[str, str]]:
        """Build the first summary prompt from recalled paragraphs."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful document summarization assistant. "
                    "You write concise summaries grounded only in the provided recalled paragraphs. "
                    "Your summary should preserve concrete, document-specific information."
                ),
            },
            {
                "role": "user",
                "content": self.initial_summary_prompt(
                    round_json=round_json,
                    document_id_json=document_id_json,
                    recalled_paragraphs_json=recalled_paragraphs_json,
                ),
            },
        ]

    def initial_summary_prompt(
        self,
        *,
        round_json: str,
        document_id_json: str,
        recalled_paragraphs_json: str,
    ) -> str:
        """Return the first summary prompt."""
        return f"""Create the first draft summary using the recalled paragraphs.

Requirements:
- Use only the recalled paragraphs as evidence.
- Keep the summary at or below {self.summary_word_limit} words.
- Prioritize concrete facts, named entities, dates, locations, causes, impacts, actions, and outcomes.
- Do not mention that the summary is based on recalled paragraphs.
- Return only the summary text.

Current round:
{round_json}

Document:
{document_id_json}

Recalled paragraphs:
{recalled_paragraphs_json}
"""

    def submit_decision_messages(
        self,
        *,
        round_json: str,
        document_id_json: str,
        latest_summary_json: str,
        recalled_paragraphs_json: str,
    ) -> list[dict[str, str]]:
        """Build the prompt for deciding whether to submit the summary."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict summary quality reviewer. "
                    "Decide whether the current summary is sufficient, or whether more evidence is needed."
                ),
            },
            {
                "role": "user",
                "content": self.submit_decision_prompt(
                    round_json=round_json,
                    document_id_json=document_id_json,
                    latest_summary_json=latest_summary_json,
                    recalled_paragraphs_json=recalled_paragraphs_json,
                ),
            },
        ]

    def submit_decision_prompt(
        self,
        *,
        round_json: str,
        document_id_json: str,
        latest_summary_json: str,
        recalled_paragraphs_json: str,
    ) -> str:
        """Return the submit-decision prompt."""
        return f"""Review the current summary and decide whether it is ready to submit.

Decision criteria:
- The summary should be at or below {self.summary_word_limit} words.
- The summary should preserve the central facts, entities, dates, places, causes, impacts, actions, and outcomes visible in the recalled paragraphs.
- The summary should not include unsupported claims.
- If important evidence appears missing or the summary seems incomplete, do not submit.
- If you do not submit, you must provide additional specific keywords that can retrieve missing paragraphs.
- Additional keywords must be document-specific, concise, and different from already covered concepts when possible.

Return only a JSON object in this exact format:
{{
  "should_submit": true,
  "reason": "brief reason",
  "additional_keywords": []
}}

If not ready, use:
{{
  "should_submit": false,
  "reason": "brief reason",
  "additional_keywords": ["specific missing concept", "specific entity or event"]
}}

Current round:
{round_json}

Document:
{document_id_json}

Current summary:
{latest_summary_json}

Recalled paragraphs already available:
{recalled_paragraphs_json}
"""

    def revision_summary_messages(
        self,
        *,
        round_json: str,
        document_id_json: str,
        latest_summary_json: str,
        recalled_paragraphs_json: str,
        latest_decision_json: str,
    ) -> list[dict[str, str]]:
        """Build the prompt for revising a summary with newly recalled paragraphs."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful summary revision assistant. "
                    "Improve the previous summary using newly recalled evidence while keeping it concise."
                ),
            },
            {
                "role": "user",
                "content": self.revision_summary_prompt(
                    round_json=round_json,
                    document_id_json=document_id_json,
                    latest_summary_json=latest_summary_json,
                    recalled_paragraphs_json=recalled_paragraphs_json,
                    latest_decision_json=latest_decision_json,
                ),
            },
        ]

    def revision_summary_prompt(
        self,
        *,
        round_json: str,
        document_id_json: str,
        latest_summary_json: str,
        recalled_paragraphs_json: str,
        latest_decision_json: str,
    ) -> str:
        """Return the revision summary prompt."""
        return f"""Revise the previous summary using the newly recalled paragraphs.

This is not the first summary. Your task is to improve the existing summary, not restart from scratch.

Requirements:
- Keep useful correct information from the previous summary.
- Add missing important facts from the newly recalled paragraphs.
- Remove or correct unsupported, vague, or redundant statements.
- Keep the final summary at or below {self.summary_word_limit} words.
- Do not mention the revision process, recalled paragraphs, or model decisions.
- Return only the revised summary text.

Current round:
{round_json}

Document:
{document_id_json}

Previous summary:
{latest_summary_json}

Latest decision:
{latest_decision_json}

Newly recalled paragraphs:
{recalled_paragraphs_json}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt messages for a document.")
    parser.add_argument("document_file", help="Path to a plain-text document file.")
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum keyword count. Default: 3.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=7,
        help="Maximum keyword count. Default: 7.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.document_file, "r", encoding="utf-8") as f:
        document_text = f.read()
    manager = PromptManager(
        keyword_min_count=args.min_count,
        keyword_max_count=args.max_count,
    )
    print(
        json.dumps(
            manager.keyword_extraction_messages(document_text),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
