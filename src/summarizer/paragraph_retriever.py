#!/usr/bin/env python3
"""Hybrid paragraph retrieval for one Repliqa document JSON file."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from .paragraph_store import ParagraphStore
except ImportError:
    from paragraph_store import ParagraphStore


class ParagraphRetriever:
    """Recall relevant paragraphs by keywords with deduplication and hit counts."""

    def __init__(
        self,
        json_file: str | Path,
        *,
        fuzzy_threshold: float = 0.88,
    ) -> None:
        self.json_file = Path(json_file)
        self.fuzzy_threshold = fuzzy_threshold
        self.store = ParagraphStore(self.json_file)
        self.document = self.store.document_record()
        self.paragraphs = self.document["paragraphs"]
        self._paragraph_tokens = [
            self._tokenize(paragraph["paragraph"]) for paragraph in self.paragraphs
        ]
        self._avg_doc_len = self._average_doc_len()
        self._idf = self._build_idf()

    def recall(
        self,
        keywords: list[str],
        *,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Return deduplicated relevant paragraphs and their hit counts."""
        normalized_keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        if not normalized_keywords:
            raise ValueError("at least one keyword is required")
        effective_top_k = self._dynamic_top_k()

        results: list[dict[str, Any]] = []
        query_tokens = self._query_tokens(normalized_keywords)

        for paragraph, tokens in zip(self.paragraphs, self._paragraph_tokens):
            text = paragraph["paragraph"]
            exact_hits = self._exact_hits(text, normalized_keywords)
            fuzzy_hits = self._fuzzy_hits(tokens, normalized_keywords)
            bm25_score = self._bm25_score(tokens, query_tokens)

            exact_hit_count = sum(exact_hits.values())
            fuzzy_hit_count = sum(fuzzy_hits.values())
            matched_keywords = sorted(
                keyword
                for keyword in normalized_keywords
                if exact_hits.get(keyword, 0) > 0 or fuzzy_hits.get(keyword, 0) > 0
            )

            # Exact hits are strongest. BM25 keeps paragraphs with related query terms
            # recallable even when the full keyword phrase is not present verbatim.
            score = (3.0 * exact_hit_count) + (1.0 * fuzzy_hit_count) + bm25_score
            if score <= min_score or (exact_hit_count == 0 and fuzzy_hit_count == 0 and bm25_score == 0):
                continue

            results.append(
                {
                    "paragraph_id": paragraph["paragraph_id"],
                    "paragraph": text,
                    "hit_count": exact_hit_count + fuzzy_hit_count,
                    "exact_hit_count": exact_hit_count,
                    "fuzzy_hit_count": fuzzy_hit_count,
                    "bm25_score": round(bm25_score, 6),
                    "score": round(score, 6),
                    "matched_keywords": matched_keywords,
                    "exact_hits": exact_hits,
                    "fuzzy_hits": fuzzy_hits,
                }
            )

        results.sort(key=lambda item: (-item["score"], item["paragraph_id"]))
        results = results[: max(0, effective_top_k)]

        return {
            "document": self.document["document"],
            "query": {
                "keywords": normalized_keywords,
                "top_k": effective_top_k,
                "top_k_strategy": "dynamic_20_percent_min_3_max_15",
                "min_score": min_score,
                "fuzzy_threshold": self.fuzzy_threshold,
            },
            "retrieved_count": len(results),
            "paragraphs": results,
        }

    def _dynamic_top_k(self) -> int:
        paragraph_count = len(self.paragraphs)
        return min(15, max(3, math.ceil(paragraph_count * 0.2)))

    def _average_doc_len(self) -> float:
        if not self._paragraph_tokens:
            return 0.0
        return sum(len(tokens) for tokens in self._paragraph_tokens) / len(self._paragraph_tokens)

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self._paragraph_tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self._paragraph_tokens:
            document_frequency.update(set(tokens))

        return {
            token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def _bm25_score(self, tokens: list[str], query_tokens: list[str]) -> float:
        if not tokens or not query_tokens:
            return 0.0

        k1 = 1.5
        b = 0.75
        counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for token in query_tokens:
            freq = counts[token]
            if freq == 0:
                continue
            idf = self._idf.get(token, 0.0)
            denominator = freq + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1.0))
            score += idf * (freq * (k1 + 1)) / denominator
        return score

    def _exact_hits(self, text: str, keywords: list[str]) -> dict[str, int]:
        hits: dict[str, int] = {}
        for keyword in keywords:
            variants = {keyword}
            tokenized_keyword = " ".join(self._tokenize(keyword))
            if tokenized_keyword:
                variants.add(tokenized_keyword)
            count = 0
            for variant in variants:
                pattern = re.compile(re.escape(variant), re.IGNORECASE)
                count += len(pattern.findall(text))
            if count:
                hits[keyword] = count
        return hits

    def _fuzzy_hits(self, tokens: list[str], keywords: list[str]) -> dict[str, int]:
        hits: dict[str, int] = {}
        token_counts = Counter(tokens)
        for keyword in keywords:
            keyword_tokens = self._tokenize(keyword)
            if not keyword_tokens:
                continue
            fuzzy_count = 0
            for keyword_token in keyword_tokens:
                if keyword_token in token_counts:
                    continue
                fuzzy_count += sum(
                    count
                    for token, count in token_counts.items()
                    if self._similarity(keyword_token, token) >= self.fuzzy_threshold
                )
            if fuzzy_count:
                hits[keyword] = fuzzy_count
        return hits

    def _query_tokens(self, keywords: list[str]) -> list[str]:
        tokens: list[str] = []
        for keyword in keywords:
            tokens.extend(self._tokenize(keyword))
        return sorted(set(tokens))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
        return re.findall(r"[A-Za-z0-9]+", spaced.lower())

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        if abs(len(left) - len(right)) > 3:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recall relevant paragraphs from one document by multiple keywords."
    )
    parser.add_argument("json_file", type=Path, help="Path to one <document_id>.json file.")
    parser.add_argument("keywords", nargs="+", help="Keyword(s) to retrieve with.")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = ParagraphRetriever(
        args.json_file,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    result = retriever.recall(
        args.keywords,
        min_score=args.min_score,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
