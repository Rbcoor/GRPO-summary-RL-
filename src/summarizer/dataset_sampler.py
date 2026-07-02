#!/usr/bin/env python3
"""Random sampling and batch loading for exported Repliqa document JSON files."""

from __future__ import annotations

import argparse
import json
import random
import secrets
from pathlib import Path
from typing import Any, Iterator

try:
    from .paragraph_excutor import ParagraphExcutor
except ImportError:
    from paragraph_excutor import ParagraphExcutor


class DatasetSampler:
    """Load exported Repliqa JSON documents with random sampling and batching."""

    def __init__(
        self,
        dataset_root: str | Path,
        splits: list[str] | None = None,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.splits = splits
        self.seed = secrets.randbits(32) if seed is None else seed
        self.seed_was_auto_generated = seed is None
        self.shuffle = shuffle
        self.files = self._discover_files()

    def _discover_files(self) -> list[Path]:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset root not found: {self.dataset_root}")

        split_dirs = (
            [self.dataset_root / split for split in self.splits]
            if self.splits
            else sorted(path for path in self.dataset_root.iterdir() if path.is_dir())
        )

        files: list[Path] = []
        for split_dir in split_dirs:
            if not split_dir.exists():
                raise FileNotFoundError(f"split directory not found: {split_dir}")
            files.extend(sorted(split_dir.glob("*.json")))

        if not files:
            raise ValueError(f"no json documents found under {self.dataset_root}")
        return files

    def _ordered_files(self) -> list[Path]:
        files = list(self.files)
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(files)
        return files

    def load_document(self, json_file: str | Path) -> dict[str, Any]:
        """Load one document JSON file and return numbered paragraphs."""
        path = Path(json_file)
        excutor = ParagraphExcutor(path)
        doc = excutor.reader.document
        return {
            "document_path": str(path),
            "document_id": doc.get("document_id"),
            "document_topic": doc.get("document_topic"),
            "source_pdf": doc.get("document_path"),
            "paragraphs": excutor.execute(),
        }

    def sample(self, n: int) -> list[dict[str, Any]]:
        """Randomly sample n documents and return paragraph-loaded records."""
        if n <= 0:
            return []
        rng = random.Random(self.seed)
        selected = rng.sample(self.files, k=min(n, len(self.files)))
        return [self.load_document(path) for path in selected]

    def iter_batches(
        self,
        batch_size: int,
        limit: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Iterate loaded documents in batches.

        batch_size counts documents, not paragraphs.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        files = self._ordered_files()
        if limit is not None:
            files = files[: max(0, limit)]

        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            yield [self.load_document(path) for path in batch_files]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample exported Repliqa documents and split them into paragraphs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/tmp/repliqa_documents_by_file"),
        help="Root directory containing repliqa_* split folders.",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="Split to include, e.g. --split repliqa_0. Can be repeated.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Limit number of documents to iterate.")
    parser.add_argument("--seed", type=int, help="Random seed. Omit for non-reproducible random sampling.")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep deterministic sorted file order instead of shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampler = DatasetSampler(
        dataset_root=args.dataset_root,
        splits=args.splits,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    output = {
        "dataset_root": str(args.dataset_root),
        "total_documents": len(sampler.files),
        "batch_size": args.batch_size,
        "seed": sampler.seed,
        "seed_was_auto_generated": sampler.seed_was_auto_generated,
        "batches": [],
    }
    for batch_id, batch in enumerate(
        sampler.iter_batches(batch_size=args.batch_size, limit=args.limit),
        start=1,
    ):
        output["batches"].append(
            {
                "batch_id": batch_id,
                "documents": batch,
            }
        )

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
