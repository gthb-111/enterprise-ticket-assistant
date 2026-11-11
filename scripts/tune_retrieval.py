from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate import evaluate_retrieval, load_cases  # noqa: E402
from serviceflow.config import Settings  # noqa: E402
from serviceflow.knowledge import KnowledgeBase  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    cases = load_cases(ROOT / "data" / "evaluation.jsonl")
    rows = []
    for weight in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        kb = KnowledgeBase.load(
            settings.knowledge_dir,
            vector_weight=weight,
            vector_dimensions=settings.vector_dimensions,
        )
        hybrid = evaluate_retrieval(kb, cases, "hybrid")
        reranked = evaluate_retrieval(kb, cases, "hybrid", rerank=True)
        rows.append(
            {
                "vector_weight": weight,
                "hybrid_hit_at_1": hybrid["hit_at_1"],
                "hybrid_hit_at_3": hybrid["hit_at_3"],
                "rerank_hit_at_1": reranked["hit_at_1"],
                "rerank_hit_at_3": reranked["hit_at_3"],
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
