from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serviceflow.classifier import CATEGORY_LABELS, classify  # noqa: E402
from serviceflow.config import Settings  # noqa: E402
from serviceflow.knowledge import KnowledgeBase  # noqa: E402
from serviceflow.retrieval import heuristic_rerank, rewrite_query  # noqa: E402


def load_cases(path: Path) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    values: list[float] = []
    for label in labels:
        true_positive = sum(e == label and p == label for e, p in zip(expected, predicted))
        false_positive = sum(e != label and p == label for e, p in zip(expected, predicted))
        false_negative = sum(e == label and p != label for e, p in zip(expected, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append((2 * true_positive / denominator) if denominator else 0.0)
    return statistics.mean(values) if values else 0.0


def evaluate_retrieval(
    kb: KnowledgeBase,
    cases: list[dict[str, str]],
    mode: str,
    rerank: bool = False,
) -> dict[str, object]:
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    failures: list[dict[str, object]] = []
    for case in cases:
        rewritten = rewrite_query(case["query"])
        started = time.perf_counter()
        matches = kb.search(
            rewritten.rewritten_query,
            case["category"],
            top_k=20,
            mode=mode,
        )
        if rerank:
            matches = heuristic_rerank(
                rewritten.rewritten_query,
                matches,
                limit=20,
            )
        latencies.append((time.perf_counter() - started) * 1000)
        docs = [item.chunk.doc_id for item in matches]
        expected = case["document"]
        hit_at_1 += int(bool(docs) and docs[0] == expected)
        hit_at_3 += int(expected in docs[:3])
        rank = next(
            (index + 1 for index, document in enumerate(docs) if document == expected),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank is None or rank > 3:
            failures.append(
                {
                    "query": case["query"],
                    "expected_document": expected,
                    "rank": rank,
                    "retrieved_documents": docs[:3],
                }
            )
    total = len(cases)
    name = f"{mode}_rerank" if rerank else mode
    return {
        "name": name,
        "hit_at_1": round(hit_at_1 / total, 4),
        "hit_at_3": round(hit_at_3 / total, 4),
        "mrr_at_5": round(sum(reciprocal_ranks) / total, 4),
        "average_latency_ms": round(statistics.mean(latencies), 3),
        "failures": failures,
    }


def run(path: Path) -> dict[str, object]:
    settings = Settings.from_env()
    kb = KnowledgeBase.load(settings.knowledge_dir)
    cases = load_cases(path)
    expected_categories: list[str] = []
    predicted_categories: list[str] = []
    category_correct = 0
    priority_correct = 0
    priority_under_escalated = 0
    priority_over_escalated = 0
    failures: list[dict[str, object]] = []
    priority_failures: list[dict[str, object]] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    retrieval_modes: list[dict[str, object]] = []

    for case in cases:
        result = classify(case["query"])
        matches = kb.search(case["query"], result.category, top_k=3)
        doc_ids = [item.chunk.doc_id for item in matches]
        expected_categories.append(case["category"])
        predicted_categories.append(result.category)
        confusion[case["category"]][result.category] += 1
        category_ok = result.category == case["category"]
        category_correct += int(category_ok)
        priority_ok = result.priority == case["priority"]
        priority_correct += int(priority_ok)
        priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        if priority_rank[result.priority] > priority_rank[case["priority"]]:
            priority_under_escalated += 1
        elif priority_rank[result.priority] < priority_rank[case["priority"]]:
            priority_over_escalated += 1
        if not priority_ok:
            priority_failures.append(
                {
                    "query": case["query"],
                    "expected": case["priority"],
                    "predicted": result.priority,
                }
            )
        if not category_ok or not doc_ids or doc_ids[0] != case["document"]:
            failures.append(
                {
                    "query": case["query"],
                    "expected_category": case["category"],
                    "predicted_category": result.category,
                    "expected_document": case["document"],
                    "retrieved_documents": doc_ids,
                    "confidence": result.confidence,
                }
            )

    total = len(cases)
    for mode, rerank in (
        ("lexical", False),
        ("vector", False),
        ("hybrid", False),
        ("hybrid", True),
    ):
        retrieval_modes.append(evaluate_retrieval(kb, cases, mode, rerank))
    primary_retrieval = next(
        item for item in retrieval_modes if item["name"] == "hybrid_rerank"
    )
    return {
        "cases": total,
        "category_accuracy": round(category_correct / total, 4),
        "category_macro_f1": round(macro_f1(expected_categories, predicted_categories), 4),
        "retrieval_hit_at_1": primary_retrieval["hit_at_1"],
        "retrieval_hit_at_3": primary_retrieval["hit_at_3"],
        "retrieval_mrr_at_5": primary_retrieval["mrr_at_5"],
        "retrieval_modes": retrieval_modes,
        "priority_accuracy": round(priority_correct / total, 4),
        "priority_under_escalated": priority_under_escalated,
        "priority_over_escalated": priority_over_escalated,
        "category_labels": CATEGORY_LABELS,
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "failures": failures,
        "priority_failures": priority_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ServiceFlow offline evaluation")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "evaluation.jsonl",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.data)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Evaluation written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()

