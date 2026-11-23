"""Evaluate retrieval on the question set: Recall@k and MRR."""

import json
import sys

from tenk_rag.config import PROJECT_ROOT
from tenk_rag.search import Searcher
from tenk_rag.rerank import Reranker

EVAL_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
K_VALUES = (1, 3, 5, 10, 50)


def load_questions(path=EVAL_PATH):
	"""Read one question per line, skipping blank lines."""
	with path.open(encoding="utf-8") as f:
		return [json.loads(line) for line in f if line.strip()]


def is_hit(chunk, evidence):
	"""A chunk counts as correct if it contains every evidence string."""
	text = chunk["text"].lower()
	return all(e.lower() in text for e in evidence)


def first_hit_rank(results, evidence):
	"""Rank (1-based) of the first correct chunk, or None if none was retrieved."""
	for rank, (chunk, _) in enumerate(results, start=1):
		if is_hit(chunk, evidence):
			return rank
	return None


def main():
	args = [a for a in sys.argv[1:] if not a.startswith("--")]
	name = args[0] if args else "latest"
	use_rerank = "--rerank" in sys.argv

	# No-answer questions have nothing to retrieve; they are scored later on the LLM's answer
	questions = [q for q in load_questions() if q["evidence"]]
	searcher = Searcher()
	reranker = Reranker() if use_rerank else None
	# Retrieves 10 even though the system uses 5, to tell near misses apart from comlete misses
	max_k = max(K_VALUES)

	ranks = []
	for q in questions:
		results = searcher.search(q["question"], q["ticker"], k=max_k)
		if reranker:
			results = reranker.rerank(q["question"], results)   # reorder the top 50
		rank = first_hit_rank(results, q["evidence"])
		ranks.append(rank)
		shown = "miss" if rank is None else rank
		print(f"{q['id']:<9} {q['category']:<12} {shown:>4}  {q['question'][:60]}")

	n = len(ranks)
	print(f"\n{n} questions")
	for k in K_VALUES:
		recall = sum(r is not None and r <= k for r in ranks) / n
		print(f"Recall@{k:<2}: {recall:.2f}")
	mrr = sum(1 / r for r in ranks if r is not None) / n
	print(f"MRR@{max_k}  : {mrr:.2f}")

	# Recall@5 by question category and by company
	for field in ("category", "ticker"):
		groups = {}
		for q, r in zip(questions, ranks):
			# Groups ranks by category (or ticker)
			groups.setdefault(q[field], []).append(r)
		print(f"\nRecall@5 by {field}:")
		for name, rs in sorted(groups.items()):
			hits = sum(r is not None and r <= 5 for r in rs)
			print(f"  {name:<12} {hits}/{len(rs)}")

	# Save results so experiments can be compared later
	name = sys.argv[1] if len(sys.argv) > 1 else "latest"
	out = PROJECT_ROOT / "experiments" / f"{name}.json"
	out.parent.mkdir(exist_ok=True)
	summary = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in K_VALUES}
	summary[f"mrr@{max_k}"] = mrr
	record = {
		"name": name,
		"rerank": use_rerank,
		"n_questions": n,
		"summary": summary,
		"ranks": {q["id"]: r for q, r in zip(questions, ranks)},
	}
	out.write_text(json.dumps(record, indent=2))
	print(f"\nSaved to {out.relative_to(PROJECT_ROOT)}")


if __name__=="__main__":
	main()