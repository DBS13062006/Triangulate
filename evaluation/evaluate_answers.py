"""
evaluation/evaluate_answers.py : run the WHOLE pipeline (search + language model) on a sample of benchmark questions
and grade the answers against the correct ones.

Grading is simple: the model's short answer counts as correct if it equals the correct answer or contains it
as whole words (case and punctuation ignored). That is strict for long answers, so read the printed failures.

It also tells you WHY wrong answers were wrong:
  evidence missing from the passages  -> a search problem
  evidence present but still wrong    -> a reading / reasoning problem

Run (from the repository root):
      python -m evaluation.evaluate_answers --n 40                  (10 questions of each of the 4 types, default mode)
      python -m evaluation.evaluate_answers --n 100 --mode hybrid_rerank --k 8   (the earlier setup, for comparison)
Expect roughly 5 to 20 seconds per question on a laptop.
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict

from rag.benchmark import load_benchmark, load_chunks
from config import CONTEXT_CHUNKS, DATA, DEFAULT_MODE
from rag.generator import confident_enough, generate
from rag.retriever import MODES, Retriever

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=40, help="total questions, split equally over the 4 question types")
parser.add_argument("--mode", default=DEFAULT_MODE, choices=MODES)
parser.add_argument("--k", type=int, default=CONTEXT_CHUNKS, help="passages given to the language model")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

TYPE_ORDER = ["inference_query", "comparison_query", "temporal_query", "null_query"]
NULL_ANSWER = "Insufficient information."


def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def is_correct(gold, predicted):
    g, p = clean(gold), clean(predicted)
    return bool(g) and (g == p or re.search(rf"\b{re.escape(g)}\b", p) is not None)


def abstained(predicted):
    return "insufficient information" in clean(predicted)


# ---------------------------------------------------------------- pick a balanced sample
chunks = load_chunks()
questions = load_benchmark(chunks)
by_type = defaultdict(list)
for q in questions:
    by_type[q["question_type"]].append(q)
rng = random.Random(args.seed)
per_type = max(1, args.n // len(TYPE_ORDER))
sample = [q for t in TYPE_ORDER for q in rng.sample(by_type[t], min(per_type, len(by_type[t])))]
print(f"{len(sample)} questions, mode={args.mode}, {args.k} passages each\n")

retriever = Retriever()
rows, seconds = [], 0.0

# ---------------------------------------------------------------- run the pipeline
for number, q in enumerate(sample, start=1):
    start = time.perf_counter()
    results = retriever.retrieve(q["query"], mode=args.mode, k=args.k)
    if confident_enough(results, args.mode):
        reply = generate(q["query"], results)
        predicted, warning = reply["short"], reply["warning"]
    else:
        predicted, warning = NULL_ANSWER, None
    elapsed = time.perf_counter() - start
    seconds += elapsed

    context_ids = {r["chunk"]["chunk_id"] for r in results}
    evidence_in_context = (all(ids & context_ids for ids in q["evidence_chunks"]) if q["answerable"] else None)
    row = {
        "type": q["question_type"], "question": q["query"], "gold": q["answer"], "predicted": predicted,
        "correct": is_correct(q["answer"], predicted), "abstained": abstained(predicted),
        "evidence_in_context": evidence_in_context, "seconds": round(elapsed, 1), "warning": warning,
        "chunk_ids": [r["chunk"]["chunk_id"] for r in results],
    }
    rows.append(row)
    print(f"  {number:>3}/{len(sample)}  {'OK ' if row['correct'] else 'BAD'}  {q['question_type']:<17} "
          f"gold={q['answer'][:25]!r:<28} got={predicted[:35]!r}  ({elapsed:.0f}s)")

# ---------------------------------------------------------------- report
print("\n=== ACCURACY BY QUESTION TYPE ===")
for t in TYPE_ORDER:
    group = [r for r in rows if r["type"] == t]
    if group:
        print(f"  {t:<18} {sum(r['correct'] for r in group):>3}/{len(group):<3} = {sum(r['correct'] for r in group) / len(group):.0%}")
print(f"  {'ALL':<18} {sum(r['correct'] for r in rows):>3}/{len(rows):<3} = {sum(r['correct'] for r in rows) / len(rows):.0%}")

answerable = [r for r in rows if r["type"] != "null_query"]
null = [r for r in rows if r["type"] == "null_query"]
print("\n=== REFUSALS ('Insufficient information') ===")
if null:
    print(f"  null questions correctly refused      : {sum(r['abstained'] for r in null)}/{len(null)}")
print(f"  answerable questions wrongly refused  : {sum(r['abstained'] for r in answerable)}/{len(answerable)}")

wrong = [r for r in answerable if not r["correct"]]
print(f"\n=== WHY WERE ANSWERABLE QUESTIONS WRONG? ({len(wrong)} wrong) ===")
print(f"  evidence NOT all in the passages (search problem)    : {sum(r['evidence_in_context'] is False for r in wrong)}")
print(f"  evidence WAS in the passages (reading/reasoning)     : {sum(r['evidence_in_context'] is True for r in wrong)}")
print(f"\naverage time per question: {seconds / len(rows):.1f}s")
if any(r["warning"] for r in rows):
    print("WARNING: some prompts nearly filled the model's context window; see 'warning' in the saved file.")

print("\n=== A FEW WRONG ANSWERS (read these!) ===")
for r in [r for r in rows if not r["correct"]][:6]:
    print(f"  [{r['type']}] {r['question'][:140]}")
    print(f"      correct: {r['gold'][:80]!r}   model said: {r['predicted'][:120]!r}   evidence in passages: {r['evidence_in_context']}")

out_file = DATA / f"eval_answers_{args.mode}_k{args.k}.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nSaved {out_file}")
