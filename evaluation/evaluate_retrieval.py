"""
evaluation/evaluate_retrieval.py : measure how well each search mode FINDS the evidence for the benchmark questions.

For every answerable question we check, among the top-k chunks returned, which of its evidence facts appear
(an evidence fact counts as found if ANY chunk that contains it is in the top k).
Two numbers per k:
  fact recall   = average share of a question's evidence facts that were found
  all evidence  = share of questions where EVERY needed fact was found (the strict multi-hop score)

Run (from the repository root):
      python -m evaluation.evaluate_retrieval                  (dense, keyword, hybrid; a few minutes)
      python -m evaluation.evaluate_retrieval --rerank         (also the three reranker modes; slower)
      python -m evaluation.evaluate_retrieval --limit 300      (quick test on a random sample of questions)
      python -m evaluation.evaluate_retrieval --limit 300 --modes smart_rerank   (only the modes you list)
"""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict

import numpy as np

from rag.benchmark import load_benchmark, load_chunks, score_question
from config import DATA
from rag.retriever import MODES, Retriever

KS = [1, 3, 5, 8, 10, 20]            # 8 = how many passages the language model reads
TYPES = ["inference_query", "comparison_query", "temporal_query"]

parser = argparse.ArgumentParser()
parser.add_argument("--rerank", action="store_true", help="also evaluate hybrid_rerank, outlet_rerank and smart_rerank (slow)")
parser.add_argument("--modes", default="", help="comma-separated list of modes to run, e.g. outlet_rerank")
parser.add_argument("--limit", type=int, default=0, help="evaluate only this many random questions")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def question_vectors(retriever, questions):
    """Embed every question once and remember the result on disk."""
    texts = [q["query"] for q in questions]
    key = hashlib.md5("\n".join(texts).encode()).hexdigest()
    vec_file, key_file = DATA / "query_vectors.npy", DATA / "query_vectors.key"
    if vec_file.exists() and key_file.exists() and key_file.read_text() == key:
        print("Using saved question vectors.")
        return np.load(vec_file)
    print(f"Embedding {len(texts):,} questions with Ollama (saved afterwards for next time) ...")
    start = time.perf_counter()
    vectors = retriever.embed_queries(texts)
    np.save(vec_file, vectors)
    key_file.write_text(key)
    print(f"  done in {time.perf_counter() - start:.0f}s")
    return vectors


def suggest_threshold(name, answerable_scores, null_scores):
    """If the best passage scores below T we say 'Insufficient information'. Which T separates the two groups best?"""
    answerable, null = np.array(answerable_scores), np.array(null_scores)
    best = None
    for t in np.unique(np.concatenate([answerable, null])):
        null_caught = float((null < t).mean())          # null questions we correctly refuse
        wrongly_refused = float((answerable < t).mean())  # answerable questions we wrongly refuse
        gain = null_caught - wrongly_refused
        if best is None or gain > best[0]:
            best = (gain, float(t), null_caught, wrongly_refused)
    _, t, caught, wrong = best
    print(f"\n  {name}: best-passage score, typical answerable {np.median(answerable):.3f} vs typical null {np.median(null):.3f}")
    print(f"    suggested ABSTAIN_BELOW = {t:.3f}  ->  refuses {caught:.0%} of null questions, "
          f"but also wrongly refuses {wrong:.0%} of answerable ones")


# ---------------------------------------------------------------- setup
chunks = load_chunks()
questions = load_benchmark(chunks)
unmapped = sum(1 for q in questions for ids in q["evidence_chunks"] if not ids)
print(f"{len(questions):,} questions ({sum(q['answerable'] for q in questions):,} answerable, "
      f"{sum(not q['answerable'] for q in questions)} null). Evidence facts not found in any chunk: {unmapped} (should be 0)\n")

retriever = Retriever()
vectors = question_vectors(retriever, questions)

indices = list(range(len(questions)))
if args.limit:
    random.Random(args.seed).shuffle(indices)
    indices = indices[:args.limit]
    print(f"Using a random sample of {len(indices)} questions.\n")

if args.modes:
    modes = [m.strip() for m in args.modes.split(",")]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown mode(s) {bad}; choose from {MODES}")
else:
    modes = ["dense", "keyword", "hybrid"] + (["hybrid_rerank", "outlet_rerank", "smart_rerank"] if args.rerank else [])
summary, top_scores = {}, {}

# ---------------------------------------------------------------- run every mode
for mode in modes:
    print(f"=== {mode} ===")
    start = time.perf_counter()
    stats = defaultdict(lambda: {"n": 0, "recall": defaultdict(float), "all": defaultdict(int)})
    top_scores[mode] = {"answerable": [], "null": []}

    for done, i in enumerate(indices, start=1):
        q = questions[i]
        results = retriever.retrieve(q["query"], mode=mode, k=max(KS), query_vector=vectors[i])
        if results:
            top_scores[mode]["answerable" if q["answerable"] else "null"].append(max(r["score"] for r in results))
        if q["answerable"]:
            ranked = [r["chunk"]["chunk_id"] for r in results]
            scores = score_question(q, ranked, KS)
            for group in (q["question_type"], "ALL"):
                stats[group]["n"] += 1
                for k in KS:
                    stats[group]["recall"][k] += scores[k][0]
                    stats[group]["all"][k] += scores[k][1]
        if done % 250 == 0:
            print(f"  {done:,}/{len(indices):,} questions  ({time.perf_counter() - start:.0f}s)")

    print(f"  finished in {time.perf_counter() - start:.0f}s\n")
    print(f"  {'question type':<18}{'n':>5} | fact recall at k: " + "".join(f"{'@' + str(k):>6}" for k in KS)
          + " | ALL evidence found at k: " + "".join(f"{'@' + str(k):>6}" for k in KS[2:]))
    summary[mode] = {}
    for group in TYPES + ["ALL"]:
        s = stats[group]
        if not s["n"]:
            continue
        recall = {k: s["recall"][k] / s["n"] for k in KS}
        every = {k: s["all"][k] / s["n"] for k in KS}
        summary[mode][group] = {"n": s["n"], "recall": recall, "all": every}
        print(f"  {group:<18}{s['n']:>5} | {'':<17}" + "".join(f"{recall[k]:>6.0%}" for k in KS)
              + " | " + " " * 25 + "".join(f"{every[k]:>6.0%}" for k in KS[2:]))
    print()

# ---------------------------------------------------------------- side-by-side
print("=== SIDE BY SIDE (all answerable questions) ===")
print(f"  {'mode':<15}{'recall@5':>10}{'recall@8':>10}{'recall@10':>11}{'recall@20':>11}{'all@5':>8}{'all@8':>8}{'all@10':>8}{'all@20':>8}")
for mode in modes:
    a = summary[mode].get("ALL")
    if a:
        print(f"  {mode:<15}{a['recall'][5]:>10.0%}{a['recall'][8]:>10.0%}{a['recall'][10]:>11.0%}{a['recall'][20]:>11.0%}"
              f"{a['all'][5]:>8.0%}{a['all'][8]:>8.0%}{a['all'][10]:>8.0%}{a['all'][20]:>8.0%}")

# ---------------------------------------------------------------- 'I don't know' gate
print("\n=== CAN SCORES TELL ANSWERABLE FROM NULL QUESTIONS? ===")
for mode in ("dense", "hybrid_rerank", "outlet_rerank", "smart_rerank"):
    if mode in top_scores and top_scores[mode]["answerable"] and top_scores[mode]["null"]:
        suggest_threshold(mode, top_scores[mode]["answerable"], top_scores[mode]["null"])
print("  (If you want the gate, paste the number into ABSTAIN_BELOW in config.py. Only scores from 'dense', "
      "'hybrid_rerank', 'outlet_rerank' and 'smart_rerank' are suitable.)")

json.dump(summary, open(DATA / "eval_retrieval.json", "w"), indent=1)
print("\nSaved data/eval_retrieval.json")
