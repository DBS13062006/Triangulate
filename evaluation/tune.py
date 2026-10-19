"""
evaluation/tune.py : find the best way to pick the final passages, and compare reranker models. Two commands.

1. cache  (slow, once per reranker model): the reranker scores every candidate passage; the scores are SAVED.
     python -m evaluation.tune cache --limit 300
     python -m evaluation.tune cache --limit 300 --seed 1                                  (a second, different sample, to double-check)
     python -m evaluation.tune cache --limit 300 --model BAAI/bge-reranker-v2-m3           (a stronger reranker; slower, ~2 GB download)
   It can be stopped with Ctrl+C and restarted: it carries on where it stopped.

2. grid   (seconds): replay many "selection rules" on the saved scores. No models needed, so you can try anything.
     python -m evaluation.tune grid
     python -m evaluation.tune grid --cache data/rerank_cache_<model>_n300_s0.jsonl --confirm data/rerank_cache_<model>_n300_s1.jsonl

A selection rule = which candidate piles to use, how many guaranteed seats per named outlet / named day / clause,
and how many passages one article may take.
"""
import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict

import numpy as np

from rag.benchmark import load_benchmark, load_chunks, score_question
from config import (CLAUSE_QUOTA, DATA, DATE_QUOTA, MAX_PER_DOC, OUTLET_QUOTA, RERANK_MAX_LENGTH, RERANK_MODEL,
                    RERANK_POOL)
from rag.retriever import Retriever, WordMatch, find_dates, find_outlets, select_passages, split_clauses

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=["cache", "grid"])
parser.add_argument("--limit", type=int, default=300, help="cache: how many random questions (same sample as evaluation/evaluate_retrieval.py)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", default=RERANK_MODEL, help="cache: which reranker to score with")
parser.add_argument("--cache", default="", help="grid: the cache file to tune on (default: the newest one)")
parser.add_argument("--confirm", default="", help="grid: a second cache file used to double-check the winner")
args = parser.parse_args()

TYPES = ["inference_query", "comparison_query", "temporal_query"]
BASE_MAX = 50                       # the cache keeps the best 50 hybrid candidates; rules may use only the best 30 of them
# pile name -> (how many hybrid candidates, add the per-outlet pile?, add the per-day pile?)
PILES = {"30": (30, False, False), "30 + outlets": (30, True, False), "30 + outlets + days": (30, True, True),
         "50": (50, False, False), "50 + outlets": (50, True, False), "50 + outlets + days": (50, True, True)}


def cache_path(model, limit, seed):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
    return DATA / f"rerank_cache_{slug}_n{limit}_s{seed}.jsonl"


def sample_indices(questions, limit, seed):
    """Exactly the sample evaluation/evaluate_retrieval.py --limit N uses (shuffle everything, keep the first N)."""
    indices = list(range(len(questions)))
    random.Random(seed).shuffle(indices)
    return indices[:limit]


# ================================================================ 1. score and save
def make_cache():
    questions = load_benchmark(load_chunks())
    retriever = Retriever()
    texts = [q["query"] for q in questions]
    key = hashlib.md5("\n".join(texts).encode()).hexdigest()
    vec_file, key_file = DATA / "query_vectors.npy", DATA / "query_vectors.key"
    if vec_file.exists() and key_file.exists() and key_file.read_text() == key:
        vectors = np.load(vec_file)
    else:
        print("Embedding the questions (once) ...")
        vectors = retriever.embed_queries(texts)
        np.save(vec_file, vectors)
        key_file.write_text(key)

    from sentence_transformers import CrossEncoder
    print(f"Loading reranker {args.model} (the first time this downloads it) ...")
    retriever._reranker = CrossEncoder(args.model, max_length=RERANK_MAX_LENGTH)

    path = cache_path(args.model, args.limit, args.seed)
    done = {json.loads(line)["i"] for line in open(path)} if path.exists() else set()
    todo = [i for i in sample_indices(questions, args.limit, args.seed) if questions[i]["answerable"] and i not in done]
    print(f"{len(done)} questions already saved, {len(todo)} to go.  File: {path}\n")

    start = time.perf_counter()
    with open(path, "a") as out:
        for n, i in enumerate(todo, start=1):
            q = questions[i]["query"]
            outlets, dates = find_outlets(q, retriever.sources), find_dates(q)
            fused = retriever._hybrid(q, vectors[i])
            position = {e["row"]["chunk_id"]: n for n, e in enumerate(fused)}      # rank in the plain hybrid list
            pool = retriever.candidates(q, vectors[i], BASE_MAX, outlets, dates, fused)
            ranked = retriever._rerank(q, list(pool.values()))
            cands = [[e["row"]["chunk_id"], e["row"]["doc_id"], e["row"]["source"], e["row"]["published_at"][:10],
                      round(e["rerank"], 5), sorted(e["origins"]), position.get(e["row"]["chunk_id"], 999)] for e in ranked]
            out.write(json.dumps({"i": i, "outlets": outlets, "dates": dates, "cands": cands}) + "\n")
            out.flush()
            if n % 20 == 0 or n == len(todo):
                spent = time.perf_counter() - start
                print(f"  {n}/{len(todo)} questions   {spent / 60:.1f} min spent, about {(spent / n) * (len(todo) - n) / 60:.1f} min to go")
    print("\nDone. Now run:  python -m evaluation.tune grid")


# ================================================================ 2. replay selection rules
def load_cache(path, chunks, questions, word_match):
    """Saved records + the word-match score of every candidate for every clause (computed once, used by all rules)."""
    text_of = {c["chunk_id"]: c["embed_text"] for c in chunks}
    data = []
    for rec in map(json.loads, open(path)):
        q = questions[rec["i"]]
        clause_scores = [[word_match.score(clause, text_of[c[0]]) for c in rec["cands"]] for clause in split_clauses(q["query"])]
        data.append((q, rec, clause_scores))
    return data


def in_pile(candidate, pile):
    """Is this saved candidate part of the pile? (candidate = [chunk_id, doc_id, source, day, score, origins, hybrid rank])"""
    base_n, use_outlets, use_days = PILES[pile]
    return (candidate[6] < base_n) or (use_outlets and "outlet" in candidate[5]) or (use_days and "date" in candidate[5])


def run_policy(data, rule, k):
    """Replay one selection rule on the saved scores. Returns {question_type or ALL: (n, recall, all-found)}."""
    pile, outlet_seats, date_seats, per_doc, clause_seats = rule
    totals = defaultdict(lambda: [0, 0.0, 0.0])
    for q, rec, clause_scores in data:
        ranked = [{"row": {"chunk_id": c[0], "doc_id": c[1], "source": c[2], "published_at": c[3]}, "rerank": c[4], "n": n}
                  for n, c in enumerate(rec["cands"]) if in_pile(c, pile)]                  # already sorted best-first
        clause_lists = [sorted(ranked, key=lambda e: -s[e["n"]]) for s in clause_scores] if clause_seats else []
        chosen = select_passages(ranked, rec["outlets"], rec["dates"], k, outlet_seats, date_seats, per_doc,
                                 clause_lists, clause_seats)
        recall, every = score_question(q, [e["row"]["chunk_id"] for e in chosen], [k])[k]
        for group in (q["question_type"], "ALL"):
            totals[group][0] += 1
            totals[group][1] += recall
            totals[group][2] += every
    return {g: (n, r / n, a / n) for g, (n, r, a) in totals.items()}


def describe(rule):
    pile, outlet_seats, date_seats, per_doc, clause_seats = rule
    cap = f"   cap {per_doc} per article" if per_doc else ""
    return f"candidates {pile:<20} seats: outlet {outlet_seats}  day {date_seats}  clause {clause_seats}{cap}"


def complexity(rule):
    pile, outlet_seats, date_seats, per_doc, clause_seats = rule
    base_n, use_outlets, use_days = PILES[pile]
    return (base_n == 50) + use_outlets + use_days + outlet_seats + date_seats + (1 if per_doc else 0) + clause_seats


def show(rows, title):
    print(f"\n  {title}")
    print(f"  {'rule':<84}{'recall':>8}{'all found':>11}   all found by type: infer / compare / temporal")
    for rule, result in rows:
        a = result["ALL"]
        by_type = " / ".join(f"{result[t][2]:.0%}" if t in result else "-" for t in TYPES)
        print(f"  {describe(rule):<84}{a[1]:>8.0%}{a[2]:>11.0%}   {by_type}")


def grid():
    chunks = load_chunks()
    questions = load_benchmark(chunks)
    word_match = WordMatch([c["embed_text"] for c in chunks])
    path = args.cache or str(max(DATA.glob("rerank_cache_*.jsonl"), key=lambda p: p.stat().st_mtime))
    data = load_cache(path, chunks, questions, word_match)
    print(f"Tuning on {len(data)} answerable questions from {path}")
    confirm = load_cache(args.confirm, chunks, questions, word_match) if args.confirm else None
    if confirm:
        print(f"Double-checking on {len(confirm)} other questions from {args.confirm}")

    rules = []
    for pile, (base_n, use_outlets, use_days) in PILES.items():
        for outlet_seats in ((0, 1, 2, 3) if use_outlets else (0,)):
            for date_seats in ((0, 1, 2) if use_days else (0,)):
                rules += [(pile, outlet_seats, date_seats, cap, clause) for cap in (0, 2) for clause in (0, 1, 2)]

    plain = ("30", 0, 0, 0, 0)
    shipped_pile = "50 + outlets + days" if RERANK_POOL >= 50 else "30 + outlets + days"
    shipped = (shipped_pile, OUTLET_QUOTA, DATE_QUOTA, MAX_PER_DOC, CLAUSE_QUOTA)
    rules = list(dict.fromkeys(rules + [shipped]))
    for k in (8, 10, 12):
        results = {rule: run_policy(data, rule, k) for rule in rules}
        ranking = sorted(results.items(), key=lambda kv: (-kv[1]["ALL"][2], -kv[1]["ALL"][1]))
        print("\n" + "=" * 112 + f"\nPASSAGES GIVEN TO THE LANGUAGE MODEL: k = {k}")
        show([(plain, results[plain]), (shipped, results[shipped])], "for reference: plain hybrid_rerank, and smart_rerank with your config.py settings")
        show(ranking[:5], "the 5 best rules")
        best = ranking[0][1]["ALL"][2]
        simple = min((kv for kv in results.items() if kv[1]["ALL"][2] >= best - 0.01), key=lambda kv: (complexity(kv[0]), -kv[1]["ALL"][1]))
        show([simple], "the simplest rule within 1 point of the best (simple rules are less likely to be a fluke)")
        if confirm:
            show([(plain, run_policy(confirm, plain, k)), (shipped, run_policy(confirm, shipped, k)),
                  (simple[0], run_policy(confirm, simple[0], k))],
                 "DOUBLE-CHECK on the other sample: plain, your config.py settings, the simplest good rule")

    print("\n" + "=" * 112 + "\nCEILING = what the reranker could find if it never made a mistake (everything in the pile kept):")
    for pile in PILES:
        ceiling = run_policy(data, (pile, 0, 0, 0, 0), 500)["ALL"]
        print(f"  candidates {pile:<22} recall {ceiling[1]:.0%}   all found {ceiling[2]:.0%}")


make_cache() if args.command == "cache" else grid()
