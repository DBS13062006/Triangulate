"""
rag/benchmark.py : loads the MultiHop-RAG questions and works out WHICH chunks hold each piece of evidence.
Used by the two evaluation scripts.
"""
import json
import re
from collections import defaultdict

from config import DATA


def normalise(text):
    """Same cleaning used when we verified the chunks: drop invisible U+FEFF, squash whitespace."""
    return re.sub(r"\s+", " ", text.replace("﻿", "")).strip()


def load_chunks():
    return [json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8")]


def load_benchmark(chunks=None):
    """
    Returns the list of questions. Each gets two extra fields:
      answerable      : False for the 'null' questions (no evidence exists)
      evidence_chunks : one set of chunk_ids per evidence fact = the chunks that contain that fact
    """
    chunks = chunks if chunks is not None else load_chunks()
    by_url = defaultdict(list)
    for c in chunks:
        by_url[c["url"]].append((c["chunk_id"], normalise(c["text"])))

    questions = json.load(open(DATA / "MultiHopRAG.json", encoding="utf-8"))
    for q in questions:
        q["answerable"] = len(q["evidence_list"]) > 0
        q["evidence_chunks"] = []
        for ev in q["evidence_list"]:
            fact = normalise(ev["fact"])
            q["evidence_chunks"].append({cid for cid, text in by_url.get(ev["url"], []) if fact in text})
    return questions


def score_question(q, ranked_chunk_ids, ks):
    """For each k: (share of this question's evidence facts found in the top-k, were ALL found?)"""
    result = {}
    for k in ks:
        top = set(ranked_chunk_ids[:k])
        found = sum(1 for ids in q["evidence_chunks"] if ids & top)
        result[k] = (found / len(q["evidence_chunks"]), found == len(q["evidence_chunks"]))
    return result
