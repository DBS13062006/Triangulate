"""
ask.py : the finished RAG app. Ask a question, see which passages were found, get a grounded answer.

Run:  python ask.py                         (the default mode from config.py: smart_rerank)
      python ask.py --mode hybrid_rerank    (hybrid search + reranker, without the outlet / date / clause seats)
      python ask.py --mode hybrid           (no reranker; starts faster)
In smart_rerank, passages that won a guaranteed seat ([seat outlet: ...], [seat date: ...], [seat clause: ...]) are
listed first, so the scores are not always in falling order.

Commands while it runs (type them instead of a question):
  /mode dense|keyword|hybrid|hybrid_rerank|outlet_rerank|smart_rerank     change the search mode
  /k 10                                        how many passages the language model gets
  /source The Verge                            only this outlet        (/source alone clears it)
  /after 2023-11-01   /before 2023-11-30       date limits             (alone clears them)
  /text on|off                                 show each passage's text too
Press Enter on an empty line to quit.
"""
import argparse
import time

from config import ABSTAIN_BELOW, CONTEXT_CHUNKS, DEFAULT_MODE
from rag.generator import confident_enough, generate
from rag.retriever import MODES, Retriever

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default=DEFAULT_MODE, choices=MODES)
args = parser.parse_args()

retriever = Retriever()
print(f"Database ready: {retriever.table.count_rows():,} chunks. Mode: {args.mode}\n")

mode, k, source, after, before, show_text = args.mode, CONTEXT_CHUNKS, None, None, None, False

while True:
    line = input("Question (blank to quit): ").strip()
    if not line:
        break

    # ---- settings commands -------------------------------------------------
    if line.startswith("/"):
        command, _, value = line[1:].partition(" ")
        value = value.strip()
        if command == "mode" and value in MODES:
            mode = value
        elif command == "k" and value.isdigit():
            k = int(value)
        elif command == "source":
            source = value or None
        elif command == "after":
            after = value or None
        elif command == "before":
            before = value or None
        elif command == "text":
            show_text = value == "on"
        else:
            print("  unknown command")
        print(f"  settings: mode={mode} k={k} source={source} after={after} before={before} text={show_text}\n")
        continue

    # ---- 1. search ---------------------------------------------------------
    start = time.perf_counter()
    try:
        results = retriever.retrieve(line, mode=mode, k=k, source=source, after=after, before=before)
    except Exception as error:
        print("  Search failed:", error, "\n")
        continue
    search_seconds = time.perf_counter() - start

    print(f"\n  PASSAGES FOUND ({mode}, {search_seconds:.1f}s)")
    for r in results:
        c = r["chunk"]
        print(f"   [{r['rank']}] score {r['score']:.3f}  {c['chunk_id']:<13} found by: {r['found_by']}")
        print(f"       {c['title'][:80]}  ({c['source']}, {c['published_at']})")
        if show_text:
            print(f"       {c['text'][:300].replace(chr(10), ' ')}...")
    if not results:
        print("   nothing matched your filters\n")
        continue

    # ---- 2. answer ---------------------------------------------------------
    if not confident_enough(results, mode):
        print(f"\n  ANSWER\n   Insufficient information.  (best passage scored {max(r['score'] for r in results):.3f}, "
              f"below the gate {ABSTAIN_BELOW})\n")
        continue

    start = time.perf_counter()
    try:
        reply = generate(line, results)
    except Exception as error:
        print("  The language model failed:", error, "\n")
        continue
    print(f"\n  ANSWER ({time.perf_counter() - start:.1f}s, prompt {reply['prompt_tokens']} tokens)")
    for text_line in reply["raw"].splitlines():
        print("   " + text_line)
    if reply["warning"]:
        print("   WARNING:", reply["warning"])
    print()
