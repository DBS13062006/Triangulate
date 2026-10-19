"""
pipeline/step6_embed.py : embed every chunk and save the vectors.
Reads : data/chunks.jsonl, data/MultiHopRAG.json
Writes: data/embeddings.npy (the vectors), data/embedding_ids.json (which chunk is which row)
Run   : python -m pipeline.step6_embed     (from the repository root; the Ollama app must be running)
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import ollama

DATA = Path("data")
MODEL = "nomic-embed-text"
BATCH_SIZE = 32                      # chunks sent to Ollama per request
DOC_PREFIX = "search_document: "     # nomic-embed-text wants this in front of every chunk
QUERY_PREFIX = "search_query: "      # ...and this in front of every question


def embed_batch(texts):
    """Return (vectors as a NumPy table, number of tokens the model read)."""
    response = ollama.embed(
        model=MODEL,
        input=texts,
        truncate=False,              # too-long text -> error, never silent cutting
        options={"num_ctx": 2048},
    )
    vectors = np.array(response["embeddings"], dtype=np.float32)
    tokens = getattr(response, "prompt_eval_count", None) or 0
    return vectors, tokens


def normalise(table):
    return table / np.linalg.norm(table, axis=1, keepdims=True)


# ---------------------------------------------------------------- load chunks
chunks = [json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8")]
texts = [DOC_PREFIX + c["embed_text"] for c in chunks]
n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"{len(texts):,} chunks to embed, {BATCH_SIZE} per request = {n_batches} requests")

# ---------------------------------------------------------------- embed all
all_vectors = []
total_tokens = 0
start = time.perf_counter()
try:
    for b in range(n_batches):
        batch = texts[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        vectors, tokens = embed_batch(batch)
        all_vectors.append(vectors)
        total_tokens += tokens
        if (b + 1) % 10 == 0 or b + 1 == n_batches:
            done = min((b + 1) * BATCH_SIZE, len(texts))
            elapsed = time.perf_counter() - start
            left = elapsed / done * (len(texts) - done)
            print(f"  {done:>6,}/{len(texts):,} chunks | {elapsed:5.0f}s elapsed | ~{left:4.0f}s left")
except Exception as error:
    print("\nStopped:", error)
    print("Check that the Ollama app is running. Nothing was saved; just run the script again.")
    raise SystemExit(1)

matrix = normalise(np.vstack(all_vectors))
print(f"\nDone in {time.perf_counter() - start:.0f}s")

# ---------------------------------------------------------------- checks
print("\nCHECKS")
print("  matrix shape (chunks, numbers each):", matrix.shape)
print("  one row per chunk?                  :", matrix.shape[0] == len(chunks))
print("  any NaN / broken numbers?           :", bool(np.isnan(matrix).any()))
lengths = np.linalg.norm(matrix, axis=1)
print(f"  vector lengths (should all be 1.0)  : min {lengths.min():.4f}  max {lengths.max():.4f}")
print(f"  total tokens the model read         : {total_tokens:,}  (average {total_tokens / len(chunks):.0f} per chunk)")
dupes = len(matrix) - len({row.tobytes() for row in matrix})
print("  identical vectors (repeat chunks)   :", dupes)

# ---------------------------------------------------------------- save
np.save(DATA / "embeddings.npy", matrix)
json.dump([c["chunk_id"] for c in chunks], open(DATA / "embedding_ids.json", "w"))
print(f"\nSaved data/embeddings.npy ({(DATA / 'embeddings.npy').stat().st_size / 1e6:.1f} MB) and data/embedding_ids.json")


# ---------------------------------------------------------------- token audit
# Which chunks are the most "token-heavy"?  Rough guess: count words + punctuation marks.
def rough_tokens(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


print("\nTOKEN AUDIT: the 8 chunks that look biggest, measured one by one by the real model")
print("  chunk_id        words  rough-guess  REAL tokens")
heaviest = sorted(range(len(chunks)), key=lambda i: -rough_tokens(texts[i]))[:8]
for i in heaviest:
    _, real = embed_batch([texts[i]])
    print(f"  {chunks[i]['chunk_id']:<14} {chunks[i]['n_words']:>6} {rough_tokens(texts[i]):>12} {real:>12}")

# ---------------------------------------------------------------- a first taste of search
questions = json.load(open(DATA / "MultiHopRAG.json", encoding="utf-8"))
item = questions[0]
q_vec = normalise(embed_batch([QUERY_PREFIX + item["query"]])[0])[0]
scores = matrix @ q_vec                      # one number per chunk: how close to the question
top5 = np.argsort(-scores)[:5]               # positions of the 5 highest scores
evidence_urls = {e["url"] for e in item["evidence_list"]}

print("\nPREVIEW OF SEARCH: question #0")
print("  Q:", item["query"][:150])
print("  correct answer:", item["answer"])
for rank, i in enumerate(top5, 1):
    mark = "<-- evidence article" if chunks[i]["url"] in evidence_urls else ""
    print(f"  {rank}. {scores[i]:.3f} {chunks[i]['chunk_id']:<14} {chunks[i]['title'][:55]} {mark}")