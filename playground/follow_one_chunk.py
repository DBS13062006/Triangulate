"""
playground/follow_one_chunk.py : follow ONE chunk and ONE question through the whole embedding pipeline.
Needs: data/chunks.jsonl, data/embeddings.npy, data/embedding_ids.json (from pipeline/step6_embed.py)
Run  : python -m playground.follow_one_chunk
"""
import json
from pathlib import Path

import numpy as np
import ollama

DATA = Path("data")
MODEL = "nomic-embed-text"
CHUNK_ID = "doc_0045_c0"      # change me to follow a different chunk
QUESTION = "Which player says he tried to delete his old tweets before they went viral?"


def embed(text):
    """Embed ONE string. Returns (vector with 768 numbers, tokens the model read)."""
    response = ollama.embed(model=MODEL, input=[text], truncate=False, options={"num_ctx": 2048})
    vector = np.array(response["embeddings"][0], dtype=np.float32)
    tokens = getattr(response, "prompt_eval_count", None) or 0
    return vector, tokens


def unit(vector):
    return vector / np.linalg.norm(vector)


chunks = [json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8")]
matrix = np.load(DATA / "embeddings.npy")
ids = json.load(open(DATA / "embedding_ids.json"))

# ------------------------------------------------------------- A. the chunk
row = ids.index(CHUNK_ID)
chunk = chunks[row]
print(f"A) THE CHUNK  ({CHUNK_ID}) is row {row} of the table")
print("   words in the chunk:", chunk["n_words"])
print("   text starts      :", repr(chunk["text"][:120]))

# ------------------------------------------------------------- B. text sent
sent = "search_document: " + chunk["embed_text"]
print("\nB) THE EXACT STRING SENT TO THE MODEL (prefix + label + text)")
print("  ", repr(sent[:230]))

# ------------------------------------------------------------- C. the vector
fresh, tokens = embed(sent)
print("\nC) WHAT CAME BACK")
print("   tokens the model read     :", tokens, f"(for {chunk['n_words']} words)")
print("   numbers in the vector     :", fresh.shape[0])
print("   first 8 numbers           :", np.round(fresh[:8], 3))
print(f"   smallest / largest number : {fresh.min():.3f} / {fresh.max():.3f}")
print("   (no single number means anything alone; the meaning is in the pattern of all 768)")

# ------------------------------------------------------------- D. same as stored?
stored = matrix[row]
print("\nD) IS THE SAVED ROW THE SAME AS A FRESH EMBEDDING?")
print(f"   similarity(saved row, fresh) = {float(stored @ unit(fresh)):.5f}   (about 1.0 means yes)")

# ------------------------------------------------------------- E. the question
q_sent = "search_query: " + QUESTION
q_vec, q_tokens = embed(q_sent)
q_vec = unit(q_vec)
print("\nE) THE QUESTION")
print("  ", repr(q_sent), f"-> {q_tokens} tokens")

# ------------------------------------------------------------- F. one score by hand
by_hand = sum(float(a) * float(b) for a, b in zip(q_vec, stored))
by_numpy = float(q_vec @ stored)
print("\nF) ONE SCORE, CALCULATED BY HAND (multiply 768 pairs, add them up)")
print(f"   by hand: {by_hand:.5f}    by NumPy: {by_numpy:.5f}")

# ------------------------------------------------------------- G. search all chunks
scores = matrix @ q_vec                       # 6,415 scores in one go
order = np.argsort(-scores)                   # positions, best first
rank_of_chunk = int(np.where(order == row)[0][0]) + 1
print("\nG) SEARCHING ALL", f"{len(chunks):,}", "CHUNKS")
print(f"   lowest / average / highest score: {scores.min():.3f} / {scores.mean():.3f} / {scores.max():.3f}")
print(f"   our chunk {CHUNK_ID} came in at rank {rank_of_chunk} with score {scores[row]:.3f}")
print("   top 3:")
for i in order[:3]:
    print(f"     {scores[i]:.3f}  {chunks[i]['chunk_id']:<13} {chunks[i]['title'][:60]}")