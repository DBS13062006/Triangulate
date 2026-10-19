"""
playground/embeddings_playground.py : play with embeddings on a few sentences.
No files needed. Run:  python -m playground.embeddings_playground
"""
import numpy as np
import ollama

MODEL = "nomic-embed-text"


def embed(texts):
    """Turn a list of strings into a NumPy table: one row (768 numbers) per string."""
    response = ollama.embed(
        model=MODEL,
        input=texts,
        truncate=False,              # if a text is too long, raise an error instead of silently cutting it
        options={"num_ctx": 2048},   # how many tokens the model may read at once
    )
    return np.array(response["embeddings"], dtype=np.float32)


def normalise(table):
    """Scale every row so its length is exactly 1 (then dot product == cosine similarity)."""
    lengths = np.linalg.norm(table, axis=1, keepdims=True)
    return table / lengths


# ---- three "documents" and one "question" -------------------------------
docs = [
    "Sam Bankman-Fried, the founder of FTX, is on trial for fraud.",             # A
    "The crypto exchange's former boss faces a criminal court case.",            # B (same meaning, different words)
    "The new Pixel 8 phone has a much better camera.",                           # C (unrelated)
]
question = "Who is on trial over the collapse of FTX?"

# ---- 1. embed WITH the prefixes nomic-embed-text expects ----------------
try:
    raw_docs = embed(["search_document: " + d for d in docs])
    raw_question = embed(["search_query: " + question])
except Exception as error:
    print("Could not get embeddings from Ollama:", error)
    print("Is the Ollama app running? (look for the llama icon in the menu bar)")
    raise SystemExit(1)

print("1) WHAT AN EMBEDDING LOOKS LIKE")
print("   table shape (texts, numbers per text):", raw_docs.shape)
print("   first 6 numbers of document A:", np.round(raw_docs[0][:6], 4))
print("   length of that vector before scaling:", round(float(np.linalg.norm(raw_docs[0])), 3))

doc_vecs = normalise(raw_docs)
q_vec = normalise(raw_question)[0]

# ---- 2. similarity between documents -------------------------------------
print("\n2) HOW SIMILAR ARE THE DOCUMENTS TO EACH OTHER? (1.0 = identical meaning)")
sim = doc_vecs @ doc_vecs.T          # every document against every document
labels = ["A", "B", "C"]
for i in range(3):
    for j in range(i + 1, 3):
        print(f"   {labels[i]} vs {labels[j]}: {sim[i, j]:.3f}")

# ---- 3. question against documents ---------------------------------------
print("\n3) QUESTION vs EACH DOCUMENT (this is the heart of RAG search)")
print("   question:", question)
scores = doc_vecs @ q_vec
for i in np.argsort(-scores):        # best first
    print(f"   {scores[i]:.3f}  {labels[i]}: {docs[i]}")

# ---- 4. what happens WITHOUT the prefixes? --------------------------------
plain_docs = normalise(embed(docs))
plain_q = normalise(embed([question]))[0]
plain_scores = plain_docs @ plain_q
print("\n4) SAME THING WITHOUT THE PREFIXES")
print("   with prefixes   :", np.round(scores, 3))
print("   without prefixes:", np.round(plain_scores, 3))