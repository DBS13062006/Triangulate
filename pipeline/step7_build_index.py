"""
pipeline/step7_build_index.py : build the search database (run once, or again whenever chunks/embeddings change).

Reads : data/chunks.jsonl, data/embeddings.npy, data/embedding_ids.json
Writes: data/lancedb/   (a folder; this folder IS the database)
        - a table with every chunk, its metadata and its vector
        - a keyword (BM25) index over the chunk text + its label
Run   : python -m pipeline.step7_build_index     (from the repository root)
"""
import json
import random
import warnings

import lancedb
import numpy as np
import pyarrow as pa

from config import DATA, DB_DIR, TABLE

# ---------------------------------------------------------------- load
chunks = [json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8")]
matrix = np.load(DATA / "embeddings.npy").astype(np.float32)
saved_ids = json.load(open(DATA / "embedding_ids.json"))
if saved_ids != [c["chunk_id"] for c in chunks]:
    raise SystemExit("embeddings.npy and chunks.jsonl are out of sync. Re-run  python -m pipeline.step6_embed")
dim = matrix.shape[1]
print(f"{len(chunks):,} chunks, {dim} numbers per vector")

# ---------------------------------------------------------------- the table's columns
schema = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("url", pa.string()),
    pa.field("title", pa.string()),
    pa.field("source", pa.string()),
    pa.field("author", pa.string()),
    pa.field("published_at", pa.string()),         # 'YYYY-MM-DD', easy to filter by date
    pa.field("category", pa.string()),
    pa.field("text", pa.string()),                 # what the language model will read
    pa.field("embed_text", pa.string()),           # label + text (what we embedded and keyword-index)
    pa.field("n_words", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), dim)),
])
vectors = pa.FixedSizeListArray.from_arrays(pa.array(matrix.ravel(), type=pa.float32()), dim)
table_data = pa.table(
    {
        "chunk_id": [c["chunk_id"] for c in chunks],
        "doc_id": [c["doc_id"] for c in chunks],
        "chunk_index": [c["chunk_index"] for c in chunks],
        "url": [c["url"] for c in chunks],
        "title": [c["title"] for c in chunks],
        "source": [c["source"] for c in chunks],
        "author": [c["author"] or "" for c in chunks],
        "published_at": [c["published_at"][:10] for c in chunks],
        "category": [c["category"] for c in chunks],
        "text": [c["text"] for c in chunks],
        "embed_text": [c["embed_text"] for c in chunks],
        "n_words": [c["n_words"] for c in chunks],
        "vector": vectors,
    },
    schema=schema,
)

# ---------------------------------------------------------------- create the database + keyword index
db = lancedb.connect(str(DB_DIR))
table = db.create_table(TABLE, data=table_data, mode="overwrite")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                # LanceDB marks this call 'deprecated' but it is the documented one
    table.create_fts_index("embed_text", replace=True)   # BM25 keyword index (lower-case, stemming, stop-words removed)
print(f"Created table '{TABLE}' with {table.count_rows():,} rows and a keyword index in {DB_DIR}/")

# ---------------------------------------------------------------- CHECK 1: meaning search == plain NumPy?
print("\nCHECK 1: database vs plain NumPy search, 100 random chunks used as questions")
random.seed(1)
same_set = same_order = 0
sample = random.sample(range(len(chunks)), 100)
for row_number in sample:
    q = matrix[row_number]
    numpy_ids = [chunks[i]["chunk_id"] for i in np.argsort(-(matrix @ q))[:5]]
    found = table.search(q).metric("cosine").limit(5).select(["chunk_id", "_distance"]).to_list()
    lance_ids = [r["chunk_id"] for r in found]
    same_set += set(numpy_ids) == set(lance_ids)
    same_order += numpy_ids == lance_ids
print(f"  same 5 chunks     : {same_set}/100")
print(f"  same order as well: {same_order}/100   (tiny score ties can swap neighbours; harmless)")

# ---------------------------------------------------------------- CHECK 2: filters
print("\nCHECK 2: filters keep only matching rows")
q = matrix[sample[0]]
hits = table.search(q).metric("cosine").where("source = 'The Verge'", prefilter=True).limit(10).select(["source", "_distance"]).to_list()
print("  asked for The Verge, got sources:", sorted({h["source"] for h in hits}), f"({len(hits)} rows)")
hits = table.search(q).metric("cosine").where("published_at >= '2023-12-01'", prefilter=True).limit(10).select(["published_at", "_distance"]).to_list()
print("  asked for 2023-12-01 or later, earliest date returned:", min(h["published_at"] for h in hits))

# ---------------------------------------------------------------- CHECK 3: keyword search
print("\nCHECK 3: keyword search finds a rare word")
hits = table.search("Bankman-Fried", query_type="fts").select(["chunk_id", "text", "_score"]).limit(5).to_list()
print("  top 5 chunks for 'Bankman-Fried' that really contain the word:",
      sum("Bankman" in h["text"] for h in hits), "/ 5")
