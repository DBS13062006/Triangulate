# pipeline/step5_chunk.py : paragraph-aware chunking of the cleaned corpus -> data/chunks.jsonl
# Run (from the repository root):  python -m pipeline.step5_chunk
import json
import re
from collections import Counter

# =============================================================================
# SETTINGS (the knobs we will tune later against retrieval results)
# =============================================================================
TARGET_WORDS = 200        # keep adding paragraphs to a chunk until it would pass this size
MAX_PARA_WORDS = 200      # a paragraph longer than this is cut into sentence groups first
MIN_CHUNK_WORDS = 60      # never close a chunk smaller than this, even if the next paragraph overshoots
MIN_LAST_WORDS = 40       # a leftover last chunk smaller than this is glued onto the previous one
OVERLAP_MIN_WORDS = 8     # the sentence we repeat at the start of the next chunk must have
OVERLAP_MAX_WORDS = 40    #   between 8 and 40 words (not a fragment, not a monster)

# A sentence ends at . ! ? or a closing quote, then spaces, then a capital letter/digit/opening quote.
# (Imperfect: it also splits after 'U.S.' or 'Mr.'. We only use it on giant paragraphs and for overlap.)
SENT_SPLIT = re.compile(r'(?<=[.!?”’"])\s+(?=[“"‘(]?[A-Z0-9])')

def n_words(text):
    return len(text.split())

# ---- step A: turn a text into 'units' (paragraphs, or sentence groups for giant paragraphs) ----
def split_long_paragraph(paragraph, limit):
    pieces, current, current_words = [], [], 0
    for sentence in SENT_SPLIT.split(paragraph):
        w = n_words(sentence)
        if current and current_words + w > limit:
            pieces.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += w
    if current:
        pieces.append(" ".join(current))
    return pieces

def make_units(text):
    units, giant = [], 0
    for paragraph in text.split("\n\n"):
        if n_words(paragraph) > MAX_PARA_WORDS:
            giant += 1
            units.extend(split_long_paragraph(paragraph, MAX_PARA_WORDS))
        else:
            units.append(paragraph)
    return units, giant

# ---- step B: pack units into chunks of about TARGET_WORDS ----------------------------------
def pack(units):
    chunks, current, current_words = [], [], 0
    for unit in units:
        w = n_words(unit)
        if current_words >= MIN_CHUNK_WORDS and current_words + w > TARGET_WORDS:
            chunks.append(current)
            current, current_words = [], 0
        current.append(unit)
        current_words += w
    if current:
        chunks.append(current)
    # a tiny leftover at the end is glued onto the previous chunk
    if len(chunks) > 1 and sum(n_words(u) for u in chunks[-1]) < MIN_LAST_WORDS:
        chunks[-2].extend(chunks.pop())
    return ["\n\n".join(c) for c in chunks]

# ---- step C: overlap = repeat the last sentence of the previous chunk -----------------------
def overlap_sentence(previous_core):
    last = SENT_SPLIT.split(previous_core.replace("\n\n", " "))[-1]
    return last if OVERLAP_MIN_WORDS <= n_words(last) <= OVERLAP_MAX_WORDS else ""

# =============================================================================
# RUN
# =============================================================================
with open("data/corpus_clean.jsonl", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]
with open("data/MultiHopRAG.json", encoding="utf-8") as f:
    queries = json.load(f)

chunks, cores_by_doc = [], {}
giant_paragraphs = 0
for doc in docs:
    units, giant = make_units(doc["text"])
    giant_paragraphs += giant
    cores = pack(units)
    cores_by_doc[doc["doc_id"]] = cores
    header = f'{doc["title"]} | {doc["source"]} | {doc["published_at"][:10]}'
    for i, core in enumerate(cores):
        carried = overlap_sentence(cores[i - 1]) if i > 0 else ""
        text = (carried + "\n\n" + core) if carried else core
        chunks.append({
            "chunk_id": f'{doc["doc_id"]}_c{i}',
            "doc_id": doc["doc_id"],
            "chunk_index": i,
            "n_chunks_in_doc": len(cores),
            "url": doc["url"],
            "title": doc["title"],
            "source": doc["source"],
            "author": doc["author"],
            "published_at": doc["published_at"],
            "category": doc["category"],
            "text": text,                                  # what the LLM will read
            "embed_text": header + "\n" + text,            # what the embedding model will read
            "n_words": n_words(text),
            "has_overlap": bool(carried),
        })

with open("data/chunks.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# =============================================================================
# REPORTS
# =============================================================================
def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p / 100))]

sizes = [c["n_words"] for c in chunks]
per_doc = Counter(c["doc_id"] for c in chunks)
print(f"Articles: {len(docs)}   Chunks: {len(chunks):,}   (giant paragraphs split by sentence: {giant_paragraphs})")
print(f"Chunks per article: min {min(per_doc.values())}, median {pct(list(per_doc.values()), 50)}, max {max(per_doc.values())}")
print(f"Words per chunk:    min {min(sizes)}, p10 {pct(sizes, 10)}, median {pct(sizes, 50)}, p90 {pct(sizes, 90)}, max {max(sizes)}")
print(f"Chunks starting with an overlap sentence: {sum(c['has_overlap'] for c in chunks):,}")
print(f"Longest embed_text is about {int(max(n_words(c['embed_text']) for c in chunks) * 1.4)} tokens "
      f"(the embedding model's limit is 512)")

print("\nChunk size histogram (words):")
for lo in range(0, 360, 40):
    n = sum(lo <= s < lo + 40 for s in sizes)
    print(f"  {lo:3d}-{lo + 39:3d} | {'#' * (n // 40)} {n}")
print(f"  360+    | {sum(s >= 360 for s in sizes)}")

# ---- check 1: lossless. Gluing the chunk cores back together must give the original article ----
bad = [d["doc_id"] for d in docs if " ".join(cores_by_doc[d["doc_id"]]).split() != d["text"].split()]
print(f"\nCheck 1 (nothing lost or invented): articles whose chunks do not rebuild the original = {len(bad)}")

# ---- check 2: every evidence sentence must sit completely inside at least one chunk ----------
squash = lambda s: " ".join(s.replace("\ufeff", "").split())
chunks_of_url = {}
for c in chunks:
    chunks_of_url.setdefault(c["url"], []).append(squash(c["text"]))
total = found = 0
broken = []
for q in queries:
    for ev in q.get("evidence_list") or []:
        total += 1
        fact = squash(ev["fact"])
        if any(fact in t for t in chunks_of_url[ev["url"]]):
            found += 1
        else:
            broken.append(fact)
print(f"Check 2 (evidence sentences intact inside one chunk): {found:,}/{total:,}")
for fact in broken[:3]:
    print("   cut in half:", fact[:110])

# ---- look at real chunks ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("THE FIRST THREE CHUNKS OF doc_0045 (Travis Kelce article)")
for c in [c for c in chunks if c["doc_id"] == "doc_0045"][:3]:
    print(f'\n[{c["chunk_id"]}]  {c["n_words"]} words  overlap={c["has_overlap"]}')
    print("  embed_text starts:", repr(c["embed_text"][:150]))
    print("  text ends:        ", repr(c["text"][-110:]))