# pipeline/step1_explore.py : first look at the raw MultiHop-RAG files.
# Run (from the repository root):  python -m pipeline.step1_explore
import json
from collections import Counter
from statistics import mean, median

CORPUS_PATH = "data/corpus.json"
QUERIES_PATH = "data/MultiHopRAG.json"

# ---- 1. Load both files -------------------------------------------------
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)       # expected: a list of dicts, one per article

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)      # expected: a list of dicts, one per question

print("Types:", type(corpus).__name__, type(queries).__name__)
print(f"Articles:  {len(corpus):,}")
print(f"Questions: {len(queries):,}")

# ---- 2. Look at the structure of one article ---------------------------
doc = corpus[1]
print("\nFields in an article:", list(doc.keys()))
for key, value in doc.items():
    if key == "body":
        text = value or ""
        print(f"  body: {text[:200]!r} ... ({len(text):,} characters)")
    else:
        print(f"  {key}: {value!r}")

# ---- 3. How long are the articles? -------------------------------------
word_counts = [len((d.get("body") or "").split()) for d in corpus]
print("\nWords per article (min / median / mean / max):",
      min(word_counts), median(word_counts), round(mean(word_counts)), max(word_counts))

# ---- 4. Is the metadata complete? ---------------------------------------
print("\nMissing values per field:")
for field in ["title", "author", "published_at", "source", "category", "url"]:
    missing = sum(1 for d in corpus if not d.get(field))
    print(f"  {field}: {missing}")

# ---- 5. Who wrote it, when, and about what? -----------------------------
print("\nTop sources:", Counter(d.get("source") for d in corpus).most_common(8))
print("Categories:", dict(Counter(d.get("category") for d in corpus)))
dates = sorted(d["published_at"] for d in corpus if d.get("published_at"))
print("Date range:", dates[0], "->", dates[-1])

# ---- 6. Any duplicates? -------------------------------------------------
urls = [d.get("url") for d in corpus]
titles = [d.get("title") for d in corpus]
print(f"\nUnique URLs:   {len(set(urls))} of {len(urls)}")
print(f"Unique titles: {len(set(titles))} of {len(titles)}")

# ---- 7. The questions ---------------------------------------------------
print("\nQuestion types:", dict(Counter(q["question_type"] for q in queries)))
ev_counts = Counter(len(q.get("evidence_list") or []) for q in queries)
print("Articles of evidence per question:", sorted(ev_counts.items()))

# ---- 8. Can every piece of evidence be found in the corpus? -------------
corpus_urls = set(urls)
total = found = 0
for q in queries:
    for ev in q.get("evidence_list") or []:
        total += 1
        if ev.get("url") in corpus_urls:
            found += 1
print(f"\nEvidence items whose URL exists in the corpus: {found:,} / {total:,}")

# ---- 9. One example of each question type -------------------------------
seen = set()
for q in queries:
    t = q["question_type"]
    if t not in seen:
        seen.add(t)
        print(f"\n[{t}]\nQ: {q['query']}\nA: {q['answer']}")