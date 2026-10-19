# pipeline/step3_find_junk.py : find web-page junk (ads, newsletter boxes, share bars) repeated across articles.
# Usage:  python -m pipeline.step3_find_junk                            -> scan the RAW corpus
#         python -m pipeline.step3_find_junk data/corpus_clean.jsonl    -> scan the CLEANED corpus (what is left?)
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.json"

# ---- load: works for the raw .json and for our cleaned .jsonl ----------------
if path.endswith(".jsonl"):
    with open(path, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    docs = [{"source": d["source"], "body": d["text"]} for d in docs]
else:
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
print(f"Scanning {path}: {len(docs)} articles\n")

def paragraphs(body):
    # In this dataset a paragraph break is a newline (checked: 609/609 articles contain one)
    return [p.strip() for p in re.split(r"\n+", body) if p.strip()]

# ---- 1. The same paragraph in many articles ------------------------------------
where = defaultdict(set)                     # paragraph text -> set of article numbers
for i, d in enumerate(docs):
    for p in paragraphs(d["body"]):
        where[p].add(i)

print("=" * 78)
print("1. PARAGRAPHS that appear word-for-word in 3+ articles")
print("   ONE outlet only -> probably that outlet's boilerplate.  Many outlets -> look closer.")
print("=" * 78)
rows = sorted(((len(s), p) for p, s in where.items() if len(s) >= 3), reverse=True)
for n, p in rows[:70]:
    outlets = Counter(docs[i]["source"] for i in where[p])
    top = outlets.most_common(1)[0][0][:22]
    print(f"{n:3d} articles | {len(outlets)} outlet(s) | {top:22s} | {p[:90]!r}")

# ---- 2. Paragraph 'families': same first 30 characters, same outlet -------------
print("\n" + "=" * 78)
print("2. PARAGRAPH FAMILIES: same outlet + same first 30 characters in 3+ articles")
print("   (catches boilerplate whose ending changes, e.g. 'Football email {{ ... }}')")
print("=" * 78)
fam = defaultdict(set)
for i, d in enumerate(docs):
    for p in paragraphs(d["body"]):
        fam[(d["source"], p[:30])].add(i)
for (src, start), s in sorted(fam.items(), key=lambda kv: -len(kv[1])):
    if len(s) >= 3 and len(start) >= 12:
        print(f"{len(s):3d} articles | {src[:24]:24s} | {start!r}")

# ---- 3. Do articles open or close with the same words? --------------------------
print("\n" + "=" * 78)
print("3. COMMON ARTICLE OPENINGS and ENDINGS (first / last paragraph, 50 chars)")
print("=" * 78)
for label, idx in [("OPENINGS", 0), ("ENDINGS", -1)]:
    c = Counter(paragraphs(d["body"])[idx][:50] for d in docs if paragraphs(d["body"]))
    print(label)
    for text, n in c.most_common(8):
        if n >= 2:
            print(f"  {n:3d}x  {text!r}")

# ---- 4. Suspect phrases: counts AND real contexts --------------------------------
print("\n" + "=" * 78)
print("4. SUSPECT PHRASES (articles containing them) + 2 contexts each")
print("=" * 78)
SUSPECTS = ["Advertisement", "ADVERTISEMENT", "CLICK HERE", "newsletter", "Sign up", "Subscribe",
            "{{", "Loading", "Table of Contents", "affiliate", "commission", "©",
            "Related Pages", "MORE:", "Also Read", "Photograph:", "Email (required)"]
for ph in SUSPECTS:
    hits = [(d["source"], d["body"], m.start()) for d in docs for m in re.finditer(re.escape(ph), d["body"])]
    n_art = len({id(b) for _, b, _ in hits})
    print(f"\n{ph!r}: {len(hits)} times in {n_art} articles")
    for src, body, pos in hits[:2]:
        print(f"    [{src[:18]}] ...{body[max(0, pos - 50): pos + 70]!r}...")

# ---- 5. Which outlets use the word 'Loading'? -------------------------------------
print("\n" + "=" * 78)
print("5. 'Loading' (a web-page placeholder) -- where is it, and where inside the text?")
print("=" * 78)
where_loading = Counter()
for d in docs:
    for m in re.finditer(r"\bLoading\b", d["body"]):
        at_start = m.start() == 0 or d["body"][m.start() - 1] == "\n"
        where_loading[(d["source"], "start of paragraph" if at_start else "in the middle of a paragraph")] += 1
for (src, kind), n in where_loading.most_common():
    print(f"  {n:3d}  {src:32s} {kind}")

# ---- 6. Unusual characters ---------------------------------------------------------
print("\n" + "=" * 78)
print("6. UNUSUAL CHARACTERS (top 12)")
print("=" * 78)
SAFE = set(" .,;:!?'\"()-$%&/\n")
census = Counter(ch for d in docs for ch in d["body"] if not (ch.isascii() and (ch.isalnum() or ch in SAFE)))
for ch, n in census.most_common(12):
    print(f"  {n:6d}  U+{ord(ch):04X}  {unicodedata.name(ch, '?'):38s} {ch!r}")
print(f"  invisible U+FEFF appears {census.get(chr(0xFEFF), 0)} times")