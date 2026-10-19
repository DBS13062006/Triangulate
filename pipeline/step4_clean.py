# pipeline/step4_clean.py : remove web-page junk paragraph by paragraph (rules written from a scan of all 609 articles).
# Reads data/corpus.json + data/MultiHopRAG.json, writes data/corpus_clean.jsonl and data/removed_log.txt
# Run (from the repository root):  python -m pipeline.step4_clean
import json
import re
from collections import Counter

# =============================================================================
# RULES.  Every rule acts on ONE PARAGRAPH at a time.  Edit only this block.
# TIER 1 = web-page chrome (never part of the story)       -> always on
# TIER 2 = judgement calls (credits, cross-links, disclosures) -> switch off with USE_TIER2 = False
# =============================================================================
USE_TIER2 = True

# --- TIER 1 ------------------------------------------------------------------
# (a) drop the paragraph if it is EXACTLY this text
DROP_EXACT = {
    "Advertisement", "ADVERTISEMENT", "ADVERTISEMENT Article continues below this ad",
    "Facebook Twitter Whatsapp Reddit Email Share",
    "America’s Favorite Video Today",
    "CLICK HERE TO GET THE FOX NEWS APP", "APP USERS CLICK HERE", "APP USERS CLICK HERE TO VIEW POST",
    "Related Pages:", "Table of Contents",
    "Loading . . .", "Like this: Like Loading...",
    "watch now", "Most read",
}
# (b) drop the paragraph if it STARTS with this text
DROP_STARTS = [
    "CLICK HERE TO SIGN UP FOR", "LIKE WHAT YOU’RE READING? CLICK HERE",
    "Read this article for free! Plus get unlimited access",
    "Stay ahead of the trend in fashion and beyond",                  # Independent newsletter box
    "Sign up to Miguel Delaney", "Sign up to Simon Calder",
    "skip past newsletter promotion",                                 # Guardian newsletter box
    "Quick Guide How do I sign up for sport breaking news alerts?",   # Guardian app-help box (4 paragraphs)
    "If you already have the Guardian app, make sure you’re on the most recent version.",
    "In the Guardian app, tap the Menu button at the bottom right",
    "Turn on sport notifications. Was this helpful?",
    "Table of Contents ",                                             # Mashable menu glued into one line
    "Mashable Deals Want more hand-picked deals",
    "Verge Deals / Sign up for Verge Deals", "Verge Deals on X (formerly known as Twitter)",
    "Educators and Parents, Sign Up for The Cheat Sheet",
    "TC+ subscribers get access",
    "Pick Six Newsletter Crafted By The Best NFL Experts",
    "This Instagram post cannot be displayed in your browser",
]
# (c) drop the paragraph if it CONTAINS this text anywhere
DROP_CONTAINS = ["{{ #verifyErrors }}", "Please enter a valid email address", "Email (required)"]
# (d) cut this glued-on header off the START of a paragraph, keep the rest (The Age / SMH)
STRIP_HEADER = re.compile(
    r"^Save Log in , register or subscribe to save articles for later\. Save articles for later "
    r"Add articles to your saved list and come back to them any time\. Got it "
    r"Normal text size Larger text size Very large text size\s*")
# (e) 'Loading' is a leftover placeholder in these outlets: delete the word everywhere
LOADING_OUTLETS = {"The Age", "The Sydney Morning Herald"}
# (f) fix the 'drop-cap' glitch:  'ikeLike many ...' -> 'Like many ...'  (see fix_dropcap)

# --- TIER 2 ------------------------------------------------------------------
TIER2_STARTS = [
    "Engadget has been testing and reviewing consumer tech since 2004",
    "All products recommended by Engadget are selected by our editorial team",
    "This article contains affilate links", "Follow @EngadgetDeals on Twitter",
    "Your Black Friday Shopping Guide: See all of Yahoo",
    "If you buy something from a Polygon link, Vox Media may earn a commission",
    "Polygon Recommends is our way of endorsing",
    "Your time is precious, and your options are endless",
    "If you buy something using links in our stories",
    "© Copyright of Globes", "Published by Globes, Israel", "© AAP",
    "MORE: Win BIG with SuperDraft", "Win BIG with SuperDraft", "MORE SUPERDRAFT DFS",
    "Please note that card details are accurate as of the publish date",
]
TIER2_CONTAINS = ["Vox Media has affiliate partnerships",
                  "This article contains links to affiliate partners"]
TIER2_REGEX = [                                   # short cross-link headlines and credit lines
    re.compile(r"^(MORE|More|See more|SEE MORE|Also Read|ALSO READ)\s*:.{0,220}$"),
    re.compile(r"^Photograph:\s.{0,60}$"),
    re.compile(r"^\(Getty Images\)$"),
]
# =============================================================================

def normalize(text):
    text = text.replace("\ufeff", "").replace("\u00a0", " ")        # invisible char, non-breaking space
    text = re.sub(r"[\u200b\u200c\u200d]", "", text)
    return text

def fix_dropcap(p):
    # A web-page glitch repeats a word's tail:  'tudentsStudents' = 'tudents' + 'Students'
    m = re.match(r"^([a-z]{2,})([A-Z][a-z]+)", p)
    if m and m.group(1) == m.group(2)[1:]:
        return m.group(2) + p[m.end():], True
    return p, False

def clean_article(body, source, log, doc_id):
    kept = []
    for p in re.split(r"\n+", normalize(body)):
        p = re.sub(r"[ \t]+", " ", p).strip()
        if not p:
            continue
        m = STRIP_HEADER.match(p)
        if m:
            log.append((doc_id, "strip_header", m.group(0)[:50]))
            p = p[m.end():]
        if p in DROP_EXACT:
            log.append((doc_id, "drop_exact", p)); continue
        if any(p.startswith(s) for s in DROP_STARTS):
            log.append((doc_id, "drop_starts", p)); continue
        if any(s in p for s in DROP_CONTAINS):
            log.append((doc_id, "drop_contains", p)); continue
        if USE_TIER2:
            if (any(p.startswith(s) for s in TIER2_STARTS) or any(s in p for s in TIER2_CONTAINS)
                    or any(r.match(p) for r in TIER2_REGEX)):
                log.append((doc_id, "drop_tier2", p)); continue
        if source in LOADING_OUTLETS and re.search(r"\bLoading\b", p):
            log.append((doc_id, "remove_loading_word", p[:80]))
            p = re.sub(r"\s*\bLoading\b\s*", " ", p).strip()
        p, changed = fix_dropcap(p)
        if changed:
            log.append((doc_id, "fix_dropcap", p[:50]))
        if p:
            kept.append(p)
    return "\n\n".join(kept)

# ---- load --------------------------------------------------------------------
with open("data/corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)
with open("data/MultiHopRAG.json", encoding="utf-8") as f:
    queries = json.load(f)

# ---- clean and package as documents ----------------------------------------------
log, docs = [], []
for i, d in enumerate(corpus):
    doc_id = f"doc_{i:04d}"
    docs.append({
        "doc_id": doc_id, "title": d["title"], "source": d["source"],
        "author": d.get("author"), "published_at": d["published_at"],
        "category": d["category"], "url": d["url"],
        "text": clean_article(d.get("body") or "", d["source"], log, doc_id),
    })

with open("data/corpus_clean.jsonl", "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
with open("data/removed_log.txt", "w", encoding="utf-8") as f:
    for doc_id, rule, text in log:
        f.write(f"{doc_id} | {rule} | {text[:300]!r}\n")

# ---- report 1: what did we change? ------------------------------------------------
w_before = sum(len(d["body"].split()) for d in corpus)
w_after = sum(len(x["text"].split()) for x in docs)
print(f"Words before: {w_before:,}   after: {w_after:,}   removed: {w_before - w_after:,} "
      f"({100 * (w_before - w_after) / w_before:.2f}%)")
print("Actions per rule:", dict(Counter(rule for _, rule, _ in log)))
print("Articles touched:", len({doc_id for doc_id, _, _ in log}), "of", len(docs))
print("Empty articles after cleaning:", sum(1 for x in docs if not x["text"].strip()))

shrink = sorted(((1 - len(x["text"].split()) / max(1, len(d["body"].split())), d["source"], d["title"])
                 for d, x in zip(corpus, docs)), reverse=True)[:5]
print("\nArticles that lost the biggest share of words (read these!):")
for frac, src, title in shrink:
    print(f"  {100 * frac:5.1f}%  [{src[:18]}] {title[:70]}")

# ---- report 2: the safety net --------------------------------------------------------
# Compare facts and articles with whitespace squeezed AND invisible characters removed on BOTH sides.
squash = lambda s: " ".join(normalize(s).split())
orig_text = {d["url"]: squash(d["body"]) for d in corpus}
new_text = {x["url"]: squash(x["text"]) for x in docs}

total = before = after = 0
lost = []
for q in queries:
    for ev in q.get("evidence_list") or []:
        total += 1
        fact = squash(ev["fact"])
        in_before = fact in orig_text[ev["url"]]
        in_after = fact in new_text[ev["url"]]
        before += in_before
        after += in_after
        if in_before and not in_after:
            lost.append((ev["url"], ev["fact"]))
print(f"\nEvidence sentences found in their article:  before {before:,}/{total:,}   after {after:,}/{total:,}")
print("Evidence destroyed by cleaning:", len(lost))
for url, fact in lost[:5]:
    print("   ", fact[:110])