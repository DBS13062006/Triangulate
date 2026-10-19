# pipeline/step2_charts.py : draw the dataset charts into docs/figures/.
# Run (from the repository root):  python -m pipeline.step2_charts
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")            # draw into image files, no pop-up windows
import matplotlib.pyplot as plt

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", FIG_DIR / name)

# ---- load and turn the lists of dicts into tables --------------------------
with open("data/corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)
with open("data/MultiHopRAG.json", encoding="utf-8") as f:
    queries = json.load(f)

articles = pd.DataFrame(corpus)
articles["words"] = articles["body"].fillna("").str.split().str.len()
articles["published_at"] = pd.to_datetime(articles["published_at"], utc=True)

qdf = pd.DataFrame(queries)
qdf["n_evidence"] = qdf["evidence_list"].apply(lambda ev: len(ev or []))

# ---- 1. How long are the articles? ----------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(articles["words"].clip(upper=4000), bins=40)
ax.axvline(articles["words"].median(), linestyle="--", label=f"median {articles['words'].median():.0f}")
ax.axvline(articles["words"].mean(), linestyle=":", label=f"mean {articles['words'].mean():.0f}")
ax.set_title("Words per article (longer ones capped at 4,000)")
ax.set_xlabel("words")
ax.set_ylabel("number of articles")
ax.legend()
save(fig, "1_article_length.png")
print("Articles longer than 4,000 words:", int((articles["words"] > 4000).sum()))

# ---- 2. Which outlets? ------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
articles["source"].value_counts().head(12).sort_values().plot.barh(ax=ax)
ax.set_title("Top 12 sources")
ax.set_xlabel("number of articles")
save(fig, "2_sources.png")

# ---- 3. Which categories? ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
articles["category"].value_counts().plot.bar(ax=ax)
ax.set_title("Articles per category")
ax.set_ylabel("number of articles")
plt.setp(ax.get_xticklabels(), rotation=0)
save(fig, "3_categories.png")

# ---- 4. When were they published? ------------------------------------------
weekly = articles.set_index("published_at").resample("W").size()
fig, ax = plt.subplots(figsize=(8, 4.5))
weekly.plot(ax=ax, marker="o")
ax.set_title("Articles published per week")
ax.set_ylabel("number of articles")
save(fig, "4_timeline.png")

# ---- 5. Question types and how many articles each needs --------------------
table = pd.crosstab(qdf["question_type"], qdf["n_evidence"])
print("\nQuestions by type and number of evidence articles:\n", table)
fig, ax = plt.subplots(figsize=(8, 4.5))
table.plot.bar(stacked=True, ax=ax)
ax.set_title("Question types, split by articles of evidence needed")
ax.set_ylabel("number of questions")
ax.legend(title="articles of evidence")
plt.setp(ax.get_xticklabels(), rotation=0)
save(fig, "5_question_types.png")

# ---- 6. How often is each article used as evidence? ------------------------
cited = Counter(ev["url"] for q in queries for ev in (q.get("evidence_list") or []))
articles["times_cited"] = articles["url"].map(cited).fillna(0).astype(int)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(articles["times_cited"], bins=range(0, int(articles["times_cited"].max()) + 2))
ax.set_title("How many questions cite each article")
ax.set_xlabel("questions citing the article")
ax.set_ylabel("number of articles")
save(fig, "6_citations.png")
print("\nArticles never cited:", int((articles["times_cited"] == 0).sum()))
print(articles.sort_values("times_cited", ascending=False)
      [["times_cited", "source", "title"]].head(5).to_string(index=False))

# ---- 7. What do answers look like? -----------------------------------------
print("\nMost common answers:\n", qdf["answer"].value_counts().head(10))