"""
rag/retriever.py : the search engine. Not run directly; other scripts do `from rag.retriever import Retriever`.

Six ways to search (the `mode`):
  dense          meaning search: the question's vector vs every chunk vector
  keyword        BM25 keyword search (exact words, names, numbers, outlet names)
  hybrid         run both, then merge the two rankings with Reciprocal Rank Fusion (RRF)
  hybrid_rerank  hybrid, then a second model re-reads the best candidates one by one and re-sorts them
  outlet_rerank  hybrid_rerank + reserved seats for every news outlet the question names ("The Verge", ...)
  smart_rerank   outlet_rerank + the same trick for DATES the question names ("October 13th, 2023")
                 + one seat for each CLAUSE of the question (the default mode)
"""
import datetime
import math
import re
from collections import Counter

import lancedb
import numpy as np
import ollama

from config import (CANDIDATES, CLAUSE_QUOTA, DATE_POOL, DATE_QUOTA, DB_DIR, DEFAULT_YEAR, EMBED_CTX, EMBED_MODEL, MAX_PER_DOC,
                    OUTLET_ALIASES, OUTLET_POOL, OUTLET_QUOTA, QUERY_PREFIX, RERANK_BATCH, RERANK_MAX_LENGTH,
                    RERANK_MODEL, RERANK_POOL, RRF_K, TABLE)

MODES = ("dense", "keyword", "hybrid", "hybrid_rerank", "outlet_rerank", "smart_rerank")
COLUMNS = ["chunk_id", "doc_id", "chunk_index", "url", "title", "source", "author",
           "published_at", "category", "text", "embed_text", "n_words"]


def quote(value):
    """Make a text value safe inside a filter:  O'Brien -> 'O''Brien'"""
    return "'" + str(value).replace("'", "''") + "'"


def build_filter(source=None, after=None, before=None):
    conditions = []
    if source:
        conditions.append(f"source = {quote(source)}")
    if after:
        conditions.append(f"published_at >= {quote(after)}")
    if before:
        conditions.append(f"published_at <= {quote(before)}")
    return " AND ".join(conditions) if conditions else None


def day_filter(day, source=None):
    """All chunks published on one calendar day ('2023-10-04'), optionally only from one outlet."""
    return build_filter(source, after=day, before=day + "T99")      # '2023-10-04T23:32' <= '2023-10-04T99'


# ---------------------------------------------------------------- spotting outlet names in a question
def name_variants(source):
    """Ways a question may write an outlet: 'Cnbc | World Business News Leader' is also written 'Cnbc'."""
    names = {source}
    for separator in (" | ", ":"):
        if separator in source:
            names.add(source.split(separator)[0].strip())
    if source.startswith("The "):
        names.add(source[4:])
    names.update(alias for alias, full in OUTLET_ALIASES.items() if full == source)
    return names


def find_outlets(question, sources):
    """
    Which outlets does the question name? Whole-word and CASE-SENSITIVE on purpose, so that
    'the age of 16' does not count as the outlet 'The Age'.
    """
    hits = {}                                            # outlet -> the text that matched
    for source in sources:
        for name in sorted(name_variants(source), key=len, reverse=True):
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question):
                hits[source] = name
                break
    # 'The Independent - Travel' must not also trigger a shorter outlet whose name sits inside it
    return [s for s, name in hits.items() if not any(name != other and name in other for other in hits.values())]


# ---------------------------------------------------------------- spotting dates in a question
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DATE_PATTERN = re.compile(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
                          r"Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
                          r"(?:,?\s+(20\d\d))?\b")
ISO_DATE = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)")      # '2023-12-12T22:38:05+00:00' -> 2023-12-12


def find_dates(question):
    """'... on October 13th, 2023 ...' or '... at 2023-10-13T21:30:00 ...' -> ['2023-10-13'].  No year -> DEFAULT_YEAR."""
    days = set()
    for year, month, day in ISO_DATE.findall(question):
        try:
            days.add(datetime.date(int(year), int(month), int(day)).isoformat())
        except ValueError:
            pass
    for match in DATE_PATTERN.finditer(question):
        year = int(match.group(3)) if match.group(3) else DEFAULT_YEAR
        try:
            days.add(datetime.date(year, MONTHS[match.group(1)[:3].lower()], int(match.group(2))).isoformat())
        except ValueError:                               # 'February 31' and the like
            pass
    return sorted(days)


# ---------------------------------------------------------------- splitting a long question into clauses
CLAUSE_SPLIT = re.compile(r",?\s+(?:while|whereas|but|compared (?:to|with)|in contrast to|contrasted with|versus|"
                          r"and the (?:subsequent|later|following|previous)|and then|and again|and do|and does|and did|"
                          r"as reported by|according to|and in|and by|and from)\s+|;\s+|,\s+and\s+|,\s+", re.I)


def split_clauses(question, max_clauses=6):
    """
    Questions glue 2 to 4 facts together ("Does The Verge say A, while TechCrunch says B ...").
    A reranker compares ONE passage with the WHOLE question, so each fact looks only partly relevant.
    Cutting the question at the glue words gives short, focused queries. Returns [] if it cannot be cut.
    """
    clauses = [c.strip() for c in CLAUSE_SPLIT.split(question) if len(c.split()) >= 4]
    return clauses[:max_clauses] if len(clauses) >= 2 else []


class WordMatch:
    """
    How well does a passage cover a clause, judged by shared words? Rare words count more than common ones
    (like BM25's IDF): sharing 'Bankman' says much more than sharing 'company'. Score 0 to 1. No model needed.
    """
    STOP = set("the a an of and or in on to for by with as at is was were be been that this which who whom whose "
               "from it its their his her after before between did does do not has have had than then also into about "
               "over under up out article report reported news source suggest suggests indicate indicates".split())

    def __init__(self, texts):
        counts = Counter(word for text in texts for word in self.words(text))
        self.idf = {word: math.log(1 + len(texts) / (1 + n)) for word, n in counts.items()}
        self.unseen = math.log(1 + len(texts))                   # a word that appears nowhere is maximally rare

    @classmethod
    def words(cls, text):
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in cls.STOP and len(w) > 2}

    def score(self, clause, passage):
        clause_words = self.words(clause)
        weight = lambda w: self.idf.get(w, self.unseen)
        total = sum(weight(w) for w in clause_words) or 1.0
        return sum(weight(w) for w in clause_words & self.words(passage)) / total

    def clause_lists(self, question, ranked):
        """One list per clause of the question: the candidates, best match for that clause first."""
        return [sorted(ranked, key=lambda e: -self.score(clause, e["row"]["embed_text"])) for clause in split_clauses(question)]


# ---------------------------------------------------------------- choosing the final passages (pure logic, no models)
def select_passages(ranked, outlets, dates, k, outlet_quota=2, date_quota=1, max_per_doc=0,
                    clause_lists=(), clause_quota=0):
    """
    `ranked`: candidate entries, best first; each has entry["row"] (chunk_id, doc_id, source, published_at).
    1. SEATS: the best `outlet_quota` chunks of every named outlet, then the best `date_quota` chunks of every
       named date, then (optional) the best `clause_quota` chunks of every clause (`clause_lists` = one list of
       entries per clause, each best-first), taken round-robin (everybody's best first, then everybody's second ...).
    2. Everything else by score, until k are chosen.
    3. `max_per_doc` (0 = no limit): one article may not take more than this many passages
       (if that would leave empty places, the cap is dropped for the last places).
    Returns the chosen entries: seats first, then the rest best-first. The order does not depend on k, so the
    first 8 of a k=20 answer are exactly what k=8 would choose (evaluation/evaluate_retrieval.py relies on that).
    """
    order, used, per_doc = [], set(), Counter()

    def take(entry, why, respect_cap=True):
        chunk_id, doc_id = entry["row"]["chunk_id"], entry["row"]["doc_id"]
        if chunk_id in used or (respect_cap and max_per_doc and per_doc[doc_id] >= max_per_doc):
            return
        used.add(chunk_id)
        per_doc[doc_id] += 1
        entry["seat"] = why
        order.append(entry)

    groups = [(o, "outlet", outlet_quota, [e for e in ranked if e["row"]["source"] == o]) for o in outlets]
    groups += [(d, "date", date_quota, [e for e in ranked if e["row"]["published_at"][:10] == d]) for d in dates]
    groups += [(f"{n}", "clause", clause_quota, members) for n, members in enumerate(clause_lists, 1)]
    for position in range(max(outlet_quota, date_quota, clause_quota) if groups else 0):
        for name, kind, quota, members in groups:
            if position < quota and position < len(members):
                take(members[position], f"{kind}: {name}")
    for entry in ranked:
        take(entry, "")
    for entry in ranked:                                 # cap left empty places: fill them without the cap
        take(entry, "", respect_cap=False)
    return order[:k]


class Retriever:
    def __init__(self):
        try:
            self.table = lancedb.connect(str(DB_DIR)).open_table(TABLE)
        except Exception as error:
            raise SystemExit(f"Could not open the database in {DB_DIR}/ ({error}).\n"
                             "Run  python -m pipeline.step7_build_index  first.")
        sources = self.table.search().select(["source"]).limit(self.table.count_rows()).to_arrow()
        self.sources = sorted(set(sources.column("source").to_pylist()))      # the ~50 outlet names
        texts = self.table.search().select(["embed_text"]).limit(self.table.count_rows()).to_arrow()
        self.word_match = WordMatch(texts.column("embed_text").to_pylist())   # for the clause seats
        self._reranker = None                      # loaded the first time it is needed

    # ------------------------------------------------------------ questions -> vectors
    def embed_queries(self, questions, batch_size=64):
        vectors = []
        for i in range(0, len(questions), batch_size):
            response = ollama.embed(
                model=EMBED_MODEL,
                input=[QUERY_PREFIX + q for q in questions[i:i + batch_size]],
                truncate=False,
                options={"num_ctx": EMBED_CTX},
            )
            vectors.extend(response["embeddings"])
        table = np.array(vectors, dtype=np.float32)
        return table / np.linalg.norm(table, axis=1, keepdims=True)

    def embed_query(self, question):
        return self.embed_queries([question])[0]

    # ------------------------------------------------------------ the two basic searches
    def _dense(self, query_vector, n, where):
        search = self.table.search(query_vector).metric("cosine")
        if where:
            search = search.where(where, prefilter=True)       # filter FIRST, then rank
        rows = search.select(COLUMNS + ["_distance"]).limit(n).to_list()
        for row in rows:
            row["score"] = 1.0 - row.pop("_distance")          # distance -> cosine similarity
        return rows

    def _keyword(self, question, n, where):
        search = self.table.search(question, query_type="fts")
        if where:
            search = search.where(where, prefilter=True)
        rows = search.select(COLUMNS + ["_score"]).limit(n).to_list()
        for row in rows:
            row["score"] = row.pop("_score")                   # BM25 score
        return rows

    # ------------------------------------------------------------ merging two rankings (RRF)
    @staticmethod
    def _fuse(dense_rows, keyword_rows):
        """A chunk earns 1/(RRF_K + rank) from each list it appears in. Appearing high in BOTH wins."""
        fused = {}
        for name, rows in (("dense", dense_rows), ("keyword", keyword_rows)):
            for rank, row in enumerate(rows, start=1):
                entry = fused.setdefault(row["chunk_id"], {"row": row, "rrf": 0.0, "ranks": {}})
                entry["rrf"] += 1.0 / (RRF_K + rank)
                entry["ranks"][name] = rank
        return sorted(fused.values(), key=lambda e: -e["rrf"])

    def _hybrid(self, question, query_vector, where=None):
        return self._fuse(self._dense(query_vector, CANDIDATES, where), self._keyword(question, CANDIDATES, where))

    # ------------------------------------------------------------ second-opinion model
    def _rerank(self, question, entries):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder      # imported only when needed (it is heavy)
            print(f"  (loading reranker {RERANK_MODEL}; the very first time this downloads it)")
            self._reranker = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
        pairs = [(question, e["row"]["embed_text"]) for e in entries]
        scores = self._reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
        for entry, score in zip(entries, scores):
            entry["rerank"] = float(score)
        return sorted(entries, key=lambda e: -e["rerank"])

    # ------------------------------------------------------------ candidate pool for the outlet / date modes
    def candidates(self, question, query_vector, k, outlets, dates, fused=None):
        """
        The pile of chunks the reranker will read:
          'base'    the usual best RERANK_POOL chunks of the whole corpus
          'outlet'  the best OUTLET_POOL chunks found INSIDE each named outlet
          'date'    the best DATE_POOL chunks published on each named day (and inside each named outlet on that day)
        Returns {chunk_id: entry}; entry["origins"] says which of the three brought it in.
        """
        pool = {}

        def add(entries, origin, limit):
            for entry in entries[:limit]:
                chunk_id = entry["row"]["chunk_id"]
                if chunk_id not in pool:
                    entry["origins"] = set()
                    pool[chunk_id] = entry
                pool[chunk_id]["origins"].add(origin)

        add(fused if fused is not None else self._hybrid(question, query_vector), "base", max(RERANK_POOL, k))
        for outlet in outlets:
            add(self._hybrid(question, query_vector, build_filter(outlet)), "outlet", OUTLET_POOL)
        for day in dates:
            add(self._hybrid(question, query_vector, day_filter(day)), "date", DATE_POOL)
            for outlet in outlets:
                add(self._hybrid(question, query_vector, day_filter(day, outlet)), "date", DATE_POOL // 2)
        return pool

    # ------------------------------------------------------------ the one function everyone calls
    def retrieve(self, question, mode="hybrid_rerank", k=10, query_vector=None,
                 source=None, after=None, before=None):
        """
        Returns up to k results, best first. Each result is
          {"rank", "score", "found_by", "chunk": {...all columns...}}
        `query_vector` lets callers pass a pre-computed vector (the evaluators do, to save time).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        where = build_filter(source, after, before)

        if mode == "keyword":
            rows = self._keyword(question, k, where)
            return [self._pack(r, i, r["score"], f"keyword#{i}") for i, r in enumerate(rows, 1)]

        if query_vector is None:
            query_vector = self.embed_query(question)

        if mode == "dense":
            rows = self._dense(query_vector, k, where)
            return [self._pack(r, i, r["score"], f"dense#{i}") for i, r in enumerate(rows, 1)]

        # hybrid, hybrid_rerank, outlet_rerank and smart_rerank all start by fusing the two searches
        fused = self._hybrid(question, query_vector, where)
        if mode == "hybrid":
            return [self._pack(e["row"], i, e["rrf"], self._found_by(e)) for i, e in enumerate(fused[:k], 1)]

        if mode in ("outlet_rerank", "smart_rerank") and not where:    # /source, /after, /before set by hand: skip
            outlets = find_outlets(question, self.sources)
            dates = find_dates(question) if mode == "smart_rerank" else []
            pool = self.candidates(question, query_vector, k, outlets, dates, fused)    # no outlet or date named: just the usual pile
            ranked = self._rerank(question, list(pool.values()))
            smart = mode == "smart_rerank"
            clause_lists = self.word_match.clause_lists(question, ranked) if smart and CLAUSE_QUOTA else []
            chosen = select_passages(ranked, outlets, dates, k, OUTLET_QUOTA, DATE_QUOTA if smart else 0, MAX_PER_DOC,
                                     clause_lists, CLAUSE_QUOTA if smart else 0)
            return [self._pack(e["row"], i, e["rerank"], self._found_by(e)) for i, e in enumerate(chosen, 1)]

        pool = self._rerank(question, fused[:max(RERANK_POOL, k)])
        return [self._pack(e["row"], i, e["rerank"], self._found_by(e)) for i, e in enumerate(pool[:k], 1)]

    # ------------------------------------------------------------ small helpers
    @staticmethod
    def _found_by(entry):
        text = " ".join(f"{name}#{rank}" for name, rank in entry["ranks"].items())
        if entry.get("origins") and "base" not in entry["origins"]:
            text = f"in-{'/'.join(sorted(entry['origins']))} {text}"       # found only by the extra searches
        if entry.get("seat"):
            text += f"  [seat {entry['seat']}]"
        return text

    @staticmethod
    def _pack(row, rank, score, found_by):
        chunk = {c: row[c] for c in COLUMNS}
        return {"rank": rank, "score": float(score), "found_by": found_by, "chunk": chunk}
