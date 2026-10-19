"""
config.py : every setting in one place. All other scripts import from here.
"""
from pathlib import Path

# ---------------------------------------------------------------- files
DATA = Path("data")
DB_DIR = DATA / "lancedb"          # the database folder (built by pipeline/step7_build_index.py)
TABLE = "chunks"

# ---------------------------------------------------------------- embedding model (via Ollama)
EMBED_MODEL = "nomic-embed-text"
QUERY_PREFIX = "search_query: "     # chunks were embedded with "search_document: " (pipeline/step6_embed.py)
EMBED_CTX = 2048                    # max tokens per text (our longest chunk is ~590)

# ---------------------------------------------------------------- reranker (via sentence-transformers)
RERANK_MODEL = "BAAI/bge-reranker-base"    # ~1 GB download. Lighter option: "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANK_MAX_LENGTH = 512             # tokens of (question + passage) the reranker reads
RERANK_BATCH = 16

# ---------------------------------------------------------------- language model (via Ollama)
LLM_MODEL = "llama3.1:8b"
LLM_CTX = 8192                      # must be bigger than the prompt, or Ollama silently cuts the start off
LLM_TEMPERATURE = 0.0               # 0 = same question, same answer

# ---------------------------------------------------------------- retrieval settings
CANDIDATES = 40                     # chunks each search (meaning / keyword) contributes before fusing
RRF_K = 60                          # standard constant for Reciprocal Rank Fusion
RERANK_POOL = 30                    # fused candidates sent to the reranker
CONTEXT_CHUNKS = 10                 # chunks handed to the language model (10 finds all the evidence more often than 8)
DEFAULT_MODE = "smart_rerank"       # the search mode ask.py and evaluate_answers.py use unless told otherwise

# ---------------------------------------------------------------- "I don't know" gate
# After running  python -m evaluation.evaluate_retrieval --rerank it prints a suggested number. Paste it here.
# None = no gate (the language model alone decides whether the passages are enough).
ABSTAIN_BELOW = 0.039

# ---------------------------------------------------------------- outlet-aware search (modes "outlet_rerank", "smart_rerank")
# If a question names news outlets ("The Verge", "TechCrunch"), each one gets its own search and its best chunks
# are guaranteed a seat among the passages the language model reads.
OUTLET_QUOTA = 2                    # guaranteed seats per named outlet (try 3)
OUTLET_POOL = 15                    # candidates per named outlet sent to the reranker
OUTLET_ALIASES = {"CNBC": "Cnbc | World Business News Leader"}     # other ways questions write an outlet's name

# ---------------------------------------------------------------- date-aware search (mode "smart_rerank")
# Dates in a question ("October 13th, 2023") are the publication dates of the evidence articles (checked: 99%).
DEFAULT_YEAR = 2023                 # used when a question writes "October 13" without a year
DATE_POOL = 10                      # candidates per named day sent to the reranker
DATE_QUOTA = 1                      # guaranteed seats per named day
MAX_PER_DOC = 0                     # one article may fill at most this many seats (0 = no limit; tested: a limit did not help)

# ---------------------------------------------------------------- clause seats (mode "smart_rerank")
# Questions glue 2-4 facts together ("Does The Verge say A, while TechCrunch says B?"). The question is cut into
# clauses, and the candidate that shares the most (rare) words with each clause gets a guaranteed seat.
CLAUSE_QUOTA = 1                    # guaranteed seats per clause (0 = off; tested: 1 is best)
