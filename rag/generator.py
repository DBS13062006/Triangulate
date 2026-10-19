"""
rag/generator.py : turns a question + retrieved passages into an answer, using the local language model.
Not run directly; other scripts do `from rag.generator import generate`.
"""
import re

import ollama

from config import ABSTAIN_BELOW, LLM_CTX, LLM_MODEL, LLM_TEMPERATURE

SYSTEM_PROMPT = """You answer questions about news articles using ONLY the numbered passages you are given.

Rules:
- Use only the passages. Do not use outside knowledge, and do not guess.
- Many questions need facts from SEVERAL passages: combine them. For comparison or timing questions,
  compare what each outlet says and the publication dates shown with each passage.
- If the passages do not contain enough information to answer, the answer is exactly: Insufficient information.
- Reply in exactly this format:
Answer: <the shortest complete answer: a name, a company, Yes, No, or a short phrase>
Explanation: <one to three sentences that cite passages like [1] or [2][4]>"""


def format_context(results):
    """Number each passage and show where and when it was published."""
    blocks = []
    for number, result in enumerate(results, start=1):
        c = result["chunk"]
        blocks.append(f'[{number}] "{c["title"]}" ({c["source"]}, {c["published_at"][:10]})\n{c["text"]}')
    return "\n\n".join(blocks)


def confident_enough(results, mode):
    """Optional gate: if the best passage scores below ABSTAIN_BELOW, don't even ask the model."""
    if ABSTAIN_BELOW is None or not results or mode not in ("dense", "hybrid_rerank", "outlet_rerank", "smart_rerank"):
        return True
    return max(r["score"] for r in results) >= ABSTAIN_BELOW


def short_answer(raw):
    """Pull out the text after 'Answer:' (or the whole reply if the format wasn't followed)."""
    match = re.search(r"answer\s*:\s*(.+?)(?:\n\s*explanation\s*:|\Z)", raw, re.I | re.S)
    return (match.group(1) if match else raw).strip()


def generate(question, results):
    """Returns {"raw", "short", "prompt_tokens", "warning"}."""
    user_message = f"Passages:\n\n{format_context(results)}\n\nQuestion: {question}"
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_message}],
        options={"temperature": LLM_TEMPERATURE, "num_ctx": LLM_CTX},
    )
    raw = response["message"]["content"].strip()
    prompt_tokens = getattr(response, "prompt_eval_count", None) or 0

    warning = None
    if prompt_tokens >= LLM_CTX - 300:    # nearly full: Ollama may have cut the START of the prompt
        warning = f"prompt used {prompt_tokens} of {LLM_CTX} tokens; lower CONTEXT_CHUNKS or raise LLM_CTX"
    return {"raw": raw, "short": short_answer(raw), "prompt_tokens": prompt_tokens, "warning": warning}
