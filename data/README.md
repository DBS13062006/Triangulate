# data/

Everything in this folder is downloaded or generated on your machine, so it is not committed to git.

## 1. Download the dataset (MultiHop-RAG)

```bash
curl -L -o data/corpus.json      https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json
curl -L -o data/MultiHopRAG.json https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json
```

| File | What it is |
|---|---|
| `corpus.json` | 609 news articles (Sep–Dec 2023) from about 50 outlets |
| `MultiHopRAG.json` | 2,556 questions with their correct answers and the evidence sentences that support them |

The dataset is by Yixuan Tang and Yi Yang ([paper](https://arxiv.org/abs/2401.15391), [GitHub](https://github.com/yixuantt/MultiHop-RAG)) and is licensed under ODC-BY.

## 2. Files the pipeline creates here

| File | Created by |
|---|---|
| `corpus_clean.jsonl`, `removed_log.txt` | `python -m pipeline.step4_clean` |
| `chunks.jsonl` | `python -m pipeline.step5_chunk` |
| `embeddings.npy`, `embedding_ids.json` | `python -m pipeline.step6_embed` |
| `lancedb/` (the search database) | `python -m pipeline.step7_build_index` |
| `query_vectors.npy`, `eval_*.json(l)`, `rerank_cache_*.jsonl` | the scripts in `evaluation/` |
