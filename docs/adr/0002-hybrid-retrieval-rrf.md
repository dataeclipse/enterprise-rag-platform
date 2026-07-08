# ADR 0002: Reciprocal Rank Fusion for Hybrid Retrieval

## Status

Accepted

## Context

Dense retrieval (bge-m3) misses exact keyword matches (IDs, product codes,
legal terms); BM25 misses paraphrases. Fusing both requires combining scores
from incompatible scales: cosine similarity in [0, 1] versus unbounded BM25
scores. Weighted-sum fusion needs per-corpus weight tuning and score
normalization.

## Decision

Reciprocal Rank Fusion (RRF) with k=60 combines dense and sparse result
lists using ranks only, followed by an optional cross-encoder reranker
(bge-reranker-v2-m3) over the fused candidates.

## Consequences

- No score normalization or weight tuning; RRF is rank-based and works
  unchanged when the embedding model or BM25 parameters change.
- Chunks found by both retrievers are boosted, which measurably improved
  MRR on the golden dataset versus either mode alone.
- The reranker is the accuracy lever: fusion produces a candidate pool,
  the cross-encoder orders it. It is optional (`RAG_RERANKER__ENABLED`)
  because it dominates retrieval latency on CPU.
- BM25 index lives in process memory and is rebuilt from PostgreSQL chunk
  storage at startup; acceptable up to roughly a million chunks, after which
  an external sparse index (Elasticsearch) becomes the next step.
