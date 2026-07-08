# Retrieval Evaluation Report

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
Corpus: 19 chunks from 3 documents, 15 golden questions, top-k = 5

| Mode | Hit@1 | Hit@3 | MRR@5 |
|------|-------|-------|-------|
| bm25 | 0.933 | 0.933 | 0.950 |
| dense | 1.000 | 1.000 | 1.000 |
| hybrid_rrf | 1.000 | 1.000 | 1.000 |

