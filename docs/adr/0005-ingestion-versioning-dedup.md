# ADR 0005: Content-Hash Versioning and MinHash Deduplication

## Status

Accepted

## Context

Enterprise document sets contain exact re-uploads, trivially edited copies,
and legitimate new versions of the same source. Indexing duplicates poisons
retrieval (identical chunks crowd the top-k) and wastes embedding compute.

## Decision

Three-stage ingestion policy:

1. Exact duplicates: SHA-256 over whitespace-normalized text; a hit returns
   the existing document without re-indexing.
2. Near duplicates: 128-permutation MinHash signatures over 5-word shingles;
   Jaccard similarity above the threshold (default 0.9) is treated as a
   duplicate of the existing document.
3. New versions: same `source` with different content increments the version,
   indexes the new chunks, then deletes the previous version's chunks from
   the vector store and BM25 index and marks the record `superseded`.

## Consequences

- Signatures persist in PostgreSQL and load into an in-memory index at
  worker startup, so deduplication survives restarts.
- MinHash is implemented in ~60 lines with a seeded RNG (no datasketch
  dependency); signatures are deterministic across processes.
- The supersede step runs after the new version is fully indexed, so a crash
  mid-ingestion leaves the old version serving queries and the new one
  marked `failed` - no window with zero coverage.
- Threshold 0.9 favors precision; lowering it toward 0.8 catches more edited
  copies at the risk of swallowing legitimate revisions uploaded under a new
  source name.
