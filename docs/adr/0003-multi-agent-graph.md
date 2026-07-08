# ADR 0003: LangGraph StateGraph with a Bounded Self-Correction Loop

## Status

Accepted

## Context

A single RAG prompt cannot reliably classify intent, ground answers, and
attach citations. Agent frameworks considered: plain LCEL chains, LangGraph,
custom orchestration. The pipeline needs an explicit revision loop where a
critic can send a draft back for correction, which is a cyclic graph.

## Decision

A LangGraph `StateGraph` with five nodes: router, retriever, reasoner,
critic, citation. The critic returns a structured verdict; a conditional
edge routes `revise` back to the reasoner. Revisions are capped by
`max_correction_rounds` (default 2).

## Consequences

- The self-correction loop is bounded: worst case is
  `1 + max_correction_rounds` LLM reasoning calls plus one critic call per
  round, keeping latency and cost predictable.
- Router and critic use structured JSON output parsed into pydantic models;
  parse failures fall back to safe defaults (factual routing, approve) so a
  malformed LLM response degrades quality, never availability.
- Out-of-scope queries short-circuit to a static refusal without touching
  retrieval, which keeps junk out of metrics and saves tokens.
- State is a TypedDict, so every node is a plain async callable that is unit
  testable without the graph.
