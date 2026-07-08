# ADR 0004: Deterministic Guardrails Before the Agent Graph

## Status

Accepted

## Context

User queries can contain PII (emails, card numbers) that must not reach an
external LLM provider, and prompt-injection attempts that try to override
agent instructions. Options: LLM-based moderation (extra call per request,
non-deterministic), Microsoft Presidio (heavy NER dependency), or
deterministic pattern matching.

## Decision

Two in-process deterministic guardrails run before the agent graph:

1. PII redaction: typed regex detectors (email, phone, credit card with Luhn
   validation, SSN, IBAN, IP) replace matches with placeholders.
2. Injection detection: weighted rule set (instruction override, prompt
   leak, role hijack, delimiter escape, encoded payloads) combined with a
   noisy-or score against a configurable threshold.

## Consequences

- Zero added latency budget: both checks are sub-millisecond regex passes,
  versus 100ms+ for an LLM moderation call.
- Deterministic behavior is unit-testable and auditable; every block is
  counted in Prometheus by guardrail type.
- Regex-based PII detection has known recall limits (names, addresses).
  The `PIIRedactor` interface accepts pluggable detectors, so Presidio can
  be added as an optional backend without API changes.
- Injection scoring is heuristic and will not stop novel attacks; it is a
  cheap first layer, not a substitute for the critic's groundedness check.
