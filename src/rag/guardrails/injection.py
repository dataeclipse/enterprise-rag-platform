import math
import re

from pydantic import BaseModel


class InjectionAssessment(BaseModel):
    score: float
    matched_rules: list[str]
    blocked: bool


_RULES: list[tuple[str, re.Pattern[str], float]] = [
    (
        "override_instructions",
        re.compile(
            r"\b(ignore|forget|disregard)\b.{0,40}\b(previous|prior|above|all|any|earlier)\b"
            r".{0,40}\b(instruction|prompt|rule|context)",
            re.IGNORECASE | re.DOTALL,
        ),
        0.6,
    ),
    (
        "reveal_system_prompt",
        re.compile(
            r"\b(reveal|show|print|repeat|output|leak)\b.{0,40}\b(system prompt|initial prompt|"
            r"hidden instruction|your instruction)",
            re.IGNORECASE | re.DOTALL,
        ),
        0.6,
    ),
    (
        "role_hijack",
        re.compile(
            r"\b(you are now|pretend (to be|you are)|act as|new persona|jailbreak|"
            r"developer mode|dan mode)\b",
            re.IGNORECASE,
        ),
        0.45,
    ),
    (
        "delimiter_escape",
        re.compile(
            r"<\|[^|]{1,30}\|>|\[/?(INST|SYS|SYSTEM)\]|^#{1,4}\s*(system|instruction)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        0.5,
    ),
    (
        "safety_bypass",
        re.compile(
            r"\b(bypass|disable|override|remove)\b.{0,40}\b(safety|filter|guardrail|"
            r"restriction|moderation)",
            re.IGNORECASE | re.DOTALL,
        ),
        0.6,
    ),
    (
        "encoded_payload",
        re.compile(r"[A-Za-z0-9+/]{80,}={0,2}"),
        0.35,
    ),
]


class InjectionDetector:
    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._threshold = threshold

    def scan(self, text: str) -> InjectionAssessment:
        matched = [name for name, pattern, _ in _RULES if pattern.search(text)]
        weights = [weight for name, _, weight in _RULES if name in matched]
        score = 1.0 - math.prod(1.0 - weight for weight in weights) if weights else 0.0
        return InjectionAssessment(
            score=round(score, 4),
            matched_rules=matched,
            blocked=score >= self._threshold,
        )
