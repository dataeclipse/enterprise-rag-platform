import re
from enum import StrEnum

from pydantic import BaseModel


class PIIType(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"


class PIIMatch(BaseModel):
    type: PIIType
    value: str
    start: int
    end: int


class RedactionResult(BaseModel):
    text: str
    matches: list[PIIMatch]


_PATTERNS: dict[PIIType, re.Pattern[str]] = {
    PIIType.EMAIL: re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    PIIType.CREDIT_CARD: re.compile(r"(?<![\dA-Za-z])(?:\d[ -]?){13,19}(?![\dA-Za-z])"),
    PIIType.SSN: re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    PIIType.IBAN: re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    PIIType.IP_ADDRESS: re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    PIIType.PHONE: re.compile(
        r"(?<![\dA-Za-z])\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{0,4}(?![\dA-Za-z])"
    ),
}

_DETECTION_ORDER = [
    PIIType.EMAIL,
    PIIType.CREDIT_CARD,
    PIIType.SSN,
    PIIType.IBAN,
    PIIType.IP_ADDRESS,
    PIIType.PHONE,
]


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or len(digits) < 13:
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _valid_ip(value: str) -> bool:
    return all(0 <= int(octet) <= 255 for octet in value.split("."))


def _digit_count(value: str) -> int:
    return sum(1 for char in value if char.isdigit())


class PIIRedactor:
    def __init__(self, enabled_types: frozenset[PIIType] | None = None) -> None:
        self._enabled = enabled_types or frozenset(PIIType)

    def detect(self, text: str) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        claimed: list[tuple[int, int]] = []
        for pii_type in _DETECTION_ORDER:
            if pii_type not in self._enabled:
                continue
            for found in _PATTERNS[pii_type].finditer(text):
                start, end = found.span()
                value = found.group()
                if pii_type is PIIType.CREDIT_CARD and not luhn_valid(re.sub(r"[ -]", "", value)):
                    continue
                if pii_type is PIIType.IP_ADDRESS and not _valid_ip(value):
                    continue
                if pii_type is PIIType.PHONE and not 10 <= _digit_count(value) <= 15:
                    continue
                if any(start < c_end and end > c_start for c_start, c_end in claimed):
                    continue
                claimed.append((start, end))
                matches.append(PIIMatch(type=pii_type, value=value, start=start, end=end))
        matches.sort(key=lambda match: match.start)
        return matches

    def redact(self, text: str) -> RedactionResult:
        matches = self.detect(text)
        redacted = text
        for match in reversed(matches):
            placeholder = f"[{match.type.value.upper()}]"
            redacted = redacted[: match.start] + placeholder + redacted[match.end :]
        return RedactionResult(text=redacted, matches=matches)
