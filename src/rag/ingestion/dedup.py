import hashlib
import random
import re
from collections.abc import Iterable

_WORD = re.compile(r"\w+")
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def content_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _shingles(text: str, size: int) -> set[int]:
    words = [word.lower() for word in _WORD.findall(text)]
    if len(words) < size:
        return {hash(" ".join(words)) & _MAX_HASH} if words else set()
    return {
        int.from_bytes(
            hashlib.blake2b(" ".join(words[i : i + size]).encode(), digest_size=4).digest(),
            "big",
        )
        for i in range(len(words) - size + 1)
    }


class MinHasher:
    def __init__(self, num_permutations: int = 128, shingle_size: int = 5, seed: int = 42) -> None:
        rng = random.Random(seed)
        self._shingle_size = shingle_size
        self._params = [
            (rng.randint(1, _MERSENNE_PRIME - 1), rng.randint(0, _MERSENNE_PRIME - 1))
            for _ in range(num_permutations)
        ]

    def signature(self, text: str) -> tuple[int, ...]:
        shingles = _shingles(text, self._shingle_size)
        if not shingles:
            return tuple(0 for _ in self._params)
        return tuple(
            min((a * shingle + b) % _MERSENNE_PRIME & _MAX_HASH for shingle in shingles)
            for a, b in self._params
        )

    @staticmethod
    def similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        matches = sum(1 for a, b in zip(left, right, strict=True) if a == b)
        return matches / len(left)


class DeduplicationIndex:
    def __init__(self, hasher: MinHasher, threshold: float = 0.9) -> None:
        self._hasher = hasher
        self._threshold = threshold
        self._exact: dict[str, str] = {}
        self._signatures: dict[str, tuple[int, ...]] = {}

    def add(self, key: str, text: str) -> None:
        self._exact[content_hash(text)] = key
        self._signatures[key] = self._hasher.signature(text)

    def load(self, entries: Iterable[tuple[str, str, tuple[int, ...]]]) -> None:
        for key, text_hash, signature in entries:
            self._exact[text_hash] = key
            self._signatures[key] = signature

    def find_duplicate(self, text: str) -> str | None:
        exact = self._exact.get(content_hash(text))
        if exact is not None:
            return exact
        signature = self._hasher.signature(text)
        best_key: str | None = None
        best_score = self._threshold
        for key, candidate in self._signatures.items():
            score = MinHasher.similarity(signature, candidate)
            if score >= best_score:
                best_key = key
                best_score = score
        return best_key
