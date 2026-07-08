from rag.ingestion.dedup import DeduplicationIndex, MinHasher, content_hash

BASE_TEXT = (
    "The quarterly report shows revenue growth across all regions. "
    "European sales increased by twelve percent while Asian markets "
    "remained stable throughout the reporting period. Management "
    "expects continued expansion in the next fiscal year."
)

NEAR_DUPLICATE = (
    "The quarterly report shows revenue growth across most regions. "
    "European sales increased by twelve percent while Asian markets "
    "remained stable throughout the reporting period. Management "
    "expects continued expansion in the next fiscal year."
)

DIFFERENT_TEXT = (
    "Employees must complete security awareness training annually. "
    "The policy covers phishing, password hygiene and incident "
    "reporting procedures for all corporate systems."
)


def test_content_hash_normalizes_whitespace_and_case() -> None:
    assert content_hash("Hello  World") == content_hash("hello world")
    assert content_hash("a") != content_hash("b")


def test_minhash_similarity_bounds() -> None:
    hasher = MinHasher(num_permutations=64)
    signature = hasher.signature(BASE_TEXT)
    assert MinHasher.similarity(signature, signature) == 1.0
    assert MinHasher.similarity(signature, ()) == 0.0


def test_minhash_near_duplicate_scores_high() -> None:
    hasher = MinHasher(num_permutations=128)
    base = hasher.signature(BASE_TEXT)
    near = hasher.signature(NEAR_DUPLICATE)
    other = hasher.signature(DIFFERENT_TEXT)
    assert MinHasher.similarity(base, near) > 0.6
    assert MinHasher.similarity(base, other) < 0.2


def test_minhash_deterministic_across_instances() -> None:
    first = MinHasher(seed=7).signature(BASE_TEXT)
    second = MinHasher(seed=7).signature(BASE_TEXT)
    assert first == second


def test_index_exact_duplicate() -> None:
    index = DeduplicationIndex(MinHasher(), threshold=0.9)
    index.add("doc-1", BASE_TEXT)
    assert index.find_duplicate(BASE_TEXT) == "doc-1"
    assert index.find_duplicate(BASE_TEXT.upper()) == "doc-1"


def test_index_near_duplicate() -> None:
    index = DeduplicationIndex(MinHasher(), threshold=0.5)
    index.add("doc-1", BASE_TEXT)
    assert index.find_duplicate(NEAR_DUPLICATE) == "doc-1"


def test_index_no_match_for_different_text() -> None:
    index = DeduplicationIndex(MinHasher(), threshold=0.5)
    index.add("doc-1", BASE_TEXT)
    assert index.find_duplicate(DIFFERENT_TEXT) is None


def test_index_load_precomputed() -> None:
    hasher = MinHasher()
    index = DeduplicationIndex(hasher, threshold=0.9)
    index.load([("doc-1", content_hash(BASE_TEXT), hasher.signature(BASE_TEXT))])
    assert index.find_duplicate(BASE_TEXT) == "doc-1"
