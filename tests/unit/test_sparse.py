from uuid import uuid4

from rag.models import Chunk
from rag.retrieval.sparse import BM25Index, tokenize


def make_chunk(text: str, index: int = 0, document_id: object = None) -> Chunk:
    doc_id = document_id or uuid4()
    return Chunk(id=f"{doc_id}:{index}", document_id=doc_id, text=text, index=index)


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Hello, World! It's 42.") == ["hello", "world", "it", "s", "42"]


def test_empty_index_returns_nothing() -> None:
    assert BM25Index().search("anything", top_k=5) == []


def test_search_ranks_relevant_chunk_first() -> None:
    index = BM25Index()
    index.rebuild(
        [
            make_chunk("the revenue report shows quarterly growth"),
            make_chunk("employees attended the security training session"),
            make_chunk("quarterly revenue exceeded all growth projections"),
        ]
    )
    results = index.search("quarterly revenue growth", top_k=2)
    assert len(results) == 2
    assert all(result.origin == "sparse" for result in results)
    assert "revenue" in results[0].chunk.text


def test_search_filters_zero_scores() -> None:
    index = BM25Index()
    index.rebuild([make_chunk("cats and dogs"), make_chunk("stocks and bonds")])
    results = index.search("zebra unicorn", top_k=5)
    assert results == []


def test_add_and_remove_document() -> None:
    index = BM25Index()
    keep_id, other_id, third_id, drop_id = uuid4(), uuid4(), uuid4(), uuid4()
    index.add([make_chunk("keep this text about cats", 0, keep_id)])
    index.add([make_chunk("finance report for the quarter", 0, other_id)])
    index.add([make_chunk("security policy for employees", 0, third_id)])
    index.add([make_chunk("drop this other text about cats", 0, drop_id)])
    assert len(index) == 4
    index.remove_document(drop_id)
    assert len(index) == 3
    results = index.search("cats", top_k=5)
    assert [r.chunk.document_id for r in results] == [keep_id]
