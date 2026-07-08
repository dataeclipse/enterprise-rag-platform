import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from rag.ingestion.chunkers import RecursiveChunker
from rag.ingestion.embedders import Embedder
from rag.models import Chunk
from rag.retrieval.sparse import BM25Index
from rag.retrieval.vector_store import InMemoryVectorStore

DATASET_DIR = Path(__file__).parent / "dataset"
CORPUS_DIR = DATASET_DIR / "corpus"
GOLDEN_PATH = DATASET_DIR / "golden_qa.jsonl"
REPORT_PATH = Path(__file__).parent / "report.md"


@dataclass(frozen=True)
class GoldenExample:
    question: str
    answer: str
    source: str


def load_golden() -> list[GoldenExample]:
    examples = []
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            examples.append(
                GoldenExample(
                    question=payload["question"],
                    answer=payload["answer"],
                    source=payload["source"],
                )
            )
    return examples


async def build_indexes(
    embedder: Embedder, chunk_size: int = 512, chunk_overlap: int = 64
) -> tuple[InMemoryVectorStore, BM25Index, list[Chunk]]:
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = InMemoryVectorStore()
    bm25 = BM25Index()
    all_chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = chunker.split(uuid4(), text)
        for chunk in chunks:
            chunk.metadata["source"] = path.name
        vectors = await embedder.embed([chunk.text for chunk in chunks])
        await store.upsert(chunks, vectors)
        all_chunks.extend(chunks)
    bm25.rebuild(all_chunks)
    return store, bm25, all_chunks
