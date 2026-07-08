from uuid import UUID

from pydantic import BaseModel

from rag.ingestion.chunkers import Chunker
from rag.ingestion.dedup import DeduplicationIndex, MinHasher, content_hash
from rag.ingestion.embedders import Embedder
from rag.ingestion.loaders import LoaderFactory
from rag.ingestion.storage import DocumentRepository
from rag.models import DocumentMeta, DocumentStatus
from rag.observability.logging import get_logger
from rag.observability.metrics import Metrics
from rag.retrieval.sparse import BM25Index
from rag.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class IngestResult(BaseModel):
    document: DocumentMeta
    deduplicated: bool = False


class IngestionPipeline:
    def __init__(
        self,
        repository: DocumentRepository,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        hasher: MinHasher | None = None,
        dedup_threshold: float = 0.9,
        metrics: Metrics | None = None,
    ) -> None:
        self._repository = repository
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._dedup_index = DeduplicationIndex(hasher or MinHasher(), threshold=dedup_threshold)
        self._hasher = hasher or MinHasher()
        self._metrics = metrics
        self._warmed = False

    async def warm_up(self) -> None:
        await self._vector_store.ensure_ready()
        self._dedup_index.load(await self._repository.all_signatures())
        self._bm25_index.rebuild(await self._repository.all_chunks())
        self._warmed = True

    async def ingest(self, filename: str, data: bytes, source: str | None = None) -> IngestResult:
        if not self._warmed:
            await self.warm_up()
        source_name = source or filename
        format = LoaderFactory.detect_format(filename)
        loader = LoaderFactory.for_format(format)
        text = loader.load(data)
        text_hash = content_hash(text)

        exact = await self._repository.get_by_content_hash(text_hash)
        if exact is not None:
            self._count("duplicate")
            logger.info("ingest_duplicate", source=source_name, existing=str(exact.id))
            return IngestResult(document=exact, deduplicated=True)
        near_key = self._dedup_index.find_duplicate(text)
        if near_key is not None:
            existing = await self._repository.get_by_id(UUID(near_key))
            if existing is not None:
                self._count("near_duplicate")
                logger.info("ingest_near_duplicate", source=source_name, existing=str(existing.id))
                return IngestResult(document=existing, deduplicated=True)

        previous = await self._repository.latest_for_source(source_name)
        meta = DocumentMeta(
            source=source_name,
            format=format,
            content_hash=text_hash,
            version=previous.version + 1 if previous else 1,
            status=DocumentStatus.PROCESSING,
            extra={"filename": filename},
        )
        await self._repository.save_document(meta, signature=self._hasher.signature(text))
        try:
            chunks = self._chunker.split(meta.id, text)
            for chunk in chunks:
                chunk.metadata.setdefault("source", source_name)
            vectors = await self._embedder.embed([chunk.text for chunk in chunks])
            await self._vector_store.upsert(chunks, vectors)
            await self._repository.add_chunks(chunks)
            self._bm25_index.add(chunks)
        except Exception:
            await self._repository.update_status(meta.id, DocumentStatus.FAILED)
            self._count("failed")
            logger.exception("ingest_failed", source=source_name)
            raise
        if previous is not None:
            await self._supersede(previous)
        await self._repository.update_status(meta.id, DocumentStatus.INDEXED)
        self._dedup_index.add(str(meta.id), text)
        self._count("indexed")
        logger.info(
            "ingest_indexed",
            source=source_name,
            document_id=str(meta.id),
            version=meta.version,
            chunks=len(chunks),
        )
        return IngestResult(
            document=meta.model_copy(update={"status": DocumentStatus.INDEXED}),
            deduplicated=False,
        )

    async def delete_document(self, meta: DocumentMeta) -> None:
        await self._vector_store.delete_document(meta.id)
        self._bm25_index.remove_document(meta.id)
        await self._repository.delete_chunks(meta.id)

    async def _supersede(self, previous: DocumentMeta) -> None:
        await self.delete_document(previous)
        await self._repository.update_status(previous.id, DocumentStatus.SUPERSEDED)
        logger.info("ingest_superseded", document_id=str(previous.id))

    def _count(self, status: str) -> None:
        if self._metrics is not None:
            self._metrics.documents_ingested_total.labels(status=status).inc()
