from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, Uuid, delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rag.models import Chunk, DocumentFormat, DocumentMeta, DocumentStatus


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    source: Mapped[str] = mapped_column(String(512), index=True)
    format: Mapped[str] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    signature: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    extra: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer)
    meta: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


def _to_meta(record: DocumentRecord) -> DocumentMeta:
    return DocumentMeta(
        id=record.id,
        source=record.source,
        format=DocumentFormat(record.format),
        content_hash=record.content_hash,
        version=record.version,
        status=DocumentStatus(record.status),
        created_at=record.created_at,
        extra=dict(record.extra or {}),
    )


class DocumentRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save_document(
        self, meta: DocumentMeta, signature: tuple[int, ...] | None = None
    ) -> None:
        async with self._session_factory() as session, session.begin():
            session.add(
                DocumentRecord(
                    id=meta.id,
                    source=meta.source,
                    format=meta.format.value,
                    content_hash=meta.content_hash,
                    version=meta.version,
                    status=meta.status.value,
                    created_at=meta.created_at,
                    signature=list(signature) if signature else None,
                    extra=meta.extra,
                )
            )

    async def get_by_content_hash(self, content_hash: str) -> DocumentMeta | None:
        async with self._session_factory() as session:
            record = await session.scalar(
                select(DocumentRecord).where(DocumentRecord.content_hash == content_hash)
            )
            return _to_meta(record) if record else None

    async def get_by_id(self, document_id: UUID) -> DocumentMeta | None:
        async with self._session_factory() as session:
            record = await session.get(DocumentRecord, document_id)
            return _to_meta(record) if record else None

    async def latest_for_source(self, source: str) -> DocumentMeta | None:
        async with self._session_factory() as session:
            record = await session.scalar(
                select(DocumentRecord)
                .where(
                    DocumentRecord.source == source,
                    DocumentRecord.status != DocumentStatus.SUPERSEDED.value,
                )
                .order_by(DocumentRecord.version.desc())
                .limit(1)
            )
            return _to_meta(record) if record else None

    async def update_status(self, document_id: UUID, status: DocumentStatus) -> None:
        async with self._session_factory() as session, session.begin():
            record = await session.get(DocumentRecord, document_id)
            if record is not None:
                record.status = status.value

    async def list_documents(self) -> list[DocumentMeta]:
        async with self._session_factory() as session:
            records = await session.scalars(
                select(DocumentRecord).order_by(DocumentRecord.created_at)
            )
            return [_to_meta(record) for record in records]

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        async with self._session_factory() as session, session.begin():
            for chunk in chunks:
                session.add(
                    ChunkRecord(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        position=chunk.index,
                        meta=chunk.metadata,
                    )
                )

    async def delete_chunks(self, document_id: UUID) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(delete(ChunkRecord).where(ChunkRecord.document_id == document_id))

    async def all_chunks(self) -> list[Chunk]:
        async with self._session_factory() as session:
            records = await session.scalars(select(ChunkRecord).order_by(ChunkRecord.position))
            return [
                Chunk(
                    id=record.id,
                    document_id=record.document_id,
                    text=record.text,
                    index=record.position,
                    metadata=dict(record.meta or {}),
                )
                for record in records
            ]

    async def all_signatures(self) -> list[tuple[str, str, tuple[int, ...]]]:
        async with self._session_factory() as session:
            records = await session.scalars(
                select(DocumentRecord).where(
                    DocumentRecord.status != DocumentStatus.SUPERSEDED.value
                )
            )
            return [
                (str(record.id), record.content_hash, tuple(record.signature or ()))
                for record in records
            ]
