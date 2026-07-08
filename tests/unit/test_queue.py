from unittest.mock import AsyncMock, MagicMock

from rag.ingestion.queue import IngestionJob, handle_message


def make_message(body: bytes) -> MagicMock:
    message = MagicMock()
    message.body = body
    message.ack = AsyncMock()
    message.reject = AsyncMock()
    return message


def test_job_payload_roundtrip() -> None:
    job = IngestionJob.from_bytes("report.pdf", b"\x00binary\xff", source="reports")
    restored = IngestionJob.model_validate_json(job.model_dump_json())
    assert restored.payload() == b"\x00binary\xff"
    assert restored.filename == "report.pdf"
    assert restored.source == "reports"


async def test_handle_message_acks_on_success() -> None:
    job = IngestionJob.from_bytes("a.txt", b"content")
    message = make_message(job.model_dump_json().encode())
    handler = AsyncMock()
    await handle_message(message, handler)
    handler.assert_awaited_once()
    message.ack.assert_awaited_once()
    message.reject.assert_not_awaited()


async def test_handle_message_rejects_invalid_json() -> None:
    message = make_message(b"not json")
    handler = AsyncMock()
    await handle_message(message, handler)
    handler.assert_not_awaited()
    message.reject.assert_awaited_once_with(requeue=False)


async def test_handle_message_rejects_on_handler_failure() -> None:
    job = IngestionJob.from_bytes("a.txt", b"content")
    message = make_message(job.model_dump_json().encode())
    handler = AsyncMock(side_effect=RuntimeError("boom"))
    await handle_message(message, handler)
    message.reject.assert_awaited_once_with(requeue=False)
    message.ack.assert_not_awaited()
