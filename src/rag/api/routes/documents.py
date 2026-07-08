from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from rag.api.auth import get_current_subject
from rag.api.container import Container
from rag.api.schemas import UploadResponse
from rag.exceptions import LoaderError

router = APIRouter(tags=["documents"])

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("/documents", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    subject: str = Depends(get_current_subject),
) -> UploadResponse:
    container: Container = request.app.state.container
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="filename is required")
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="file too large")
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty file")
    try:
        result = await container.pipeline.ingest(file.filename, data)
    except LoaderError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return UploadResponse(
        document_id=result.document.id,
        status=result.document.status.value,
        version=result.document.version,
        deduplicated=result.deduplicated,
    )
