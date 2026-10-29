from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field as ModelField
from typing import Literal
from app.services.video_processor import (
    process_upload,
    get_status,
    cancel_session,
    retry_uploads,
)

router = APIRouter(tags=["Video Management"])


class KickResponse(BaseModel):
    total_kicks: int = ModelField(ge=0)
    cloudinary_urls: list[str]


class UploadReceipt(BaseModel):
    session_id: str


class ProgressResponse(BaseModel):
    stage: Literal["queued", "preparing", "detecting", "analyzing", "rendering", "uploading"]
    current_frame: int = ModelField(ge=0)
    total_frames: int = ModelField(ge=0)
    progress_pct: float = ModelField(ge=0, le=100)


@router.post("/upload", status_code=202, response_model=UploadReceipt, responses={
    400: {"description": "Invalid video"},
    413: {"description": "Upload exceeds the configured size limit"},
    500: {"description": "Video processing failed"},
    503: {"description": "Processing capacity is full or processing was cancelled"},
})
def upload_video(
    file: UploadFile = File(
        ..., description="Video file containing player kicking a football."
    ),
):
    """
    Validate and save the video, queue processing, then return a session ID.
    """
    return process_upload(file)


@router.get("/status/{session_id}", response_model=ProgressResponse | KickResponse)
def video_status(session_id: str):
    """Return stage/frame progress, or only the completed count and clip URLs."""
    return get_status(session_id)


@router.post("/cancel/{session_id}")
def cancel_job(session_id: str):
    """Request cancellation of a running pipeline job."""
    return cancel_session(session_id)


@router.post("/retry-uploads/{session_id}")
def retry_session_uploads(session_id: str):
    """Retry any Cloudinary uploads that failed during initial processing."""
    return retry_uploads(session_id)


@router.get("/health")
def health_check():
    """Check if the video upload service is running."""
    return {"status": "healthy", "service": "video_upload"}
