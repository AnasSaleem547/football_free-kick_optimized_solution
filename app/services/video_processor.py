"""Bounded upload admission and minimal progress/result polling."""

import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.core.config import get_settings, session_path
from .save_video import save_uploaded_file
from .pipeline import run_pipeline
from . import job_manager as jm

log = logging.getLogger(__name__)
_slots = threading.BoundedSemaphore(get_settings().max_pending_jobs)
_executor = None
_executor_lock = threading.Lock()


def _get_executor():
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="football")
        return _executor


def _run_in_background(session_id, video_path, clips_dir, video_name, admission=None):
    try:
        job = jm.get_job(session_id)
        if job is None:
            raise ValueError("Missing job state")
        jm.mark_processing(session_id)
        if job.cancel_event.is_set():
            raise InterruptedError("Cancelled before processing")
        result = run_pipeline(
            session_id=session_id,
            video_path=video_path,
            clips_dir=clips_dir,
            video_name=video_name,
            cancel_event=job.cancel_event,
        )
        jm.mark_done(session_id, result)
        if (
            jm.get_job(session_id).status == "done"
            and not get_settings().keep_source_video
        ):
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                log.exception("Could not remove completed session source")
    except InterruptedError:
        jm.mark_cancelled(session_id)
    except Exception as error:
        log.exception("Football processing failed for %s", session_id)
        jm.mark_failed(session_id, str(error))
    finally:
        if admission is not None:
            admission.release()


def process_upload(file: UploadFile):
    if not _slots.acquire(blocking=False):
        raise HTTPException(
            503, "Processing capacity is full; retry after a current job finishes"
        )
    submitted = False
    session_id = None
    job_created = False
    try:
        session_id, path, clips, name = save_uploaded_file(file)
        jm.create_job(session_id)
        job_created = True
        _get_executor().submit(
            _run_in_background, session_id, path, clips, name, _slots
        )
        submitted = True
        return {"session_id": session_id}
    except HTTPException:
        raise
    except Exception as error:
        if session_id:
            if job_created:
                jm.mark_failed(session_id, str(error))
            else:
                # No queryable job owns this newly copied source. Do not leave
                # an orphan when initial status persistence fails.
                shutil.rmtree(session_path(session_id), ignore_errors=True)
        raise HTTPException(500, "Could not process video") from error
    finally:
        if not submitted:
            _slots.release()


def _get_job(session_id):
    try:
        job = jm.get_job(session_id)
    except ValueError as error:
        raise HTTPException(400, "Invalid session identifier") from error
    if job is None:
        raise HTTPException(404, "Session not found")
    return job


def get_status(session_id):
    job = _get_job(session_id)
    if job.status in jm.ACTIVE_STATUSES:
        return {
            "stage": job.stage,
            "current_frame": job.current_frame,
            "total_frames": job.total_frames,
            "progress_pct": job.progress,
        }
    if job.status == "cancelled":
        raise HTTPException(503, "Video processing was cancelled")
    if job.status != "done" or job.result is None:
        raise HTTPException(500, "Video processing failed; upload the source again")
    result = job.result
    clips = sorted(result["clips"], key=lambda clip: clip["kick_number"])
    urls = [clip.get("cloud_url") for clip in clips]
    if (result.get("upload_status") != "complete"
            or len(urls) != result["total_kicks"] or not all(urls)):
        log.warning("Incomplete Cloudinary delivery for session %s", session_id)
        raise HTTPException(
            502, "Not all kick clips were delivered to Cloudinary; local results were retained"
        )
    return {"total_kicks": result["total_kicks"], "cloudinary_urls": urls}


def retry_uploads(session_id):
    from .cloud_uploader import retry_failed_uploads

    try:
        directory = session_path(session_id)
    except ValueError as error:
        raise HTTPException(400, "Invalid session identifier") from error
    if not directory.is_dir():
        raise HTTPException(404, "Session not found")
    job = jm.get_job(session_id)
    if job and job.status in ("pending", "processing"):
        raise HTTPException(
            409, "Wait for processing to finish before retrying uploads"
        )
    outcome = retry_failed_uploads(str(directory))
    jm.refresh_result(session_id)
    return {"session_id": session_id, **outcome}


def cancel_session(session_id):
    job = _get_job(session_id)
    requested = jm.request_cancel(session_id)
    return {
        "session_id": session_id,
        "message": "Cancellation requested; processing stops after the current inference or network operation."
        if requested
        else f"Nothing to cancel — job is already '{job.status}'.",
    }


def shutdown_workers():
    global _executor
    jm.cancel_all()
    with _executor_lock:
        executor = _executor
        _executor = None
    if executor:
        executor.shutdown(wait=True)
