"""Single-process job coordination with atomic on-disk terminal results/status."""

import copy
import json
import threading
from dataclasses import dataclass, field, replace
from typing import Optional
from app.core.config import session_path
from .storage import atomic_json


@dataclass
class Job:
    session_id: str
    status: str = "pending"
    stage: str = "queued"
    current_frame: int = 0
    total_frames: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def progress(self):
        if self.status == "done":
            return 100.0
        return (
            round(100 * self.current_frame / self.total_frames, 2)
            if self.total_frames else 0.0
        )


_store = {}
_lock = threading.RLock()
ACTIVE_STATUSES = {"pending", "processing"}


def _remember(job):
    _store.pop(job.session_id, None)
    _store[job.session_id] = job
    # Terminal results stay on disk; only recent snapshots need memory.
    for key in list(_store):
        if len(_store) <= 32:
            break
        if _store[key].status in ("done", "failed", "cancelled"):
            del _store[key]


def _persist(job):
    atomic_json(
        session_path(job.session_id) / "job.json",
        {
            key: getattr(job, key)
            for key in (
                "session_id",
                "status",
                "stage",
                "current_frame",
                "total_frames",
                "error",
            )
        },
    )


def _commit(job, **changes):
    """Publish a new snapshot only after its status has been saved."""
    updated = replace(job, **changes)
    _persist(updated)
    _remember(updated)


def _lookup(session_id):
    directory = session_path(session_id)
    if session_id in _store:
        return _store[session_id]
    path = directory / "job.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data.pop("progress", None)  # Older checkpoints also stored a percentage.
    job = Job(**data)
    if job.status in ACTIVE_STATUSES:
        job = replace(
            job, status="failed",
            error="Processing interrupted by application restart; upload the source again.",
        )
        _persist(job)
    if job.status == "done":
        job.result = json.loads((directory / "result.json").read_text())
    _remember(job)
    return job


def create_job(session_id):
    with _lock:
        if _lookup(session_id) is not None:
            raise ValueError("Session already exists")
        job = Job(session_id)
        _persist(job)
        _remember(job)
        return copy.copy(job)


def get_job(session_id):
    with _lock:
        job = _lookup(session_id)
        if job is None:
            return None
        snapshot = copy.copy(job)
        snapshot.result = copy.deepcopy(job.result)
        return snapshot


def mark_processing(session_id):
    with _lock:
        job = _lookup(session_id)
        if job and job.status == "pending":
            _commit(job, status="cancelled" if job.cancel_event.is_set() else "processing",
                    stage="preparing")


def update_progress(session_id, current_frame, total_frames, *, stage=None):
    with _lock:
        job = _lookup(session_id)
        if job and job.status == "processing":
            updated = replace(
                job,
                current_frame=current_frame, total_frames=total_frames,
                stage=stage or job.stage,
            )
            if updated.stage != job.stage or int(updated.progress) > int(job.progress):
                _persist(updated)
            _remember(updated)


def mark_done(session_id, result):
    with _lock:
        job = _lookup(session_id)
        if job and job.status in ACTIVE_STATUSES:
            if job.cancel_event.is_set():
                _commit(job, status="cancelled")
                return
            # Complete result is durable before any reader can see done.
            atomic_json(session_path(session_id) / "result.json", result)
            _commit(job, result=copy.deepcopy(result), status="done", stage="done")


def mark_failed(session_id, error):
    with _lock:
        job = _lookup(session_id)
        if job and job.status in ACTIVE_STATUSES:
            _commit(job, status="failed", error=str(error))


def mark_cancelled(session_id):
    with _lock:
        job = _lookup(session_id)
        if job and job.status in ACTIVE_STATUSES:
            _commit(job, status="cancelled")


def request_cancel(session_id):
    with _lock:
        job = _lookup(session_id)
        if job and job.status in ACTIVE_STATUSES:
            job.cancel_event.set()
            return True
        return False


def refresh_result(session_id):
    with _lock:
        job = _lookup(session_id)
        if job and job.status == "done":
            job.result = json.loads(
                (session_path(session_id) / "result.json").read_text()
            )


def cancel_all():
    with _lock:
        for job in _store.values():
            if job.status in ACTIVE_STATUSES:
                job.cancel_event.set()
