"""Cloudinary delivery with a durable per-session pending/failed upload record."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from weakref import WeakValueDictionary
from .storage import atomic_json
from .video_io import check_cancel
from app.core.config import get_settings

MAX_RETRIES = 3
RETRY_BACKOFF_BASE_S = 2.0
MAX_PARALLEL_UPLOADS = 4
CLOUD_ROOT_FOLDER = "football_kicks"
FAILED_UPLOADS_FILE = "failed_uploads.json"
_credentials_loaded = False
_credentials_lock = threading.Lock()
_delivery_locks = WeakValueDictionary()
_delivery_guard = threading.Lock()


def _session_lock(directory):
    key = str(Path(directory).resolve())
    with _delivery_guard:
        return _delivery_locks.setdefault(key, threading.RLock())


def _ensure_credentials():
    global _credentials_loaded
    with _credentials_lock:
        if _credentials_loaded:
            return
        from dotenv import load_dotenv

        load_dotenv()
        values = [
            os.getenv(k)
            for k in (
                "CLOUDINARY_CLOUD_NAME",
                "CLOUDINARY_API_KEY",
                "CLOUDINARY_API_SECRET",
            )
        ]
        if not all(values):
            raise RuntimeError("Cloudinary credentials are not configured")
        import cloudinary

        cloudinary.config(
            cloud_name=values[0], api_key=values[1], api_secret=values[2], secure=True
        )
        _credentials_loaded = True


def _upload_one(clip_path, folder, public_id, cancel_event=None):
    if not Path(clip_path).is_file():
        return {"cloud_url": None, "error": "Local clip is missing"}
    import cloudinary.uploader

    for attempt in range(MAX_RETRIES):
        check_cancel(cancel_event)
        try:
            response = cloudinary.uploader.upload(
                clip_path,
                resource_type="video",
                folder=folder,
                public_id=public_id,
                overwrite=True,
                timeout=60,
            )
            url = response.get("secure_url")
            if not url:
                raise RuntimeError("Cloudinary returned no secure video URL")
            return {"cloud_url": url, "error": None}
        except Exception as error:
            message = str(error)
            if attempt + 1 < MAX_RETRIES:
                delay = RETRY_BACKOFF_BASE_S * 2**attempt
                if cancel_event is not None:
                    if cancel_event.wait(delay):
                        check_cancel(cancel_event)
                else:
                    time.sleep(delay)
    return {"cloud_url": None, "error": message}


def _persist_failed(session_dir, video_name, folder, failed):
    path = Path(session_dir) / FAILED_UPLOADS_FILE
    if failed:
        atomic_json(
            path,
            {
                "video_name": video_name,
                "cloud_folder": folder,
                "failed_uploads": failed,
            },
        )
    else:
        path.unlink(missing_ok=True)


def _delivered(directory):
    path = Path(directory) / "uploads.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["uploads"]


def _record_delivery(directory, entry, url):
    records = {v["kick_number"]: v for v in _delivered(directory)}
    records[entry["kick_number"]] = {
        "kick_number": entry["kick_number"],
        "cloud_url": url,
    }
    atomic_json(Path(directory) / "uploads.json", {"uploads": list(records.values())})


def make_uploader(video_name, session_dir, session_id, cancel_event=None):
    try:
        if not get_settings().cloud_uploads:
            raise RuntimeError("Cloudinary delivery is disabled")
        _ensure_credentials()
        enabled = True
        init_error = None
    except Exception as error:
        enabled = False
        init_error = str(error)
    folder = f"{CLOUD_ROOT_FOLDER}/{video_name}/{session_id}"
    executor = (
        ThreadPoolExecutor(
            max_workers=MAX_PARALLEL_UPLOADS, thread_name_prefix="cloudinary"
        )
        if enabled
        else None
    )
    entries = []
    futures = []
    lock = _session_lock(session_dir)

    def deliver(entry):
        try:
            value = _upload_one(
                entry["local_path"], folder, f"kick_{entry['kick_number']:02d}", cancel_event
            )
        except InterruptedError:
            raise  # Leave the pending record available for a deliberate retry.
        except Exception as error:
            value = {"cloud_url": None, "error": str(error)}
        with lock:
            if value["cloud_url"]:
                # Remote completion is recorded even if the caller cancelled
                # before collecting this future. Clear pending only afterward.
                _record_delivery(session_dir, entry, value["cloud_url"])
                entries.remove(entry)
            else:
                entry["error"] = value["error"]
            _persist_failed(session_dir, video_name, folder, entries)
        return value

    def submit_all(clips):
        with lock:
            if entries or futures:
                raise RuntimeError("Clip batch has already been submitted")
            entries.extend(
                {
                    "kick_number": clip["kick_number"],
                    "local_path": clip["file_path"],
                    "error": init_error or "Upload pending",
                }
                for clip in clips
            )
            # Register the complete local batch before cancellation, executor
            # failure, or a network response can leave only a partial retry list.
            _persist_failed(session_dir, video_name, folder, entries)
            for entry in entries:
                check_cancel(cancel_event)
                future = executor.submit(deliver, entry) if executor else None
                futures.append((entry, future))

    def wait_all():
        results = []
        for entry, future in futures:
            check_cancel(cancel_event)
            try:
                value = (
                    future.result()
                    if future
                    else {"cloud_url": None, "error": init_error}
                )
            except InterruptedError:
                raise
            except Exception as error:
                value = {"cloud_url": None, "error": str(error)}
            results.append({"kick_number": entry["kick_number"], **value})
        return results

    def shutdown():
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)

    return {
        "submit_all": submit_all,
        "wait_all": wait_all,
        "shutdown": shutdown,
    }


def _merge_recovered(directory, recovered):
    path = directory / "result.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    urls = {item["kick_number"]: item["cloud_url"] for item in recovered}
    for clip in result.get("clips", []):
        if clip["kick_number"] in urls:
            clip["cloud_url"] = urls[clip["kick_number"]]
            clip["upload_error"] = None
    result["upload_status"] = (
        "complete"
        if all(c.get("cloud_url") for c in result.get("clips", []))
        else "incomplete"
    )
    atomic_json(path, result)


def retry_failed_uploads(session_dir):
    directory = Path(session_dir)
    with _session_lock(directory):
        recovered = _delivered(directory)
        _merge_recovered(directory, recovered)
        path = directory / FAILED_UPLOADS_FILE
        if not path.exists():
            return (
                {"status": "ok", "recovered": recovered, "still_failed": []}
                if recovered
                else {
                    "status": "nothing_to_retry",
                    "message": "No pending or failed clip uploads",
                }
            )
        try:
            payload = json.loads(path.read_text())
            known = {v["kick_number"] for v in recovered}
            failed = [
                v for v in payload["failed_uploads"] if v["kick_number"] not in known
            ]
            _persist_failed(
                directory, payload["video_name"], payload["cloud_folder"], failed
            )
            if failed:
                _ensure_credentials()
        except Exception as error:
            return {"status": "error", "message": str(error), "recovered": recovered}
        for item in list(failed):
            clip = Path(item["local_path"]).resolve()
            if not clip.is_relative_to(directory.resolve()):
                value = {"cloud_url": None, "error": "Clip path is outside the session"}
            else:
                value = _upload_one(
                    str(clip),
                    payload["cloud_folder"],
                    f"kick_{item['kick_number']:02d}",
                )
            if value["cloud_url"]:
                _record_delivery(directory, item, value["cloud_url"])
                recovered = _delivered(directory)
                # Both the URL ledger and any complete result precede retry-record removal.
                _merge_recovered(directory, recovered)
                failed.remove(item)
            else:
                item["error"] = value["error"]
            _persist_failed(
                directory, payload["video_name"], payload["cloud_folder"], failed
            )
        return {
            "status": "ok",
            "recovered": recovered,
            "still_failed": failed,
            "summary": f"{len(recovered)} delivered, {len(failed)} still failed",
        }
