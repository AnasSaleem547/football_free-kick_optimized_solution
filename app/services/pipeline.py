"""Coordinate evidence, per-ball decisions, annotated clip export and delivery."""

from pathlib import Path
from .analysis import ALGORITHM_REVISION, detect_attempts
from .evidence import extract_evidence
from .rendering import render_outputs
from .cloud_uploader import make_uploader
from .video_io import check_cancel
from . import job_manager as jm


def run_pipeline(session_id, video_path, clips_dir, video_name=None, cancel_event=None):
    directory = Path(video_path).parent

    def progress(stage):
        return lambda current, total: jm.update_progress(
            session_id, current, total, stage=stage
        )

    check_cancel(cancel_event)
    evidence = extract_evidence(
        video_path,
        directory / "evidence",
        cancel_event=cancel_event,
        progress=progress("detecting"),
    )
    check_cancel(cancel_event)
    expected = evidence["decoded_frames"]
    jm.update_progress(session_id, expected, expected, stage="analyzing")
    result = detect_attempts(evidence)
    jm.update_progress(session_id, 0, expected, stage="rendering")
    clips = render_outputs(
        video_path,
        evidence,
        result,
        clips_dir,
        cancel_event=cancel_event,
        progress=progress("rendering"),
    )
    check_cancel(cancel_event)
    uploaded = []
    jm.update_progress(session_id, expected, expected, stage="uploading")
    if clips:
        uploader = make_uploader(
            video_name=video_name or Path(video_path).stem,
            session_dir=str(directory),
            session_id=session_id,
            cancel_event=cancel_event,
        )
        try:
            uploader["submit_all"](clips)
            uploaded = uploader["wait_all"]()
            check_cancel(cancel_event)
        finally:
            uploader["shutdown"]()
    by_number = {v["kick_number"]: v for v in uploaded}
    for clip in clips:
        upload = by_number.get(clip["kick_number"], {})
        clip["cloud_url"] = upload.get("cloud_url")
        clip["upload_error"] = (
            upload.get("error")
            if clip["cloud_url"]
            else upload.get("error", "Upload did not return a URL")
        )
    return {
        "total_kicks": len(result["events"]),
        "clips": clips,
        "upload_status": "complete"
        if all(c["cloud_url"] for c in clips)
        else "incomplete",
        "quality": result["quality"],
        "ball_coverage": result["ball_coverage"],
        "pose_coverage": result["pose_coverage"],
        "events": result["events"],
        "incomplete_attempts": result.get("incomplete_attempts", []),
        "visual_tracking": result.get("visual_tracking", {}),
        "diagnostics": result["diagnostics"],
        "algorithm_revision": ALGORITHM_REVISION,
        "source_sha256": evidence.get("sha256"),
        "decoded_frames": evidence["decoded_frames"],
    }
