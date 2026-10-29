"""One source decode produces measured annotations and independent kick clips."""

from pathlib import Path
import cv2
import numpy as np
from .annotator import update_ball_trails, draw_ball_scene, DisplayBallLabels, shot_trail_ids
from .flight_tracker import FlightTracker, flight_playback_observations
from .kick_extractor import clip_windows, ClipExtractor
from .video_io import open_video, decoded_frames, check_cancel


def render_outputs(
    video_path,
    evidence,
    result,
    clips_dir,
    *,
    width=640,
    trail_seconds=1.0,
    show_candidates=False,
    cancel_event=None,
    progress=None,
):
    expected = evidence["decoded_frames"]
    if any(
        len(result[k]) != expected
        for k in ("observations", "details", "ball_observations")
    ):
        raise ValueError("Rendering evidence is incomplete")
    check_cancel(cancel_event)
    if not result["events"]:
        result["visual_tracking"] = {
            "frames_with_visual_measurements": 0,
            "used_for_kick_decisions": False,
        }
        return []
    with open_video(video_path) as (capture, metadata):
        if metadata["total_frames"] != expected:
            raise ValueError("Video frame count differs from evidence")
        width = max(2, 2 * round(min(width, metadata["width"]) / 2))
        height = max(2, 2 * round(metadata["height"] / metadata["width"] * width / 2))
        scale = width / metadata["width"]
        duration = expected / evidence["fps"]
        windows = clip_windows(result["events"], duration)
        trails = {}
        labels = DisplayBallLabels()
        flight = FlightTracker()
        visual_frames = 0
        with ClipExtractor(clips_dir, windows, width, height, evidence["fps"]) as clips:
            for index, _, frame in decoded_frames(capture, metadata, cancel_event):
                obs = result["observations"][index]
                detail = result["details"][index]
                shift = np.asarray(detail["shift"])
                visual = flight.update(
                    frame,
                    obs.t,
                    shift,
                    result["ball_observations"][index],
                    result["events"],
                    camera_unreliable=detail["camera_unreliable"],
                )
                playback, markers = flight_playback_observations(
                    result["ball_observations"][index], detail["balls"], visual
                )
                visual_frames += bool(visual)
                frame = cv2.resize(frame, (width, height))
                active = detail["active_ball_id"]
                if visual and (
                    active is None or any(v["ball_id"] == active for v in visual)
                ):
                    active = visual[0]["ball_id"]
                paths = update_ball_trails(
                    trails,
                    playback,
                    obs.t,
                    active,
                    detail["camera_unreliable"],
                    trail_seconds,
                    shot_ball_ids=shot_trail_ids(result["events"], obs.t),
                )
                confirmed = {
                    e["ball_id"]: i + 1
                    for i, e in enumerate(result["events"])
                    if e["confirmed_at"] <= obs.t
                }
                markers = labels.update(markers, confirmed)
                feet = [np.asarray(foot) + shift for foot in obs.feet]
                draw_ball_scene(
                    frame,
                    paths,
                    markers,
                    active,
                    confirmed,
                    scale,
                    shift,
                    feet,
                    show_candidates=show_candidates,
                )
                clips.write(obs.t, frame)
                if progress:
                    progress(index + 1, expected)
            check_cancel(cancel_event)
        if len(clips.clips) != len(windows):
            for clip in clips.clips:
                Path(clip["file_path"]).unlink(missing_ok=True)
            raise ValueError("One or more kick windows produced no clip")
        result["visual_tracking"] = {
            "frames_with_visual_measurements": visual_frames,
            "used_for_kick_decisions": False,
        }
        return clips.clips
