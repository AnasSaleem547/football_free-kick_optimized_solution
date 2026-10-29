"""Measured-only ball markers and short, independent ball trajectories."""

import math
from collections import deque
import cv2
import numpy as np


class DisplayBallLabels:
    """Compact visible labels; internal tracking IDs remain untouched."""

    def __init__(self):
        self.labels = {}
        self.kicks = {}
        self.next_label = 1

    def update(self, balls, confirmed):
        output = []
        for original in balls:
            ball = dict(original)
            if (ball.get("validated") and ball.get("raw_ball") is not None
                    and ball["source"] in ("detector", "visual")):
                identity = ball["ball_id"]
                kick = ball.get("kick_number", confirmed.get(identity))
                if kick in self.kicks:
                    self.labels[identity] = self.kicks[kick]
                if identity not in self.labels:
                    self.labels[identity] = self.next_label
                    self.next_label += 1
                ball["display_id"] = self.labels[identity]
                if kick is not None:
                    self.kicks[kick] = self.labels[identity]
            output.append(ball)
        return output


class FadingSegment(list):
    def __init__(self):
        super().__init__()
        self.opacity = []


def shot_trail_ids(events, t, window_s=3.0):
    """Only known shot windows produce trails, never ordinary ball retrieval."""
    return {e["ball_id"] for e in events if e["t"] <= t <= e["t"] + window_s}


class TrajectoryTrail:
    def __init__(self, lifetime_s=1.0, maximum_join_gap_s=0.18):
        self.lifetime_s = lifetime_s
        self.maximum_join_gap_s = maximum_join_gap_s
        self.points = deque()
        self.segment = 0
        self.last_measurement_t = None
        self.break_before_next = False

    def update(self, t, point, source, confidence):
        if source == "camera_unreliable":
            self.points.clear()
            self.last_measurement_t = None
        while self.points and t - self.points[0][0] > self.lifetime_s:
            self.points.popleft()
        measured = point is not None and (
            (source == "detector" and confidence >= 0.45)
            or (source == "visual" and confidence >= 0.62)
        )
        if measured:
            if self.break_before_next or (
                self.last_measurement_t is not None
                and t - self.last_measurement_t > self.maximum_join_gap_s
            ):
                self.segment += 1
            self.points.append((t, tuple(map(float, point)), self.segment))
            self.last_measurement_t = t
            self.break_before_next = False
        else:
            self.break_before_next = True
        segments = []
        previous_segment = None
        for measured_t, p, segment in self.points:
            if segment != previous_segment:
                segments.append(FadingSegment())
                previous_segment = segment
            segments[-1].append(p)
            # One age fade only. Losing evidence breaks the line, but does not
            # erase already measured positions or revive them on reacquisition.
            segments[-1].opacity.append(max(0.0, 1.0 - (t - measured_t) / self.lifetime_s))
        return segments


def draw_trajectory(frame, segments, scale, shift):
    """Thin cyan measured segments, fading with age and after lost evidence."""
    shift = np.asarray(shift, dtype=float)
    for segment in segments:
        if len(segment) < 2:
            continue
        projected = (np.asarray(segment) + shift) * scale
        if np.max(np.linalg.norm(projected - projected[0], axis=1)) < 1.5:
            continue
        points = np.rint(projected).astype(np.int32)
        opacity = getattr(segment, "opacity", [1.0] * len(points))
        for i, (a, b) in enumerate(zip(points, points[1:])):
            alpha = (opacity[i] + opacity[i + 1]) / 2
            if alpha <= 0:
                continue
            low = np.maximum(np.minimum(a, b) - 3, (0, 0))
            high = np.minimum(np.maximum(a, b) + 4, (frame.shape[1], frame.shape[0]))
            if np.any(high <= low):
                continue
            roi = frame[low[1]:high[1], low[0]:high[0]]
            overlay = roi.copy()
            cv2.line(overlay, tuple(a - low), tuple(b - low), (25, 25, 25), 2, cv2.LINE_AA)
            cv2.line(overlay, tuple(a - low), tuple(b - low), (255, 220, 80), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def update_ball_trails(
    trails, observations, t, active_id, camera_unreliable, lifetime_s=1.0,
    shot_ball_ids=None,
):
    if camera_unreliable:
        trails.clear()
    eligible = ({active_id} if active_id is not None else set()) if shot_ball_ids is None else shot_ball_ids
    for identity in eligible:
        if identity not in trails:
            trails[identity] = TrajectoryTrail(lifetime_s=lifetime_s)
    output = {}
    for ball_id in list(trails):
        obs = observations.get(ball_id) if shot_ball_ids is None or ball_id in eligible else None
        points = trails[ball_id].update(
            t,
            obs.ball if obs else None,
            obs.source if obs else "missing",
            obs.confidence if obs else 0.0,
        )
        if not points and ball_id != active_id:
            del trails[ball_id]
        else:
            output[ball_id] = points
    return output


def draw_ball_markers(frame, balls, active_id, confirmed, scale, show_candidates=False):
    for ball in balls:
        if ball["source"] not in ("detector", "visual") or ball["raw_ball"] is None:
            continue
        ball_id = ball["ball_id"]
        p = tuple(int(v * scale) for v in ball["raw_ball"])
        if not ball.get("validated", False):
            if show_candidates:
                cv2.drawMarker(
                    frame,
                    p,
                    (160, 160, 160),
                    cv2.MARKER_TILTED_CROSS,
                    6,
                    1,
                    cv2.LINE_AA,
                )
                origin = (
                    max(0, min(p[0] + 5, frame.shape[1] - 110)),
                    max(64, p[1] - 6),
                )
                cv2.putText(
                    frame,
                    f"candidate {ball['confidence']:.2f}",
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.33,
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )
            continue
        radius = max(1, int(math.ceil(ball["diameter"] * scale / 2))) + 3
        active = ball_id == active_id
        color = (255, 220, 80) if active else (235, 235, 235)
        for angle in (10, 100, 190, 280):
            cv2.ellipse(frame, p, (radius, radius), 0, angle, angle + 45, (25, 25, 25), 2, cv2.LINE_AA)
            cv2.ellipse(frame, p, (radius, radius), 0, angle, angle + 45, color, 1, cv2.LINE_AA)
        label = f"B{ball.get('display_id', ball_id)}" if ball_id is not None else "?"
        if ball_id in confirmed:
            label += f" KICK {confirmed[ball_id]}"
        elif ball.get("kick_number") is not None:
            label += f" KICK {ball['kick_number']}"
        if show_candidates:
            label += f" {ball['source']} {ball['confidence']:.2f}"
        origin = (
            max(0, min(p[0] - radius, frame.shape[1] - 110)),
            max(64, p[1] - radius - 5),
        )
        cv2.putText(
            frame,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA
        )


def draw_ball_scene(
    frame,
    paths,
    balls,
    active_id,
    confirmed,
    scale,
    shift,
    feet=(),
    show_candidates=False,
):
    """Keep original ball pixels visible beneath paths and foot markers."""
    saved = []
    for ball in balls:
        if (
            ball["source"] not in ("detector", "visual")
            or ball["raw_ball"] is None
            or not ball.get("validated", False)
        ):
            continue
        cx, cy = (int(v * scale) for v in ball["raw_ball"])
        radius = max(1, int(math.ceil(ball["diameter"] * scale / 2)))
        x1 = max(0, cx - radius)
        x2 = min(frame.shape[1], cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(frame.shape[0], cy + radius + 1)
        if x1 >= x2 or y1 >= y2:
            continue
        yy, xx = np.ogrid[y1:y2, x1:x2]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        region = frame[y1:y2, x1:x2]
        saved.append((region, mask, region[mask].copy()))
    for segments in paths.values():
        draw_trajectory(frame, segments, scale, shift)
    for foot in feet:
        p = tuple(int(v * scale) for v in foot)
        cv2.circle(frame, p, 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 2, (255, 255, 0), -1, cv2.LINE_AA)
    draw_ball_markers(
        frame, balls, active_id, confirmed, scale, show_candidates=show_candidates
    )
    for region, mask, pixels in saved:
        region[mask] = pixels
