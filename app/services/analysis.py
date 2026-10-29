"""Per-ball kick decisions from measured evidence, independent of model execution."""

import numpy as np
from .tracking import MultiBallTracker, BALL_VALIDATION_SETTINGS
from .kick_detector import Observation, ShotConfig, StationaryShotDetector, _distance

ALGORITHM_REVISION = "phase-aware-evidence-v10"


def quality_verdict(decoded, ball_measured, pose_measured, expected_frames):
    if decoded <= 0 or (expected_frames > 0 and decoded < expected_frames - 2):
        raise ValueError(f"Incomplete decode: {decoded}/{expected_frames} frames")
    if ball_measured / decoded < 0.50:
        return "review_required:ball_coverage"
    if pose_measured / decoded < 0.10:
        return "review_required:pose_coverage"
    return "ok"


class StationarySetupMemory:
    """Retain measured setup proof through a bounded, unique reacquisition.

    No missing interval is credited as stationary evidence. Recovery requires
    a validated new identity and a fresh measured stationary suffix. Actual
    movement, ambiguity, stale proof and camera discontinuity prevent reuse.
    """

    def __init__(self, config):
        self.config = config
        self.entries = {}
        self.extents = {}

    @staticmethod
    def _stationary_suffix(history):
        # A stable return has no resolved directional drift. Compare fitted
        # movement with three standard errors of its measured scatter rather
        # than accepting a slowly rolling sequence inside a position radius.
        times = np.asarray([p.t for p in history], dtype=float)
        times -= times.mean()
        spread = float(times @ times)
        points = np.asarray([p.ball for p in history], dtype=float)
        centered = points - points.mean(axis=0)
        velocity = times @ centered / spread
        residual = centered - times[:, None] * velocity
        variance = np.square(residual).sum() / max(1, len(history) - 2)
        return float(np.linalg.norm(velocity)) <= 3 * np.sqrt(variance / spread) + np.finfo(float).eps

    def update(self, t, observations, detectors, camera_unreliable):
        c = self.config
        if camera_unreliable:
            self.entries.clear()
            self.extents.clear()
            return
        self.extents = {bid: extent for bid, extent in self.extents.items() if bid in observations}
        for bid, obs in observations.items():
            if obs.ball is not None and obs.source in ('detector', 'candidate'):
                x, y = obs.ball
                extent = self.extents.setdefault(bid, (x, y, x, y))
                self.extents[bid] = (min(x, extent[0]), min(y, extent[1]),
                                     max(x, extent[2]), max(y, extent[3]))
        for identity, memory in list(self.entries.items()):
            obs = observations.get(memory.get('reacquiring_id', identity))
            measured = obs is not None and obs.ball is not None and obs.source in ('detector', 'candidate')
            moved = measured and _distance(obs.ball, memory['setup']['anchor']) > c.stationary_radius_body * memory['setup']['height']
            detector = detectors.get(identity)
            deadline = memory['seen'] + c.stationary_window_s
            if memory.get('pending') is not None:
                deadline = max(deadline, memory['pending'] + BALL_VALIDATION_SETTINGS['window_s'])
            if (t > deadline + 1e-9 or moved
                    or detector is not None and detector._candidate is not None):
                del self.entries[identity]
        matches = {}
        for identity, memory in self.entries.items():
            setup = memory['setup']
            matches[identity] = [bid for bid, obs in observations.items()
                if obs.ball is not None and obs.source in ('detector', 'candidate')
                and obs.confidence >= c.validated_setup_min_confidence
                and (bid == identity or all(_distance((x, y), setup['anchor'])
                     <= c.stationary_radius_body * setup['height']
                     for x in (self.extents[bid][0], self.extents[bid][2])
                     for y in (self.extents[bid][1], self.extents[bid][3])))
                and _distance(obs.ball, setup['anchor']) <= c.stationary_radius_body * setup['height']
                and .5 <= obs.diameter / setup['diameter'] <= 1.5]
        for identity, choices in matches.items():
            if len(choices) != 1:
                continue
            current = choices[0]
            if sum(current in group for group in matches.values()) != 1:
                continue
            obs, detector = observations[current], detectors[current]
            memory = self.entries[identity]
            memory['reacquiring_id'] = current
            if obs.confidence >= c.min_confidence:
                memory['seen'] = t
                memory['pending'] = None
            elif (memory.get('pending') is None
                    and t <= memory['seen'] + c.stationary_window_s + 1e-9):
                # A weak measured return may start identity validation, but
                # cannot keep extending its own deadline without strong support.
                memory['pending'] = t
            history = detector._history
            setup = memory['setup']
            if (detector._setup is None and detector._candidate is None
                    and obs.validated and obs.confidence >= c.min_confidence
                    and len(history) >= c.stationary_min_samples
                    and history[-1].t - history[0].t >= BALL_VALIDATION_SETTINGS['minimum_span_s'] - 1e-9
                    and self._stationary_suffix(history)
                    and all(_distance(p.ball, setup['anchor']) <= c.stationary_radius_body * setup['height']
                            and .5 <= p.diameter / setup['diameter'] <= 1.5 for p in history)):
                detector._setup = dict(setup)
                detector._log(t, 'stationary_setup_recovered')
                if current != identity:
                    del self.entries[identity]
                    self.entries[current] = memory
        for identity, detector in detectors.items():
            obs = observations.get(identity)
            setup = detector._setup
            if (setup is not None and detector._candidate is None and obs is not None
                    and obs.ball is not None and obs.source == 'detector' and obs.validated
                    and obs.confidence >= c.validated_setup_min_confidence
                    and _distance(obs.ball, setup['anchor']) <= c.stationary_radius_body * setup['height']):
                self.entries[identity] = {'setup': dict(setup), 'seen': t}



def all_ball_observations(evidence):
    tracker = MultiBallTracker()
    players = {}
    frames_out = []
    details = []
    last_pose_t = -1e9
    # Reuse the same camera-adjusted detections in overlapping lookahead windows.
    adjusted = []
    for frame in evidence["frames"]:
        shift = np.asarray(frame.get("shift", [0.0, 0.0]), dtype=float)
        adjusted.append((shift, [
            (b[0] - shift[0], b[1] - shift[1], b[2], b[3]) for b in frame["balls"]
        ]))
    for frame_index, frame in enumerate(evidence["frames"]):
        t = float(frame["t"])
        shift, candidates = adjusted[frame_index]
        unreliable = frame.get("camera_cut", False) or not frame.get(
            "camera_reliable", True
        )
        if unreliable:
            tracker.reset()
            players.clear()
        # A short look-ahead uses saved detections only; no additional inference.
        future = []
        for next_index in range(frame_index + 1, len(evidence['frames'])):
            next_frame = evidence['frames'][next_index]
            next_t = float(next_frame['t'])
            if (next_t - t > BALL_VALIDATION_SETTINGS['window_s'] + 1e-9
                    or next_frame.get('camera_cut', False)
                    or not next_frame.get('camera_reliable', True)):
                break
            future.append((next_t, adjusted[next_index][1]))
        tracks = tracker.update(t, candidates, future)
        poses = frame.get("poses", [])
        if poses:
            last_pose_t = t
        observations = {}
        visible = []
        for track in tracks:
            ball_id = track["ball_id"]
            ball = track["ball"]
            old = players.get(ball_id)
            if poses:

                def score(p):
                    feet = np.asarray(p["ankles"], dtype=float).reshape(-1, 2) - shift
                    height = max(40.0, p["box"][3] - p["box"][1])
                    if ball is None or not len(feet):
                        return 1e6
                    return (
                        min(float(np.linalg.norm(f - np.asarray(ball))) for f in feet)
                        / height
                    )

                player = min(poses, key=score)
                height = max(40.0, player["box"][3] - player["box"][1])
                hip = np.asarray(player["hip"]) - shift
                feet = np.asarray(player["ankles"], dtype=float).reshape(-1, 2) - shift
                swing = 0.0
                if (
                    old
                    and 0 < t - old["t"] <= 0.2
                    and len(feet) == len(old["feet"]) == 2
                    and np.linalg.norm(hip - old["hip"]) < 0.65 * height
                ):
                    swing = float(
                        np.max(
                            np.linalg.norm(
                                (feet - hip) - (old["feet"] - old["hip"]), axis=1
                            )
                        )
                        / (t - old["t"])
                        / height
                    )
                    if swing > 12.0:
                        swing = 0.0
                old = {
                    "t": t,
                    "height": height,
                    "hip": hip,
                    "feet": feet,
                    "swing": swing,
                }
                players[ball_id] = old
            fresh = old is not None and t - old["t"] <= 0.12
            obs = Observation(
                t=t,
                ball=ball,
                diameter=track["diameter"],
                confidence=track["confidence"],
                validated=track["validated"],
                feet=tuple(map(tuple, old["feet"])) if fresh else (),
                body_height=old["height"] if old else 180.0,
                foot_speed=old["swing"] if fresh else 0.0,
                source="camera_unreliable"
                if unreliable
                else (
                    "candidate"
                    if track["source"] == "detector" and not track["validated"]
                    else track["source"]
                ),
            )
            observations[ball_id] = obs
            visible.append(
                {
                    "ball_id": ball_id,
                    "raw_ball": None
                    if ball is None
                    else [ball[0] + shift[0], ball[1] + shift[1]],
                    "diameter": obs.diameter,
                    "confidence": obs.confidence,
                    "source": "camera_unreliable" if unreliable else track["source"],
                    "pose_fresh": fresh,
                    "validated": track["validated"],
                }
            )
        for b in frame["balls"]:
            if not any(
                v["source"] == "detector"
                and np.linalg.norm(np.asarray(v["raw_ball"]) - b[:2])
                < 0.45 * min(v["diameter"], b[2])
                for v in visible
            ):
                visible.append(
                    {
                        "ball_id": None,
                        "raw_ball": list(b[:2]),
                        "diameter": b[2],
                        "confidence": b[3],
                        "source": "detector",
                        "pose_fresh": False,
                        "validated": False,
                    }
                )
        alive = set(observations)
        players = {k: v for k, v in players.items() if k in alive}
        frames_out.append(observations)
        details.append(
            {
                "balls": visible,
                "shift": list(map(float, shift)),
                "pose_fresh": t - last_pose_t <= 0.12,
                "camera_unreliable": unreliable,
                "t": t,
            }
        )
    return frames_out, details


def focus_ball(observations):
    if not observations:
        return None
    measured = [
        (bid, o)
        for bid, o in observations.items()
        if o.source == "detector" and o.confidence >= 0.45 and o.ball is not None
    ]
    if not measured:
        weak = [
            bid
            for bid, o in observations.items()
            if o.source == "detector" and o.ball is not None and o.confidence >= 0.18
        ]
        return max(weak, key=lambda bid: observations[bid].confidence) if weak else None
    nearby = []
    for bid, obs in measured:
        distance = min(
            (_distance(foot, obs.ball) / obs.body_height for foot in obs.feet),
            default=2.0,
        )
        if distance <= 0.35:
            nearby.append((distance, -obs.confidence, bid))
    if nearby:
        return min(nearby)[2]
    # Display focus before approach is a foreground preference, not a kick decision.
    return max(measured, key=lambda item: item[1].diameter * item[1].confidence)[0]




def detect_attempts(evidence, config=None):
    cfg = config or ShotConfig()
    frames, details = all_ball_observations(evidence)
    detectors = {}
    setup_memory = StationarySetupMemory(cfg)
    events = []
    diagnostics = []
    observations = []
    accepted_ball = None
    accepted_t = -1e9
    for frame, detail in zip(frames, details):
        t = detail["t"]
        for ball_id in list(detectors):
            if ball_id not in frame:
                d = detectors.pop(ball_id)
                d.update(
                    Observation(
                        t=t,
                        ball=None,
                        source="camera_unreliable"
                        if detail["camera_unreliable"]
                        else "missing",
                    )
                )
                d.finish(reason="track_lost")
                diagnostics.extend(
                    {"ball_id": ball_id, **entry} for entry in d.diagnostics
                )
        for ball_id, obs in frame.items():
            if ball_id not in detectors:
                detectors[ball_id] = StationaryShotDetector(cfg)
            d = detectors[ball_id]
            event = d.update(obs)
            if event is not None:
                events.append({"ball_id": ball_id, **event})
                accepted_ball = ball_id
                accepted_t = t
        setup_memory.update(t, frame, detectors, detail["camera_unreliable"])
        active = focus_ball(frame)
        confirming = [i for i, d in detectors.items() if d._candidate is not None]
        if confirming:
            active = min(confirming, key=lambda i: detectors[i]._candidate["first"])
        elif (
            accepted_ball in frame
            and frame[accepted_ball].source == "detector"
            and t - accepted_t <= 1.0
        ):
            active = accepted_ball
        obs = (
            frame[active]
            if active is not None
            else Observation(t=t, ball=None, source="missing")
        )
        detail["active_ball_id"] = active
        detail["raw_ball"] = (
            None if obs.ball is None else list(np.asarray(obs.ball) + detail["shift"])
        )
        observations.append(obs)
    incomplete_attempts = []
    for ball_id, d in detectors.items():
        pending = d.finish()
        if pending is not None:
            incomplete_attempts.append({"ball_id": ball_id, **pending})
        diagnostics.extend({"ball_id": ball_id, **entry} for entry in d.diagnostics)
    events.sort(key=lambda e: e["t"])
    diagnostics.sort(key=lambda d: d["t"] if d["t"] is not None else -1.0)
    measured = sum(
        any(
            o.source == "detector" and o.confidence >= cfg.min_confidence
            for o in frame.values()
        )
        for frame in frames
    )
    pose_measured = sum(d["pose_fresh"] for d in details)
    quality = quality_verdict(
        len(frames), measured, pose_measured, evidence.get("total_frames", len(frames))
    )
    return {
        "events": events,
        "incomplete_attempts": incomplete_attempts,
        "diagnostics": diagnostics,
        "quality": quality,
        "observations": observations,
        "ball_observations": frames,
        "details": details,
        "ball_coverage": measured / len(frames),
        "pose_coverage": pose_measured / len(frames),
    }
