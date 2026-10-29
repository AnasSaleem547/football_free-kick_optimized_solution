# @title 3. Stationary setup and launch confirmation
"""Conservative, CPU-only stationary-ball attempt decisions.

Coordinates MUST share one camera-stabilizing transform for ball and ankles.
Only source='detector' observations at sufficient confidence are evidence.
This recognizes a stationary setup, contact and sustained launch; ball/ankle
motion alone cannot perfectly separate a forceful pass from an attempted shot.
Create a new detector for every video. No filename or outcome enters decisions.
"""

from dataclasses import dataclass
from math import hypot, isfinite
from statistics import median
from typing import Optional


@dataclass(frozen=True)
class Observation:
    t: float
    ball: Optional[tuple[float, float]]
    diameter: float = 16.0
    confidence: float = 0.0
    feet: tuple = ()
    body_height: float = 180.0
    foot_speed: float = 0.0
    source: str = "detector"
    validated: bool = False


@dataclass(frozen=True)
class ShotConfig:
    min_confidence: float = 0.45
    validated_setup_min_confidence: float = 0.18
    stationary_duration_s: float = 0.65
    stationary_window_s: float = 0.85
    stationary_radius_body: float = 0.055
    stationary_min_samples: int = 6
    stationary_min_coverage: float = 0.75
    coverage_gap_credit_s: float = 0.08
    max_occlusion_s: float = 0.18
    contact_recent_s: float = 0.20
    contact_radius_body: float = 0.17
    contact_radius_diameters: float = 1.4
    launch_speed_body_s: float = 0.8
    launch_displacement_body: float = 0.055
    foot_swing_body_s: float = 1.2
    strong_impulse_body_s: float = 2.8
    confirmation_min_s: float = 0.15
    confirmation_max_s: float = 0.45
    confirmation_min_samples: int = 4
    confirmation_min_progress_body: float = 0.25
    confirmation_min_advancing_intervals: int = 3
    advancing_step_min_body: float = 0.01
    departure_min_body: float = 0.50
    depth_departure_min_diameters: float = 2.0
    depth_progress_min_diameters: float = 1.25
    depth_diameter_shrink_fraction: float = 0.35
    path_min_efficiency: float = 0.82
    direction_min_cosine: float = 0.70
    reversal_tolerance_body: float = 0.03
    control_min_samples: int = 3
    control_min_span_s: float = 0.10
    control_max_gap_s: float = 0.12
    control_feet_travel_body: float = 0.10
    control_release_samples: int = 2
    control_release_span_s: float = 0.03

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not (self.stationary_window_s >= self.stationary_duration_s > 0):
            raise ValueError("stationary window must include required duration")
        if self.confirmation_max_s <= self.confirmation_min_s:
            raise ValueError("confirmation maximum must exceed minimum")
        if any(
            value > 1
            for value in (
                self.min_confidence,
                self.stationary_min_coverage,
                self.path_min_efficiency,
                self.direction_min_cosine,
            )
        ):
            raise ValueError(
                "confidence, coverage and direction fractions must be <= 1"
            )


def _distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def _point_valid(point):
    try:
        return len(point) == 2 and all(isfinite(v) for v in point)
    except (TypeError, ValueError):
        return False


class PlayerControlEvidence:
    """Hold a launch when both feet travel with a nearby measured ball.

    Missing ankles cannot establish or clear this hold. One kicking foot's
    follow-through alone is insufficient; both feet must move with the ball.
    All distances use the frozen setup height, not changing player box sizes.
    """

    def __init__(self, config, height):
        self.config = config
        self.height = height
        self.origin = None
        self.close_times = []
        self.separated_times = []
        self.suspected = False
        self.blocked = False
        self.metrics = {}
        self.last_observed_t = None

    def update(self, obs):
        c = self.config
        feet = [f for f in obs.feet if _point_valid(f)]
        if (len(feet) != 2 or self.last_observed_t is not None
                and obs.t - self.last_observed_t > c.control_max_gap_s + 1e-9):
            self.origin = None
            self.close_times.clear()
            self.separated_times.clear()
        if len(feet) != 2:
            self.last_observed_t = None
            return
        self.last_observed_t = obs.t
        separation = min(_distance(obs.ball, foot) for foot in feet)
        near = separation <= max(c.contact_radius_body * self.height,
                                 c.contact_radius_diameters * obs.diameter)
        self.metrics["ball_foot_separation_body"] = separation / self.height
        if not near:
            self.close_times.clear()
            self.origin = None
            self.separated_times.append(obs.t)
            if (len(self.separated_times) >= c.control_release_samples
                    and obs.t - self.separated_times[0] >= c.control_release_span_s - 1e-9):
                self.blocked = False
            return
        self.separated_times.clear()
        if self.origin is None:
            self.origin = (obs.t, obs.ball, tuple(feet))
        self.close_times.append(obs.t)
        first_t, first_ball, first_feet = self.origin
        dx, dy = obs.ball[0] - first_ball[0], obs.ball[1] - first_ball[1]
        distance = hypot(dx, dy)
        if distance == 0:
            return
        # Ankles can exchange detection order. Use the shorter one-to-one match.
        matched = min((feet, feet[::-1]), key=lambda pair: sum(
            _distance(a, b) for a, b in zip(first_feet, pair)))
        travel = min(((b[0]-a[0])*dx + (b[1]-a[1])*dy) / distance / self.height
                     for a, b in zip(first_feet, matched))
        self.metrics.update(control_samples=len(self.close_times),
                            control_span_s=obs.t-first_t,
                            both_feet_travel_body=travel)
        if (len(self.close_times) >= c.control_min_samples
                and obs.t-first_t >= c.control_min_span_s - 1e-9
                and travel >= c.control_feet_travel_body):
            self.suspected = True
            self.blocked = True


class StationaryShotDetector:
    """One event per verified launch; a new measured stationary setup rearms.

    Times and missing-data windows are seconds. Spatial gates use the frozen
    setup player's height, so a pixel rescale accompanied by equal height/ball
    rescaling preserves decisions. Height must be locally reliable; this is
    not a guarantee under perspective changes or erroneous player association.
    """

    def __init__(self, config=ShotConfig()):
        self.config = config
        self.events = []
        self.diagnostics = []
        self._last_t = None
        self._previous = None
        self._history = []
        self._provisional_history = []
        self._setup = None
        self._candidate = None
        self._contact = None
        self._swing = None

    def _log(self, t, reason, **metrics):
        self.diagnostics.append({"t": t, "reason": reason, "metrics": metrics})

    def _reset(self):
        self._previous = None
        self._history.clear()
        self._provisional_history.clear()
        self._setup = None
        self._candidate = None
        self._contact = None
        self._swing = None

    def _reject(self, t, reason, **metrics):
        self._log(t, "rejected_" + reason, **metrics)
        self._reset()

    def _expire(self, t):
        if (
            self._provisional_history
            and t - self._provisional_history[-1].t > self.config.max_occlusion_s + 1e-9
        ):
            self._provisional_history.clear()
        if self._candidate is not None:
            # Confirmation has its own bounded clock. Missing observations do
            # not add evidence, but need not erase a still-recoverable launch.
            if t - self._candidate['first'] > self.config.confirmation_max_s + 1e-9:
                self._reject(t, 'confirmation_timeout')
            return
        if self._previous is None:
            return
        latest = self._previous.t
        if self._candidate is None and self._history:
            latest = max(latest, self._history[-1].t)
        if t - latest > self.config.max_occlusion_s + 1e-9:
            self._log(
                t,
                "long_loss_reset",
                gap_s=t - latest,
                candidate_active=self._candidate is not None,
            )
            self._reset()

    def _buffer_provisional_setup(self, obs):
        """Keep real, consistent measurements without granting contact or launch."""
        c = self.config
        self._provisional_history.append(obs)
        self._provisional_history = [
            p for p in self._provisional_history
            if obs.t - p.t <= c.stationary_window_s + 1e-9
        ]
        # The current measurement must agree with the entire retained suffix.
        # A location/size jump starts fresh; it cannot borrow an older setup.
        while len(self._provisional_history) > 1:
            height = median(p.body_height for p in self._provisional_history)
            if all(
                _distance(p.ball, obs.ball) <= c.stationary_radius_body * height
                and 0.5 <= p.diameter / obs.diameter <= 1.5
                for p in self._provisional_history
            ):
                break
            self._provisional_history.pop(0)

    def update(self, obs):
        """Return an event only on the measured frame that confirms it.

        Duplicate/backward times and invalid samples are ignored. Weak real
        samples cannot establish contact or launch. Validated, consistent weak
        samples may maintain setup or extend an already established departure.
        Tentative measurements are buffered; only later identity validation can
        admit their stationary history, never their contact or launch evidence.
        Missing/predicted observations can preserve a short gap, but do not
        supply geometric evidence. Finish never fabricates a final sample.
        """
        c = self.config
        try:
            time_valid = isfinite(obs.t)
        except (TypeError, ValueError):
            time_valid = False
        if not time_valid:
            self._log(None, "invalid_observation", field="timestamp")
            return None
        if self._last_t is not None and obs.t <= self._last_t:
            self._log(obs.t, "nonincreasing_timestamp")
            return None
        self._last_t = obs.t
        if obs.source == "camera_unreliable":
            self._log(obs.t, "camera_discontinuity_reset")
            self._reset()
            return None
        self._expire(obs.t)
        try:
            finite = all(
                isfinite(v)
                for v in (obs.body_height, obs.diameter, obs.confidence, obs.foot_speed)
            )
            valid = (
                finite
                and obs.body_height > 0
                and obs.diameter > 0
                and obs.foot_speed >= 0
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            self._log(obs.t, "invalid_observation", field="scale_or_quality")
            return None
        if obs.source not in ("detector", "candidate") or obs.ball is None:
            return None
        if not _point_valid(obs.ball) or obs.confidence > 1:
            self._log(obs.t, "invalid_observation", field="ball_or_confidence")
            return None

        if obs.source == "candidate":
            if (
                self._previous is None and not self._history
                and self._candidate is None
                and obs.confidence >= c.validated_setup_min_confidence
            ):
                self._buffer_provisional_setup(obs)
            return None

        if self._provisional_history:
            if (
                obs.validated and obs.confidence >= c.min_confidence
                and self._previous is None and not self._history
            ):
                self._buffer_provisional_setup(obs)
                # The normal strong-measurement path records the current frame.
                # Do not replay tentative feet, swing, or motion into that path.
                self._history = self._provisional_history[:-1]
                if self._history:
                    self._log(obs.t, "provisional_setup_admitted",
                              samples=len(self._history),
                              duration_s=obs.t - self._history[0].t)
            self._provisional_history.clear()

        if obs.confidence < c.min_confidence:
            if self._candidate is not None and obs.validated and obs.confidence > 0:
                # Association already established the identity. A weak real
                # point may extend flight only when it agrees with the measured
                # departure; it cannot create contact or start a launch.
                radius, previous_radius, _, efficiency, cosine = self._departure_geometry(obs)
                if (radius >= previous_radius - c.reversal_tolerance_body
                        and efficiency >= c.path_min_efficiency
                        and cosine >= c.direction_min_cosine):
                    result = self._advance_candidate(obs)
                    self._previous = obs
                    return result
                return None
            # Identity admission and stationary evidence are different questions.
            # Weak real measurements can maintain an established, still ball;
            # they never supply contact, swing, launch speed or flight confirmation.
            if (
                not obs.validated
                or obs.confidence < c.validated_setup_min_confidence
                or self._candidate is not None
                or self._previous is None
                or not self._history
            ):
                return None
            anchor = self._setup["anchor"] if self._setup else self._previous.ball
            height = (
                self._setup["height"] if self._setup else self._previous.body_height
            )
            if _distance(obs.ball, anchor) <= c.stationary_radius_body * height:
                self._record_setup(obs)
            return None

        previous = self._previous
        if (self._candidate is None and previous is not None
                and obs.t - previous.t > c.max_occlusion_s + 1e-9):
            # Weak stationary continuity does not extend the strong launch gap.
            previous = None
            if (
                self._setup
                and _distance(obs.ball, self._setup["anchor"])
                > c.stationary_radius_body * 1.5 * self._setup["height"]
            ):
                self._log(obs.t, "setup_lost_without_strong_launch_sample")
                self._reset()
        # Contact is checked against the settled ball anchor, never the moving
        # ball after onset. Swing history must also be recent at launch.
        anchor = self._setup["anchor"] if self._setup else obs.ball
        height = self._setup["height"] if self._setup else obs.body_height
        radius = max(
            c.contact_radius_body * height, c.contact_radius_diameters * obs.diameter
        )
        if any(
            _point_valid(foot) and _distance(foot, anchor) <= radius
            for foot in obs.feet
        ):
            self._contact = obs.t
            if obs.foot_speed >= c.foot_swing_body_s:
                self._swing = obs.t

        if self._candidate is not None:
            result = self._advance_candidate(obs)
            # A rejection clears previous; restart evidence from this frame.
            self._previous = obs
            return result

        if self._setup is not None and previous is not None:
            dt = obs.t - previous.t
            speed = _distance(obs.ball, previous.ball) / (dt * height)
            displacement = _distance(obs.ball, anchor) / height
            if (
                speed >= c.launch_speed_body_s
                and displacement >= c.launch_displacement_body
            ):
                recent_contact = (
                    self._contact is not None
                    and obs.t - self._contact <= c.contact_recent_s + 1e-9
                )
                recent_swing = (
                    self._swing is not None
                    and obs.t - self._swing <= c.contact_recent_s + 1e-9
                )
                if not recent_contact:
                    self._reject(obs.t, "no_recent_contact", launch_speed_body_s=speed)
                elif not (recent_swing or speed >= c.strong_impulse_body_s):
                    self._reject(
                        obs.t, "insufficient_kick_evidence", launch_speed_body_s=speed
                    )
                else:
                    self._candidate = {
                        "onset": previous.t,
                        "first": obs.t,
                        "anchor": anchor,
                        "height": height,
                        "points": [anchor, obs.ball],
                        "times": [obs.t],
                        "advancing": 0,
                        "initial": (obs.ball[0] - anchor[0], obs.ball[1] - anchor[1]),
                        "diameter": self._setup["diameter"],
                        "diameters": [obs.diameter],
                        "launch_speed_body_s": speed,
                        "foot_swing": recent_swing,
                        "stationary_duration_s": self._setup["duration"],
                        "stationary_coverage": self._setup["coverage"],
                        "weak_setup_samples": self._setup.get("weak_samples", 0),
                        "strong_setup_samples": self._setup.get("strong_samples", 0),
                        "control": PlayerControlEvidence(c, height),
                    }
                    self._candidate["control"].update(obs)
                    self._log(
                        obs.t,
                        "candidate_started",
                        launch_speed_body_s=speed,
                        foot_swing=recent_swing,
                    )
                self._previous = obs
                return None
            if displacement > c.stationary_radius_body * 1.5:
                self._log(obs.t, "setup_lost", displacement_body=displacement)
                self._reset()

        self._record_setup(obs)
        self._previous = obs
        return None

    def _record_setup(self, obs):
        c = self.config
        self._history.append(obs)
        self._history = [
            p for p in self._history if obs.t - p.t <= c.stationary_window_s + 1e-9
        ]
        # An outlier splits a setup window. It cannot be averaged into stability.
        points = [p.ball for p in self._history]
        center = (median(p[0] for p in points), median(p[1] for p in points))
        setup_height = median(p.body_height for p in self._history)
        while len(self._history) > 1 and any(
            _distance(p.ball, center) > c.stationary_radius_body * setup_height
            for p in self._history
        ):
            self._history.pop(0)
            points = [p.ball for p in self._history]
            center = (median(p[0] for p in points), median(p[1] for p in points))
            setup_height = median(p.body_height for p in self._history)
        duration = obs.t - self._history[0].t
        credited = sum(
            min(b.t - a.t, c.coverage_gap_credit_s)
            for a, b in zip(self._history, self._history[1:])
        )
        coverage = credited / duration if duration > 0 else 0
        if (
            duration + 1e-9 >= c.stationary_duration_s
            and len(self._history) >= c.stationary_min_samples
            and coverage >= c.stationary_min_coverage
        ):
            first_setup = self._setup is None
            self._setup = {
                "anchor": center,
                "height": setup_height,
                "duration": duration,
                "coverage": coverage,
                "diameter": median(p.diameter for p in self._history),
                "weak_samples": sum(
                    p.confidence < c.min_confidence for p in self._history
                ),
                "strong_samples": sum(
                    p.confidence >= c.min_confidence for p in self._history
                ),
            }
            if first_setup:
                self._log(
                    obs.t,
                    "setup_established",
                    stationary_duration_s=duration,
                    stationary_coverage=coverage,
                    weak_setup_samples=self._setup["weak_samples"],
                    strong_setup_samples=self._setup["strong_samples"],
                )

    def _departure_geometry(self, obs):
        """Evaluate a real next point without committing it to the flight."""
        candidate = self._candidate
        anchor, height = candidate['anchor'], candidate['height']
        points = candidate['points'] + [obs.ball]
        radius = _distance(obs.ball, anchor) / height
        previous_radius = _distance(points[-2], anchor) / height
        progress = _distance(obs.ball, points[1]) / height
        path = sum(_distance(a, b) for a, b in zip(points, points[1:])) / height
        initial = candidate['initial']
        departure = (obs.ball[0] - anchor[0], obs.ball[1] - anchor[1])
        norm = hypot(*initial) * hypot(*departure)
        cosine = sum(a*b for a, b in zip(initial, departure)) / norm if norm else 0
        return radius, previous_radius, progress, radius / path if path else 0, cosine

    def _advance_candidate(self, obs):
        c = self.config
        candidate = self._candidate
        control = candidate["control"]
        was_suspected, was_blocked = control.suspected, control.blocked
        control.update(obs)
        if control.suspected and not was_suspected:
            self._log(obs.t, "player_control_hold", **control.metrics)
        elif was_blocked and not control.blocked:
            self._log(obs.t, "player_control_release", **control.metrics)
        anchor, height = candidate["anchor"], candidate["height"]
        points = candidate["points"]
        radius, previous_radius, progress, efficiency, cosine = self._departure_geometry(obs)
        if radius > previous_radius + c.advancing_step_min_body:
            candidate['advancing'] += 1
        points.append(obs.ball)
        candidate['times'].append(obs.t)
        candidate['diameters'].append(obs.diameter)
        elapsed = obs.t - candidate["first"]
        diameter = candidate["diameter"]
        shrink = 1.0 - median(candidate["diameters"][-3:]) / diameter
        depth_departure = (
            candidate["foot_swing"]
            and shrink >= c.depth_diameter_shrink_fraction
            and radius * height >= c.depth_departure_min_diameters * diameter
            and progress * height >= c.depth_progress_min_diameters * diameter
        )
        metrics = {
            "net_departure_body": radius,
            "path_efficiency": efficiency,
            "direction_cosine": cosine,
            "measured_samples": len(candidate["times"]),
            "confirmation_duration_s": elapsed,
            "measured_progress_body": progress,
            "advancing_intervals": candidate["advancing"],
            "launch_speed_body_s": candidate["launch_speed_body_s"],
            "stationary_duration_s": candidate["stationary_duration_s"],
            "stationary_coverage": candidate["stationary_coverage"],
            "weak_setup_samples": candidate.get("weak_setup_samples", 0),
            "strong_setup_samples": candidate.get("strong_setup_samples", 0),
            "foot_swing": candidate["foot_swing"],
            "diameter_shrink_fraction": shrink,
            "depth_departure": depth_departure,
            "player_control_suspected": control.suspected,
            "player_control_blocked": control.blocked,
            **control.metrics,
        }
        if (
            radius < previous_radius - c.reversal_tolerance_body
            or cosine < c.direction_min_cosine
            or efficiency < c.path_min_efficiency
        ):
            self._reject(obs.t, "inconsistent_departure", **metrics)
        elif elapsed > c.confirmation_max_s + 1e-9:
            self._reject(obs.t, "continued_player_control" if control.blocked else "confirmation_timeout", **metrics)
        elif (
            elapsed + 1e-9 >= c.confirmation_min_s
            and (radius >= c.departure_min_body or depth_departure)
            and len(candidate["times"]) >= c.confirmation_min_samples
            and (progress >= c.confirmation_min_progress_body or depth_departure)
            and candidate["advancing"] >= c.confirmation_min_advancing_intervals
            and not control.blocked
        ):
            event = {
                "t": candidate["onset"],
                "confirmed_at": obs.t,
                "type": "stationary_ball_attempt",
                "anchor": anchor,
                "metrics": metrics,
            }
            self.events.append(event)
            self._log(obs.t, "accepted", **metrics)
            self._reset()
            return event
        return None

    def finish(self, reason="end_of_video"):
        """Flag a cut-off candidate without promoting it to a confirmed kick."""
        if self._candidate is None:
            return None
        candidate = self._candidate
        incomplete = {
            "t": candidate["onset"],
            "reason": reason,
            "confirmed": False,
            "measured_samples": len(candidate["times"]),
        }
        self._reject(
            self._last_t,
            reason,
            measured_samples=len(candidate["times"]),
            onset=candidate["onset"],
        )
        return incomplete if reason == "end_of_video" else None
