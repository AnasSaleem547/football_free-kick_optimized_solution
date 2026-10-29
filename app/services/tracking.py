"""Per-video measured ball association and temporal admission."""

import math
from collections import deque
import numpy as np


class BallTrack:
    """One track per candidate. Predictions guide association, never confirm events."""

    def __init__(self, max_gap=0.45):
        self.history = deque(maxlen=6)
        self.max_gap = max_gap

    def motion(self, t):
        """Median measured velocity with a size- and time-based search radius."""
        last_t, last_p, diameter = self.history[-1]
        pairs = zip(list(self.history)[:-1], list(self.history)[1:])
        velocities = [
            (np.asarray(b[1]) - a[1]) / (b[0] - a[0])
            for a, b in pairs if b[0] > a[0]
        ]
        velocity = np.median(velocities, axis=0) if velocities else np.zeros(2)
        dt = max(0.0, t - last_t)
        prediction = np.asarray(last_p) + velocity * dt
        # The existing six-diameter gate expands with measured speed. The
        # diameter/max_gap term covers a still ball without a pixels/s constant.
        radius = 6.0 * diameter + max(
            float(np.linalg.norm(velocity)), diameter / self.max_gap
        ) * dt
        return prediction, radius, diameter

    def association_cost(self, t, candidate):
        """Compare position error, size change and model confidence once."""
        prediction, radius, diameter = self.motion(t)
        distance = float(np.linalg.norm(np.asarray(candidate[:2]) - prediction))
        ratio = candidate[2] / diameter
        cfg = BALL_VALIDATION_SETTINGS
        weak_ok = candidate[3] >= cfg["strong_confidence"] or (
            t - self.history[-1][0] <= cfg["weak_maximum_gap_s"]
            and distance <= cfg["weak_radius_diameters"] * diameter
            and cfg["weak_minimum_size_ratio"] <= ratio <= cfg["weak_maximum_size_ratio"]
        )
        if not weak_ok or distance > radius or not 0.25 <= ratio <= 4.0:
            return math.inf
        return distance / radius + 0.25 * abs(math.log(ratio)) - 0.15 * candidate[3]

    def supported_continuation(self, t, candidate, future):
        """Check an alternative motion path against later real measurements.

        This is fixed-lag association for offline videos, not new detections.
        Normal matching gets first choice. Only established identities can use
        this fallback, within the existing size-based search reach. Success
        requires the ordinary admission sample count/span and a strong later
        model observation. Predictions never become measurements.
        """
        cfg = BALL_VALIDATION_SETTINGS
        last_t, last_point, last_size = self.history[-1]
        dt = t - last_t
        _, reach, _ = self.motion(t)
        point = np.asarray(candidate[:2], dtype=float)
        previous = np.asarray(last_point, dtype=float)
        if (not 0 < dt <= cfg['weak_maximum_gap_s']
                or np.linalg.norm(point - previous) > reach
                or not 0.25 <= candidate[2] / last_size <= 4.0):
            return None
        velocity = (point - previous) / dt
        time, size, hits = t, candidate[2], 1
        path = []
        for next_t, candidates in future:
            gap = next_t - time
            if gap > cfg['weak_maximum_gap_s'] + 1e-9:
                return None
            options = []
            for other in candidates:
                if (len(other) != 4 or not all(math.isfinite(float(v)) for v in other)
                        or other[2] <= 0 or other[3] < cfg['minimum_confidence']
                        or not 0.25 <= other[2] / size <= 4.0):
                    continue
                measured = np.asarray(other[:2], dtype=float)
                step = measured - point
                # A disconnected static object or reversal cannot explain a launch.
                if float(np.dot(step, velocity)) <= 0:
                    continue
                error = float(np.linalg.norm(measured - (point + velocity * gap)))
                allowance = cfg['weak_radius_diameters'] * max(size, other[2])
                if error <= allowance:
                    options.append((error / allowance - .15 * other[3], other))
            if not options:
                continue
            cost, other = min(options, key=lambda item: item[0])
            # Overlapping detector boxes may describe the same object. Distinct
            # plausible objects are unresolved evidence, not a tie to break by order.
            if any(np.linalg.norm(np.asarray(alternative[:2]) - other[:2])
                   >= (alternative[2] + other[2]) / 2
                   for _, alternative in options if alternative is not other):
                return None
            measured = np.asarray(other[:2], dtype=float)
            velocity = (measured - point) / gap
            point, time, size = measured, next_t, other[2]
            hits += 1
            path.append((time, tuple(point), size))
            if (hits >= cfg['minimum_hits']
                    and time - t >= cfg['minimum_span_s'] - 1e-9
                    and other[3] >= cfg['strong_confidence']):
                return tuple(path)
        return None

    def observe(self, t, candidate):
        """Commit one assigned measurement, or expose a labelled prediction."""
        if candidate is not None:
            x, y, diameter, confidence = candidate
            self.history.append((float(t), (x, y), diameter))
            return dict(
                ball=(x, y), diameter=diameter, confidence=confidence, source="detector"
            )
        if self.history:
            prediction, _, diameter = self.motion(t)
            return dict(
                ball=tuple(prediction), diameter=diameter,
                confidence=0.0, source="prediction"
            )
        return dict(ball=None, diameter=0.0, confidence=0.0, source="missing")


BALL_VALIDATION_SETTINGS = {
    "minimum_confidence": 0.18,
    "strong_confidence": 0.45,
    "minimum_hits": 4,
    "minimum_span_s": 0.10,
    "window_s": 0.30,
    "maximum_strong_gap_s": 0.45,
    "weak_maximum_gap_s": 0.12,
    "weak_radius_diameters": 1.5,
    "weak_minimum_size_ratio": 0.5,
    "weak_maximum_size_ratio": 1.5,
}


class BallValidation:
    """Repeated strong measurements support a track; they do not prove object class."""

    def __init__(self):
        self.strong_times = deque()
        self.last_strong = -math.inf
        self.validated = False

    def update(self, t, source, confidence):
        cfg = BALL_VALIDATION_SETTINGS
        if t - self.last_strong > cfg["maximum_strong_gap_s"]:
            self.validated = False
        while self.strong_times and t - self.strong_times[0] > cfg["window_s"] + 1e-9:
            self.strong_times.popleft()
        if source == "detector" and confidence >= cfg["strong_confidence"]:
            if not self.strong_times or t > self.strong_times[-1] + 1e-9:
                self.strong_times.append(t)
            self.last_strong = t
            if (
                len(self.strong_times) >= cfg["minimum_hits"]
                and t - self.strong_times[0] >= cfg["minimum_span_s"] - 1e-9
            ):
                self.validated = True
        return self.validated


class MultiBallTracker:
    """One-to-one measured association. IDs survive only short occlusions.

    Each ball keeps its own BallTrack history. New detections outside every
    existing track's gate create a new ID instead of being discarded.
    """

    def __init__(self, max_gap=0.45, minimum_confidence=0.18):
        self.tracks = {}
        self.next_id = 1
        self.max_gap = max_gap
        self.minimum_confidence = minimum_confidence
        self.validation = {}

    def reset(self):
        self.tracks.clear()  # Never reuse an ID across a camera discontinuity.
        self.validation.clear()

    def update(self, t, candidates, future=()):
        from scipy.optimize import linear_sum_assignment

        self.tracks = {
            k: v for k, v in self.tracks.items() if t - v.history[-1][0] <= self.max_gap
        }
        good = []
        for c in sorted(
            candidates, key=lambda v: float(v[3]) if len(v) == 4 else -1.0, reverse=True
        ):
            if (
                len(c) != 4
                or not all(math.isfinite(float(v)) for v in c)
                or c[2] <= 0
                or not 0 < c[3] <= 1
            ):
                continue
            c = tuple(map(float, c))
            if any(
                np.linalg.norm(np.asarray(c[:2]) - v[:2]) < 0.45 * min(c[2], v[2])
                for v in good
            ):
                continue
            good.append(c)
        ids = list(self.tracks)
        assigned = {}
        used = set()

        def assign(stage_ids, stage_columns):
            stage_ids = [i for i in stage_ids if i not in assigned]
            stage_columns = [j for j in stage_columns if j not in used]
            if not stage_ids or not stage_columns:
                return
            costs = np.full((len(stage_ids), len(stage_columns) + len(stage_ids)), 1.1)
            for row, ball_id in enumerate(stage_ids):
                for col, good_index in enumerate(stage_columns):
                    cost = self.tracks[ball_id].association_cost(t, good[good_index])
                    costs[row, col] = cost if math.isfinite(cost) else 1e6
            rows, cols = linear_sum_assignment(costs)
            for row, col in zip(rows, cols):
                if col < len(stage_columns) and costs[row, col] < 1.1:
                    assigned[stage_ids[row]] = stage_columns[col]
                    used.add(stage_columns[col])

        cfg = BALL_VALIDATION_SETTINGS
        established = [
            i
            for i in ids
            if i in self.validation
            and self.validation[i].validated
            and t - self.validation[i].last_strong <= cfg["maximum_strong_gap_s"]
        ]
        tentative = [i for i in ids if i not in established]
        # Match established identities before tentative tracks, but compare
        # position, size and score jointly. A strong unrelated ball must not
        # override the exact weak continuation of an existing ball.
        regular_columns = [j for j, value in enumerate(good)
                           if value[3] >= self.minimum_confidence]
        # Lower-score boxes can continue one established identity, but cannot
        # start or validate an object. Reject ambiguous low-score matches.
        low_pairs = [(identity, j) for identity in established
                     for j, value in enumerate(good)
                     if value[3] < self.minimum_confidence
                     and math.isfinite(self.tracks[identity].association_cost(t, value))]
        low_columns = [j for identity, j in low_pairs
                       if sum(other == identity for other, _ in low_pairs) == 1
                       and sum(column == j for _, column in low_pairs) == 1]
        assign(established, regular_columns + low_columns)
        # Only unmatched established tracks may offer an alternative motion path.
        # Reject competing identities and distinct competing paths. Overlapping
        # duplicate boxes that converge on the same later measurement are one
        # hypothesis, rather than several physical balls.
        proposals = {}
        for identity in established:
            if identity in assigned:
                continue
            options = []
            for column, candidate in enumerate(good):
                if column in used or any(np.linalg.norm(
                        np.asarray(candidate[:2]) - earlier[:2])
                        < (candidate[2] + earlier[2]) / 2
                        for earlier in good[:column]):
                    continue
                support = self.tracks[identity].supported_continuation(t, candidate, future)
                if support is not None:
                    options.append((column, support))
            if len(options) != 1:
                continue
            column, support_path = options[0]
            proposals[identity] = (column, support_path)
        def overlapping_paths(first, second):
            return any(a[0] == b[0] and np.linalg.norm(np.asarray(a[1]) - b[1])
                       < (a[2] + b[2]) / 2 for a in first for b in second)

        # Later evidence already belonging to a normally matched ball cannot
        # corroborate recovery of a different identity.
        reserved_paths = []
        if proposals:
            for identity, column in assigned.items():
                continuation = BallTrack(self.max_gap)
                continuation.history.extend(self.tracks[identity].history)
                continuation.observe(t, good[column])
                support = []
                for next_t, candidates in future:
                    if next_t - continuation.history[-1][0] > self.max_gap:
                        break
                    options = []
                    for candidate in candidates:
                        if (len(candidate) != 4 or candidate[2] <= 0
                                or candidate[3] < self.minimum_confidence
                                or not all(math.isfinite(float(v)) for v in candidate)):
                            continue
                        cost = continuation.association_cost(next_t, candidate)
                        if cost < 1.1:
                            options.append((cost, candidate))
                    if options:
                        _, candidate = min(options, key=lambda item: item[0])
                        continuation.observe(next_t, candidate)
                        support.append((next_t, tuple(candidate[:2]), candidate[2]))
                reserved_paths.append(support)
        for identity, (column, support_path) in proposals.items():
            if (not any(overlapping_paths(support_path, path) for path in reserved_paths)
                    and not any(other != identity and (
                        column == other_column or overlapping_paths(support_path, other_path))
                        for other, (other_column, other_path) in proposals.items())):
                assigned[identity] = column
                used.add(column)
        assign(tentative, regular_columns)
        output = []
        for ball_id, track in self.tracks.items():
            value = track.observe(
                t, good[assigned[ball_id]] if ball_id in assigned else None
            )
            output.append({"ball_id": ball_id, **value})
        for col, c in enumerate(good):
            if col in used or c[3] < self.minimum_confidence:
                continue
            ball_id = self.next_id
            self.next_id += 1
            track = BallTrack(self.max_gap)
            self.tracks[ball_id] = track
            output.append({"ball_id": ball_id, **track.observe(t, c)})
        self.validation = {k: v for k, v in self.validation.items() if k in self.tracks}
        for value in output:
            state = self.validation.setdefault(value["ball_id"], BallValidation())
            value["validated"] = state.update(t, value["source"], value["confidence"])
        return output
