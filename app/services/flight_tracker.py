"""Image-supported playback recovery after a confirmed kick.

This class never supplies observations to the kick detector. It follows compact
moving image regions near an established flight, after aligning recent images
for camera translation. Its quality score is not a model class probability.
"""

from collections import deque
import math
import cv2
import numpy as np
from .ball_appearance import BallAppearance
from .tracking import BALL_VALIDATION_SETTINGS


class FlightTracker:
    def __init__(self, max_gap_s=0.18, max_visual_s=2.5):
        self.max_gap_s = max_gap_s
        self.max_visual_s = max_visual_s
        self.background = deque(maxlen=32)
        self.history = {}
        self.states = {}
        self.dormant = {}
        self.seen = set()
        self.shape = None

    def update(self, image, t, shift, observations, events, camera_unreliable=False):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 960.0 / max(image.shape))
        gray = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        shift = np.asarray(shift, dtype=float) * scale
        if camera_unreliable or (self.shape is not None and self.shape != gray.shape):
            self.background.clear()
            self.history.clear()
            self.states.clear()
            self.dormant.clear()
            self.seen.update(
                (e["ball_id"], e["t"]) for e in events if e["confirmed_at"] <= t
            )
            self.shape = gray.shape
            return []
        self.shape = gray.shape
        for identity, obs in observations.items():
            if (
                obs.source == "detector"
                and obs.confidence >= BALL_VALIDATION_SETTINGS["strong_confidence"]
                and obs.ball is not None
                and obs.validated
            ):
                values = self.history.setdefault(identity, deque(maxlen=4))
                values.append((t, np.asarray(obs.ball) * scale, obs.diameter * scale))
        self.history = {
            i: v for i, v in self.history.items() if t - v[-1][0] <= self.max_gap_s
        }
        for event in events:
            key = (event["ball_id"], event["t"])
            if key in self.seen or event["t"] > t + 1e-9:
                continue
            values = list(self.history.get(event["ball_id"], ()))
            if len(values) >= 3:
                self.seen.add(key)
                self.states[event["ball_id"]] = {
                    "points": deque(values, maxlen=4),
                    "model_points": deque(values, maxlen=4),
                    "last_strong": t,
                    "kick_number": events.index(event) + 1,
                    "confirmed_at": event["confirmed_at"],
                    "appearance": BallAppearance(),
                    "image": gray,
                    "image_shift": shift,
                }
        # A missed image measurement stops drawing immediately. A short dormant
        # record permits a restart only after three fresh same-ID model anchors.
        for identity, state in list(self.dormant.items()):
            last_t, last_point, diameter = state["points"][-1]
            if t - last_t > 0.75:
                del self.dormant[identity]
                continue
            anchors = list(self.history.get(identity, ()))
            fresh = [a for a in anchors if a[0] > last_t + self.max_gap_s]
            if len(fresh) < 3:
                continue
            a, b, c = fresh[-3:]
            if (
                c[0] - a[0] <= 0.20
                and t - c[0] < 1e-6
                and all(0.5 <= x[2] / diameter <= 1.5 for x in (a, b, c))
                and max(np.linalg.norm(x[1] - last_point) for x in (a, b, c))
                <= 6.0 * diameter
                and np.linalg.norm(c[1] - a[1]) >= 0.5 * c[2]
                and np.dot(b[1] - a[1], c[1] - b[1]) > 0
            ):
                state["points"] = deque((a, b, c), maxlen=4)
                state["model_points"] = deque((a, b, c), maxlen=4)
                state["last_strong"] = t
                self.states[identity] = self.dormant.pop(identity)
        output = []
        proposals = {}
        reserved = {}
        for identity, state in list(self.states.items()):
            anchors = list(state["model_points"])
            if len(anchors) >= 3:
                a, b, c = anchors[-3:]
                first, second = b[1] - a[1], c[1] - b[1]
                if (
                    np.linalg.norm(c[1] - a[1]) >= 0.5 * c[2]
                    and np.dot(first, second) > 0
                ):
                    state["launched"] = True
            history = state["points"]
            last_t, last_point, diameter = history[-1]
            if (
                t - last_t > self.max_gap_s + 1e-9
                or t - state["last_strong"] > self.max_visual_s
            ):
                if state.get("launched") and t - last_t <= 0.75:
                    self.dormant[identity] = state
                del self.states[identity]
                continue
            if t <= last_t:
                if t == last_t:
                    state["appearance"].remember(gray, last_point+shift, diameter,
                                                 model=True)
                    state["image"], state["image_shift"] = gray, shift
                    reserved[identity] = (last_point+shift, diameter)
                continue
            a, b = history[-2], history[-1]
            velocity = (b[1] - a[1]) / max(1e-9, b[0] - a[0])
            predicted = last_point + velocity * (t - last_t) + shift
            obs = observations.get(identity)
            # A current model anchor takes priority only if it fits the established
            # flight. A far-away ID reassignment cannot drag the visual path.
            if (
                obs is not None
                and obs.source == "detector"
                and obs.confidence >= BALL_VALIDATION_SETTINGS["strong_confidence"]
                and obs.validated
                and obs.ball is not None
                and np.linalg.norm(np.asarray(obs.ball) * scale + shift - predicted)
                <= max(8.0, 2.0 * diameter)
            ):
                history.append((t, np.asarray(obs.ball) * scale, obs.diameter * scale))
                state["model_points"].append(history[-1])
                state["last_strong"] = t
                state["appearance"].remember(gray, np.asarray(obs.ball)*scale+shift,
                                             obs.diameter*scale, model=True)
                state["image"], state["image_shift"] = gray, shift
                reserved[identity] = (np.asarray(obs.ball)*scale+shift, obs.diameter*scale)
                continue
            if not state.get("launched", False):
                # Near contact, foot motion can resemble a ball. Image recovery
                # needs consistent movement across three strong model anchors.
                continue
            hints = []
            for candidate in observations.values():
                if (candidate.source in ("detector", "candidate")
                        and candidate.ball is not None
                        and candidate.confidence >= BALL_VALIDATION_SETTINGS["minimum_confidence"]
                        and BALL_VALIDATION_SETTINGS["weak_minimum_size_ratio"]
                        <= candidate.diameter*scale/diameter
                        <= BALL_VALIDATION_SETTINGS["weak_maximum_size_ratio"]):
                    hints.append((np.asarray(candidate.ball)*scale+shift,
                                  candidate.diameter*scale))
            measured = state["appearance"].locate(
                gray, state["image"], shift, state["image_shift"], last_point+shift, predicted,
                diameter, hints)
            if measured is None and not state["appearance"].ambiguous:
                measured = self._measure(gray, shift, predicted, diameter)
                if measured is not None:
                    measured = state["appearance"].support_detection(measured, hints)
            if measured is None:
                continue
            proposals[identity] = (measured, hints)

        # A single image object can update at most one flight. Suppress competing
        # claims before committing either history or an appearance update.
        blocked = set()
        for identity, (measurement, _) in proposals.items():
            point, diameter, _ = measurement
            for other, (other_point, other_diameter) in reserved.items():
                if other != identity and np.linalg.norm(point-other_point) < min(diameter,other_diameter)/2:
                    blocked.add(identity)
            for other, (other_measurement, _) in proposals.items():
                if other <= identity:
                    continue
                other_point, other_diameter, _ = other_measurement
                if np.linalg.norm(point-other_point) < min(diameter,other_diameter)/2:
                    blocked.update((identity,other))
        for identity, (measured, hints) in proposals.items():
            if identity in blocked:
                continue
            state = self.states[identity]
            history = state["points"]
            diameter = history[-1][2]
            point, new_diameter, quality = measured
            history.append((t, point - shift, 0.8 * diameter + 0.2 * new_diameter))
            state["image"], state["image_shift"] = gray, shift
            if any(np.linalg.norm(point-p) <= d/2 for p,d in hints):
                # Learn only from agreement between detector geometry and image
                # evidence. Self-labelled image guesses can learn the background.
                state["appearance"].remember(gray, point, new_diameter)
            if t + 1e-9 < state["confirmed_at"]:
                continue  # Offline playback can follow earlier frames, never display early.
            output.append(
                {
                    "ball_id": identity,
                    "ball": tuple((point - shift) / scale),
                    "raw_ball": list(point / scale),
                    "diameter": float(history[-1][2] / scale),
                    "source": "visual",
                    "confidence": float(quality),
                    "validated": True,
                    "kick_number": state["kick_number"],
                    "visual_quality": float(quality),
                }
            )
        self.background.append((gray.copy(), shift.copy()))
        return output

    def _measure(self, gray, shift, predicted, diameter):
        if len(self.background) < 3:
            return None
        radius = max(10, round(2 * diameter))
        size = 2 * radius + 1
        h, w = gray.shape
        if not (
            radius < predicted[0] < w - radius and radius < predicted[1] < h - radius
        ):
            return None
        # Fix the current ROI to the pixel grid. Subpixel ROI resampling can
        # change residual components under tiny model-box jitter.
        roi_center = np.rint(predicted)
        current = cv2.getRectSubPix(gray, (size, size), tuple(map(float, roi_center)))
        aligned = []
        for lag in (1, 3, 7, 15, 31):
            if lag > len(self.background):
                continue
            old, old_shift = self.background[-lag]
            center = roi_center - shift + old_shift
            aligned.append(
                cv2.getRectSubPix(old, (size, size), tuple(map(float, center)))
            )
        residual = cv2.absdiff(
            current, np.median(np.stack(aligned), axis=0).astype(np.uint8)
        )
        noise = float(np.median(residual))
        threshold = max(12.0, 3 * noise, 0.3 * float(residual.max()))
        mask = (residual > threshold).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for k in range(1, count):
            _, _, bw, bh, area = stats[k]
            if (
                area < 3
                or area > 4 * diameter**2
                or max(bw, bh) / max(1, min(bw, bh)) > 3
            ):
                continue
            measured_diameter = (bw + bh) / 2
            if not 0.3 * diameter <= measured_diameter <= 1.8 * diameter:
                continue
            point = roi_center + centers[k] - radius
            distance = float(np.linalg.norm(point - predicted))
            if distance > max(8, 1.5 * diameter):
                continue
            compactness = area / (bw * bh)
            contrast = float(residual[labels == k].mean())
            if compactness < 0.3 or contrast < max(20.0, 2 * noise):
                continue
            quality = (
                0.5 * math.exp(-distance / max(4.0, diameter))
                + 0.25 * compactness
                + 0.25 * min(1.0, contrast / 60.0)
            )
            candidates.append((quality, point, measured_diameter))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates or candidates[0][0] < 0.62:
            return None
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.12:
            return None
        quality, point, diameter = candidates[0]
        return point, diameter, quality


def flight_playback_observations(observations, balls, measurements):
    """Return playback copies. Never mutate the event detector's evidence."""
    from .kick_detector import Observation

    visible = dict(observations)
    replaced = {item["ball_id"] for item in measurements}
    markers = [dict(b) for b in balls if b["ball_id"] not in replaced]
    for item in measurements:
        identity = item["ball_id"]
        visible[identity] = Observation(
            t=0.0,
            ball=item["ball"],
            diameter=item["diameter"],
            confidence=item["visual_quality"],
            source="visual",
            validated=True,
        )
        overlaps = [
            marker
            for marker in markers
            if marker["source"] == "detector"
            and marker.get("validated", False)
            and marker["confidence"] >= 0.45
            and marker["raw_ball"] is not None
            and np.linalg.norm(np.asarray(marker["raw_ball"]) - item["raw_ball"])
            < 0.5 * min(marker["diameter"], item["diameter"])
        ]
        if overlaps:
            # Keep the model's measured marker if it reacquires under another ID.
            # Association here is playback-only; detection evidence remains intact.
            min(
                overlaps,
                key=lambda x: np.linalg.norm(
                    np.asarray(x["raw_ball"]) - item["raw_ball"]
                ),
            )["kick_number"] = item["kick_number"]
        else:
            markers.append(item)
    return visible, markers
