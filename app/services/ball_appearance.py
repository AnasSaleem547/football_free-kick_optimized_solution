"""Local image matching for an already established ball flight.

Detector boxes propose locations; they cannot substitute for matching pixels.
Correlation measures appearance agreement, not football-class probability.
"""
import cv2
import numpy as np


class BallAppearance:
    def __init__(self, minimum_match=0.70, minimum_margin=0.10):
        self.minimum_match = minimum_match
        self.minimum_margin = minimum_margin
        self.reference = None
        self.recent = None
        self.ambiguous = False

    def remember(self, gray, center, diameter, *, model=False):
        size = 2 * max(1, round(0.75 * diameter)) + 1
        radius = size // 2
        x, y = np.rint(center).astype(int)
        if not (radius <= x < gray.shape[1]-radius
                and radius <= y < gray.shape[0]-radius):
            return
        patch = gray[y-radius:y+radius+1, x-radius:x+radius+1].copy()
        if np.std(patch) <= np.finfo(np.float32).eps:
            return
        self.recent = patch
        if model:
            self.reference = self.recent

    def locate(self, gray, previous, shift, previous_shift, last, predicted,
               diameter, hints=()):
        self.ambiguous = False
        if self.recent is None or previous is None:
            return None
        # Search around the last measured position as well as the old direction.
        # A turn needs new pixel evidence, not a permanently wider velocity gate.
        reach = np.linalg.norm(predicted-last) + 2 * diameter
        low = np.maximum(np.floor(last-reach).astype(int), 0)
        high = np.minimum(np.ceil(last+reach+1).astype(int), gray.shape[::-1])
        search = gray[low[1]:high[1], low[0]:high[0]]
        # A small image pyramid tests neighbouring sizes. Box-size jitter on a
        # tiny ball must not force every template into one wrongly sized patch.
        sizes = {2*max(1,round(0.75*diameter*factor))+1
                 for factor in (2**-0.5,1.0,2**0.5)}
        response = np.full(search.shape, -1, np.float32)
        matched_sizes = np.zeros(search.shape, np.int32)
        for template in (self.reference, self.recent):
            if template is None:
                continue
            for size in sorted(sizes):
                if min(search.shape) < size:
                    continue
                patch = cv2.resize(template, (size, size))
                scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
                radius = size//2
                target = response[radius:search.shape[0]-radius,
                                  radius:search.shape[1]-radius]
                dimensions = matched_sizes[radius:search.shape[0]-radius,
                                           radius:search.shape[1]-radius]
                better = scores > target
                target[better] = scores[better]
                dimensions[better] = size
        yy, xx = np.indices(response.shape)
        centers = np.stack((xx, yy), axis=-1) + low
        possible = np.linalg.norm(centers-last, axis=-1) <= reach

        # Check all competing image peaks before consulting detector hints. A
        # weak box must not hide an equally plausible visible ball.
        scores = np.where(possible, response, -1).astype(np.float32)
        _, best, _, location = cv2.minMaxLoc(scores)
        if best < self.minimum_match:
            return None
        remaining = scores.copy()
        cv2.circle(remaining, location, max(1, round(diameter/2)), -1, -1)
        if best - float(remaining.max()) < self.minimum_margin:
            self.ambiguous = True
            return None
        point = np.array(location, dtype=float) + low
        size = int(matched_sizes[location[1],location[0]])
        supported_hints = [
            (p, d) for p, d in hints if np.linalg.norm(point-p) <= d/2
        ]
        if supported_hints:
            # Image matching verifies identity; the detector box recentres the
            # target instead of teaching the template a displaced background.
            point, measured_diameter = min(
                supported_hints, key=lambda hint: np.linalg.norm(point-hint[0])/hint[1]
            )
        else:
            measured_diameter = diameter

        # Reject an unchanged distractor and global illumination changes. Compare
        # the target's actual pixel change with the surrounding change level.
        old_center = point-shift+previous_shift
        current_patch = cv2.getRectSubPix(gray, (size, size), tuple(map(float, point)))
        old_patch = cv2.getRectSubPix(previous, (size, size), tuple(map(float, old_center)))
        previous_roi_center = (low+high-1)/2 - shift + previous_shift
        old_search = cv2.getRectSubPix(previous, search.shape[::-1],
                                     tuple(map(float, previous_roi_center)))
        # Track the candidate back into the last measured image. A background
        # lookalike already present elsewhere must not replace a missing ball.
        backward = cv2.matchTemplate(old_search, current_patch, cv2.TM_CCOEFF_NORMED)
        _, back_score, _, back_location = cv2.minMaxLoc(backward)
        back_point = np.array(back_location) + low + size//2
        competing = backward.copy()
        cv2.circle(competing, back_location, max(1,round(diameter/2)), -1, -1)
        runner = float(competing.max())
        if runner >= self.minimum_match and back_score-runner < self.minimum_margin:
            self.ambiguous = True
            return None
        if np.linalg.norm(back_point-last) > diameter:
            return None
        changes = cv2.absdiff(search, old_search).astype(np.float32)
        local_means = cv2.boxFilter(changes, -1, (size, size), normalize=True)
        radius = size//2
        background = local_means[radius:-radius, radius:-radius]
        noise = float(np.median(background))
        # Compare patch means with patch means. A mean-versus-median pixel test
        # accepts ordinary sensor noise as movement on a static lookalike.
        target_change = float(np.mean(cv2.absdiff(current_patch, old_patch)))
        pixel_median = float(np.median(changes))
        pixel_spread = 1.4826 * float(np.median(np.abs(changes-pixel_median)))
        mean_uncertainty = pixel_spread / size
        if target_change <= noise + 3 * mean_uncertainty:
            return None
        return point, measured_diameter, float(best)

    @staticmethod
    def support_detection(measurement, hints):
        """A measured moving region can also verify a nearby detector box."""
        point, diameter, quality = measurement
        supported = [(p,d) for p,d in hints if np.linalg.norm(point-p) <= d/2]
        if not supported:
            return measurement
        point, diameter = min(supported,
                              key=lambda hint: np.linalg.norm(point-hint[0])/hint[1])
        return point, diameter, quality
