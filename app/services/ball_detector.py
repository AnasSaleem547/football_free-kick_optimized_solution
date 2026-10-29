"""Ball model output conversion. Association belongs to tracking.py."""

import math


class BallDetector:
    def __init__(self, model):
        self.model = model

    def predict(self, frames, *, device, half, imgsz, conf):
        results = self.model.predict(
            frames,
            imgsz=imgsz,
            conf=conf,
            classes=[32],
            device=device,
            half=half,
            verbose=False,
        )
        converted = []
        for frame, result in zip(frames, results):
            candidates = []
            for box, score in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
            ):
                x1, y1, x2, y2 = map(float, box)
                diameter = (x2 - x1 + y2 - y1) / 2
                if (
                    all(math.isfinite(v) for v in (x1, y1, x2, y2, score))
                    and 3 <= diameter <= 0.15 * frame.shape[1]
                ):
                    candidates.append(
                        [(x1 + x2) / 2, (y1 + y2) / 2, diameter, float(score)]
                    )
            converted.append(candidates)
        if len(converted) != len(frames):
            raise RuntimeError("Ball model returned incomplete batch")
        return converted
