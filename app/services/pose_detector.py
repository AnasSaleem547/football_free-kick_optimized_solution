"""YOLO-pose output conversion to the notebook's ankle/hip evidence schema."""

import numpy as np


class PoseDetector:
    def __init__(self, model):
        self.model = model

    def predict(self, frames, *, device, half, imgsz):
        if not frames:
            return []
        results = self.model.predict(
            frames, imgsz=imgsz, conf=0.30, device=device, half=half, verbose=False
        )
        converted = []
        for result in results:
            poses = []
            if result.keypoints is not None:
                for box, points, confidence in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.keypoints.xy.cpu().numpy(),
                    result.keypoints.conf.cpu().numpy(),
                ):
                    ankles = [
                        list(map(float, points[k]))
                        for k in (15, 16)
                        if confidence[k] >= 0.35
                    ]
                    hips = [points[k] for k in (11, 12) if confidence[k] >= 0.35]
                    if ankles and hips:
                        poses.append(
                            {
                                "box": list(map(float, box)),
                                "ankles": ankles,
                                "hip": list(map(float, np.mean(hips, axis=0))),
                            }
                        )
            converted.append(poses)
        if len(converted) != len(frames):
            raise RuntimeError("Pose model returned incomplete batch")
        return converted
