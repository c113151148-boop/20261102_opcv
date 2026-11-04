# AI 模型服務

from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetectorService:
    """人臉偵測模型服務類別"""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型檔案: {self.model_path}")

    def detect(self, img_rgb: np.ndarray, min_confidence: float = 0.5):
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence
        )

        detector = vision.FaceDetector.create_from_options(options)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)

            annotated_image = img_rgb.copy()
            h, w, _ = img_rgb.shape

            for detection in results.detections:
                score = detection.categories[0].score
                bbox = detection.bounding_box

                x1, y1 = max(0, bbox.origin_x), max(0, bbox.origin_y)
                x2, y2 = min(w, bbox.origin_x + bbox.width), min(h, bbox.origin_y + bbox.height)

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_y = max(20, y1 - 10)
                cv2.putText(annotated_image, f"{score:.2f}", (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return annotated_image, len(results.detections)
        finally:
            detector.close()