from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetectorService:
    """人臉偵測模型服務類別"""

    def __init__(self, model_path):
        """初始化物件

        Args: 
            model_path (Path): MediaPipe 人臉偵測模型的檔案路徑 (.tflite 檔)

        Raises:
            FileNotFoundError: 找不到 MediaPipe 人臉偵測模型的檔案
        """
        self.model_path = model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型檔案: {self.model_path}")

    def detect(self, img_rgb, min_confidence=0.5):
        """偵測人臉並與信心度一起標示在影像上

        Args:
            img_rgb (np.ndarray): RGB 影像
            min_confidence (float): 最低信心度

        Returns:
            回傳一個 tuple (np.ndarray, int)
            np.ndarray: 標示好的圖片
            int: 偵測到的人臉數
        """
        # 設定模型的實際檔案位置
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))

        # 設定人臉偵測指定參數
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence
        )

        # 建立人臉偵測器物件
        detector = vision.FaceDetector.create_from_options(options)

        try:
            # 把 RGB 影像轉換成 MediaPipe 影像格式
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # 將影像丟進模型運算
            results = detector.detect(mp_image)

            # 在影像上標示人臉偵測結果與信心度
            annotated_image = img_rgb.copy()
            h, w, _ = img_rgb.shape

            # 將每一張辨識出的人臉標示出來
            for detection in results.detections:
                score = detection.categories[0].score # 信心度
                bbox = detection.bounding_box # 框的座標資訊

                # 計算框的左上角與右下角座標後畫出綠色框
                x1, y1 = max(0, bbox.origin_x), max(0, bbox.origin_y)
                x2, y2 = min(w, bbox.origin_x + bbox.width), min(h, bbox.origin_y + bbox.height)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 標示信心度
                text_y = max(20, y1 - 10)
                cv2.putText(annotated_image, f"{score:.2f}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 回傳畫好的圖片、找到的人臉數
            return annotated_image, len(results.detections)
        
        finally:
            detector.close() # 釋放資源