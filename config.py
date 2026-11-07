from pathlib import Path

# 檔案路徑
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "blaze_face_full_range.tflite"

# 預設參數們
DEFAULT_CONFIDENCE = 0.5