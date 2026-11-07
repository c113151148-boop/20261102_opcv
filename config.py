from pathlib import Path

# 檔案路徑設定
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "blaze_face_full_range.tflite"

# 預設參數
DEFAULT_CONFIDENCE = 0.5