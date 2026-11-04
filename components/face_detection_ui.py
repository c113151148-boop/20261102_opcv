# UI 元件層 (功能 4)

import streamlit as st
import numpy as np
from config import MODEL_PATH, DEFAULT_CONFIDENCE
from core.face_detector import FaceDetectorService

def render_face_detection_section(img_rgb: np.ndarray):
    st.header("功能 4: 人臉偵測 (MediaPipe Tasks API)")
    min_confidence = st.slider("最低信心度閾值", min_value=0.1, max_value=1.0, value=DEFAULT_CONFIDENCE, step=0.05)

    if st.button("執行人臉偵測"):
        try:
            detector = FaceDetectorService(MODEL_PATH)
            annotated_image, face_count = detector.detect(img_rgb, min_confidence)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(annotated_image, caption="人臉偵測結果", use_container_width=True)
            with c2:
                st.info(f"**偵測到的人臉總數:** {face_count}")
                st.write(f"設定的信心度閾值: {min_confidence}")
                if face_count == 0:
                    st.warning("未偵測到人臉，請嘗試調低信心度閾值。")
        except Exception as e:
            st.error(f"MediaPipe 人臉偵測發生錯誤: {e}")