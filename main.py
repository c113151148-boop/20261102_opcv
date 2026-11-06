import streamlit as st
import numpy as np
import cv2

from components.image_info import render_image_info_section
from components.color_histogram import render_color_histogram_section
from components.convolution_psnr import render_convolution_psnr_section
from components.face_detection_ui import render_face_detection_section

st.set_page_config(page_title="基於 AI 輔助之互動式影像處理與人臉分析工具", layout="wide")
st.title("互動式影像處理與人臉分析工具")

# 檔案上傳
with st.sidebar:
    st.header("設定與控制區")
    uploaded_file = st.file_uploader("上傳影像檔 (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        # 讀取影像
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            st.error("無法讀取影像，請確認檔案格式是否正確。")
            st.stop()
        
        # 轉換為 RGB 影像
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

if img_rgb is not None:
    # 渲染各個功能模組 UI
    render_image_info_section(img_rgb, uploaded_file.size)
    st.markdown("---")

    render_color_histogram_section(img_rgb)
    st.markdown("---")

    render_convolution_psnr_section(img_rgb)
    st.markdown("---")

    render_face_detection_section(img_rgb)