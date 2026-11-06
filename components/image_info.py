import streamlit as st
import numpy as np

def render_image_info_section(img_rgb, file_size):
    """Streamlit 介面 1

    顯示原始影像與基本中繼資訊

    Args:
        img_rgb (np.ndarray): RGB 影像
        ile_size (int): 影像檔案大小
    """
    st.header("功能 1: 影像載入與基礎解析")
    h, w, c = img_rgb.shape
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="原始影像", use_container_width=True)
    with col2:
        st.subheader("影像中繼資訊")
        st.write(f"**解析度 (寬 x 高):** {w} x {h} pixels")
        st.write(f"**色彩維度:** {c}")
        st.write(f"**檔案大小:** {file_size / 1024:.2f} KB")
        st.write(f"**資料型態:** {img_rgb.dtype}")