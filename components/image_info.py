import streamlit as st
import numpy as np

def render_image_info_section(img_rgb, file_size):
    """Streamlit 介面 1

    顯示原始影像與基本中繼資訊

    Args:
        img_rgb (np.ndarray): RGB 影像
        ile_size (int): 影像檔案大小
    """
    st.header("功能 1: 影像基本中繼資訊")
    h, w, c = img_rgb.shape

    # 顯示
    col1, col_space, col2 = st.columns([1.2, 0.1, 1])
    with col1:
        st.markdown(
            """
            <style>
            .centered-image-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
            }
            .centered-image-container img {
                max-height: 150px !important;
                width: auto !important;
                object-fit: contain;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="centered-image-container">', unsafe_allow_html=True)
        st.image(img_rgb, caption="原始影像", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_space:
        st.empty()

    with col2:
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        st.subheader("影像中繼資訊")
        st.markdown("---")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="影像寬度", value=f"{w} px")
            st.metric(label="色彩維度", value=f"{c}")
        with m_col2:
            st.metric(label="影像高度", value=f"{h} px")
            st.metric(label="檔案大小", value=f"{file_size / 1024:.2f} KB")
        
        st.markdown(
            f"""
            <div style="background-color: rgba(128, 128, 128, 0.1); padding: 10px; border-radius: 6px; margin-top: 10px;">
                <span style="font-size: 14px; color: gray;">資料型態 (dtype)：</span>
                <span style="font-size: 14px; font-weight: bold;">{img_rgb.dtype}</span>
            </div>
            """,
            unsafe_allow_html=True
        )