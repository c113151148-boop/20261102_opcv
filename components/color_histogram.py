# UI 元件層 (功能 2)
import numpy as np
import pandas as pd
import streamlit as st
from core.image_processor import ImageProcessor

def _render_channel(col, image: np.ndarray, caption: str, channel_data: np.ndarray, hex_color: str):
    """內部輔助函式：在一欄內依序渲染圖片與對應直方圖，杜絕迴圈快取問題。"""
    with col:
        st.empty().image(image, caption=caption, use_container_width=True)
        hist_data = ImageProcessor.calc_histogram(channel_data)
        st.caption(f"{caption.split(' ')[0]} Histogram")
        st.bar_chart(hist_data, color=hex_color, height=200)

def render_color_histogram_section(img_rgb_input: np.ndarray):
    st.header("功能 2: 多色彩空間與直方圖")
    
    img_rgb = np.copy(img_rgb_input)
    
    color_space = st.selectbox(
        "選擇檢視模式", 
        ["原始影像 (RGB)", "R/G/B 分離頻道", "YUV 色彩空間"],
        key="color_space_selector"
    )

    if color_space == "原始影像 (RGB)":
        with st.container():
            col_img, col_hist = st.columns(2)
            with col_img:
                st.image(img_rgb, caption="RGB 影像", use_container_width=True)
            with col_hist:
                r_h = ImageProcessor.calc_histogram(img_rgb[:, :, 0])
                g_h = ImageProcessor.calc_histogram(img_rgb[:, :, 1])
                b_h = ImageProcessor.calc_histogram(img_rgb[:, :, 2])
                df_hist = pd.DataFrame({"R": r_h, "G": g_h, "B": b_h})
                st.caption("RGB 直方圖 (Intensity vs Count)")
                st.line_chart(df_hist, color=["#FF0000", "#00FF00", "#0000FF"])

    elif color_space == "R/G/B 分離頻道":
        with st.container():
            # 從核心層取得預先處理好的資料
            chs = ImageProcessor.split_rgb_channels(img_rgb)
            
            # 使用靜態展開呼叫輔助函式，保證 DOM 不丟失
            c1, c2, c3 = st.columns(3)
            _render_channel(c1, chs["R"]["image"], "R 通道", chs["R"]["channel"], "#FF4B4B")
            _render_channel(c2, chs["G"]["image"], "G 通道", chs["G"]["channel"], "#21C354")
            _render_channel(c3, chs["B"]["image"], "B 通道", chs["B"]["channel"], "#0068C9")

    elif color_space == "YUV 色彩空間":
        with st.container():
            # 從核心層取得預先處理好的資料
            chs = ImageProcessor.split_yuv_channels(img_rgb)
            
            # 靜態展開渲染
            c1, c2, c3 = st.columns(3)
            _render_channel(c1, chs["Y"]["image"], "Y (亮度)", chs["Y"]["channel"], "#808080")
            _render_channel(c2, chs["U"]["image"], "U (色差)", chs["U"]["channel"], "#0068C9")
            _render_channel(c3, chs["V"]["image"], "V (色差)", chs["V"]["channel"], "#FF4B4B")