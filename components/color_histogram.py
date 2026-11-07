import streamlit as st
import pandas as pd
import numpy as np
from core.image_processor import ImageProcessor

def _render_channel(col, image, caption, channel, hex_color):
    """(輔助函式) 在同一欄內依序渲染圖片與對應的單通道直方圖
    
    Args:
        col (streamlit.delta_generator.DeltaGenerator): 要顯示的欄位
        image (np.ndarray): 要顯示的圖片
        caption (str): 圖片下方的說明文字
        channel (np.ndarray): 單通道的像素矩陣
        hex_color (str): 直方圖的顏色
    """
    with col:
        st.empty().image(image, caption=caption, use_container_width=True)

        # 取得直方圖數據
        hist_data = ImageProcessor.calc_histogram(channel) 

        st.caption(f"{caption.split(' ')[0]} Histogram")
        st.bar_chart(hist_data, color=hex_color, height=200)


def render_color_histogram_section(img_rgb_input):
    """Streamlit 介面 2

    提供下拉式選單選擇三種不同模式，根據對應模式顯示對應通道的灰階/強度直方圖與視覺化圖片
    
    Args:
        img_rgb_input (np.ndarray): RGB 影像
    """
    st.header("功能 2: 多色彩空間轉換與同步直方圖分析")
    
    img_rgb = np.copy(img_rgb_input)
    
    color_space = st.selectbox("選擇顯示模式",
                               ["原始影像 (Original)","RGB 各分量獨立檢視", "YUV 色彩空間轉換"],
                               key="color_space_selector")

    if color_space == "原始影像 (Original)":
        with st.container():
            col_img, col_hist = st.columns(2)

            with col_img:
                st.image(img_rgb, caption="原始影像 (Original)", use_container_width=True)

            with col_hist:
                # 取得各通道直方圖數據
                r_h = ImageProcessor.calc_histogram(img_rgb[:, :, 0])
                g_h = ImageProcessor.calc_histogram(img_rgb[:, :, 1])
                b_h = ImageProcessor.calc_histogram(img_rgb[:, :, 2])
                
                # 整理成 DataFrame
                df_hist = pd.DataFrame({"R": r_h, "G": g_h, "B": b_h})

                # 顯示折線圖
                st.caption("RGB 折線圖 (Intensity vs Count)")
                st.line_chart(df_hist, color=["#FF0000", "#00FF00", "#0000FF"])

    elif color_space == "RGB 各分量獨立檢視":
        with st.container():
            # 取得拆分好的 R、G、B 各通道的像素矩陣與視覺化圖片
            chs = ImageProcessor.split_rgb_channels(img_rgb)
            
            # 使用靜態展開呼叫輔助函式渲染視覺化圖片與對應直方圖，維持畫面渲染的穩定性
            c1, c2, c3 = st.columns(3)
            _render_channel(c1, chs["R"]["image"], "R 通道", chs["R"]["channel"], "#FF4B4B")
            _render_channel(c2, chs["G"]["image"], "G 通道", chs["G"]["channel"], "#21C354")
            _render_channel(c3, chs["B"]["image"], "B 通道", chs["B"]["channel"], "#0068C9")

    elif color_space == "YUV 色彩空間轉換":
        with st.container():
            # 取得拆分好的 Y、U、V 各通道的像素矩陣與視覺化圖片
            chs = ImageProcessor.split_yuv_channels(img_rgb)
            
            # 靜態展開渲染
            c1, c2, c3 = st.columns(3)
            _render_channel(c1, chs["Y"]["image"], "Y (亮度)", chs["Y"]["channel"], "#808080")
            _render_channel(c2, chs["U"]["image"], "U (色度)", chs["U"]["channel"], "#0068C9")
            _render_channel(c3, chs["V"]["image"], "V (飽和度)", chs["V"]["channel"], "#FF4B4B")