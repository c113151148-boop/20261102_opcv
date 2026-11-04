# UI 元件層 (功能 2)

import cv2
import numpy as np
import pandas as pd
import streamlit as st

def render_color_histogram_section(img_rgb_input: np.ndarray):
    st.header("功能 2: 多色彩空間與直方圖")
    
    img_rgb = np.copy(img_rgb_input)
    
    color_space = st.selectbox(
        "選擇檢視模式", 
        ["原始影像 (RGB)", "R/G/B 分離頻道", "YUV 色彩空間"],
        key="color_space_selector"
    )

    if color_space == "原始影像 (RGB)":
        # 使用獨立 Container 隔絕 DOM 快取
        with st.container():
            col_img, col_hist = st.columns(2)
            with col_img:
                st.image(img_rgb, caption="RGB 影像", use_container_width=True)
            with col_hist:
                r_h, _ = np.histogram(img_rgb[:, :, 0], bins=256, range=[0, 256])
                g_h, _ = np.histogram(img_rgb[:, :, 1], bins=256, range=[0, 256])
                b_h, _ = np.histogram(img_rgb[:, :, 2], bins=256, range=[0, 256])
                df_hist = pd.DataFrame({"R": r_h, "G": g_h, "B": b_h})
                st.caption("RGB 直方圖 (Intensity vs Count)")
                st.line_chart(df_hist, color=["#FF0000", "#00FF00", "#0000FF"])

    elif color_space == "R/G/B 分離頻道":
        # 使用獨立 Container 隔絕 DOM 快取
        with st.container():
            r_ch, g_ch, b_ch = cv2.split(img_rgb)
            
            # 建立三張明確獨立的彩色 3 通道矩陣
            img_r = np.zeros_like(img_rgb)
            img_r[:, :, 0] = r_ch
            
            img_g = np.zeros_like(img_rgb)
            img_g[:, :, 1] = g_ch
            
            img_b = np.zeros_like(img_rgb)
            img_b[:, :, 2] = b_ch

            img_cols = st.columns(3)
            with img_cols[0]:
                st.image(img_r, caption="R 通道", use_container_width=True)
            with img_cols[1]:
                st.image(img_g, caption="G 通道", use_container_width=True)
            with img_cols[2]:
                st.image(img_b, caption="B 通道", use_container_width=True)
            
            hist_cols = st.columns(3)
            for col, ch_data, label_name, hex_color in zip(
                hist_cols, [r_ch, g_ch, b_ch], ["R", "G", "B"], ["#FF4B4B", "#21C354", "#0068C9"]
            ):
                with col:
                    h_data, _ = np.histogram(ch_data, bins=256, range=[0, 256])
                    st.caption(f"{label_name} Histogram")
                    st.bar_chart(h_data, color=hex_color, height=200)

    elif color_space == "YUV 色彩空間":
        # 使用獨立 Container 隔絕 DOM 快取
        with st.container():
            img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
            y_ch, u_ch, v_ch = cv2.split(img_yuv)
            
            # 構建純灰階三通道
            y_gray = np.dstack((y_ch, y_ch, y_ch))
            
            u_norm = cv2.normalize(u_ch, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            u_gray = np.dstack((u_norm, u_norm, u_norm))
            
            v_norm = cv2.normalize(v_ch, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            v_gray = np.dstack((v_norm, v_norm, v_norm))

            img_cols = st.columns(3)
            with img_cols[0]:
                st.image(y_gray, caption="Y (亮度)", use_container_width=True)
            with img_cols[1]:
                st.image(u_gray, caption="U (色差)", use_container_width=True)
            with img_cols[2]:
                st.image(v_gray, caption="V (色差)", use_container_width=True)
                    
            hist_cols = st.columns(3)
            for col, ch_data, label, hex_color in zip(
                hist_cols, [y_ch, u_ch, v_ch], ["Y (亮度)", "U (色差)", "V (色差)"], ["#808080", "#0068C9", "#FF4B4B"]
            ):
                with col:
                    h_data, _ = np.histogram(ch_data, bins=256, range=[0, 256])
                    st.caption(f"{label[0]} Histogram")
                    st.bar_chart(h_data, color=hex_color, height=200)