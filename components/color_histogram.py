# UI 元件層 (功能 2)

import cv2
import numpy as np
import pandas as pd
import streamlit as st

def render_color_histogram_section(img_rgb: np.ndarray):
    st.header("功能 2: 多色彩空間與直方圖")
    color_space = st.selectbox(
        "選擇檢視模式", 
        ["原始影像 (RGB)", "R/G/B 分離頻道", "YUV 色彩空間"],
        key="color_space_selector"
    )

    if color_space == "原始影像 (RGB)":
        col_img, col_hist = st.columns(2)
        with col_img:
            st.image(img_rgb, caption="RGB 影像", use_container_width=True)
        with col_hist:
            # 原生繪製全通道 RGB 直方圖
            r_hist, _ = np.histogram(img_rgb[:, :, 0], bins=256, range=[0, 256])
            g_hist, _ = np.histogram(img_rgb[:, :, 1], bins=256, range=[0, 256])
            b_hist, _ = np.histogram(img_rgb[:, :, 2], bins=256, range=[0, 256])
            df_hist = pd.DataFrame({"R": r_hist, "G": g_hist, "B": b_hist})
            st.caption("RGB 直方圖 (Intensity vs Count)")
            st.line_chart(df_hist, color=["#FF0000", "#00FF00", "#0000FF"])

    elif color_space == "R/G/B 分離頻道":
        r, g, b = cv2.split(img_rgb)
        channels = [r, g, b]
        labels = ["R", "G", "B"]
        hex_colors = ["#FF4B4B", "#21C354", "#0068C9"]
        
        # 第一排：通道圖片
        img_cols = st.columns(3)
        for i, col in enumerate(img_cols):
            ch_img = np.zeros_like(img_rgb)
            ch_img[:, :, i] = channels[i]
            with col:
                st.image(ch_img, caption=f"{labels[i]} 通道", use_container_width=True)
        
        # 第二排：原生直方圖 (零快取問題)
        hist_cols = st.columns(3)
        for i, col in enumerate(hist_cols):
            with col:
                hist_data, _ = np.histogram(channels[i], bins=256, range=[0, 256])
                st.caption(f"{labels[i]} Histogram")
                st.bar_chart(hist_data, color=hex_colors[i], height=200)

    elif color_space == "YUV 色彩空間":
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        
        yuv_data = [
            (y, "#808080", "Y (亮度)"), 
            (u, "#0068C9", "U (色差)"), 
            (v, "#FF4B4B", "V (色差)")
        ]
        
        # 第一排：YUV 通道圖片 (正規化拉伸強化色彩對比)
        img_cols = st.columns(3)
        for col, (ch, _, label) in zip(img_cols, yuv_data):
            with col:
                # 對 U/V 進行直方圖對比度強化，讓細節看得更清楚
                norm_ch = cv2.normalize(ch, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                gray_3ch = np.stack((norm_ch, norm_ch, norm_ch), axis=-1)
                st.image(gray_3ch, caption=label, use_container_width=True)
                
        # 第二排：原生直方圖
        hist_cols = st.columns(3)
        for col, (ch, hex_color, label) in zip(hist_cols, yuv_data):
            with col:
                hist_data, _ = np.histogram(ch, bins=256, range=[0, 256])
                st.caption(f"{label[0]} Histogram")
                st.bar_chart(hist_data, color=hex_color, height=200)