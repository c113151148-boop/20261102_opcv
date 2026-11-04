# UI 元件層 (功能 2)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from core.image_processor import ImageProcessor

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
            fig = ImageProcessor.plot_rgb_histogram(img_rgb)
            st.pyplot(fig)
            plt.close(fig)

    elif color_space == "R/G/B 分離頻道":
        r, g, b = cv2.split(img_rgb)
        channels = [r, g, b]
        colors = ["red", "green", "blue"]
        labels = ["R", "G", "B"]
        
        # 第一排：通道圖片
        img_cols = st.columns(3)
        for i, col in enumerate(img_cols):
            ch_img = np.zeros_like(img_rgb)
            ch_img[:, :, i] = channels[i]
            with col:
                st.image(ch_img, caption=f"{labels[i]} 通道", use_container_width=True)
        
        # 第二排：直方圖 (移除多餘 key 參數，避免傳給 Matplotlib savefig)
        hist_cols = st.columns(3)
        for i, col in enumerate(hist_cols):
            with col:
                fig = ImageProcessor.plot_single_histogram(channels[i], colors[i], f"{labels[i]} Histogram")
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

    elif color_space == "YUV 色彩空間":
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        
        yuv_data = [
            (y, "gray", "Y (亮度)"), 
            (u, "blue", "U (色差)"), 
            (v, "red", "V (色差)")
        ]
        
        # 第一排：YUV 通道圖片 (np.stack 確保標準灰階)
        img_cols = st.columns(3)
        for col, (ch, _, label) in zip(img_cols, yuv_data):
            with col:
                gray_3ch = np.stack((ch, ch, ch), axis=-1)
                st.image(gray_3ch, caption=label, use_container_width=True)
                
        # 第二排：YUV 直方圖
        hist_cols = st.columns(3)
        for col, (ch, color_name, label) in zip(hist_cols, yuv_data):
            with col:
                fig = ImageProcessor.plot_single_histogram(ch, color_name, f"{label[0]} Histogram")
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)