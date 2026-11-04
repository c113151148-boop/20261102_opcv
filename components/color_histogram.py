# UI 元件層 (功能 2)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from core.image_processor import ImageProcessor

def render_color_histogram_section(img_rgb: np.ndarray):
    st.header("功能 2: 多色彩空間與直方圖")
    color_space = st.selectbox("選擇檢視模式", ["原始影像 (RGB)", "R/G/B 分離頻道", "YUV 色彩空間"])

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
        
        cols = st.columns(3)
        for i, col in enumerate(cols):
            ch_img = np.zeros_like(img_rgb)
            ch_img[:, :, i] = channels[i]
            with col:
                st.image(ch_img, caption=f"{labels[i]} 通道", width=300)
                fig = ImageProcessor.plot_single_histogram(channels[i], colors[i], f"{labels[i]} Histogram")
                st.pyplot(fig)
                plt.close(fig)

    elif color_space == "YUV 色彩空間":
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        yuv_data = [(y, "gray", "Y (亮度)"), (u, "blue", "U (色差)"), (v, "red", "V (色差)")]
        
        cols = st.columns(3)
        for col, (ch, color, label) in zip(cols, yuv_data):
            with col:
                st.image(ch, caption=label, width=300, clamp=True)
                fig = ImageProcessor.plot_single_histogram(ch, color, f"{label[0]} Histogram")
                st.pyplot(fig)
                plt.close(fig)