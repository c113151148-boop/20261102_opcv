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
            r_hist, _ = np.histogram(img_rgb[:, :, 0], bins=256, range=[0, 256])
            g_hist, _ = np.histogram(img_rgb[:, :, 1], bins=256, range=[0, 256])
            b_hist, _ = np.histogram(img_rgb[:, :, 2], bins=256, range=[0, 256])
            df_hist = pd.DataFrame({"R": r_hist, "G": g_hist, "B": b_hist})
            st.caption("RGB 直方圖 (Intensity vs Count)")
            st.line_chart(df_hist, color=["#FF0000", "#00FF00", "#0000FF"])

    elif color_space == "R/G/B 分離頻道":
        r, g, b = cv2.split(img_rgb)
        
        # 1. 顯式創建 3 個獨立的 RGB 彩色影像矩陣，確保絕不相互干擾
        img_r = np.zeros_like(img_rgb)
        img_r[:, :, 0] = r
        
        img_g = np.zeros_like(img_rgb)
        img_g[:, :, 1] = g
        
        img_b = np.zeros_like(img_rgb)
        img_b[:, :, 2] = b

        rgb_imgs = [img_r, img_g, img_b]
        channels = [r, g, b]
        labels = ["R 通道", "G 通道", "B 通道"]
        hex_colors = ["#FF4B4B", "#21C354", "#0068C9"]
        
        # 第一排：專門渲染 3 張單獨色彩通道圖片
        img_cols = st.columns(3)
        for col, ch_img, label in zip(img_cols, rgb_imgs, labels):
            with col:
                st.image(ch_img, caption=label, use_container_width=True)
        
        # 第二排：專門渲染 3 張原生直方圖
        hist_cols = st.columns(3)
        for col, ch_data, label_name, hex_color in zip(hist_cols, channels, ["R", "G", "B"], hex_colors):
            with col:
                hist_data, _ = np.histogram(ch_data, bins=256, range=[0, 256])
                st.caption(f"{label_name} Histogram")
                st.bar_chart(hist_data, color=hex_color, height=200)

    elif color_space == "YUV 色彩空間":
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        
        # 對 Y (亮度)、U (色差)、V (色差) 分別進行獨立灰階矩陣建構
        y_gray = np.stack((y, y, y), axis=-1)
        
        # 對 U/V 進行對比度拉伸強化視覺效果
        u_norm = cv2.normalize(u, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        u_gray = np.stack((u_norm, u_norm, u_norm), axis=-1)
        
        v_norm = cv2.normalize(v, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        v_gray = np.stack((v_norm, v_norm, v_norm), axis=-1)

        yuv_imgs = [y_gray, u_gray, v_gray]
        channels = [y, u, v]
        yuv_labels = ["Y (亮度)", "U (色差)", "V (色差)"]
        hex_colors = ["#808080", "#0068C9", "#FF4B4B"]
        
        # 第一排：專門渲染 3 張 YUV 通道圖片
        img_cols = st.columns(3)
        for col, ch_img, label in zip(img_cols, yuv_imgs, yuv_labels):
            with col:
                st.image(ch_img, caption=label, use_container_width=True)
                
        # 第二排：專門渲染 3 張 YUV 直方圖
        hist_cols = st.columns(3)
        for col, ch_data, label, hex_color in zip(hist_cols, channels, yuv_labels, hex_colors):
            with col:
                hist_data, _ = np.histogram(ch_data, bins=256, range=[0, 256])
                st.caption(f"{label[0]} Histogram")
                st.bar_chart(hist_data, color=hex_color, height=200)