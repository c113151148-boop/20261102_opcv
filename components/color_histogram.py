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
        with st.container():
            r_ch, g_ch, b_ch = cv2.split(img_rgb)
            
            img_r = np.zeros_like(img_rgb)
            img_r[:, :, 0] = r_ch
            
            img_g = np.zeros_like(img_rgb)
            img_g[:, :, 1] = g_ch
            
            img_b = np.zeros_like(img_rgb)
            img_b[:, :, 2] = b_ch

            # 捨棄迴圈，採用最穩定的靜態展開渲染
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.empty().image(img_r, caption="R 通道", use_container_width=True)
                h_r, _ = np.histogram(r_ch, bins=256, range=[0, 256])
                st.caption("R Histogram")
                st.bar_chart(h_r, color="#FF4B4B", height=200)
                
            with c2:
                st.empty().image(img_g, caption="G 通道", use_container_width=True)
                h_g, _ = np.histogram(g_ch, bins=256, range=[0, 256])
                st.caption("G Histogram")
                st.bar_chart(h_g, color="#21C354", height=200)
                
            with c3:
                st.empty().image(img_b, caption="B 通道", use_container_width=True)
                h_b, _ = np.histogram(b_ch, bins=256, range=[0, 256])
                st.caption("B Histogram")
                st.bar_chart(h_b, color="#0068C9", height=200)

    elif color_space == "YUV 色彩空間":
        with st.container():
            img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
            y_ch, u_ch, v_ch = cv2.split(img_yuv)
            
            y_gray = np.dstack((y_ch, y_ch, y_ch))
            
            u_norm = cv2.normalize(u_ch, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            u_gray = np.dstack((u_norm, u_norm, u_norm))
            
            v_norm = cv2.normalize(v_ch, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            v_gray = np.dstack((v_norm, v_norm, v_norm))

            # 捨棄迴圈，採用最穩定的靜態展開渲染
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.empty().image(y_gray, caption="Y (亮度)", use_container_width=True)
                h_y, _ = np.histogram(y_ch, bins=256, range=[0, 256])
                st.caption("Y (亮度) Histogram")
                st.bar_chart(h_y, color="#808080", height=200)
                
            with c2:
                st.empty().image(u_gray, caption="U (色差)", use_container_width=True)
                h_u, _ = np.histogram(u_ch, bins=256, range=[0, 256])
                st.caption("U (色差) Histogram")
                st.bar_chart(h_u, color="#0068C9", height=200)
                
            with c3:
                st.empty().image(v_gray, caption="V (色差)", use_container_width=True)
                h_v, _ = np.histogram(v_ch, bins=256, range=[0, 256])
                st.caption("V (色差) Histogram")
                st.bar_chart(h_v, color="#FF4B4B", height=200)