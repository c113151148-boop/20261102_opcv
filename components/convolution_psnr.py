import streamlit as st
import pandas as pd
import numpy as np
import math
from core.image_processor import ImageProcessor

def generate_gaussian_kernel(k_size):
    """根據輸入的尺寸，動態產生二維高斯卷積核
    
    Args:
        k_size (int): 矩陣尺寸
    
    """
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return np.round(kernel, 4)


def render_convolution_psnr_section(img_rgb: np.ndarray):
    """Streamlit 介面 3

    1. 提供使用者輸入一組任意數值的二維陣列 (卷積核)，並可選擇是否正規化
    2. 並排顯示原始影像與進行卷積運算後的影像
    3. 計算並顯示兩個影像間的 PSNR 值

    Args:
        img_rgb (np.ndarray): RGB 影像
    """
    st.header("功能 3: 自訂卷積核與 PSNR 計算")

    # 提供模板選擇與尺寸設定
    c_preset, c_size = st.columns([2, 1])
    with c_preset:
        preset = st.selectbox(
            "選擇卷積核模板或完全自訂", 
            ["自訂 (Custom)", "邊緣銳化 (Edge Sharpening)", "高斯平滑 (Gaussian Blur)", "浮雕 (Emboss)", "均值平滑 (Mean Blur)"]
        )
    with c_size:
        # 卷積核尺寸通常必須是奇數，才能有明確的中心點
        k_size = st.number_input("設定卷積核大小 (N)", min_value=3, max_value=15, step=2, value=3)
        
        # 防呆：如果使用者手動鍵盤輸入了偶數，自動 +1 轉成奇數
        if k_size % 2 == 0:
            k_size += 1
            st.warning(f"卷積核尺寸必須為奇數，已自動為您調整為 {k_size}")

    # 根據選擇的尺寸與模板，動態生成預設矩陣
    if preset == "邊緣銳化 (Edge Sharpening)":
        # 產生周圍全為 -1，中心點為 N^2 的矩陣 (總和為 1)
        default_kernel = np.full((k_size, k_size), -1.0)
        default_kernel[k_size//2, k_size//2] = k_size * k_size

    elif preset == "高斯平滑 (Gaussian Blur)":
        # 動態計算高斯分佈
        default_kernel = generate_gaussian_kernel(k_size)

    elif preset == "浮雕 (Emboss)":
        # 動態產生對角線漸層 (左上負，右下正)
        default_kernel = np.zeros((k_size, k_size), dtype=float)
        center = k_size // 2
        for i in range(k_size):
            for j in range(k_size):
                default_kernel[i, j] = (i - center) + (j - center)
        default_kernel[center, center] = 1.0  # 中心保留原像素亮度

    elif preset == "均值平滑 (Mean Blur)":
        # 全為 1 的矩陣
        default_kernel = np.ones((k_size, k_size), dtype=float)

    else:
        # 完全自訂模式 (預設為 Identity 矩陣，僅中心為 1)
        default_kernel = np.zeros((k_size, k_size), dtype=float)
        default_kernel[k_size//2, k_size//2] = 1.0

    # 提供互動式表格讓使用者直接編輯卷積核數值
    st.write(f"編輯 {k_size}×{k_size} 卷積核矩陣 (可直接點擊表格修改數值):")
    edited_df = st.data_editor(pd.DataFrame(default_kernel), use_container_width=True)
    kernel = edited_df.to_numpy()

    # 正規化選項
    # 模糊濾波 (高斯/均值) 通常需要正規化，其餘不一定
    default_norm = True if "Blur" in preset else False
    normalize = st.checkbox("是否正規化 (各項數值除以矩陣總和)", value=default_norm)

    try:
        # 進行卷積運算
        filtered_img = ImageProcessor.apply_convolution(img_rgb, kernel, normalize=normalize)

        # 計算兩個影像間的 PSNR
        psnr_val = ImageProcessor.calculate_psnr(img_rgb, filtered_img)

        # 並排顯示卷積運算前後的影像
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="原始影像", use_container_width=True)
        with c2:
            st.image(filtered_img, caption="濾波後影像", use_container_width=True)

        # 顯示 PSNR 值
        if math.isinf(psnr_val):
            st.markdown("<h3 style='text-align: center;'>PSNR: ∞ dB (兩張影像完全相同)</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center;'>PSNR: {psnr_val:.2f} dB</h3>", unsafe_allow_html=True)
        # st.latex(r"PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)")

    except Exception as e:
        st.error(f"卷積處理失敗: {e}")