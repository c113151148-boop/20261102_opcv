import streamlit as st
import numpy as np
import math
from config import DEFAULT_KERNEL_STR
from core.image_processor import ImageProcessor

def render_convolution_psnr_section(img_rgb: np.ndarray):
    """Streamlit 介面 3

    1. 提供使用者輸入一組任意數值的二維陣列 (卷積核)，並可選擇是否正規化
    2. 並排顯示原始影像與進行卷積運算後的影像
    3. 計算並顯示兩個影像間的 PSNR 值

    Args:
        img_rgb (np.ndarray): RGB 影像
    """
    st.header("功能 3: 自訂卷積核與 PSNR 計算")
    kernel_input = st.text_input("卷積核矩陣 (以逗號分隔行、分號分隔列)", DEFAULT_KERNEL_STR)
    normalize = st.checkbox("是否正規化 (除以權重總和)", value=True)

    try:
        # 將輸入字串轉為 np.ndarray 並以表格顯示
        kernel = ImageProcessor.parse_kernel(kernel_input)
        st.write("目前使用的 Kernel:")
        st.dataframe(kernel)

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