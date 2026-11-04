# UI 元件層 (功能 3)

import math
import streamlit as st
import numpy as np
from config import DEFAULT_KERNEL_STR
from core.image_processor import ImageProcessor

def render_convolution_psnr_section(img_rgb: np.ndarray):
    st.header("功能 3: 自訂卷積核與 PSNR 計算")
    kernel_input = st.text_input("卷積核矩陣 (逗號分隔列，分號分隔行)", DEFAULT_KERNEL_STR)
    normalize = st.checkbox("套用正規化 (除以權重總和)", value=False)

    try:
        kernel = ImageProcessor.parse_kernel(kernel_input)
        st.write("目前使用的 Kernel:")
        st.dataframe(kernel)

        filtered_img = ImageProcessor.apply_convolution(img_rgb, kernel, normalize=normalize)
        psnr_val = ImageProcessor.calculate_psnr(img_rgb, filtered_img)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="原始影像", use_container_width=True)
        with c2:
            st.image(filtered_img, caption="濾波後影像", use_container_width=True)
            if math.isinf(psnr_val):
                st.success("PSNR: ∞ dB (兩張影像完全相同)")
            else:
                st.success(f"**PSNR:** {psnr_val:.2f} dB")
            st.latex(r"PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)")

    except Exception as e:
        st.error(f"卷積處理失敗: {e}")