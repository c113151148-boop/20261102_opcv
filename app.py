import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Streamlit 頁面設定
st.set_page_config(page_title="基於 AI 輔助之互動式影像處理與人臉分析工具", layout="wide")
st.title("互動式影像處理與人臉分析工具")


# MediaPipe 模型設定
MODEL_PATH = Path(__file__).resolve().parent / "blaze_face_full_range.tflite"


# 功能 1：載入影像與基礎解析
st.header("載入影像")
uploaded_file = st.file_uploader("上傳影像檔 (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # 將上傳的檔案轉換成 NumPy 陣列
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # 使用 OpenCV 解碼影像，預設取得 BGR 格式
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("無法讀取影像，請確認檔案格式是否正確。")
        st.stop()

    # BGR 轉換成 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 取得影像尺寸
    image_height, image_width, image_channels = img_rgb.shape


    # 功能 1：影像基本資訊
    st.header("功能 1: 影像載入與基礎解析")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="原始影像", use_container_width=True)

    with col2:
        st.subheader("影像中繼資訊")
        st.write(f"**解析度 (寬 x 高):** {image_width} x {image_height} pixels")
        st.write(f"**色彩維度 (Channels):** {image_channels}")
        st.write(f"**檔案大小:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**資料型態:** {img_rgb.dtype}")

    st.markdown("---")


    # 功能 2：多色彩空間與直方圖
    st.header("功能 2: 多色彩空間與直方圖")
    color_space = st.selectbox("選擇檢視模式", ["原始影像 (RGB)", "R/G/B 分離頻道", "YUV 色彩空間"])

    # 直方圖繪製函式
    def plot_histogram(img_channel, color, title):
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.hist(img_channel.ravel(), bins=256, range=[0, 256], color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xlim([0, 256])
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    # RGB 原始影像與直方圖
    if color_space == "原始影像 (RGB)":
        col_img, col_hist = st.columns(2)
        with col_img:
            st.image(img_rgb, caption="RGB 影像", use_container_width=True)

        with col_hist:
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ("r", "g", "b")
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                ax.plot(hist, color=color, alpha=0.7, label=color.upper())

            ax.set_title("RGB 直方圖")
            ax.set_xlim([0, 256])
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    # RGB 通道分離
    elif color_space == "R/G/B 分離頻道":
        r, g, b = cv2.split(img_rgb)

        # 將單一通道轉成 RGB 影像，讓各通道分別以紅、綠、藍色顯示
        r_img = np.zeros_like(img_rgb)
        g_img = np.zeros_like(img_rgb)
        b_img = np.zeros_like(img_rgb)
        r_img[:, :, 0] = r
        g_img[:, :, 1] = g
        b_img[:, :, 2] = b

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(r_img, caption="R 通道", width=300)
            fig = plot_histogram(r, "red", "R Histogram")
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            st.image(g_img, caption="G 通道", width=300)
            fig = plot_histogram(g, "green", "G Histogram")
            st.pyplot(fig)
            plt.close(fig)

        with c3:
            st.image(b_img, caption="B 通道", width=300)
            fig = plot_histogram(b, "blue", "B Histogram")
            st.pyplot(fig)
            plt.close(fig)

    # YUV 色彩空間
    elif color_space == "YUV 色彩空間":
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(y, caption="Y (亮度)", width=300, clamp=True)
            fig = plot_histogram(y, "gray", "Y Histogram")
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            st.image(u, caption="U (色差)", width=300, clamp=True)
            fig = plot_histogram(u, "blue", "U Histogram")
            st.pyplot(fig)
            plt.close(fig)

        with c3:
            st.image(v, caption="V (色差)", width=300, clamp=True)
            fig = plot_histogram(v, "red", "V Histogram")
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")


    # 功能 3：自訂卷積核與 PSNR 計算
    st.header("功能 3: 自訂卷積核與 PSNR 計算")
    st.write("輸入卷積核 (以逗號分隔列，分號分隔行，例如: 0,-1,0;-1,5,-1;0,-1,0)")

    kernel_input = st.text_input("卷積核矩陣", "0,-1,0;-1,5,-1;0,-1,0")
    normalize = st.checkbox("套用正規化 (除以權重總和)", value=False)

    # 解析卷積核
    try:
        rows = kernel_input.split(";")
        kernel_matrix = []
        for row in rows:
            values = [float(x.strip()) for x in row.split(",")]
            kernel_matrix.append(values)

        kernel = np.array(kernel_matrix, dtype=np.float32)

        # 確認卷積核必須為方陣
        if kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 == 0:
            raise ValueError("卷積核必須是奇數尺寸的正方形矩陣 (例如 3x3, 5x5)。")

        # 權重正規化
        if normalize:
            k_sum = np.sum(kernel)
            if k_sum != 0:
                kernel = kernel / k_sum
            else:
                st.warning("權重總和為 0，跳過正規化以避免除以零錯誤。")

        st.write("目前使用的 Kernel:")
        st.dataframe(kernel)

        # 執行影像濾波
        filtered_img = cv2.filter2D(img_rgb, -1, kernel)

        # PSNR 計算函式
        def calculate_psnr(img1, img2):
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)

            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float("inf")

            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
            return psnr

        # 計算 PSNR
        psnr_val = calculate_psnr(img_rgb, filtered_img)

        # 顯示濾波結果
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="原始影像", use_container_width=True)

        with c2:
            st.image(filtered_img, caption="濾波後影像", use_container_width=True)
            if math.isinf(psnr_val):
                st.success("PSNR: ∞ dB (兩張影像完全相同)")
            else:
                st.success(f"**PSNR (Peak Signal-to-Noise Ratio):** {psnr_val:.2f} dB")

            st.latex(r"PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)")

    except (ValueError, TypeError) as e:
        st.error(f"卷積核輸入格式錯誤，請檢查: {e}")
    except cv2.error as e:
        st.error(f"OpenCV 濾波發生錯誤: {e}")

    st.markdown("---")


    # 功能 4：新版 MediaPipe 人臉偵測
    st.header("功能 4: 人臉偵測 (MediaPipe Tasks API)")

    min_confidence = st.slider("最低信心度閾值", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    run_detection = st.button("執行人臉偵測")

    # 執行新版人臉偵測
    if run_detection:
        if not MODEL_PATH.exists():
            st.error(f"找不到模型檔案: {MODEL_PATH}")
        else:
            try:
                # 建立 MediaPipe 模型設定
                base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
                options = vision.FaceDetectorOptions(
                    base_options=base_options, 
                    min_detection_confidence=min_confidence
                )
                # 建立人臉偵測器物件
                detector = vision.FaceDetector.create_from_options(options)

                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    results = detector.detect(mp_image)

                    annotated_image = img_rgb.copy()
                    face_count = 0

                    # 繪製邊界框與信心度
                    for detection in results.detections:
                        face_count += 1
                        score = detection.categories[0].score
                        bbox = detection.bounding_box

                        x = bbox.origin_x
                        y = bbox.origin_y
                        width = bbox.width
                        height = bbox.height

                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(image_width, x + width)
                        y2 = min(image_height, y + height)

                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text_y = max(20, y1 - 10)
                        cv2.putText(
                            annotated_image, 
                            f"{score:.2f}", 
                            (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 255, 0), 
                            2
                        )
                finally:
                    detector.close()

                # 顯示人臉偵測結果
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.image(annotated_image, caption="人臉偵測結果", use_container_width=True)

                with c2:
                    st.info(f"**偵測到的人臉總數:** {face_count}")
                    st.write(f"設定的信心度閾值: {min_confidence}")
                    if face_count == 0:
                        st.warning("未偵測到人臉，請嘗試調低信心度閾值。")

            except Exception as e:
                st.error(f"MediaPipe 人臉偵測發生錯誤: {e}")