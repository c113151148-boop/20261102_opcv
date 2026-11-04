# 核心影像演算法

import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageProcessor:
    """影像處理演算法服務類別"""

    @staticmethod
    def parse_kernel(kernel_str: str) -> np.ndarray:
        """解析字串格式的卷積核並轉換為 NumPy 陣列。"""
        rows = kernel_str.split(";")
        kernel_matrix = [[float(x.strip()) for x in row.split(",")] for row in rows]
        kernel = np.array(kernel_matrix, dtype=np.float32)

        if kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 == 0:
            raise ValueError("卷積核必須是奇數尺寸的正方形矩陣 (例如 3x3, 5x5)。")
        return kernel

    @staticmethod
    def apply_convolution(img_rgb: np.ndarray, kernel: np.ndarray, normalize: bool = False) -> np.ndarray:
        """對輸入影像套用 2D 卷積遮罩處理。"""
        if normalize:
            k_sum = np.sum(kernel)
            if k_sum != 0:
                kernel = kernel / k_sum
        return cv2.filter2D(img_rgb, -1, kernel)

    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """計算兩張影像之間的峰值訊號對雜訊比 (PSNR, Peak Signal-to-Noise Ratio)。"""
        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)
        mse = np.mean((img1_f - img2_f) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(mse))

    @staticmethod
    def plot_single_histogram(img_channel: np.ndarray, color: str, title: str) -> plt.Figure:
        """繪製單一通道影像的直方圖。"""
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.hist(img_channel.ravel(), bins=256, range=[0, 256], color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xlim([0, 256])
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_rgb_histogram(img_rgb: np.ndarray) -> plt.Figure:
        """繪製三通道 (RGB) 影像的強度直方圖重疊圖。"""
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ("r", "g", "b")
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, alpha=0.7, label=color.upper())
        ax.set_title("RGB 直方圖")
        ax.set_xlim([0, 256])
        ax.legend()
        fig.tight_layout()
        return fig