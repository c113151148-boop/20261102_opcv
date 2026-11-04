# 核心影像演算法
import math
import cv2
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
        """計算兩張影像之間的 PSNR。"""
        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)
        mse = np.mean((img1_f - img2_f) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(mse))

    @staticmethod
    def calc_histogram(channel: np.ndarray) -> np.ndarray:
        """計算單通道 256 階直方圖資料陣列。"""
        hist, _ = np.histogram(channel, bins=256, range=[0, 256])
        return hist

    @staticmethod
    def split_rgb_channels(img_rgb: np.ndarray) -> dict:
        """拆解 RGB 影像，回傳用於計算直方圖的原始通道與用於 UI 渲染的單色 3 通道影像。"""
        result = {}
        names = ["R", "G", "B"]
        for idx, name in enumerate(names):
            ch = img_rgb[:, :, idx]
            colored_img = np.zeros_like(img_rgb)
            colored_img[:, :, idx] = ch
            result[name] = {"channel": ch, "image": colored_img}
        return result

    @staticmethod
    def split_yuv_channels(img_rgb: np.ndarray) -> dict:
        """將 RGB 轉為 YUV，回傳原始通道資料與正規化後的 3 通道灰階渲染影像。"""
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        result = {}
        for name, ch in zip(["Y", "U", "V"], [y, u, v]):
            # Y 亮度保持原樣，U、V 色差則作正規化拉伸強化對比度
            vis = ch if name == "Y" else cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX)
            # 轉換為標準 3 通道灰階供前端渲染
            gray_3ch = np.dstack((vis, vis, vis))
            result[name] = {"channel": ch, "image": gray_3ch}
        return result