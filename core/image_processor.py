import math
import cv2
import numpy as np

class ImageProcessor:
    """影像處理工具類別"""

    @staticmethod
    def apply_convolution(img_rgb, kernel, normalize=False):
        """對輸入影像進行卷積運算

        Args:
            img_rgb (ndarray): 原始 RGB 影像
            kernel (ndarray): 卷積核
            normalize (bool): 是否對卷積核做正規化

        Returns:
            np.ndarray: 卷積運算後的新影像
        """
        # 卷積核正規化
        if normalize:
            k_sum = np.sum(kernel)
            if k_sum != 0:
                kernel = kernel / k_sum

        # 執行卷積運算
        filtered_img = cv2.filter2D(img_rgb, -1, kernel)

        return filtered_img

    @staticmethod
    def calculate_psnr(img1, img2):
        """計算兩張影像間的 PSNR
        
        Args:
            img1 (np.ndarray): 影像 1
            img2 (np.ndarray): 影像 2

        Returns:
            float: PSNR 值
        """
        # 轉換成 float64 防止溢位
        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)

        # 計算 MSE
        mse = np.mean((img1_f - img2_f) ** 2)
        if mse < 1e-10:
            return float("inf")

        # 計算 psnr
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
        
        return psnr

    @staticmethod
    def calc_histogram(channel):
        """計算單通道的 256 階直方圖數據

        Args:
            channel (np.ndarray): 單通道像素矩陣
        
        Returns:
            np.ndarray: 該通道的直方圖數據 (每個像素強度的像素數量)
        """
        # 計算每個像素強度 (0 - 255) 的像素數量
        hist, _ = np.histogram(channel, bins=256, range=[0, 256])

        return hist

    @staticmethod
    def split_rgb_channels(img_rgb):
        """將 RGB 影像拆成 R、G、B 三個通道，並取出每個通道的像素矩陣與處理視覺化圖片
        
        Args:
            img_rgb (np.ndarray): RGB 影像
        
        Returns:
            dict: 三個通道各自的像素矩陣與視覺化圖片
        """
        result = {}
        names = ["R", "G", "B"]
        for idx, name in enumerate(names):
            # 取出單一顏色通道
            ch = img_rgb[:, :, idx]

            # 處理視覺化圖片 (變純色圖片而不是灰階，比較好看 :))
            colored_img = np.zeros_like(img_rgb)
            colored_img[:, :, idx] = ch

            # 放入字典
            result[name] = {"channel": ch, "image": colored_img}

        return result

    @staticmethod
    def split_yuv_channels(img_rgb):
        """將 RGB 影像轉成 YUV 並拆成 Y、U、V 三個通道，並取出每個通道的像素矩陣與處理視覺化圖片

        Args:
            img_rgb (np.ndarray): RGB 影像

        Returns:
            dict: 三個通道各自的像素矩陣與視覺化圖片
        """
        # RGB 轉 YUV
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)

        result = {}
        names = ["Y", "U", "V"]
        for idx, name in enumerate(names):
            # 取出單一顏色通道
            ch = img_yuv[:, :, idx]

            # 拉伸 U、V 增強畫面對比方便觀察
            if name == "Y":
                vis = ch 
            else:
                vis = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX)

            # 轉成 3 維當作視覺化圖片 (三個通道都放一樣的值)，防止後續 UI 顯示時發生格式錯誤
            gray_3ch = np.dstack((vis, vis, vis))

            # 放入字典
            result[name] = {"channel": ch, "image": gray_3ch}

        return result