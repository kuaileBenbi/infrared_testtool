"""
1.红外图像
非均匀线性校正
盲元检测与补偿
直方图拉伸
引导滤波
2.可见光图像
白平衡校正
锐化
引导滤波
"""

import os
import numpy as np
import cv2
import logging
from numpy.lib.stride_tricks import sliding_window_view

import scipy.signal as signal

# from PIL import Image
import numpy as np

TOL_factor = 0.01
pixel_type = np.uint16
pixel_max = 16383


logger = logging.getLogger(__name__)
current_dir = os.path.dirname(__file__)


class ImagePreprocessor:
    def __init__(self):
        self.bd_para = None
        self.nuc_para = None
        self.blind_mask = None

    def load_config(self, npz_path):
        try:
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"文件 {npz_path} 不存在。")
            with np.load(npz_path) as data:
                self.nuc_para = data["nuc"]
                self.bd_para = data["bd"]
        except Exception as e:
            raise ValueError(f"ImagePreprocessor load failed: {e}")

    def linear_corr(self, raw_frame, nuc_dict, bit_max):
        a_map, b_map, ga, gb = nuc_dict["a_map"], nuc_dict["b_map"], nuc_dict["ga"], nuc_dict["gb"]
        return self.apply_multi_calibration(raw_frame, a_map, b_map, ga, gb, bit_max)

    def quadrast_corr(self, raw_frame, nuc_dict, bit_max):
        a2_arr, a1_arr, a0_arr = nuc_dict["a2"], nuc_dict["a1"], nuc_dict["a0"]
        return self.apply_quadratic_correction(raw_frame, a2_arr, a1_arr, a0_arr, bit_max)
    
    def apply_quadratic_correction(
        self,
        image,
        a2_arr,
        a1_arr,
        a0_arr,
        out_max=255,
        bit_max=16383
    ):
        """
        使用二阶拟合的 NUC 校正参数对图像进行校正。

        Args:
            image: 输入的图像，(H, W) 的 numpy 数组
            a2_arr: 二阶系数数组，(H, W)
            a1_arr: 一阶系数数组，(H, W)
            a0_arr: 常数项数组，(H, W)
            out_max: 输出范围最大值，255 表示输出为 8bit

        Returns:
            corrected: 校正后的图像，(H, W) 的 numpy 数组

        Note:
            1. 输入图像应为 uint16 类型
            2. 输出图像为 uint16 类型
        """
        print(f"raw image mean: {image.mean()}")
        image = image.astype(np.float32)
        corrected = a2_arr * image**2 + a1_arr * image + a0_arr

        corrected = np.clip(corrected, 0, bit_max)
        print(f"corrected image mean: {corrected.mean()}")
        # corrected = (corrected / bit_max * out_max).astype(np.uint8)
        return np.rint(corrected).astype(np.uint16)

    def apply_multi_calibration(
        self,
        raw: np.ndarray,
        a_map: np.ndarray,
        b_map: np.ndarray,
        global_a: float = None,
        global_b: float = None,
        bit_max: int = 16383,  # 目标位宽上限（默认映射到12位）
        eps_gain: float = 1e-3,  # 与标定端一致，避免极小增益导致数值爆
    ) -> np.ndarray:
        """
        使用多点校正参数对单张原始图进行校正，并映射到 bit_max。

        y = (raw - b) / a          （像素级去非均匀）
        y = y * global_a + global_b（可选：回到“全局参考”亮度）

        Args:
            raw:      (H, W)
            a_map:    (H, W)
            b_map:    (H, W)
            global_a/global_b: 可选的全局线性再映射（保持与训练自变量的整体一致性）
            bit_max:  最终裁剪上限（默认 16383）
            eps_gain: 增益下限

        Returns:
            uint16 校正结果
        """
        raw_f = raw.astype(np.float32)

        # 增益防护
        a_safe = np.where(
            np.abs(a_map) < eps_gain, np.sign(a_map) * eps_gain, a_map
        ).astype(np.float32)

        corr = (raw_f - b_map.astype(np.float32)) / a_safe

        if global_a is not None and global_b is not None:
            corr = corr * float(global_a) + float(global_b)

        corr = np.clip(corr, 0, float(bit_max))
        return np.rint(corr).astype(np.uint16)

    def apply_autogian(self, frame):
        # 自动亮度调整：使用自动方法如直方图均衡化来自动优化图像的亮度和对比度。
        if frame.ndim == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(frame)

    def apply_manualgain(self, frame, control_mode, value):
        if frame.ndim == 2:
            logger.warning("不对灰度图做变换")
            return frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if control_mode == 1:
            v = cv2.add(v, np.full(v.shape, value, dtype=v.dtype))
        elif control_mode == 2:
            v = cv2.subtract(v, np.full(v.shape, value, dtype=v.dtype))
        else:
            logger.warning("手动调整亮度方向不存在，返回原图像")
            return frame

        v = np.clip(v, 0, 255).astype(hsv.dtype)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def apply_manualcontrast(self, frame, alpha):
        # alpha：对比度调整因子。0 < alpha < 1 时，对比度会降低；
        # alpha > 1 时，对比度会增加。
        # 默认值为 1，表示不改变对比度
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    def apply_flip(self, frame, control_mode):
        if control_mode == 0:
            return frame
        elif control_mode == 1:
            return cv2.flip(frame, 1)
        elif control_mode == 2:
            return cv2.flip(frame, 0)
        elif control_mode == 3:
            return cv2.flip(frame, -1)
        else:
            logger.warning("不存在这种翻转方式,返回原图")
            return frame

    def apply_blind_pixel_detect(self, frame, sigma=3, window_size=33):
        """
        blind_mask 的值为 True（或 1）：表示该位置的像素是盲元。
        """
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rows, cols = frame.shape
        half = window_size // 2

        # 使用滑动窗口视图避免逐像素循环
        window_view = sliding_window_view(frame, (window_size, window_size))

        # 计算每个窗口的均值和标准差
        window_flat = window_view.reshape(-1, window_size * window_size)
        means = np.mean(window_flat, axis=1).reshape(rows - 2 * half, cols - 2 * half)
        stds = np.std(window_flat, axis=1).reshape(rows - 2 * half, cols - 2 * half)

        # 提取中心像素值
        centers = frame[half:-half, half:-half]

        # 生成盲元掩码
        mask = np.abs(centers - means) > sigma * stds
        blind_mask = np.zeros((rows, cols), dtype=np.bool_)
        blind_mask[half:-half, half:-half] = mask

        self.blind_mask = blind_mask
        return blind_mask.astype(np.uint16)

    def compensate_with_filter(self, image, blind_mask, ksize: int = 5):
        print("坏点查表补偿中....")

        k = int(ksize)

        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1

        blurred = cv2.medianBlur(image, k)

        image[blind_mask] = blurred[blind_mask]

        return image.astype(np.uint16)

    def pixel_nuc(self, test_img, coef_map):
        B = coef_map["bk"]
        G = coef_map["poly"]
        ref = coef_map["ref"]
        corrected = (test_img - B) * G
        corrected = np.clip(corrected, 0, ref)
        corrected = (corrected / ref) * 4095
        return corrected.astype(np.uint16)

    def dw_nuc(self, raw, coef, bit_max):
        gain_map = coef["gain_map"]
        offset_map = coef["offset_map"]
        ref = coef["ref"]

        corrected = gain_map * raw + offset_map
        corrected = np.clip(corrected, 0, ref)
        corrected = (corrected / ref) * bit_max
        return corrected.astype(np.uint16)

    def apply_non_uniform_correction(self, test_img, coef_map):
        """应用非均匀校正"""
        b_map, a_map, global_a, global_b = (
            coef_map["b_map"],
            coef_map["a_map"],
            coef_map["ga"],
            coef_map["gb"],
        )
        raw = test_img.astype(np.float32)
        corr = (raw - b_map) / a_map
        if global_a is not None and global_b is not None:
            corr = corr * global_a + global_b
        
        corr = np.clip(corr, 0, 4095)

        return corr.astype(np.uint16)

    def imadjust(self, x, a, b, c, d, gamma=1):
        y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
        return y

    def process(self, img, gamma=1.0):
        t = img.astype(np.float32)
        t = cv2.medianBlur(t, ksize=3)
        # 角点修正
        t[0, 0] = t[0, 1]
        t[0, -1] = t[0, -2]
        t[-1, 0] = t[-2, 0]
        t[-1, -1] = t[-1, -2]
        # 阈值
        lo = np.percentile(t, 100 * TOL_factor)
        hi = np.percentile(t, 100 * (1 - TOL_factor))
        # 矢量化拉伸
        t = self.imadjust_vec(t, lo, hi, 0, 4095, gamma=gamma)
        np.clip(t, 0, 4095, out=t)
        return t.astype(np.uint16)

    @staticmethod
    def imadjust_vec(x, a, b, c, d, gamma=1.0):
        y = (np.clip((x - a) / (b - a), 0, 1) ** gamma) * (d - c) + c
        return y

    def apply_denoise(self, image):
        # 引导滤波降噪
        img_medianblur = cv2.medianBlur(image, 3)
        return cv2.ximgproc.guidedFilter(
            img_medianblur, image, radius=9, eps=0.01, dDepth=-1
        )

    def apply_whitebalance(self, image, method="SimpleWB"):
        """对可见光图像进行白平衡"""
        if method == "GrayworldWB":
            wb = cv2.xphoto.createGrayworldWB()
            return wb.balanceWhite(image)
        elif method == "SimpleWB":
            wb = cv2.xphoto.createSimpleWB()
            wb.setWBCoeffs(cv2.cvtColor(image, cv2.COLOR_BGR2XYZ))  # 设置白平衡系数
            return wb.balanceWhite(image)

    def apply_sharping(self, image):
        if image.ndim == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            blurred = cv2.GaussianBlur(y, (7, 7), 1.5)
            y_sharp = cv2.addWeighted(y, 1.5, blurred, -0.5, 0)
            sharpened = cv2.merge((y_sharp, u, v))
            return cv2.cvtColor(sharpened, cv2.COLOR_YUV2BGR)
        else:
            blurred = cv2.GaussianBlur(image, (7, 7), 1.5)
            return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    def draw_center_cross_polylines(self, image, color=(255, 255, 255), thickness=5):
        """
        使用polylines高效绘制十字线
        :param cross_radius: 十字线半径（从中心到端点的距离）
        """
        height, width = image.shape[:2]
        center_x = width // 2
        center_y = height // 2

        cross_radius = int(width * 0.1)
        # print(cross_radius)

        # 定义横线和竖线的顶点坐标
        horizontal_line = np.array(
            [
                [[center_x - cross_radius, center_y]],
                [[center_x + cross_radius, center_y]],
            ],
            dtype=np.int32,
        )

        vertical_line = np.array(
            [
                [[center_x, center_y - cross_radius]],
                [[center_x, center_y + cross_radius]],
            ],
            dtype=np.int32,
        )

        # 绘制所有线段
        cv2.polylines(
            image,
            [horizontal_line, vertical_line],
            isClosed=False,
            color=color,
            thickness=thickness,
        )
        return image
