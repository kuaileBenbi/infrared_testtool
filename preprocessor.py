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
import time
import numpy as np
import cv2
import logging
from numpy.lib.stride_tricks import sliding_window_view

import scipy.signal as signal

# from PIL import Image
import numpy as np

TOL_factor = 0.01
pixel_type = np.uint16
Level = ["off", "light", "medium", "strong"]


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
        start_time = time.time()
        a_map, b_map, ga, gb = nuc_dict["a_map"], nuc_dict["b_map"], nuc_dict["ga"], nuc_dict["gb"]
        result = self.apply_linear_calibration(raw_frame, a_map, b_map, ga, gb, bit_max)
        elapsed_time = time.time() - start_time
        print(f"linear_corr 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def quadrast_corr(self, raw_frame, nuc_dict, bit_max):
        start_time = time.time()
        a2_arr, a1_arr, a0_arr = nuc_dict["a2"], nuc_dict["a1"], nuc_dict["a0"]
        result = self.apply_quadratic_correction(raw_frame, a2_arr, a1_arr, a0_arr, bit_max)
        elapsed_time = time.time() - start_time
        print(f"quadrast_corr 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def dw_nuc(self, raw_frame, nuc_dict, bit_max):
        start_time = time.time()
        gain_map, offset_map, ref = nuc_dict["gain_map"], nuc_dict["offset_map"], nuc_dict["ref"]
        result = self.apply_dw_nuc(raw_frame, gain_map, offset_map, ref, bit_max)
        elapsed_time = time.time() - start_time
        print(f"dw_nuc 处理耗时: {elapsed_time:.4f} 秒")
        return result
    
    def apply_quadratic_correction(
        self,
        raw_arr_float,
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
        print(f"raw image mean: {raw_arr_float.mean()}")
        raw_arr_float = raw_arr_float.astype(np.float32)

        corrected_arr = np.empty_like(raw_arr_float, dtype=np.float32)

        # 原地计算：y = a2 * x^2 + a1 * x + a0
        np.multiply(
            raw_arr_float, raw_arr_float, out=corrected_arr
        )  # corrected_arr = raw_arr^2
        np.multiply(
            corrected_arr, a2_arr, out=corrected_arr
        )  # corrected_arr = a2 * raw_arr^2
        corrected_arr += a1_arr * raw_arr_float  # corrected_arr += a1 * raw_arr
        corrected_arr += a0_arr  # corrected_arr += a0

        corrected_arr = np.clip(corrected_arr, 0, bit_max)

        print(f"corrected image mean: {corrected_arr.mean()}")

        return np.rint(corrected_arr).astype(np.uint16)

    def apply_linear_calibration(
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


    def apply_dw_nuc(self, raw, gain_map, offset_map, ref, bit_max):
        start_time = time.time()
        # print(f"raw image mean: {raw.mean()}, ref: {ref}")
        corrected = gain_map * raw + offset_map
        corrected = np.clip(corrected, 0, ref)
        # print(f"corrected image mean: {corrected.mean()}")
        corrected = (corrected / ref) * bit_max
        # print(f"corrected image mean: {corrected.mean()}")
        result = corrected.astype(np.uint16)
        elapsed_time = time.time() - start_time
        print(f"apply_dw_nuc 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def apply_autogian(self, frame):
        start_time = time.time()
        # 自动亮度调整：使用自动方法如直方图均衡化来自动优化图像的亮度和对比度。
        
        if frame.dtype not in [np.uint8, np.uint16]:
            if frame.max() > 255:
                frame = frame.astype(np.uint16)
            else:
                frame = frame.astype(np.uint8)
                
        if frame.ndim == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(frame)
        
        elapsed_time = time.time() - start_time
        print(f"apply_autogian 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def apply_manualgain(self, frame, control_mode, value):
        start_time = time.time()
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
        result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        elapsed_time = time.time() - start_time
        print(f"apply_manualgain 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def apply_manualcontrast(self, frame, alpha):
        start_time = time.time()
        # alpha：对比度调整因子。0 < alpha < 1 时，对比度会降低；
        # alpha > 1 时，对比度会增加。
        # 默认值为 1，表示不改变对比度
        result = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        elapsed_time = time.time() - start_time
        print(f"apply_manualcontrast 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def apply_flip(self, frame, control_mode):
        start_time = time.time()
        if control_mode == 0:
            result = frame
        elif control_mode == 1:
            result = cv2.flip(frame, 1)
        elif control_mode == 2:
            result = cv2.flip(frame, 0)
        elif control_mode == 3:
            result = cv2.flip(frame, -1)
        else:
            logger.warning("不存在这种翻转方式,返回原图")
            result = frame
        
        elapsed_time = time.time() - start_time
        print(f"apply_flip 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def apply_blind_pixel_detect(self, frame, sigma=3, window_size=33):
        start_time = time.time()
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
        result = blind_mask.astype(np.uint16)
        elapsed_time = time.time() - start_time
        print(f"apply_blind_pixel_detect 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def compensate_with_filter(self, image, blind_mask, ksize: int = 5):
        start_time = time.time()
        print("坏点查表补偿中....")

        k = int(ksize)

        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1

        blurred = cv2.medianBlur(image, k)

        image[blind_mask] = blurred[blind_mask]

        result = image.astype(np.uint16)
        elapsed_time = time.time() - start_time
        print(f"compensate_with_filter 处理耗时: {elapsed_time:.4f} 秒")
        return result


    def stretch_u16(
        self,
        image: np.ndarray,
        max_val: int,
        downsample: int,          # 可选：分位数估计下采样（2/4能提速）
        level: Level = "medium",
        median_ksize: int = 3,                     # 可选：去噪，中值核（<=1 关闭）
    ) -> np.ndarray:
        """
        对 uint16 图像做分位数拉伸 + γ 调整（按 UI 档位选择力度）。
        输入:
            img: HxW 或 HxW x C 的 uint16
            level: "off" | "light" | "medium" | "strong"
        输出:
            与输入同形状的 uint16
        """
        start_time = time.time()

        if level == "off":
            return image

        # —— 档位预设（可按需微调）——
        if level == "light":
            q_low, q_high = 0.5, 99.5
            safe_floor, safe_ceil = 0.0, 1.0
            min_range = 0.03
            noise = 0.0 # 0.0005–0.001
            gamma_mode = "fixed"
            gamma, gamma_bounds, adapt_str = 0.9, (0.5, 1.2), 0.5
        elif level == "medium":
            q_low, q_high = 0.1, 99.9
            safe_floor, safe_ceil = 0.005, 0.99
            min_range = 0.05
            noise = 0.001 # 0.001–0.002
            gamma_mode = "fixed"
            gamma, gamma_bounds, adapt_str = 0.8, (0.5, 1.2), 0.5
        elif level == "strong":
            q_low, q_high = 0.1, 99.5
            safe_floor, safe_ceil = 0.005, 0.99 # 0.01, 0.95
            min_range = 0.15
            noise = 0.003 # 0.002–0.004
            gamma_mode = "adaptive"
            gamma, gamma_bounds, adapt_str = 0.4, (0.4, 1.2), 0.5
        else:
            raise ValueError(f"unknown level: {level}")

        eps = 1e-6

        # —— 统计域（单通道直接用；多通道用第0通道统计，保证三通道统一风格）——
        if image.ndim == 2:
            stat = image
        else:
            stat = image[..., 0]

        stat_src = stat

        if downsample and downsample > 1 and stat_src.ndim == 2:
            h, w = stat_src.shape
            stat_src = cv2.resize(stat_src, (w // downsample, h // downsample), interpolation=cv2.INTER_AREA)

        # 中值滤波（用于统计的预去噪）
        if median_ksize and median_ksize > 1:
            k = median_ksize if (median_ksize % 2 == 1) else (median_ksize + 1)
            # 注意：cv2.medianBlur 需要2D；若用了 ROI 下采样后仍是2D
            if stat_src.ndim == 2:
                stat_src = cv2.medianBlur(stat_src, ksize=k)
                image = cv2.medianBlur(image, ksize=k)

        # —— 分位数估计 + 保护 —— 
        lo = float(np.percentile(stat_src, q_low))
        hi = float(np.percentile(stat_src, q_high))
        safe_min = max(lo, max_val * safe_floor)
        safe_max = min(hi, max_val * safe_ceil)
        valid_range = max(safe_max - safe_min, max_val * min_range, eps)

        # —— 自适应 γ（仅 strong 使用）——
        if gamma_mode == "adaptive":
            # 预览均值（使用统计域）
            prev = (stat_src.astype(np.float32) - safe_min + max_val * noise) / valid_range
            prev = np.clip(prev, 0.0, 1.0)
            avg = float(prev.mean()) if prev.size > 0 else 0.5
            adj_gamma = gamma * (1.0 + adapt_str * (0.5 - avg))
            adj_gamma = float(np.clip(adj_gamma, gamma_bounds[0], gamma_bounds[1]))
        else:
            adj_gamma = gamma
        
        print(f"lo: {lo}, hi: {hi}, safe_min: {safe_min}, safe_max: {safe_max}, valid_range: {valid_range}, adj_gamma: {adj_gamma}")

        # —— 构建 LUT（uint16→uint16），然后套在所有通道 —— 
        x = np.arange(max_val + 1, dtype=np.float32)
        y = (x - safe_min + max_val * noise) / valid_range
        np.clip(y, 0.0, 1.0, out=y)
        y = np.power(y, adj_gamma) * max_val
        lut = np.clip(y, 0.0, float(max_val)).astype(np.uint16)

        # 使用numpy索引操作替代cv2.LUT（因为cv2.LUT不支持16位）
        # 先裁剪图像像素值到LUT的有效范围内，避免索引越界
        image_min, image_max = image.min(), image.max()
        if image_max > max_val:
            print(f"警告：图像像素值超出范围 [{image_min}, {image_max}]，将裁剪到 [0, {max_val}]")
        image_clipped = np.clip(image, 0, max_val)
        
        if image.ndim == 2:
            out = lut[image_clipped]
        else:
            # 对每个通道套同一 LUT，保持风格一致
            out = np.zeros_like(image)
            for c in range(image.shape[2]):
                out[..., c] = lut[image_clipped[..., c]]

        # 计算并输出耗时
        elapsed_time = time.time() - start_time
        print(f"stretch_u16 处理耗时: {elapsed_time:.4f} 秒")

        return out
    
    def process(self, img, gamma=1.0, bit_max=None):
        start_time = time.time()
        
        if bit_max is None:
            return img
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
        print(f"lo: {lo}, hi: {hi}")
        # 矢量化拉伸
        t = self.imadjust_vec(t, lo, hi, 0, bit_max, gamma=gamma)
        np.clip(t, 0, bit_max, out=t)
                
        # 计算并输出耗时
        elapsed_time = time.time() - start_time
        print(f"process 处理耗时: {elapsed_time:.4f} 秒")

        self.scale_to_gamma_alpha(img, bit_max)
        self.scale_midway(img, bit_max)
        
        return t.astype(np.uint16)

    @staticmethod
    def imadjust_vec(x, a, b, c, d, gamma=1.0):
        print(f"a: {a}, b: {b}, c: {c}, d: {d}, gamma: {gamma}")
        y = (np.clip((x - a) / (b - a), 0, 1) ** gamma) * (d - c) + c
        return y

    def scale_to_gamma_alpha(self, img, img_bit, gamma=0.4, min_range_ratio=0.15):
        start_time = time.time()
        t = img.astype(np.float32)
        # t = cv2.medianBlur(t, ksize=3)
        t = cv2.medianBlur(t, ksize=3)

        # 计算更稳定的分位数范围 (扩大采样区间)
        p05, p995 = np.percentile(t, (0.01, 99.9))  # 改用0.5%~99.5%分位数

        # 动态范围保护机制
        # max_val = (2**14 - 1) << 2  # 理论最大值65532
        safe_min = max(p05, img_bit * 0.01)  # 最低不低于最大值的1%
        safe_max = min(p995, img_bit * 0.95)  # 最高不超过最大值的95%
        valid_range = max(safe_max - safe_min, img_bit * min_range_ratio)

        # 带底噪补偿的拉伸
        img_stretch = (t - safe_min + img_bit * 0.005) / valid_range  # 添加0.5%底噪补偿
        img_stretch = np.clip(img_stretch, 0, 1)

        # 自适应gamma调整
        avg_brightness = np.mean(img_stretch)
        adj_gamma = gamma * (1 + 0.5 * (0.5 - avg_brightness))  # 根据亮度微调gamma
        img_gamma = np.power(img_stretch, np.clip(adj_gamma, 0.5, 1.2))

        print(f"p05: {p05}, p995: {p995}, safe_min: {safe_min}, safe_max: {safe_max}, valid_range: {valid_range}, avg_brightness: {avg_brightness}, adj_gamma: {adj_gamma}")

        img_16bit = img_gamma * img_bit
        np.clip(img_16bit, 0, img_bit, out=img_16bit)

        result = img_16bit.astype(np.uint16)
        elapsed_time = time.time() - start_time
        print(f"scale_to_gamma_alpha 处理耗时: {elapsed_time:.4f} 秒")
        return result


    def scale_midway(self, img, img_bit, gamma=0.8, min_range_ratio=0.05):
        start_time = time.time()
        t = img.astype(np.float32)
        t = cv2.medianBlur(t, ksize=3)

        # 使用更宽松的分位数
        p01, p999 = np.percentile(t, (0.1, 99.9))

        # 动态范围保护（比 scale_to_gamma_alpha 宽一些）
        safe_min = max(p01, img_bit * 0.005)  # 允许接近0
        safe_max = min(p999, img_bit * 0.99)  # 允许接近最大值
        valid_range = max(safe_max - safe_min, img_bit * min_range_ratio)

        # 拉伸 + 底噪补偿
        img_stretch = (t - safe_min + img_bit * 0.002) / valid_range
        img_stretch = np.clip(img_stretch, 0, 1)

        # 轻量 gamma 调整（不随亮度强烈变化）
        img_gamma = np.power(img_stretch, gamma)

        print(f"p01: {p01}, p999: {p999}, safe_min: {safe_min}, safe_max: {safe_max}, valid_range: {valid_range}")

        img_16bit = img_gamma * img_bit
        np.clip(img_16bit, 0, img_bit, out=img_16bit)
        result = img_16bit.astype(np.uint16)
        elapsed_time = time.time() - start_time
        print(f"scale_midway 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def apply_denoise(self, image):
        start_time = time.time()
        # 引导滤波降噪
        image = image.astype(np.float32)
        img_medianblur = cv2.medianBlur(image, 3)
        result = cv2.ximgproc.guidedFilter(
            img_medianblur, image, radius=9, eps=0.01, dDepth=-1
        )
        elapsed_time = time.time() - start_time
        print(f"apply_denoise 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def apply_whitebalance(self, image, method="SimpleWB"):
        start_time = time.time()
        """对可见光图像进行白平衡"""
        if method == "GrayworldWB":
            wb = cv2.xphoto.createGrayworldWB()
            result = wb.balanceWhite(image)
        elif method == "SimpleWB":
            wb = cv2.xphoto.createSimpleWB()
            wb.setWBCoeffs(cv2.cvtColor(image, cv2.COLOR_BGR2XYZ))  # 设置白平衡系数
            result = wb.balanceWhite(image)
        else:
            result = image
        
        elapsed_time = time.time() - start_time
        print(f"apply_whitebalance 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def apply_sharping(self, image):
        start_time = time.time()
        if image.ndim == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            blurred = cv2.GaussianBlur(y, (7, 7), 1.5)
            y_sharp = cv2.addWeighted(y, 1.5, blurred, -0.5, 0)
            sharpened = cv2.merge((y_sharp, u, v))
            result = cv2.cvtColor(sharpened, cv2.COLOR_YUV2BGR)
        else:
            blurred = cv2.GaussianBlur(image, (7, 7), 1.5)
            result = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        elapsed_time = time.time() - start_time
        print(f"apply_sharping 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def draw_center_cross_polylines(self, image, color=(255, 255, 255), thickness=5):
        start_time = time.time()
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
        elapsed_time = time.time() - start_time
        print(f"draw_center_cross_polylines 处理耗时: {elapsed_time:.4f} 秒")
        return image

    def apply_defog(self, image, t_min=0.1, omega=0.95, guided_filter_radius=40, guided_filter_eps=1e-3, in_max=None):
        """
        透雾算法 - 基于暗通道先验的去雾算法
        :param image: 输入图像 (BGR格式或灰度图)
        :param t_min: 透射率的最小值，防止过度去雾
        :param omega: 保留雾气的比例，0-1之间
        :param guided_filter_radius: 引导滤波半径
        :param guided_filter_eps: 引导滤波正则化参数
        :return: 去雾后的图像
        """
        start_time = time.time()
        
        orig_dtype = image.dtype
        is_grayscale = (image.ndim == 2)

        # ---- 统一为3通道参与计算，输出再还原 ----
        if is_grayscale:
            img3 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img3 = image

        img3 = img3.astype(np.float32)

        # ---- 识别/设定输入满量程 ----
        def _auto_in_max(arr_u16):
            vmax = int(arr_u16.max())
            # 常见有效位宽就近上限（保留“头部空间”，避免把没打满的图压得过亮）
            if vmax <= 4095:   return 4095.0   # 12-bit
            if vmax <= 16383:  return 16383.0  # 14-bit
            return 65535.0                     # 16-bit（默认）
        if in_max is None:
            if orig_dtype == np.uint16:
                in_scale = _auto_in_max(image)
            elif orig_dtype == np.uint8:
                in_scale = 255.0
            else:
                # float 输入：按当前最大值近似为满量程，至少防零
                cur_max = float(np.max(img3)) if img3.size else 1.0
                in_scale = max(cur_max, 1.0)
        else:
            in_scale = float(in_max)

        # ---- 归一化到[0,1] ----
        img = np.clip(img3 / max(in_scale, 1.0), 0.0, 1.0)

        # ---- 内部函数 ----
        def get_dark_channel(im, patch_size=15):
            k = max(3, int(patch_size) | 1)  # 奇数核
            min_channel = np.min(im, axis=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            return cv2.erode(min_channel, kernel)

        def estimate_atmospheric_light(im, dark, top_percent=0.001):
            h, w = im.shape[:2]
            n = max(1, int(h * w * top_percent))
            flat_dark = dark.reshape(-1)
            idx = np.argpartition(flat_dark, -n)[-n:]
            cand = im.reshape(-1, 3)[idx]
            A = cand[np.argmax(np.linalg.norm(cand, axis=1))]
            return np.maximum(A, 1e-3)

        def estimate_transmission(im, A, omega=0.95):
            normalized = im / (A[None, None, :] + 1e-6)
            dark_norm = get_dark_channel(normalized)
            return 1.0 - omega * dark_norm

        def guided_filter(guide_rgb, src, radius, eps):
            # guide/src 已在[0,1]，不再 /255
            guide = guide_rgb
            if guide.ndim == 3:
                guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
            guide = guide.astype(np.float32)
            src = src.astype(np.float32)

            k = int(radius)
            k = (2*k + 1, 2*k + 1)  # 以“半径”定义窗口
            mean_g  = cv2.boxFilter(guide, -1, k, normalize=True)
            mean_s  = cv2.boxFilter(src,   -1, k, normalize=True)
            mean_gs = cv2.boxFilter(guide * src, -1, k, normalize=True)

            cov_gs = mean_gs - mean_g * mean_s
            var_g  = cv2.boxFilter(guide * guide, -1, k, normalize=True) - mean_g * mean_g

            a = cov_gs / (var_g + eps)
            b = mean_s - a * mean_g

            mean_a = cv2.boxFilter(a, -1, k, normalize=True)
            mean_b = cv2.boxFilter(b, -1, k, normalize=True)
            return mean_a * guide + mean_b

        def recover_image(im, t, A, t_floor=0.1):
            t = np.clip(np.nan_to_num(t, nan=1.0, posinf=1.0, neginf=1.0), t_floor, 0.999)
            J = (im - A[None, None, :]) / t[..., None] + A[None, None, :]
            return J

        try:
            dark = get_dark_channel(img)
            A = estimate_atmospheric_light(img, dark, top_percent=0.001)
            t = estimate_transmission(img, A, omega=omega)
            t = guided_filter(img, t, guided_filter_radius, guided_filter_eps)
            J = recover_image(img, t, A, t_floor=t_min)
            J = np.clip(J, 0.0, 1.0)

            # ---- 还原到原始量程 & dtype ----
            if orig_dtype == np.uint16:
                out3 = (J * in_scale + 0.5).astype(np.uint16)
            elif orig_dtype == np.uint8:
                out3 = (J * 255.0 + 0.5).astype(np.uint8)
            else:
                # float：维持 float32，同量程（0~in_scale）方便后续处理
                out3 = (J * in_scale).astype(np.float32)

            out = cv2.cvtColor(out3, cv2.COLOR_BGR2GRAY) if is_grayscale else out3

            print(f"apply_defog 耗时: {time.time() - start_time:.4f}s, dtype={orig_dtype}, in_scale={in_scale}")
            return out

        except Exception as e:
            print(f"透雾算法失败: {e}")
            print(f"apply_defog 耗时: {time.time() - start_time:.4f}s (fallback)")
            return image  # 保持原始dtype直接返回
