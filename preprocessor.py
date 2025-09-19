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
from typing import Optional, Dict, Any

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
        
        # 缓存的校正参数，避免每次调用时重复提取
        self._linear_params = None
        self._quadratic_params = None
        self._dw_nuc_params = None

    def load_config(self, npz_path):
        try:
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"文件 {npz_path} 不存在。")
            with np.load(npz_path) as data:
                self.nuc_para = data["nuc"]
                self.bd_para = data["bd"]
                # 加载配置后立即提取和缓存校正参数
                self._extract_correction_params()
        except Exception as e:
            raise ValueError(f"ImagePreprocessor load failed: {e}")
    
    def _extract_correction_params(self):
        """提取并缓存校正参数，避免每次调用时重复提取"""
        if self.nuc_para is None:
            return
            
        try:
            # 提取线性校正参数
            if all(key in self.nuc_para for key in ["a_map", "b_map", "ga", "gb"]):
                self._linear_params = {
                    "a_map": self.nuc_para["a_map"],
                    "b_map": self.nuc_para["b_map"], 
                    "ga": self.nuc_para["ga"],
                    "gb": self.nuc_para["gb"]
                }
            
            # 提取二次校正参数
            if all(key in self.nuc_para for key in ["a2", "a1", "a0"]):
                self._quadratic_params = {
                    "a2": self.nuc_para["a2"],
                    "a1": self.nuc_para["a1"],
                    "a0": self.nuc_para["a0"]
                }
            
            # 提取明暗校正参数
            if all(key in self.nuc_para for key in ["gain_map", "offset_map", "ref"]):
                self._dw_nuc_params = {
                    "gain_map": self.nuc_para["gain_map"],
                    "offset_map": self.nuc_para["offset_map"],
                    "ref": self.nuc_para["ref"]
                }
                
        except Exception as e:
            logger.warning(f"提取校正参数时出错: {e}")
            # 如果提取失败，清空缓存
            self._linear_params = None
            self._quadratic_params = None
            self._dw_nuc_params = None
    
    def clear_correction_params_cache(self):
        """清除校正参数缓存，强制下次调用时重新提取"""
        self._linear_params = None
        self._quadratic_params = None
        self._dw_nuc_params = None
        logger.info("校正参数缓存已清除")

    def linear_corr(self, raw_frame, nuc_dict, bit_max):
        start_time = time.time()
        
        # 使用缓存的参数，如果缓存为空则从nuc_dict中提取并缓存
        if self._linear_params is None:
            if nuc_dict is not None and all(key in nuc_dict for key in ["a_map", "b_map", "ga", "gb"]):
                self._linear_params = {
                    "a_map": nuc_dict["a_map"],
                    "b_map": nuc_dict["b_map"], 
                    "ga": nuc_dict["ga"],
                    "gb": nuc_dict["gb"]
                }
            else:
                raise ValueError("线性校正参数不可用")
        
        a_map = self._linear_params["a_map"]
        b_map = self._linear_params["b_map"]
        ga = self._linear_params["ga"]
        gb = self._linear_params["gb"]
        
        result = self.apply_linear_calibration(raw_frame, a_map, b_map, ga, gb, bit_max)
        elapsed_time = time.time() - start_time
        print(f"linear_corr 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def quadrast_corr(self, raw_frame, nuc_dict, bit_max):
        start_time = time.time()
        
        # 使用缓存的参数，如果缓存为空则从nuc_dict中提取并缓存
        if self._quadratic_params is None:
            if nuc_dict is not None and all(key in nuc_dict for key in ["a2", "a1", "a0"]):
                self._quadratic_params = {
                    "a2": nuc_dict["a2"],
                    "a1": nuc_dict["a1"],
                    "a0": nuc_dict["a0"]
                }
            else:
                raise ValueError("二次校正参数不可用")
        
        a2_arr = self._quadratic_params["a2"]
        a1_arr = self._quadratic_params["a1"]
        a0_arr = self._quadratic_params["a0"]
        
        result = self.apply_quadratic_correction(raw_frame, a2_arr, a1_arr, a0_arr, bit_max)
        elapsed_time = time.time() - start_time
        print(f"quadrast_corr 处理耗时: {elapsed_time:.4f} 秒")
        return result

    def dw_nuc(self, raw_frame, nuc_dict, bit_max):
        start_time = time.time()
        
        # 使用缓存的参数，如果缓存为空则从nuc_dict中提取并缓存
        if self._dw_nuc_params is None:
            if nuc_dict is not None and all(key in nuc_dict for key in ["gain_map", "offset_map", "ref"]):
                self._dw_nuc_params = {
                    "gain_map": nuc_dict["gain_map"],
                    "offset_map": nuc_dict["offset_map"],
                    "ref": nuc_dict["ref"]
                }
            else:
                raise ValueError("明暗校正参数不可用")
        
        gain_map = self._dw_nuc_params["gain_map"]
        offset_map = self._dw_nuc_params["offset_map"]
        ref = self._dw_nuc_params["ref"]
        
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
        t1 = time.time()
        raw_f = raw.astype(np.float32)

        # 增益防护
        a_safe = np.where(
            np.abs(a_map) < eps_gain, np.sign(a_map) * eps_gain, a_map
        ).astype(np.float32)

        corr = (raw_f - b_map.astype(np.float32)) / a_safe

        if global_a is not None and global_b is not None:
            corr = corr * float(global_a) + float(global_b)

        corr = np.clip(corr, 0, float(bit_max))
        t2 = time.time()
        print(f"apply_linear_calibration 处理耗时: {t2 - t1:.4f} 秒")
        return np.rint(corr).astype(np.uint16)


    def apply_dw_nuc(self, raw, gain_map, offset_map, ref, bit_max):
        start_time = time.time()
        # print(f"raw image mean: {raw.mean()}, ref: {ref}")
        raw = raw.astype(np.float32)
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
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
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


    def stretch_u16_adaptive(
        self,
        image: np.ndarray,
        max_val: int,
        downsample: int = 2,  # 分位统计下采样
        median_ksize: int = 3,  # 统计前中值去噪；<=1 关闭
        print_debug: bool = False,
        video_state: Optional[Dict[str, Any]] = None,  # 跨帧稳态上下文：{} / None
        config: Optional[Dict[str, Any]] = None,  # 覆盖默认稳态参数
    ) -> np.ndarray:
        """
        分段自适应 uint16 拉伸（稳健动态范围分段 + 自适应γ）。
        - 单帧模式: video_state=None
        - 视频模式: 提供一个可变的 video_state=dict即可自动启用防闪烁稳

        返回：与输入同形状的 uint16
        """
        logger = logging.getLogger(__name__)
        t0 = time.time()
        eps = 1e-6

        if image.dtype != np.uint16:
            raise TypeError("image 必须为 uint16")

        # ---------- 工具函数（内部） ----------
        def _hist_u16(img_u16: np.ndarray, _max: int, bins: int) -> np.ndarray:
            h, _ = np.histogram(img_u16, bins=bins, range=(0, _max + 1))
            return h.astype(np.float32)

        def _pct_from_hist(H: np.ndarray, q: float, _max: int, bins: int) -> float:
            # q ∈ [0,100]
            cdf = np.cumsum(H, dtype=np.float64)
            s = cdf[-1] + 1e-12
            p = q / 100.0
            idx = int(np.searchsorted(cdf, p * s))
            if idx <= 0:
                return 0.0
            if idx >= bins:
                return float(_max)
            bin_w = (_max + 1) / bins
            return idx * bin_w

        def _hist_bhat_dist(
            H1: Optional[np.ndarray], H2: Optional[np.ndarray]
        ) -> float:
            if H1 is None or H2 is None:
                return 0.0
            h1 = H1 / (H1.sum() + 1e-6)
            h2 = H2 / (H2.sum() + 1e-6)
            bc = float(np.sum(np.sqrt(h1 * h2)))
            return float(np.sqrt(max(1e-12, 1.0 - bc)))

        def _decide_segment(
            R: float, prev_seg: Optional[str], upf: float, dnf: float
        ) -> str:
            if prev_seg is None:
                if R < 256:
                    return "tiny"
                if R < 512:
                    return "small"
                if R < 1024:
                    return "mid"
                if R < 2048:
                    return "large"
                return "huge"
            s = prev_seg
            if s == "tiny":
                return "small" if R >= 256 * upf else "tiny"
            if s == "small":
                if R >= 512 * upf:
                    return "mid"
                if R < 256 * dnf:
                    return "tiny"
                return "small"
            if s == "mid":
                if R >= 1024 * upf:
                    return "large"
                if R < 512 * dnf:
                    return "small"
                return "mid"
            if s == "large":
                if R >= 2048 * upf:
                    return "huge"
                if R < 1024 * dnf:
                    return "mid"
                return "large"
            if s == "huge":
                return "large" if R < 2048 * dnf else "huge"

        def _limit_step(
            prev: Optional[float], cur: float, d_abs: float, d_rel: float
        ) -> float:
            if prev is None:
                return float(cur)
            max_step = max(d_abs, abs(prev) * d_rel)
            return float(np.clip(cur, prev - max_step, prev + max_step))

        def _ema(prev: Optional[float], cur: float, a: float) -> float:
            return float(cur) if prev is None else float((1 - a) * prev + a * cur)

        # ---------- 分段参数表（与你原来一致） ----------
        params_table = {
            "tiny": dict(
                q_low=5.0,
                q_high=99.8,
                safe_floor=0.010,
                safe_ceil=0.990,
                min_range=0.020,
                noise=0.0018,
                gamma_mode="adaptive",
                base_gamma=0.60,
                gamma_bounds=(0.40, 1.00),
                adapt_str=0.60,
            ),
            "small": dict(
                q_low=1.0,
                q_high=99.8,
                safe_floor=0.008,
                safe_ceil=0.990,
                min_range=0.030,
                noise=0.0015,
                gamma_mode="adaptive",
                base_gamma=0.70,
                gamma_bounds=(0.50, 1.10),
                adapt_str=0.55,
            ),
            "mid": dict(
                q_low=0.5,
                q_high=99.5,
                safe_floor=0.005,
                safe_ceil=0.990,
                min_range=0.050,
                noise=0.0010,
                gamma_mode="adaptive",
                base_gamma=0.80,
                gamma_bounds=(0.60, 1.20),
                adapt_str=0.50,
            ),
            "large": dict(
                q_low=0.2,
                q_high=99.8,
                safe_floor=0.003,
                safe_ceil=0.995,
                min_range=0.080,
                noise=0.0008,
                gamma_mode="adaptive",
                base_gamma=0.95,
                gamma_bounds=(0.70, 1.30),
                adapt_str=0.40,
            ),
            "huge": dict(
                q_low=0.1,
                q_high=99.9,
                safe_floor=0.000,
                safe_ceil=1.000,
                min_range=0.120,
                noise=0.0005,
                gamma_mode="adaptive",
                base_gamma=1.05,
                gamma_bounds=(0.80, 1.40),
                adapt_str=0.30,
            ),
        }

        # ---------- 统计域（与你原代码一致） ----------
        if image.ndim == 2:
            stat = image
        else:
            stat = image[..., 0]

        stat_src = stat
        if downsample and downsample > 1 and stat_src.ndim == 2:
            h, w = stat_src.shape
            stat_src = cv2.resize(
                stat_src,
                (max(1, w // downsample), max(1, h // downsample)),
                interpolation=cv2.INTER_AREA,
            )

        if median_ksize and median_ksize > 1:
            k = median_ksize if (median_ksize % 2 == 1) else (median_ksize + 1)
            if stat_src.ndim == 2:
                stat_src = cv2.medianBlur(stat_src, ksize=k)
                # 若仅想稳统计而不动原图，可注释下一行
                image = cv2.medianBlur(image, ksize=k)

        # =============== 分两种路径：单帧 vs. 视频稳态 ===============
        if video_state is None:
            # -------- 单帧模式：与你原来基本一致（直接 per-frame 分位） --------
            rb_lo = float(np.percentile(stat_src, 0.1))
            rb_hi = float(np.percentile(stat_src, 99.9))
            rb_range = max(rb_hi - rb_lo, eps)

            if rb_range < 256:
                seg = "tiny"
            elif rb_range < 512:
                seg = "small"
            elif rb_range < 1024:
                seg = "mid"
            elif rb_range < 2048:
                seg = "large"
            else:
                seg = "huge"

            P = params_table[seg]
            lo = float(np.percentile(stat_src, P["q_low"]))
            hi = float(np.percentile(stat_src, P["q_high"]))

            safe_min = max(lo, max_val * P["safe_floor"])
            safe_max = min(hi, max_val * P["safe_ceil"])
            valid_range = max(safe_max - safe_min, max_val * P["min_range"], eps)

            if P["gamma_mode"] == "adaptive":
                prev = (
                    stat_src.astype(np.float32) - safe_min + max_val * P["noise"]
                ) / valid_range
                prev = np.clip(prev, 0.0, 1.0)
                avg = float(prev.mean()) if prev.size > 0 else 0.5
                adj_gamma = P["base_gamma"] * (1.0 + P["adapt_str"] * (0.5 - avg))
                adj_gamma = float(
                    np.clip(adj_gamma, P["gamma_bounds"][0], P["gamma_bounds"][1])
                )
            else:
                adj_gamma = P["base_gamma"]

            # 构建 LUT 并应用
            x = np.arange(max_val + 1, dtype=np.float32)
            y = (x - safe_min + max_val * P["noise"]) / valid_range
            np.clip(y, 0.0, 1.0, out=y)
            y = np.power(y, adj_gamma) * max_val
            lut = np.clip(y, 0.0, float(max_val)).astype(np.uint16)

            img_min, img_max = image.min(), image.max()
            if img_max > max_val and print_debug:
                logger.warning(
                    f"警告：图像像素值超出范围 [{img_min}, {img_max}]，将裁剪到 [0, {max_val}]"
                )

            img_clip = np.clip(image, 0, max_val)
            if img_clip.ndim == 2:
                out = lut[img_clip]
            else:
                out = np.empty_like(img_clip)
                for c in range(img_clip.shape[2]):
                    out[..., c] = lut[img_clip[..., c]]

            if print_debug:
                logger.info(
                    f"[single] seg={seg} R={rb_range:.1f} lo/hi={lo:.1f}/{hi:.1f} "
                    f"smin/smax={safe_min:.1f}/{safe_max:.1f} V={valid_range:.1f} gamma={adj_gamma:.3f} "
                    f"time={(time.time()-t0):.4f}s"
                )
            return out

        # ================== 视频稳态路径（防闪烁） ==================
        # —— 默认稳态配置，可被 config 覆盖 ——
        C = dict(
            bins=1024,
            ds=4,
            rho_hist=0.25,
            alpha_param=0.15,
            beta_lut=0.20,
            delta_abs=6.0,
            delta_rel=0.10,
            hys_up=1.10,
            hys_dn=0.90,
            scene_jump_th=0.30,
        )
        if config:
            C.update(config)

        # 初始化 video_state 的键
        H_ema = video_state.get("hist_ema")
        prev_lut = video_state.get("prev_lut")
        prev_params = video_state.get("prev_params")  # (smin, smax, gamma)
        prev_seg = video_state.get("segment")
        prev_R = video_state.get("prev_R")

        # 统计直方图用的图：再次轻模糊 + 额外下采样
        stat_hist = stat_src
        if C["ds"] and C["ds"] > 1:
            hh, ww = stat_hist.shape
            stat_hist = cv2.resize(
                stat_hist,
                (max(1, ww // C["ds"]), max(1, hh // C["ds"])),
                interpolation=cv2.INTER_AREA,
            )
        stat_hist = cv2.GaussianBlur(stat_hist, (5, 5), 0)

        H_now = _hist_u16(stat_hist, max_val, bins=C["bins"])
        dist = _hist_bhat_dist(H_ema, H_now)
        if H_ema is None:
            H_ema = H_now.copy()
        else:
            H_ema = (1 - C["rho_hist"]) * H_ema + C["rho_hist"] * H_now

        # 用 EMA 直方图取非常鲁棒的 R（0.1~99.9）
        rb_lo = _pct_from_hist(H_ema, 0.1, max_val, C["bins"])
        rb_hi = _pct_from_hist(H_ema, 99.9, max_val, C["bins"])
        rb_range = max(rb_hi - rb_lo, eps)

        # 分段（带滞回）
        seg = _decide_segment(rb_range, prev_seg, C["hys_up"], C["hys_dn"])
        P = params_table[seg]

        # 用 EMA 直方图取该分段下的 lo/hi
        lo = _pct_from_hist(H_ema, P["q_low"], max_val, C["bins"])
        hi = _pct_from_hist(H_ema, P["q_high"], max_val, C["bins"])

        smin_raw = max(lo, max_val * P["safe_floor"])
        smax_raw = min(hi, max_val * P["safe_ceil"])
        V_raw = max(smax_raw - smin_raw, max_val * P["min_range"], eps)

        # 自适应 γ（先算“原始目标γ”）
        if P["gamma_mode"] == "adaptive":
            prev = (
                stat_src.astype(np.float32) - smin_raw + max_val * P["noise"]
            ) / V_raw
            prev = np.clip(prev, 0.0, 1.0)
            avg = float(prev.mean()) if prev.size > 0 else 0.5
            gamma_raw = P["base_gamma"] * (1.0 + P["adapt_str"] * (0.5 - avg))
            gamma_raw = float(
                np.clip(gamma_raw, P["gamma_bounds"][0], P["gamma_bounds"][1])
            )
        else:
            gamma_raw = P["base_gamma"]

        # 场景突变 → 暂时加快收敛
        alpha_param = C["alpha_param"]
        beta_lut = C["beta_lut"]
        if dist > C["scene_jump_th"]:
            alpha_param = min(0.8, alpha_param * (1 + 2.0 * dist))
            beta_lut = min(0.8, beta_lut * (1 + 2.0 * dist))

        # 参数限幅 + EMA
        if prev_params is None:
            smin = smin_raw
            smax = smax_raw
            gamma = gamma_raw
        else:
            p_smin, p_smax, p_gamma = prev_params
            smin = _limit_step(p_smin, smin_raw, C["delta_abs"], C["delta_rel"])
            smax = _limit_step(p_smax, smax_raw, C["delta_abs"], C["delta_rel"])
            gamma = _limit_step(p_gamma, gamma_raw, C["delta_abs"], C["delta_rel"])
            smin = _ema(p_smin, smin, alpha_param)
            smax = _ema(p_smax, smax, alpha_param)
            gamma = _ema(p_gamma, gamma, alpha_param)

        # 构建目标 LUT 并与前一帧 LUT 渐变
        x = np.arange(max_val + 1, dtype=np.float32)
        V = max(smax - smin, eps)
        y = (x - smin + max_val * P["noise"]) / V
        np.clip(y, 0.0, 1.0, out=y)
        y = np.power(y, gamma) * max_val
        lut_target = np.clip(y, 0.0, float(max_val)).astype(np.uint16)

        if prev_lut is None:
            lut = lut_target
        else:
            lut = (
                (1 - beta_lut) * prev_lut.astype(np.float32)
                + beta_lut * lut_target.astype(np.float32)
            ).astype(np.uint16)

        # 应用 LUT
        img_min, img_max = image.min(), image.max()
        if img_max > max_val and print_debug:
            logger.warning(
                f"警告：图像像素值超出范围 [{img_min}, {img_max}]，将裁剪到 [0, {max_val}]"
            )

        img_clip = np.clip(image, 0, max_val)
        if img_clip.ndim == 2:
            out = lut[img_clip]
        else:
            out = np.empty_like(img_clip)
            for c in range(img_clip.shape[2]):
                out[..., c] = lut[img_clip[..., c]]

        # 更新 video_state（原地修改）
        video_state["hist_ema"] = H_ema
        video_state["prev_lut"] = lut
        video_state["prev_params"] = (smin, smax, gamma)
        video_state["segment"] = seg
        video_state["prev_R"] = rb_range

        if print_debug:
            logger.info(
                f"[video] seg={seg} R={rb_range:.1f} lo/hi={lo:.1f}/{hi:.1f} "
                f"smin/smax={smin:.1f}/{smax:.1f} V={V:.1f} gamma={gamma:.3f} "
                f"hist_dist={dist:.3f} alpha={alpha_param:.2f} beta_lut={beta_lut:.2f} "
                f"time={(time.time()-t0):.4f}s"
            )

        return out
    
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
            gamma_mode = "adaptive"
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
            print(f"avg: {avg}")
            adj_gamma = gamma * (1.0 + adapt_str * (0.5 - avg))
            print(f"adj_gamma: {adj_gamma}")
            adj_gamma = float(np.clip(adj_gamma, gamma_bounds[0], gamma_bounds[1]))

        else:
            adj_gamma = gamma
        
        print(f"lo: {lo}, hi: {hi}, safe_min: {safe_min}, safe_max: {safe_max}, valid_range: {valid_range}, adj_gamma: {adj_gamma}")

        # —— 构建 LUT（uint16→uint16），然后套在所有通道 —— 
        x = np.arange(max_val + 1, dtype=np.float32)
        y = (x - safe_min + max_val * noise) / valid_range
        np.clip(y, 0.0, 1.0, out=y)
        y = np.power(y, adj_gamma) * max_val

        midtone_boost = 0.08  # 建议 0.05~0.12，小幅提亮
        y_norm = y / max_val
        y_norm = y_norm + midtone_boost * y_norm * (1.0 - y_norm)  # 只抬中灰
        y = y_norm * max_val

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

        # self.scale_to_gamma_alpha(img, bit_max)
        # self.scale_midway(img, bit_max)
        
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
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        result = cv2.addWeighted(image, 2, blurred, -1.0, 0)
        elapsed_time = time.time() - start_time
        print(f"apply_sharping 处理耗时: {elapsed_time:.4f} 秒")
        return result
    
    def apply_sharping_alpah(self, image):

        start_time = time.time()

        # ========= 加速档位（按需切换） =========
        FAST_MODE   = True   # True: 单尺度 USM，取消形态学“过冲保护”，大幅提速
        ULTRA_FAST  = False  # True: 3x3 拉普拉斯锐化（最快），适度有锐化边缘纹理

        # —— 配置（FAST/正常模式通用）——
        amount = 1.5          # 锐化强度
        radius_small = 1      # USM 半径（kernel=3）
        radius_large = 3      # 多尺度时的大半径（kernel=7）
        thr_rel = 0.003       # 相对阈值(满量程比例)，抑制噪声
        multiscale = (not FAST_MODE)  # FAST_MODE 关闭多尺度以提速
        work_in_y = True      # 彩色时仅在Y通道锐化

        orig_dtype = image.dtype
        if np.issubdtype(orig_dtype, np.integer):
            max_val = np.iinfo(orig_dtype).max
        else:
            vmax = float(np.max(image)) if image.size else 1.0
            max_val = vmax if vmax > 1.0 else 1.0
        thr_abs = float(thr_rel) * float(max_val)

        # 预创建 3x3 形态学核（仅在需要时使用）
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # ========= ULTRA_FAST：拉普拉斯（最快）=========
        if ULTRA_FAST:
            # 彩色则转Y通道，灰度直接处理
            if image.ndim == 3 and work_in_y:
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                y, u, v = cv2.split(yuv)
                y32 = y.astype(np.float32)
                lap = cv2.Laplacian(y32, ddepth=cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE)
                # 阈值抑噪
                if thr_abs > 0:
                    np.putmask(lap, np.abs(lap) < thr_abs, 0.0)
                out = y32 + amount * lap
                out = np.clip(out, 0, max_val).astype(orig_dtype)
                result = cv2.cvtColor(cv2.merge((out, u, v)), cv2.COLOR_YUV2BGR)
            else:
                ch = image if image.ndim == 2 else cv2.split(image)
                def _lap_fast(c):
                    c32 = c.astype(np.float32)
                    lap = cv2.Laplacian(c32, ddepth=cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE)
                    if thr_abs > 0:
                        np.putmask(lap, np.abs(lap) < thr_abs, 0.0)
                    o = c32 + amount * lap
                    return np.clip(o, 0, max_val).astype(orig_dtype)
                if image.ndim == 2:
                    result = _lap_fast(ch)
                else:
                    b, g, r = ch
                    result = cv2.merge((_lap_fast(b), _lap_fast(g), _lap_fast(r)))
            print(f"apply_sharping 处理耗时: {time.time() - start_time:.4f} 秒")
            return result

        # ========= FAST / 正常：USM =========
        def _box_blur(src_f32, r: int):
            k = 2 * r + 1
            return cv2.boxFilter(src_f32, ddepth=-1, ksize=(k, k),
                                normalize=True, borderType=cv2.BORDER_REPLICATE)

        # 是否启用过冲保护（FAST_MODE 下关闭以省去 erode/dilate）
        OVERDRIVE_PROTECT = (not FAST_MODE)

        def _usm_once(src_f32, r: int, amt: float, thr_abs_: float):
            blur = _box_blur(src_f32, r)
            mask = src_f32 - blur
            if thr_abs_ > 0:
                np.putmask(mask, np.abs(mask) < thr_abs_, 0.0)  # 硬阈值，快
            dst = src_f32 + amt * mask
            if OVERDRIVE_PROTECT:
                lo = cv2.erode(src_f32, kernel3)
                hi = cv2.dilate(src_f32, kernel3)
                dst = np.minimum(np.maximum(dst, lo), hi)
            return dst

        def _sharpen_single_channel(ch):
            ch_f = ch.astype(np.float32)
            if multiscale:
                s1 = _usm_once(ch_f, radius_small, amount * 0.85, thr_abs)
                s2 = _usm_once(ch_f, radius_large, amount * 0.35, thr_abs * 0.5)
                out = 0.7 * s1 + 0.3 * s2
            else:
                out = _usm_once(ch_f, radius_small, amount, thr_abs)
            return np.clip(out, 0, max_val).astype(orig_dtype)

        if image.ndim == 2:  # 灰度
            result = _sharpen_single_channel(image)
        else:                # 彩色（BGR）
            if work_in_y:
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                y, u, v = cv2.split(yuv)
                y_sharp = _sharpen_single_channel(y)
                result = cv2.cvtColor(cv2.merge((y_sharp, u, v)), cv2.COLOR_YUV2BGR)
            else:
                b, g, r = cv2.split(image)
                result = cv2.merge((_sharpen_single_channel(b),
                                    _sharpen_single_channel(g),
                                    _sharpen_single_channel(r)))

        print(f"apply_sharping 处理耗时: {time.time() - start_time:.4f} 秒")
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

    def apply_defog(
        self,
        image,
        t_min=0.12,                  # 略提高下限，减少过度去雾导致变暗
        omega=0.90,                  # 略降低去雾强度
        guided_filter_radius=20,     # 半径调小，+ 透射率降采样，整体更快
        guided_filter_eps=1e-3,
        in_max=None,
        # ---- 新增：加速与稳态选项 ----
        fast_ds=2,                   # 透射率在 1/fast_ds 分辨率估计
        blend_power=0.5,             # 混合权重指数，越小越“温和”
        blend_min=0.15,              # 最小混合权重
        blend_max=0.85,              # 最大混合权重
        do_exposure_comp=True,       # 去雾后做一次全局曝光匹配
        comp_p=0.75,                 # 用第 p 分位数做匹配，0.6~0.85 均可
        comp_clip=(0.8, 1.25)        # 增益裁剪范围，避免过度提亮/压暗
    ):
        """
        暗通道去雾（IR灰度/三通道；保持原始dtype与量程；快速&不易变黑）
        """
        import time, cv2, numpy as np
        start_time = time.time()

        orig_dtype = image.dtype
        is_gray = (image.ndim == 2)

        # ---- 构造 3 通道参与计算（灰度不做颜色增强，只是走流程）----
        if is_gray:
            img3 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img3 = image
        img3 = img3.astype(np.float32)

        # ---- 识别/设定输入满量程 ----
        def _auto_in_max(arr_u16):
            vmax = int(arr_u16.max()) if arr_u16.size else 0
            if vmax <= 4095:  return 4095.0   # 12-bit
            if vmax <= 16383: return 16383.0  # 14-bit
            return 65535.0                    # 16-bit 默认
        if in_max is None:
            if orig_dtype == np.uint16:
                in_scale = _auto_in_max(image)
            elif orig_dtype == np.uint8:
                in_scale = 255.0
            else:
                cur_max = float(img3.max()) if img3.size else 1.0
                in_scale = max(cur_max, 1.0)
        else:
            in_scale = float(in_max)

        # ---- 归一化到[0,1] ----
        img = np.clip(img3 / max(in_scale, 1.0), 0.0, 1.0)

        # ---- 内部函数 ----
        def get_dark_channel(im, patch=15):
            k = max(3, int(patch) | 1)
            min_c = np.min(im, axis=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            return cv2.erode(min_c, kernel)

        def estimate_atmospheric_light(im, dark, top_percent=0.001):
            h, w = im.shape[:2]
            n = max(1, int(h * w * top_percent))
            flat_dark = dark.reshape(-1)
            idx = np.argpartition(flat_dark, -n)[-n:]
            cand = im.reshape(-1, 3)[idx]
            A = cand[np.argmax(np.linalg.norm(cand, axis=1))]
            return np.maximum(A, 1e-3)

        def estimate_transmission(im, A, omega_=0.95, patch=15):
            normalized = im / (A[None, None, :] + 1e-6)
            dark_norm = get_dark_channel(normalized, patch)
            return 1.0 - omega_ * dark_norm

        def guided_filter(guide_rgb, src, radius, eps):
            guide = guide_rgb
            if guide.ndim == 3:
                guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
            guide = guide.astype(np.float32)  # 已在[0,1]
            src = src.astype(np.float32)

            k = int(radius)
            k = (2*k + 1, 2*k + 1)  # 半径->窗口
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

        # ---- 1) 暗通道（全分辨率）----
        dark_full = get_dark_channel(img, patch=15)

        # ---- 2) 大气光（全分辨率top0.1%）----
        A = estimate_atmospheric_light(img, dark_full, top_percent=0.001)

        # ---- 3) 透射率：低分辨率估计 -> 上采样 -> 引导滤波细化（更快）----
        if fast_ds and fast_ds > 1:
            H, W = img.shape[:2]
            h, w = (H + fast_ds - 1) // fast_ds, (W + fast_ds - 1) // fast_ds
            img_lr = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            # patch 与半径随分辨率缩放
            patch_lr = max(3, (15 // fast_ds) | 1)
            t_lr = estimate_transmission(img_lr, A, omega_=omega, patch=patch_lr)
            t0 = cv2.resize(t_lr, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            t0 = estimate_transmission(img, A, omega_=omega, patch=15)

        t0 = np.clip(t0, 0.0, 1.0)
        t = guided_filter(img, t0, guided_filter_radius, guided_filter_eps)
        t = np.nan_to_num(t, nan=1.0, posinf=1.0, neginf=1.0)

        # ---- 4) 恢复图像（广播，无循环）----
        t_clip = np.clip(t, t_min, 0.999)
        J = (img - A[None, None, :]) / t_clip[..., None] + A[None, None, :]

        # ---- 5) 自适应“柔性去雾”：按 t 做混合，避免晴朗区域被过度处理 ----
        # w = clamp( (1 - t)^blend_power , blend_min, blend_max )
        w = np.clip((1.0 - t) ** float(blend_power), float(blend_min), float(blend_max))
        # 对灰度而言三通道等同；对可见光也能生效
        J_soft = w[..., None] * J + (1.0 - w)[..., None] * img

        # ---- 6) 曝光匹配（分位数）以避免整体偏暗 ----
        Jx = J_soft
        if do_exposure_comp:
            # 用分位数（默认 75%）在[0,1]域做一次全局增益
            q_src = np.quantile(img,  comp_p)
            q_dst = np.quantile(Jx,   comp_p)
            gain = q_src / max(q_dst, 1e-6)
            gmin, gmax = comp_clip
            gain = float(np.clip(gain, gmin, gmax))
            Jx = np.clip(Jx * gain, 0.0, 1.0)

        # ---- 7) 回写原量程 & dtype ----
        if orig_dtype == np.uint16:
            out3 = (Jx * in_scale + 0.5).astype(np.uint16)
        elif orig_dtype == np.uint8:
            out3 = (Jx * 255.0 + 0.5).astype(np.uint8)
        else:
            out3 = (Jx * in_scale).astype(np.float32)

        out = cv2.cvtColor(out3, cv2.COLOR_BGR2GRAY) if is_gray else out3

        print(f"apply_defog 耗时: {time.time() - start_time:.4f}s | dtype={orig_dtype} | in_scale={in_scale} | ds={fast_ds}")
        return out

