import os
import shlex
import signal
import subprocess
import threading
import queue
import time
import traceback
import cv2
import numpy as np
from PIL import Image, ImageTk
from collections import deque
from preprocessor import ImagePreprocessor

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    GST_AVAILABLE = True
except ImportError:
    # 如果没有gi模块，创建一个模拟的Gst类
    class Gst:
        class State:
            NULL = 0
            PLAYING = 1

        @staticmethod
        def parse_launch(*args, **kwargs):
            return None

        @staticmethod
        def init(*args, **kwargs):
            pass

    GST_AVAILABLE = False


class CameraFunctions:
    """相机功能类，包含所有相机控制和图像处理功能"""

    def __init__(self):

        self.init_param()

        # 视频源线程
        self.running = True
        self.fps = 30
        self.source_thread = threading.Thread(
            target=self.video_source_loop, name="VideoSourceThread", daemon=True
        )

        # 预处理线程（不自动启动，由main_app控制）
        self.process_thread = None

    def init_param(self):
        """初始化参数"""
        # 基本参数 - 只在未设置时才初始化
        if not hasattr(self, 'device_num'):
            self.device_num = "0"
        if not hasattr(self, 'real'):
            self.real = "0"
        if not hasattr(self, 'bit_max'):
            self.bit_max = 4095
        if not hasattr(self, 'wave'):
            self.wave = "mwir"
            
        self.whonpz = "multi_point_correction.npz"
        self.bp_npz_path = "blind_pixels.npz"
        self.frame_count = 0
        self.running = True

        # 处理标志
        self.non_uniform_0_enabled = False
        self.non_uniform_1_enabled = False
        self.non_uniform_3_enabled = False  # 多点校正
        self.non_uniform_4_enabled = False
        self.bp_correction_enabled = False
        self.bp_tab_correction_enabled = False

        self.linear_nuc_enabled = False
        self.quadrast_nuc_enabled = False

        self.enhance_enabled = False
        self.sharpen_enabled = False
        self.manual_saving = False  # 手动保存标志
        self.adjust_enabled = False
        self.denoise_enabled = False

        # 变量初始化
        self.blind_mask = None  # 盲元表存储
        self.blind_tab_mask = None  # 坏点查表补偿
        self.detection_running = False  # 检测状态标志
        self.nuc_para = None  # 非均匀校正参数

        self.init_bk_frame()
        self.current_frame = None
        self.whiteground_frame = None

        # 视频捕获相关变量
        self.v4l2_proc = None
        self.gst_proc = None
        self.pipeline = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dw_npz_path = os.path.join(script_dir, "dark_white_calibration.npz")

        # 滚动时间戳队列: 用于帧率平滑
        self.timestamps = deque(maxlen=30)
        self.frame_count = 0

        # 预处理初始化
        self.precessor = ImagePreprocessor()

        self.raw_queue = queue.Queue(maxsize=3)  # 原始帧队列
        self.play_queue = queue.Queue(maxsize=1)  # 处理后帧队列

        self.crosshair_enabled = False  # 控制是否显示十字星
        self.display_image_size = (640, 512)  # 显示图像的宽高
        self.image_area_rect = (0, 0, 640, 512)  # Canvas 中图像区域（用于缩放计算）

        # 离线图片模式
        self.offline_image_mode = False
        self.offline_image = None

    def init_bk_frame(self):
        """初始化背景帧"""
        try:
            if os.path.exists("background_frame.png"):
                self.background_frame = cv2.imread("background_frame.png", -1)
                print(f"背景帧已加载: {self.background_frame.mean()}")
        except Exception:
            pass

    def choose_camera_mode(self, real="0"):
        """选择相机模式"""
        # 保存当前参数（如果已设置）
        saved_device_num = getattr(self, 'device_num', None)
        saved_wave = getattr(self, 'wave', None)
        saved_bit_max = getattr(self, 'bit_max', None)
        
        # 先停止当前数据源
        self.cleanup()

        # 重新初始化参数
        self.init_param()
        
        # 恢复保存的参数（如果存在）
        if saved_device_num is not None:
            self.device_num = saved_device_num
        if saved_wave is not None:
            self.wave = saved_wave
        if saved_bit_max is not None:
            self.bit_max = saved_bit_max

        # 清除离线图片模式
        self.offline_image_mode = False
        self.offline_image = None

        if real == "0":
            # 测试视频模式
            self.running = True
            self.source_thread = threading.Thread(
                target=self.video_source_loop, name="VideoSourceThread", daemon=True
            )
            self.source_thread.start()
            print("测试视频模式已启动")
        elif real == "1":
            # 摄像头模式
            self.running = True
            self._start_v4l2_stream()
            self.init_camera()
            print("摄像头模式已启动")

    def _start_v4l2_stream(self):
        """启动v4l2流"""

        if self.wave == "mwir":
            cmd = f"""
            v4l2-ctl -d /dev/video{self.device_num}
                --set-fmt-video=width=640,height=512,pixelformat='Y12 '
                --stream-mmap=4
                --set-selection=target=crop,flags=0,top=0,left=0,width=640,height=512
                --stream-to=-
            """
        elif self.wave == "lwir":
            cmd = f"""
            v4l2-ctl -d /dev/video{self.device_num}
                --set-fmt-video=width=1280,height=512,pixelformat=GREY
                --stream-mmap=4
                --set-selection=target=crop,flags=0,top=0,left=0,width=1280,height=512
                --stream-to=-
            """
        try:
            self.v4l2_proc = subprocess.Popen(
                shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception as e:
            print(f"开启硬件控制出错:{e},详细内容如下：")
            print(traceback.format_exc())

    def init_camera(self):
        """初始化相机"""
        if not GST_AVAILABLE:
            print("GStreamer不可用，跳过相机初始化")
            return

        pipeline_str = f"""
            fdsrc fd={self.v4l2_proc.stdout.fileno()}
            ! videoparse format=gray16-le width=640 height=512 framerate=30/1
            ! appsink drop=1
        """
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsink = self.pipeline.get_by_name("appsink0")
            if not self.appsink:
                self.appsink = Gst.ElementFactory.make("appsink", "appsink0")
                self.pipeline.add(self.appsink)

            self.appsink.set_property("emit-signals", True)
            self.appsink.set_property("sync", False)
            self.appsink.connect("new-sample", self.on_new_sample, None)

            self.pipeline.set_state(Gst.State.PLAYING)
        except Exception as e:
            print(f"开启流水线出错:{e},详细内容如下：")
            print(traceback.format_exc())

    def on_new_sample(self, sink, data):
        """处理新的样本"""
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        if not buffer:
            return Gst.FlowReturn.ERROR

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        self.frame_count += 1

        # 处理大端16位灰度数据
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        frame = np.frombuffer(map_info.data, dtype=np.uint16).reshape(height, width)
        print(f"min: {frame.min()}, max: {frame.max()}, mean: {np.mean(frame)}")

        try:
            self.raw_queue.put_nowait(frame)
        except queue.Full:
            self.raw_queue.get()
            self.raw_queue.put(frame)

        buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def video_source_loop(self):
        """模拟以 self.fps 帧率随机生成 16位图像"""
        interval = 1.0 / self.fps
        try:
            if os.path.exists("display_20250517_085815_929.png"):
                frame_cached = cv2.imread("display_20250517_085815_929.png", -1)
            else:
                frame_cached = np.random.randint(0, 65536, (512, 640), dtype=np.uint16)
        except Exception:
            frame_cached = None

        start_time = time.perf_counter()
        while self.running:
            frame_16bit = np.copy(frame_cached)
            self.frame_count += 1
            # print(f"{self.frame_count} frame_16bit mean: {frame_16bit.mean()}")
            try:
                self.raw_queue.put(frame_16bit, timeout=0.1)
            except queue.Full:
                self.raw_queue.get(timeout=0.1)
                self.raw_queue.put(frame_16bit, timeout=0.1)
            except Exception as e:
                print(f"处理帧出错:{e},详细内容如下：")
                print(traceback.format_exc())

            # 高精度休眠
            elapsed = time.perf_counter() - start_time
            sleep_time = max(interval - elapsed, 0)
            time.sleep(sleep_time)
            start_time = time.perf_counter()

    def preprocess_ctrl(self, frame):
        """预处理控制"""
        processed_frame = frame.copy()

        # 根据标志决定处理流程
        if self.non_uniform_0_enabled and self.background_frame is not None:

            self.non_uniform_1_enabled = False
            self.non_uniform_3_enabled = False
            self.non_uniform_4_enabled = False
            self.linear_nuc_enabled = False
            self.quadrast_nuc_enabled = False

            processed_frame = cv2.subtract(self.background_frame, processed_frame)

        if self.non_uniform_1_enabled and self.background_frame is not None:
            self.non_uniform_0_enabled = False
            self.non_uniform_3_enabled = False
            self.non_uniform_4_enabled = False
            self.linear_nuc_enabled = False
            self.quadrast_nuc_enabled = False

            processed_frame = cv2.subtract(processed_frame, self.background_frame)

        if self.non_uniform_3_enabled and self.nuc_para is not None:
            self.non_uniform_0_enabled = False
            self.non_uniform_1_enabled = False
            self.non_uniform_4_enabled = False
            self.linear_nuc_enabled = False
            self.quadrast_nuc_enabled = False

            # 多点校正
            processed_frame = self.precessor.apply_non_uniform_correction(
                processed_frame, self.nuc_para
            )

        if self.non_uniform_4_enabled and self.nuc_para is not None:
            self.non_uniform_0_enabled = False
            self.non_uniform_1_enabled = False
            self.non_uniform_3_enabled = False
            self.linear_nuc_enabled = False
            self.quadrast_nuc_enabled = False

            processed_frame = self.precessor.dw_nuc(
                processed_frame, self.nuc_para, bit_max=self.bit_max
            )

        if self.linear_nuc_enabled and self.nuc_para is not None:
            self.non_uniform_0_enabled = False
            self.non_uniform_1_enabled = False
            self.non_uniform_3_enabled = False
            self.non_uniform_4_enabled = False
            self.quadrast_nuc_enabled = False

            processed_frame = self.precessor.linear_corr(
                processed_frame, self.nuc_para, bit_max=self.bit_max
            )

        if self.quadrast_nuc_enabled and self.nuc_para is not None:
            self.non_uniform_0_enabled = False
            self.non_uniform_1_enabled = False
            self.non_uniform_3_enabled = False
            self.non_uniform_4_enabled = False
            self.linear_nuc_enabled = False

            processed_frame = self.precessor.quadrast_corr(
                processed_frame, self.nuc_para, bit_max=self.bit_max
            )

        if self.bp_tab_correction_enabled and self.blind_tab_mask is not None:
            print("坏点查表补偿中....")
            processed_frame = self.precessor.compensate_with_filter(
                processed_frame, self.blind_tab_mask
            )
            print("坏点查表补偿完成！")

        if self.bp_correction_enabled and self.blind_mask is not None:
            processed_frame = self.precessor.compensate_with_filter(
                processed_frame, self.blind_mask
            )

        if self.adjust_enabled:
            processed_frame = self.precessor.process(processed_frame, gamma=1.0, bit_max=self.bit_max)

        if self.denoise_enabled:
            processed_frame = self.precessor.apply_denoise(processed_frame)

        if self.enhance_enabled:
            processed_frame = self.precessor.apply_autogian(processed_frame)

        if self.sharpen_enabled:
            processed_frame = self.precessor.apply_sharping(processed_frame)

        return processed_frame

    # 图像处理功能
    def draw_crosshair(self, image, color=(255, 0, 0), thickness=2):
        """在图像中心绘制十字星"""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        length = min(w, h) // 50  # 十字长度为图像短边的 1/10

        # 横线
        cv2.line(
            image,
            (center_x - length, center_y),
            (center_x + length, center_y),
            color,
            thickness,
        )
        # 竖线
        cv2.line(
            image,
            (center_x, center_y - length),
            (center_x, center_y + length),
            color,
            thickness,
        )
        return image

    def scale_to_8bit(self, frame_16bit, max_val=None):
        """将 16 位图像拉伸到 8 位"""
        if max_val is None:
            max_val = self.bit_max
        t = frame_16bit.astype(np.float32)
        scaled = (t / max_val) * 255.0
        scaled = np.clip(scaled, 0, 255)
        return scaled.astype(np.uint8)

    def apply_flip(self, frame, flip_mode="none"):
        """应用翻转变换"""
        if flip_mode == "none":
            return frame
        elif flip_mode == "horizontal":
            return cv2.flip(frame, 1)  # 水平翻转
        elif flip_mode == "vertical":
            return cv2.flip(frame, 0)  # 垂直翻转
        elif flip_mode == "both":
            return cv2.flip(frame, -1)  # 水平垂直翻转
        else:
            return frame

    def normalize_to_8bit_for_jpg(self, frame_16bit, max_val=None):
        """将16位图像归一化到0-1再乘以255，用于JPG保存，不截断数据"""
        if max_val is None:
            max_val = self.bit_max
        t = frame_16bit.astype(np.float64)  # 使用更高精度
        # 归一化到0-1
        normalized = t / max_val
        # 乘以255并转换为8位
        jpg_frame = (normalized * 255.0).astype(np.uint8)
        return jpg_frame

    # 存储功能
    def save_background(self):
        """保存背景图"""
        try:
            if hasattr(self, "current_frame"):
                self.background_frame = self.current_frame.copy()
                # 保存16位PNG
                cv2.imwrite(
                    "background_frame.png",
                    self.background_frame.copy().astype(np.uint16),
                )
                # 保存8位JPG
                jpg_frame = self.normalize_to_8bit_for_jpg(self.background_frame)
                cv2.imwrite("background_frame.jpg", jpg_frame)
                print("Info", "Background frame saved successfully! (PNG + JPG)")
            else:
                print("Error", "No frame available to save!")
        except Exception as e:
            print(f"保存背景图出错:{e},详细内容如下：")
            print(traceback.format_exc)

    def save_whiteground(self):
        """保存亮场图"""
        try:
            if hasattr(self, "current_frame"):
                self.whiteground_frame = self.current_frame.copy()
                # 保存16位PNG
                cv2.imwrite(
                    "whiteground_frame.png",
                    self.whiteground_frame.astype(np.uint16),
                )
                # 保存8位JPG
                jpg_frame = self.normalize_to_8bit_for_jpg(self.whiteground_frame)
                cv2.imwrite("whiteground_frame.jpg", jpg_frame)
                print("Info", "Whiteground frame saved successfully! (PNG + JPG)")
            else:
                print("Error", "No frame available to save!")
        except Exception as e:
            print(f"保存背景图出错:{e},详细内容如下：")
            print(traceback.format_exc)

    def auto_save_100(self, save_count=100):
        """自动保存指定张数图片"""
        t = threading.Thread(
            target=self._auto_save_thread, args=(save_count,), daemon=True
        )
        t.start()

    def _auto_save_thread(self, save_count):
        """自动保存指定张数图片的线程"""
        folder = "auto_images"
        os.makedirs(folder, exist_ok=True)  # 如果文件夹不存在则创建

        count = 0
        # 循环保存指定张数图片
        try:
            while count < save_count:
                if self.current_frame is not None:
                    display_frame = self.current_frame.copy().astype(np.uint16)
                    # 使用时间戳命名（加上计数后缀避免同一秒重复）
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    
                    # 保存16位PNG
                    png_filename = f"display_{timestamp}_{count:03d}.png"
                    png_filepath = os.path.join(folder, png_filename)
                    cv2.imwrite(png_filepath, display_frame)
                    
                    # 保存8位JPG
                    jpg_filename = f"display_{timestamp}_{count:03d}.jpg"
                    jpg_filepath = os.path.join(folder, jpg_filename)
                    jpg_frame = self.normalize_to_8bit_for_jpg(self.current_frame)
                    cv2.imwrite(jpg_filepath, jpg_frame)
                    
                    count += 1
                    print(f"已保存第 {count}/{save_count} 张图片 (PNG + JPG)")
                time.sleep(0.03)  # 默认保存间隔0.03秒
        except Exception as e:
            print(f"自动保存出错:{e},详细内容如下：")
            print(traceback.format_exc)

    def save_stop_image(self):
        """保存/停止图像"""
        if not hasattr(self, "manual_saving"):
            self.manual_saving = False

        # 切换状态：若当前未保存，则开始保存；反之则停止保存
        if not self.manual_saving:
            self.manual_saving = True
            # 启动后台线程保存图像
            t = threading.Thread(target=self._manual_save_thread, daemon=True)
            t.start()
        else:
            self.manual_saving = False

    def _manual_save_thread(self):
        """手动保存线程"""
        folder = "manual"
        os.makedirs(folder, exist_ok=True)
        try:
            while self.manual_saving:
                if self.current_frame is not None:
                    display_frame = self.current_frame.copy().astype(np.uint16)
                    # 使用时间戳命名，添加毫秒部分以保证唯一性
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    ms = int(time.time() * 1000) % 1000
                    
                    # 保存16位PNG
                    png_filename = f"display_{timestamp}_{ms:03d}.png"
                    png_filepath = os.path.join(folder, png_filename)
                    cv2.imwrite(png_filepath, display_frame)
                    
                    # 保存8位JPG
                    jpg_filename = f"display_{timestamp}_{ms:03d}.jpg"
                    jpg_filepath = os.path.join(folder, jpg_filename)
                    jpg_frame = self.normalize_to_8bit_for_jpg(self.current_frame)
                    cv2.imwrite(jpg_filepath, jpg_frame)
                    
                # 控制保存频率，与界面刷新同步
                time.sleep(0.03)
        except Exception as e:
            print(f"手动保存出错:{e},详细内容如下：")
            print(traceback.format_exc)

    # 校正功能
    def dark_white_two_point_calibration(self, dark, white):
        """明暗两点校正"""
        L1, L2 = np.mean(dark), np.mean(white)
        gain_map = (L2 - L1) / (white - dark + 1e-6)
        offset_map = L1 - gain_map * dark
        np.savez_compressed(
            self.save_dw_npz_path, gain_map=gain_map, offset_map=offset_map, ref=L2 - L1
        )
        print(f"[INFO] save to  {self.save_dw_npz_path}")
        self.whonpz = self.save_dw_npz_path
        print(f"已设置校正文件路径为计算结果路径: {self.save_dw_npz_path}")

    def correction_bw(self):
        """计算黑白场校正"""
        if self.background_frame is not None and self.whiteground_frame is not None:
            nuc_t = threading.Thread(
                target=self.dark_white_two_point_calibration,
                args=(
                    self.background_frame.copy(),
                    self.whiteground_frame.copy(),
                ),
                daemon=True,
            )
            nuc_t.start()
        else:
            print("缺少校正数据源！")

    def async_blind_detection(self, frame):
        """异步盲元检测线程"""
        self.detection_running = True
        try:
            print("坏点检测中....")
            new_mask = self.precessor.apply_blind_pixel_detect(frame)
            self.blind_mask = new_mask  # 更新盲元表
            print("盲元检测完成！")
        except Exception as e:
            print("盲元检测错误:", e)
        finally:
            self.detection_running = False

    # 控制功能
    def non_uniform_correction_0(self):
        """单点校正(b-)"""
        self.non_uniform_0_enabled = not self.non_uniform_0_enabled
        state = "enabled" if self.non_uniform_0_enabled else "disabled"
        
        # 如果启用单点校正但没有背景帧，使用当前帧作为背景帧
        if self.non_uniform_0_enabled and self.background_frame is None and self.current_frame is not None:
            self.background_frame = self.current_frame.copy()
            print("Info", "使用当前帧作为背景帧进行单点校正")
        
        print("Info", f"Non-uniform correction {state}")
        
        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def non_uniform_correction_1(self):
        """单点校正(-b)"""
        self.non_uniform_1_enabled = not self.non_uniform_1_enabled
        state = "enabled" if self.non_uniform_1_enabled else "disabled"
        
        # 如果启用单点校正但没有背景帧，使用当前帧作为背景帧
        if self.non_uniform_1_enabled and self.background_frame is None and self.current_frame is not None:
            self.background_frame = self.current_frame.copy()
            print("Info", "使用当前帧作为背景帧进行单点校正")
        
        print("Info", f"Non-uniform correction {state}")
        
        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def bp_correction(self):
        """坏点检测"""
        if not self.bp_correction_enabled:
            # 首次启用时启动检测
            if self.blind_mask is None and not self.detection_running:
                if self.current_frame is not None:
                    # 启动后台检测线程
                    detection_frame = self.current_frame.copy()
                    threading.Thread(
                        target=self.async_blind_detection,
                        args=(detection_frame,),
                        daemon=True,
                    ).start()
                else:
                    print("Warning", "请先获取有效图像帧")
                    return

            # 切换状态
            self.bp_correction_enabled = True
            state = "enabled"
        else:
            # 关闭功能
            self.bp_correction_enabled = False
            state = "disabled"

        print("Info", f"坏点校正 {state}")

    def bp_table_compensation(self):
        """查表补偿"""
        if not self.bp_tab_correction_enabled:
            # 启用时加载文件
            try:
                if not os.path.exists(self.bp_npz_path):
                    print("文件 'blind_pixels.npz' 不存在。")
                    return

                # 加载npz文件
                data = np.load(self.bp_npz_path)
                self.blind_tab_mask = data["blind"].astype(bool)

                # 标记为启用坏点查表补偿
                self.bp_tab_correction_enabled = True

                # 如果是离线图片模式，重新处理图片
                if self.offline_image_mode and self.offline_image is not None:
                    self._trigger_offline_reprocess()

                print("成功加载坏点查表文件，坏点补偿已启用。")
            except Exception as e:
                print(f"加载坏点查表文件时出错: {e}")

        else:
            # 关闭功能
            self.bp_tab_correction_enabled = False
            self.blind_tab_mask = None
            print("坏点查表补偿已禁用。")

    def linear_multi_point_correction(self):
        """线性校正"""
        if not self.linear_nuc_enabled:
            try:
                if not os.path.exists(self.whonpz):
                    print(f"文件 '{self.whonpz}' 不存在。")
                    return

                self.nuc_para = np.load(self.whonpz, allow_pickle=True)
                self.linear_nuc_enabled = True
                print("线性校正已启用。")

                # 如果是离线图片模式，重新处理图片
                if self.offline_image_mode and self.offline_image is not None:
                    self._trigger_offline_reprocess()

            except Exception as e:
                print(f"线性校正时出错: {e}")

        else:
            # 关闭功能
            self.linear_nuc_enabled = False
            self.nuc_para = None
            print("线性校正已禁用。")

            # 如果是离线图片模式，重新处理图片
            if self.offline_image_mode and self.offline_image is not None:
                self._trigger_offline_reprocess()

    def quadrast_multi_point_correction(self):
        """非线性校正"""
        if not self.quadrast_nuc_enabled:
            try:
                if not os.path.exists(self.whonpz):
                    print(f"文件 '{self.whonpz}' 不存在。")
                    return

                self.nuc_para = np.load(self.whonpz, allow_pickle=True)
                self.quadrast_nuc_enabled = True
                print("非线性校正已启用。")

                # 如果是离线图片模式，重新处理图片
                if self.offline_image_mode and self.offline_image is not None:
                    self._trigger_offline_reprocess()

            except Exception as e:
                print(f"非线性校正时出错: {e}")

        else:
            # 关闭功能
            self.quadrast_nuc_enabled = False
            self.nuc_para = None
            print("非线性校正已禁用。")

            # 如果是离线图片模式，重新处理图片
            if self.offline_image_mode and self.offline_image is not None:
                self._trigger_offline_reprocess()

    def dark_white_correction(self):
        """明暗校正"""
        if not self.non_uniform_4_enabled:
            try:
                if not os.path.exists(self.whonpz):
                    print(f"文件 '{self.whonpz}' 不存在。")
                    return

                self.nuc_para = np.load(self.whonpz, allow_pickle=True)

                print(f"成功加载明暗校正文件: {self.whonpz}，明暗校正已启用。")

                self.non_uniform_4_enabled = True

                # 如果是离线图片模式，重新处理图片
                if self.offline_image_mode and self.offline_image is not None:
                    self._trigger_offline_reprocess()

            except Exception as e:
                print(f"明暗校正时出错: {e}")

        else:
            # 关闭功能
            self.non_uniform_4_enabled = False
            self.nuc_para = None
            print("明暗校正已禁用。")

            # 如果是离线图片模式，重新处理图片
            if self.offline_image_mode and self.offline_image is not None:
                self._trigger_offline_reprocess()

    def img_denoise(self):
        """图像去噪"""
        self.denoise_enabled = not self.denoise_enabled
        state = "enabled" if self.denoise_enabled else "disabled"
        print("Info", f"Image denoise {state}")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def img_enhance(self):
        """图像增强"""
        self.enhance_enabled = not self.enhance_enabled
        state = "enabled" if self.enhance_enabled else "disabled"
        print("Info", f"Image enhancement {state}")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def img_imadjust(self):
        """图像调整"""
        self.adjust_enabled = not self.adjust_enabled
        state = "enabled" if self.adjust_enabled else "disabled"
        print("Info", f"Image adjust {state}")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def image_sharpen(self):
        """图像锐化"""
        self.sharpen_enabled = not self.sharpen_enabled
        state = "enabled" if self.sharpen_enabled else "disabled"
        print("Info", f"Image sharpen {state}")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def toggle_crosshair(self):
        """切换十字星显示"""
        self.crosshair_enabled = not self.crosshair_enabled
        state = "已启用" if self.crosshair_enabled else "已禁用"
        print(f"Info: 十字星 {state}")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def show_original(self):
        """显示原图"""
        self.non_uniform_1_enabled = False
        self.non_uniform_0_enabled = False
        self.non_uniform_3_enabled = False
        self.non_uniform_4_enabled = False

        self.bp_correction_enabled = False
        self.bp_tab_correction_enabled = False

        self.enhance_enabled = False
        self.sharpen_enabled = False
        self.denoise_enabled = False

        self.quadrast_nuc_enabled = False
        self.linear_nuc_enabled = False

        self.adjust_enabled = False

        print("Info", "Showing original frame!")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def cleanup(self):
        """清理资源"""
        self.running = False

        # 停止视频源线程
        if (
            hasattr(self, "source_thread")
            and self.source_thread
            and self.source_thread.is_alive()
        ):
            try:
                self.source_thread.join(timeout=2)
                print("视频源线程已停止")
            except Exception as e:
                print(f"停止视频源线程时出错: {e}")

        # 停止GStreamer管道
        if GST_AVAILABLE and self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

        # 停止v4l2进程
        if self.v4l2_proc:
            try:
                self.v4l2_proc.send_signal(signal.SIGINT)
                self.v4l2_proc.wait(timeout=5)
            except Exception as e:
                print(f"停止v4l2进程时出错: {e}")
            finally:
                self.v4l2_proc = None

        # 清空队列
        try:
            while not self.raw_queue.empty():
                self.raw_queue.get_nowait()
        except:
            pass

        try:
            while not self.play_queue.empty():
                self.play_queue.get_nowait()
        except:
            pass

        print("资源清理完成")

    def _trigger_offline_reprocess(self):
        """触发离线图片重新处理"""
        # 这个方法会被main_app.py重写，用于通知UI重新处理离线图片
        pass
