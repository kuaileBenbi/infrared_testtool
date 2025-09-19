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
import tkinter as tk
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

    def __init__(self, ui=None):
        self.ui = ui
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
        self.dw_nuc_enabled = False
        self.bp_correction_enabled = False
        self.bp_tab_correction_enabled = False

        self.linear_nuc_enabled = False
        self.quadrast_nuc_enabled = False

        self.enhance_enabled = False
        self.sharpen_enabled = False
        self.manual_saving = False  # 手动保存标志
        self.adjust_enabled = False
        self.denoise_enabled = False
        self.stretch_enabled = False
        self.adaptive_stretch_enabled = False
        self.stretch_state = {}
        self.defog_enabled = False

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
        
        # 强制清理可能残留的资源
        self._force_cleanup_resources()

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
        # 检查设备是否存在
        device_path = f"/dev/video{self.device_num}"
        if not os.path.exists(device_path):
            raise FileNotFoundError(f"摄像头设备 {device_path} 不存在")

        # 检查设备是否被其他进程占用
        try:
            check_cmd = f"lsof {device_path}"
            result = subprocess.run(shlex.split(check_cmd), capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                print(f"警告: 设备 {device_path} 可能被其他进程占用")
                print(f"占用信息: {result.stdout}")
        except Exception as e:
            print(f"检查设备占用状态时出错: {e}")

        if self.wave in ["mwir", "swir"]:
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
            print(f"启动v4l2流，设备: {device_path}, 波段: {self.wave}")
            self.v4l2_proc = subprocess.Popen(
                shlex.split(cmd), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            # 检查进程是否成功启动
            if self.v4l2_proc.poll() is not None:
                stderr_output = self.v4l2_proc.stderr.read().decode('utf-8')
                raise RuntimeError(f"v4l2进程启动失败: {stderr_output}")
            
            print("v4l2流启动成功")
            
        except Exception as e:
            print(f"开启硬件控制出错:{e},详细内容如下：")
            print(traceback.format_exc())
            # 确保清理失败的进程
            if hasattr(self, 'v4l2_proc') and self.v4l2_proc:
                try:
                    self.v4l2_proc.kill()
                except:
                    pass
                self.v4l2_proc = None
            raise

    def init_camera(self):
        """初始化相机"""
        if not GST_AVAILABLE:
            print("GStreamer不可用，跳过相机初始化")
            return

        if not hasattr(self, 'v4l2_proc') or not self.v4l2_proc:
            raise RuntimeError("v4l2进程未启动，无法初始化相机")

        # 使用手动创建元素的方式，更可靠
        fd_num = self.v4l2_proc.stdout.fileno()
        
        try:
            print("初始化GStreamer管道...")
            print(f"使用文件描述符: {fd_num}")
            
            # 创建管道
            self.pipeline = Gst.Pipeline()
            if not self.pipeline:
                raise RuntimeError("GStreamer管道创建失败")
            
            # 创建元素
            fdsrc = Gst.ElementFactory.make("fdsrc", "fdsrc")
            videoparse = Gst.ElementFactory.make("videoparse", "videoparse")
            self.appsink = Gst.ElementFactory.make("appsink", "appsink0")
            
            if not fdsrc or not videoparse or not self.appsink:
                raise RuntimeError("无法创建GStreamer元素")
            
            # 设置元素属性
            fdsrc.set_property("fd", fd_num)
            videoparse.set_property("format", 1)  # GstVideoFormat.GRAY16_LE
            videoparse.set_property("width", 640)
            videoparse.set_property("height", 512)
            videoparse.set_property("framerate", Gst.Fraction(30, 1))
            
            self.appsink.set_property("emit-signals", True)
            self.appsink.set_property("sync", False)
            self.appsink.set_property("max-buffers", 1)
            self.appsink.set_property("drop", True)
            
            # 添加元素到管道
            self.pipeline.add(fdsrc)
            self.pipeline.add(videoparse)
            self.pipeline.add(self.appsink)
            
            # 连接元素
            if not fdsrc.link(videoparse):
                raise RuntimeError("无法连接fdsrc到videoparse")
            if not videoparse.link(self.appsink):
                raise RuntimeError("无法连接videoparse到appsink")
            
            print("GStreamer元素创建和连接成功")
            
            # 连接信号
            self.appsink.connect("new-sample", self.on_new_sample, None)

            print("设置管道状态为READY...")
            # 先设置为READY状态
            ret = self.pipeline.set_state(Gst.State.READY)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer管道设置为READY状态失败")
            
            # 等待READY状态
            ret = self.pipeline.get_state(timeout=3 * Gst.SECOND)
            print(f"READY状态结果: {ret}")
            if ret[0] == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(f"GStreamer管道READY状态失败: {ret}")
            
            print("设置管道状态为PLAYING...")
            # 再设置为PLAYING状态
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer管道启动失败")
            
            # 等待PLAYING状态，但不要过于严格
            ret = self.pipeline.get_state(timeout=3 * Gst.SECOND)
            print(f"PLAYING状态结果: {ret}")
            if ret[0] == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(f"GStreamer管道PLAYING状态失败: {ret}")
            
            # 如果状态是ASYNC，也认为是成功的
            if ret[0] in [Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.ASYNC]:
                print("GStreamer管道初始化成功")
            else:
                print(f"警告: GStreamer管道状态为 {ret[0]}，但继续运行")
            
        except Exception as e:
            print(f"手动创建管道失败: {e}")
            print("尝试使用parse_launch方法...")
            
            # 清理失败的管道
            if hasattr(self, 'pipeline') and self.pipeline:
                try:
                    self.pipeline.set_state(Gst.State.NULL)
                except:
                    pass
                self.pipeline = None
            
            # 尝试使用parse_launch方法
            try:
                pipeline_str = f"fdsrc fd={fd_num} ! videoparse format=gray16-le width=640 height=512 framerate=30/1 ! appsink name=appsink0 drop=1"
                print(f"备用管道字符串: {pipeline_str}")
                
                self.pipeline = Gst.parse_launch(pipeline_str)
                if not self.pipeline:
                    raise RuntimeError("GStreamer管道创建失败")
                
                # 获取appsink元素
                self.appsink = self.pipeline.get_by_name("appsink0")
                if not self.appsink:
                    raise RuntimeError("无法获取appsink元素")
                
                # 设置appsink属性
                self.appsink.set_property("emit-signals", True)
                self.appsink.set_property("sync", False)
                self.appsink.set_property("max-buffers", 1)
                self.appsink.set_property("drop", True)
                self.appsink.connect("new-sample", self.on_new_sample, None)
                
                print("使用parse_launch方法成功")
                
                # 继续状态设置
                print("设置管道状态为READY...")
                ret = self.pipeline.set_state(Gst.State.READY)
                if ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("GStreamer管道设置为READY状态失败")
                
                ret = self.pipeline.get_state(timeout=3 * Gst.SECOND)
                print(f"READY状态结果: {ret}")
                if ret[0] == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError(f"GStreamer管道READY状态失败: {ret}")
                
                print("设置管道状态为PLAYING...")
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("GStreamer管道启动失败")
                
                ret = self.pipeline.get_state(timeout=3 * Gst.SECOND)
                print(f"PLAYING状态结果: {ret}")
                if ret[0] == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError(f"GStreamer管道PLAYING状态失败: {ret}")
                
                if ret[0] in [Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.ASYNC]:
                    print("GStreamer管道初始化成功")
                else:
                    print(f"警告: GStreamer管道状态为 {ret[0]}，但继续运行")
                    
            except Exception as e2:
                print(f"parse_launch方法也失败: {e2}")
                print(traceback.format_exc())
                # 清理失败的管道
                if hasattr(self, 'pipeline') and self.pipeline:
                    try:
                        self.pipeline.set_state(Gst.State.NULL)
                    except:
                        pass
                    self.pipeline = None
                raise RuntimeError(f"所有GStreamer管道创建方法都失败: {e}, {e2}")

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
        # print(f"min: {frame.min()}, max: {frame.max()}, mean: {np.mean(frame)}")
        
        # 更新图像统计信息到status_frame
        if hasattr(self, 'ui') and hasattr(self.ui, 'image_stats_text'):
            stats_text = f"图像统计:\nmean={np.mean(frame):.1f}\nmin={frame.min()}\nmax={frame.max()}"
            self.ui.image_stats_text.config(state="normal")
            self.ui.image_stats_text.delete(1.0, tk.END)
            self.ui.image_stats_text.insert(1.0, stats_text)
            self.ui.image_stats_text.config(state="disabled")

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
            self.dw_nuc_enabled = False
            self.linear_nuc_enabled = False
            self.quadrast_nuc_enabled = False

            processed_frame = cv2.subtract(self.background_frame, processed_frame)

        if self.non_uniform_1_enabled and self.background_frame is not None:
            self.non_uniform_0_enabled = False
            self.dw_nuc_enabled = False
            self.linear_nuc_enabled = False
            self.quadrast_nuc_enabled = False

            processed_frame = cv2.subtract(processed_frame, self.background_frame)

        if self.dw_nuc_enabled and self.nuc_para is not None:
            self.non_uniform_0_enabled = False
            self.non_uniform_1_enabled = False
            self.linear_nuc_enabled = False
            self.quadrast_nuc_enabled = False

            processed_frame = self.precessor.dw_nuc(
                processed_frame, self.nuc_para, bit_max=self.bit_max
            )

        if self.linear_nuc_enabled and self.nuc_para is not None:
            self.non_uniform_0_enabled = False
            self.non_uniform_1_enabled = False

            self.dw_nuc_enabled = False
            self.quadrast_nuc_enabled = False

            processed_frame = self.precessor.linear_corr(
                processed_frame, self.nuc_para, bit_max=self.bit_max
            )

        if self.quadrast_nuc_enabled and self.nuc_para is not None:
            self.non_uniform_0_enabled = False
            self.non_uniform_1_enabled = False
            self.dw_nuc_enabled = False
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
        if self.stretch_enabled:

            bit_max = getattr(self, 'bit_max', 4095)
            
            # 获取UI参数p
            if hasattr(self.ui, 'stretch_level_var'):
                level = self.ui.stretch_level_var.get()
            else:
                level = "off"
                
            if hasattr(self.ui, 'downsample_var'):
                downsample = int(self.ui.downsample_var.get())
            else:
                downsample = 1
                
            if hasattr(self.ui, 'median_ksize_var'):
                median_ksize = int(self.ui.median_ksize_var.get())
            else:
                median_ksize = 3
            
            print(f"应用图像拉伸 - 级别: {level}, 下采样: {downsample}, 中值核: {median_ksize}")
            
            # 应用拉伸处理
            processed_frame = self.precessor.stretch_u16(
                image=processed_frame,
                max_val=bit_max,
                downsample=downsample,
                level=level,
                median_ksize=median_ksize
            )

        if self.adaptive_stretch_enabled:
            bit_max = getattr(self, 'bit_max', 4095)
            
            # 获取UI参数
            if hasattr(self.ui, 'downsample_var'):
                downsample = int(self.ui.downsample_var.get())
            else:
                downsample = 2  # 自适应拉伸默认下采样为2
                
            if hasattr(self.ui, 'median_ksize_var'):
                median_ksize = int(self.ui.median_ksize_var.get())
            else:
                median_ksize = 3  # 自适应拉伸默认中值核为3
            
            print(f"应用自适应拉伸 - 下采样: {downsample}, 中值核: {median_ksize}")
            
            # 应用自适应拉伸处理
            processed_frame = self.precessor.stretch_u16_adaptive(
                image=processed_frame,
                max_val=bit_max,
                downsample=downsample,
                median_ksize=median_ksize,
                print_debug=True,
                video_state=self.stretch_state
            )
        
        if self.denoise_enabled:
            processed_frame = self.precessor.apply_denoise(processed_frame)
        
        if self.adjust_enabled:
            processed_frame = self.precessor.process(processed_frame, gamma=1.0, bit_max=self.bit_max)

        if self.enhance_enabled:
            processed_frame = self.precessor.apply_autogian(processed_frame)

        if self.sharpen_enabled:
            processed_frame = self.precessor.apply_sharping(processed_frame)

        if self.defog_enabled:
            processed_frame = self.precessor.apply_defog(processed_frame)

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

    def auto_save_100(self, save_count=100, keyword="auto_images"):
        """自动保存指定张数图片"""
        t = threading.Thread(
            target=self._auto_save_thread, args=(save_count, keyword), daemon=True
        )
        t.start()

    def _auto_save_thread(self, save_count, keyword="auto_images"):
        """自动保存指定张数图片的线程"""
        folder = os.path.join("auto_images", keyword)
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
                if self.blind_tab_mask is None:
                    if not os.path.exists(self.bp_npz_path):
                        print("文件 'blind_pixels.npz' 不存在。")
                        return
                    else:
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
            print("坏点查表补偿已禁用。")
            if self.offline_image_mode and self.offline_image is not None:
                self._trigger_offline_reprocess()

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
        if not self.dw_nuc_enabled:
            try:
                if not os.path.exists(self.whonpz):
                    print(f"文件 '{self.whonpz}' 不存在。")
                    return

                self.nuc_para = np.load(self.whonpz, allow_pickle=True)

                print(f"成功加载明暗校正文件: {self.whonpz}，明暗校正已启用。")

                self.dw_nuc_enabled = True

                # 如果是离线图片模式，重新处理图片
                if self.offline_image_mode and self.offline_image is not None:
                    self._trigger_offline_reprocess()

            except Exception as e:
                print(f"明暗校正时出错: {e}")

        else:
            # 关闭功能
            self.dw_nuc_enabled = False
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

    def image_stretch(self):
        """应用图像拉伸处理"""
        self.stretch_enabled = not self.stretch_enabled
        state = "enabled" if self.stretch_enabled else "disabled"
        print("Info", f"Image stretch {state}")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def adaptive_stretch(self):
        """应用自适应拉伸处理"""
        self.adaptive_stretch_enabled = not self.adaptive_stretch_enabled
        state = "enabled" if self.adaptive_stretch_enabled else "disabled"

        self.stretch_state = {}
        print("Info", f"Adaptive stretch {state}")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def image_defog(self):
        """透雾处理"""
        self.defog_enabled = not self.defog_enabled
        state = "enabled" if self.defog_enabled else "disabled"
        print("Info", f"Image defog {state}")

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
        self.dw_nuc_enabled = False

        self.bp_correction_enabled = False
        self.bp_tab_correction_enabled = False

        self.enhance_enabled = False
        self.sharpen_enabled = False
        self.denoise_enabled = False
        self.stretch_enabled = False
        self.adaptive_stretch_enabled = False
        self.defog_enabled = False

        self.quadrast_nuc_enabled = False
        self.linear_nuc_enabled = False

        self.adjust_enabled = False

        print("Info", "Showing original frame!")

        # 如果是离线图片模式，重新处理图片
        if self.offline_image_mode and self.offline_image is not None:
            self._trigger_offline_reprocess()

    def cleanup(self):
        """清理资源"""
        print("开始清理资源...")
        self.running = False

        # 停止视频源线程
        if (
            hasattr(self, "source_thread")
            and self.source_thread
            and self.source_thread.is_alive()
        ):
            try:
                self.source_thread.join(timeout=3)
                print("视频源线程已停止")
            except Exception as e:
                print(f"停止视频源线程时出错: {e}")

        # 停止GStreamer管道
        if GST_AVAILABLE and hasattr(self, 'pipeline') and self.pipeline:
            try:
                print("停止GStreamer管道...")
                # 先设置为PAUSED状态
                ret = self.pipeline.set_state(Gst.State.PAUSED)
                print(f"设置PAUSED状态结果: {ret}")
                
                # 再设置为NULL状态
                ret = self.pipeline.set_state(Gst.State.NULL)
                print(f"设置NULL状态结果: {ret}")
                
                # 等待管道状态改变
                ret = self.pipeline.get_state(timeout=3 * Gst.SECOND)
                print(f"GStreamer管道最终状态: {ret}")
                
                self.pipeline = None
                print("GStreamer管道已停止")
            except Exception as e:
                print(f"停止GStreamer管道时出错: {e}")
                # 强制设置为NULL状态
                try:
                    if hasattr(self, 'pipeline') and self.pipeline:
                        self.pipeline.set_state(Gst.State.NULL)
                except:
                    pass

        # 停止v4l2进程
        if hasattr(self, 'v4l2_proc') and self.v4l2_proc:
            try:
                print("停止v4l2进程...")
                # 首先尝试优雅终止
                self.v4l2_proc.send_signal(signal.SIGINT)
                try:
                    self.v4l2_proc.wait(timeout=3)
                    print("v4l2进程已优雅终止")
                except subprocess.TimeoutExpired:
                    print("v4l2进程优雅终止超时，强制终止...")
                    self.v4l2_proc.kill()
                    self.v4l2_proc.wait(timeout=2)
                    print("v4l2进程已强制终止")
            except Exception as e:
                print(f"停止v4l2进程时出错: {e}")
            finally:
                self.v4l2_proc = None

        # 重置摄像头设备状态
        if hasattr(self, 'device_num'):
            try:
                print(f"重置摄像头设备 /dev/video{self.device_num}...")
                # 使用v4l2-ctl重置设备
                reset_cmd = f"v4l2-ctl -d /dev/video{self.device_num} --all"
                subprocess.run(shlex.split(reset_cmd), timeout=2, 
                             capture_output=True, text=True)
                print("摄像头设备状态已重置")
            except Exception as e:
                print(f"重置摄像头设备时出错: {e}")

        # 清空队列
        try:
            while not self.raw_queue.empty():
                self.raw_queue.get_nowait()
            print("原始队列已清空")
        except:
            pass

        try:
            while not self.play_queue.empty():
                self.play_queue.get_nowait()
            print("播放队列已清空")
        except:
            pass

        # 强制垃圾回收
        import gc
        gc.collect()
        
        print("资源清理完成")

    def _force_cleanup_resources(self):
        """强制清理可能残留的资源"""
        print("执行强制资源清理...")
        
        # 清理可能残留的v4l2进程
        if hasattr(self, 'device_num'):
            try:
                device_path = f"/dev/video{self.device_num}"
                # 查找占用设备的进程
                check_cmd = f"lsof {device_path}"
                result = subprocess.run(shlex.split(check_cmd), capture_output=True, text=True, timeout=2)
                if result.returncode == 0 and result.stdout.strip():
                    print(f"发现设备 {device_path} 被占用，尝试清理...")
                    # 这里可以添加更激进的清理逻辑，比如杀死占用进程
                    # 但为了安全起见，我们只记录信息
                    print(f"占用信息: {result.stdout}")
            except Exception as e:
                print(f"检查设备占用时出错: {e}")
        
        # 清理可能残留的GStreamer管道
        if GST_AVAILABLE:
            try:
                # 这里可以添加GStreamer特定的清理逻辑
                pass
            except Exception as e:
                print(f"清理GStreamer资源时出错: {e}")
        
        # 等待一小段时间让系统释放资源
        time.sleep(0.5)
        print("强制资源清理完成")

    def _trigger_offline_reprocess(self):
        """触发离线图片重新处理"""
        # 这个方法会被main_app.py重写，用于通知UI重新处理离线图片
        pass
