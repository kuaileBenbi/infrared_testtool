import os
import sys
import tkinter as tk
import threading
import queue
import time
import traceback
import cv2
import numpy as np
from PIL import Image, ImageTk
from collections import deque
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm

# 配置matplotlib中文字体
def setup_matplotlib_chinese_font():
    """设置matplotlib中文字体"""
    # 获取系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 按优先级选择中文字体，优先选择支持中文的字体
    chinese_candidates = [
        'WenQuanYi Micro Hei',      # 文泉驿微米黑
        'WenQuanYi Zen Hei',        # 文泉驿正黑
        'Noto Sans CJK SC',         # 思源黑体
        'Noto Sans CJK',            # 思源黑体
        'Source Han Sans SC',       # 思源黑体
        'Source Han Sans CN',       # 思源黑体
        'SimHei',                   # 黑体
        'Microsoft YaHei',          # 微软雅黑
        'SimSun',                   # 宋体
        'FangSong',                 # 仿宋
        'KaiTi',                    # 楷体
        'fangsong ti',              # 系统中发现的字体
        'song ti',                  # 系统中发现的字体
        'mincho',                   # 系统中发现的字体
        'clearlyu',                 # 系统中发现的字体
        'DejaVu Sans',
        'Liberation Sans', 
        'Arial Unicode MS'
    ]
    
    # 直接使用已知支持中文的字体，避免测试过程
    selected_font = None
    
    # 优先选择已知支持中文的字体
    known_chinese_fonts = [
        'Noto Sans CJK JP',         # 系统中发现的字体
        'Noto Serif CJK JP',        # 系统中发现的字体
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 
        'Noto Sans CJK SC',
        'Noto Sans CJK',
        'Source Han Sans SC',
        'Source Han Sans CN',
        'SimHei',
        'Microsoft YaHei',
        'SimSun',
        'FangSong',
        'KaiTi'
    ]
    
    for candidate in known_chinese_fonts:
        if candidate in available_fonts:
            selected_font = candidate
            break
    
    # 如果没找到已知的中文字体，尝试系统字体
    if selected_font is None:
        system_fonts = ['fangsong ti', 'song ti', 'mincho', 'clearlyu']
        for candidate in system_fonts:
            if candidate in available_fonts:
                selected_font = candidate
                break
    
    # 最后回退到DejaVu Sans
    if selected_font is None:
        selected_font = 'DejaVu Sans'
        print("警告: 未找到支持中文的字体，使用DejaVu Sans")
    
    # 设置matplotlib字体
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['figure.autolayout'] = True  # 自动调整布局
    
    # 设置字体回退列表，确保中文字符能正确显示
    if selected_font in known_chinese_fonts:
        # 如果选择了已知的中文字体，设置完整的回退列表
        font_fallback = [selected_font, 'DejaVu Sans', 'Arial', 'sans-serif']
    else:
        # 如果选择了系统字体，添加更多中文字体作为回退
        font_fallback = [selected_font, 'Noto Sans CJK JP', 'DejaVu Sans', 'Arial', 'sans-serif']
    
    # 设置字体族和回退列表
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_fallback
    
    print(f"matplotlib使用字体: {selected_font}")
    print(f"字体回退列表: {font_fallback}")
    return selected_font

# 初始化matplotlib字体
setup_matplotlib_chinese_font()

from ui_config import UIConfig
from camera_functions import CameraFunctions


class CameraApp:
    """主相机应用程序类"""

    def __init__(self, root):
        self.root = root

        # 初始化界面配置
        self.ui = UIConfig(root)

        # 初始化相机功能
        self.camera_func = CameraFunctions(self.ui)

        # 数据源状态
        self.data_source_running = False
        self.current_data_source = "camera"

        # 绑定事件
        self.bind_events()

        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 绑定鼠标点击事件
        self.ui.canvas.bind("<Button-1>", self.on_image_click)

        # 放大窗口相关
        self.zoom_window = None
        self.zoom_update_job = None
        self.zoom_roi_coords = None

        # 重写离线图片重新处理方法
        self.camera_func._trigger_offline_reprocess = self.process_offline_image
        
        # 记住上次打开的文件夹路径
        self.last_folder_path = os.getcwd()

        # 设置翻转回调函数
        self.ui.flip_callback = self.on_flip_mode_changed

        # 启动帧率更新
        self.ui.capture_fps_label.after(1000, self._update_fps)

        # 若曾用于时序噪声/FPN，现移除
        self._recent_frames = None

    def bind_events(self):
        """绑定所有按钮事件"""
        # 存储功能
        self.ui.save_background_btn.config(command=self.camera_func.save_background)
        self.ui.save_whiteground_btn.config(command=self.camera_func.save_whiteground)
        self.ui.save_stop_btn.config(command=self.save_stop_image)
        self.ui.auto_save_100_btn.config(command=self.auto_save_images)

        # 校正/盲元功能
        self.ui.non_uniform_0_btn.config(
            command=self.camera_func.non_uniform_correction_0
        )
        self.ui.non_uniform_1_btn.config(
            command=self.camera_func.non_uniform_correction_1
        )
        self.ui.bp_correction_btn.config(command=self.camera_func.bp_correction)
        self.ui.bp_table_compensation_btn.config(
            command=self.camera_func.bp_table_compensation
        )
        self.ui.load_bp_file_btn.config(command=self.load_bp_file)
        self.ui.linear_correction_btn.config(
            command=self.camera_func.linear_multi_point_correction
        )
        self.ui.quadrast_correction_btn.config(
            command=self.camera_func.quadrast_multi_point_correction
        )
        self.ui.dark_white_correction_btn.config(
            command=self.camera_func.dark_white_correction
        )
        self.ui.load_file_btn.config(command=self.load_file)

        # 图像处理功能
        self.ui.show_original_btn.config(command=self.camera_func.show_original)
        self.ui.img_enhance_btn.config(command=self.camera_func.img_enhance)
        self.ui.img_imadjust_btn.config(command=self.camera_func.img_imadjust)
        self.ui.image_sharpen_btn.config(command=self.camera_func.image_sharpen)
        self.ui.img_denoise_btn.config(command=self.camera_func.img_denoise)
        self.ui.img_defog_btn.config(command=self.camera_func.image_defog)
        self.ui.correction_bw_btn.config(command=self.camera_func.correction_bw)
        self.ui.toggle_crosshair_btn.config(command=self.camera_func.toggle_crosshair)
        self.ui.show_histogram_btn.config(command=self.show_histogram)
        self.ui.clear_roi_btn.config(command=self.clear_roi_markers)
        
        # 图像质量评价功能
        self.ui.calculate_clarity_btn.config(command=self.calculate_image_clarity)
        self.ui.calculate_brightness_btn.config(command=self.calculate_image_brightness)
        self.ui.clear_quality_btn.config(command=self.clear_quality_metrics)
        if hasattr(self.ui, 'calculate_quality_btn'):
            self.ui.calculate_quality_btn.config(command=self.calculate_image_quality_metrics)
        
        # 图像拉伸功能
        self.ui.apply_stretch_btn.config(command=self.camera_func.image_stretch)
        self.ui.adaptive_stretch_btn.config(command=self.camera_func.adaptive_stretch)
        self.ui.roi_stretch_btn.config(command=self.on_roi_stretch_click)

        # 数据源控制
        self.ui.start_stop_btn.config(command=self.on_start_stop_click)
        self.ui.on_start_stop_click = self.on_start_stop_click

    def load_file(self):
        """加载校正文件"""
        filepath = filedialog.askopenfilename(
            initialdir=self.last_folder_path,
            title="选择要加载的文件",
            filetypes=[("所有文件", "*.*")],
        )
        if not filepath:
            return  # 用户取消
        
        # 更新记住的文件夹路径
        self.last_folder_path = os.path.dirname(filepath)
        
        print(f"加载校正文件: {filepath}")
        self.camera_func.whonpz = filepath

    def load_bp_file(self):
        """加载坏点文件"""
        filepath = filedialog.askopenfilename(
            initialdir=self.last_folder_path,
            title="选择坏点文件",
            filetypes=[("NPZ文件", "*.npz"), ("所有文件", "*.*")],
        )
        if not filepath:
            return  # 用户取消

        # 更新记住的文件夹路径
        self.last_folder_path = os.path.dirname(filepath)

        self.camera_func.bp_npz_path = filepath
        print(f"成功加载坏点文件: {filepath}")

    def on_roi_stretch_click(self):
        """解析输入并启停区域拉伸"""
        try:
            text = self.ui.roi_stretch_entry.get().strip()
            # 期望格式: x1:x2,y1:y2
            xy = text.split(',')
            x_part = xy[0].split(':')
            y_part = xy[1].split(':')
            x1, x2 = int(x_part[0]), int(x_part[1])
            y1, y2 = int(y_part[0]), int(y_part[1])
            self.camera_func.toggle_roi_stretch((x1, x2, y1, y2))
        except Exception as e:
            print(f"区域拉伸参数错误: '{text}'，示例: 0:256,0:256. 错误: {e}")

    def load_local_image(self):
        """加载本地图像"""
        filepath = filedialog.askopenfilename(
            initialdir=self.last_folder_path,
            title="选择要加载的图像",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("BMP文件", "*.bmp"),
                ("TIFF文件", "*.tiff *.tif"),
                ("所有文件", "*.*"),
            ],
        )
        if not filepath:
            return  # 用户取消

        try:
            # 使用OpenCV加载图像
            image = cv2.imread(filepath, -1)
            # print(image.mean())
            
            # 更新图像统计信息到status_frame
            if hasattr(self.ui, 'image_stats_text'):
                stats_text = f"图像统计:\nmean={image.mean():.1f}\nmin={image.min()}\nmax={image.max()}"
                self.ui.image_stats_text.config(state="normal")
                self.ui.image_stats_text.delete(1.0, tk.END)
                self.ui.image_stats_text.insert(1.0, stats_text)
                self.ui.image_stats_text.config(state="disabled")
            if image is None:
                print(f"无法加载图像: {filepath}")
                return

            # 如果是彩色图像，转换为灰度
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 如果是8位图像，转换为16位
            if image.dtype == np.uint8:
                image = image.astype(np.uint16) * 256
            elif image.dtype != np.uint16:
                image = image.astype(np.uint16)

            # 设置当前帧
            self.camera_func.current_frame = image

            # 设置离线图片模式
            self.camera_func.offline_image_mode = True
            self.camera_func.offline_image = image.copy()

            # 直接进行预处理和显示，不放入队列
            self.process_offline_image()

            # 更新记住的文件夹路径
            self.last_folder_path = os.path.dirname(filepath)
            
            print(f"成功加载本地图像: {filepath}")
            print(f"图像尺寸: {image.shape}")
            print("图像已设置为离线模式，可重复进行校正处理")

        except Exception as e:
            print(f"加载本地图像时出错: {e}")
            import traceback

            traceback.print_exc()

    def process_offline_image(self):
        """处理离线图片"""
        if (
            not hasattr(self.camera_func, "offline_image")
            or self.camera_func.offline_image is None
        ):
            return

        try:
            # 使用原始图片进行预处理
            processed_frame = self.camera_func.preprocess_ctrl(
                self.camera_func.offline_image
            )

            # 更新当前帧
            self.camera_func.current_frame = processed_frame

            # 显示处理后的图像
            display_frame = self.camera_func.scale_to_8bit(processed_frame)
            # 应用翻转
            flip_mode = self.ui.flip_mode.get()
            display_frame = self.camera_func.apply_flip(display_frame, flip_mode)
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)

            # 如果启用十字星，绘制十字星
            if self.camera_func.crosshair_enabled:
                display_rgb = self.camera_func.draw_crosshair(display_rgb)

            # 如果存在ROI标记，则绘制ROI标记
            if (
                hasattr(self, "current_roi_coords")
                and self.current_roi_coords is not None
            ):
                click_x, click_y, left, top, right, bottom = self.current_roi_coords
                cv2.rectangle(display_rgb, (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.circle(display_rgb, (click_x, click_y), 2, (255, 0, 0), -1)

            # 使用新的显示系统更新图像
            pil_img = Image.fromarray(display_rgb)
            self.ui.set_image(pil_img)

            print("离线图片已重新处理")

        except Exception as e:
            print(f"处理离线图片时出错: {e}")
            traceback.print_exc()

    def auto_save_images(self):
        """自动保存指定张数图片"""
        try:
            # 从界面获取保存张数
            save_count = int(self.ui.save_count_entry.get())
            if save_count <= 0:
                print("保存张数必须大于0")
                return

            # 从界面获取保存关键词
            keyword = self.ui.save_keyword_entry.get().strip()
            if not keyword:
                keyword = "auto_images"  # 默认关键词

            print(f"开始自动保存 {save_count} 张图片到文件夹 '{keyword}'，保存间隔 0.03 秒")
            self.camera_func.auto_save_100(save_count, keyword)

        except ValueError:
            print("请输入有效的数字")
        except Exception as e:
            print(f"启动自动保存时出错: {e}")
            traceback.print_exc()

    def on_start_stop_click(self):
        """开始/停止数据源"""
        if not self.data_source_running:
            self.start_data_source()
        else:
            self.stop_data_source()

    def start_data_source(self):
        """开始数据源"""
        try:
            source = self.ui.data_source_var.get()
            self.current_data_source = source

            # 从界面获取参数
            device_num = self.ui.device_num_entry.get()
            band = self.ui.band_var.get()
            bit_max = int(self.ui.bit_max_entry.get())

            # 设置相机功能参数
            self.camera_func.device_num = device_num
            self.camera_func.wave = band
            self.camera_func.bit_max = bit_max

            if source == "camera":
                # 摄像头模式
                self.camera_func.choose_camera_mode("1")  # 使用硬件摄像头
                self.ui.data_source_status.config(text="状态: 摄像头运行中", fg="green")

            elif source == "test_video":
                # 测试视频模式
                self.camera_func.choose_camera_mode("0")  # 使用模拟视频
                self.ui.data_source_status.config(
                    text="状态: 测试视频运行中", fg="green"
                )

            elif source == "offline_file":
                # 离线文件模式
                self.load_local_image()
                self.ui.data_source_status.config(
                    text="状态: 离线文件已加载", fg="blue"
                )
                # 离线文件模式不需要启动帧处理线程，直接处理
                return

            self.data_source_running = True
            self.ui.start_stop_btn.config(text="停止")

            # 启动帧处理线程
            if (
                not hasattr(self, "frame_processing_thread")
                or not self.frame_processing_thread.is_alive()
            ):
                self.start_frame_processing()

        except Exception as e:
            print(f"启动数据源时出错: {e}")
            self.ui.data_source_status.config(text="状态: 启动失败", fg="red")

    def stop_data_source(self):
        """停止数据源"""
        try:
            # 停止相机功能
            self.camera_func.cleanup()

            # 停止数据源运行状态
            self.data_source_running = False

            # 清空队列
            try:
                while not self.camera_func.raw_queue.empty():
                    self.camera_func.raw_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                while not self.camera_func.play_queue.empty():
                    self.camera_func.play_queue.get_nowait()
            except queue.Empty:
                pass

            # 更新UI状态
            self.ui.start_stop_btn.config(text="开始")
            self.ui.data_source_status.config(text="状态: 已停止", fg="red")

            print("数据源已停止，队列已清空")

        except Exception as e:
            print(f"停止数据源时出错: {e}")
            traceback.print_exc()

    def start_frame_processing(self):
        """启动帧处理线程"""

        def frame_processor():
            while self.camera_func.running and self.data_source_running:
                try:
                    # 从原始队列获取帧
                    frame_16bit = self.camera_func.raw_queue.get(timeout=1)

                    # 进行预处理
                    processed_frame = self.camera_func.preprocess_ctrl(frame_16bit)

                    # 将处理后的帧放入播放队列
                    try:
                        self.camera_func.play_queue.put(processed_frame, timeout=0.1)
                    except queue.Full:
                        self.camera_func.play_queue.get(timeout=0.1)
                        self.camera_func.play_queue.put(processed_frame, timeout=0.1)
                    except Exception as e:
                        print(f"处理帧出错:{e},详细内容如下：")
                        print(traceback.format_exc())

                    # 通知UI更新
                    self.root.after(0, self.on_new_frame_event)

                except queue.Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    print(f"帧处理出错: {e}")
                    traceback.print_exc()
                    break

        # 启动帧处理线程
        self.frame_processing_thread = threading.Thread(
            target=frame_processor, daemon=True
        )
        self.frame_processing_thread.start()
        print("帧处理线程已启动")

    def save_stop_image(self):
        """保存/停止图像按钮"""
        if not hasattr(self.camera_func, "manual_saving"):
            self.camera_func.manual_saving = False

        # 切换状态：若当前未保存，则开始保存；反之则停止保存
        if not self.camera_func.manual_saving:
            self.camera_func.manual_saving = True
            # 更新按钮显示文字
            self.ui.save_stop_btn.config(text="停止保存")
            # 启动后台线程保存图像
            t = threading.Thread(
                target=self.camera_func._manual_save_thread, daemon=True
            )
            t.start()
        else:
            self.camera_func.manual_saving = False
            self.ui.save_stop_btn.config(text="保存(停止)图像按钮")

    def on_image_click(self, event):
        """鼠标左键点击 canvas 时获取对应图像中的坐标"""
        if self.camera_func.current_frame is None:
            print("没有可用图像")
            return

        # 获取画布和图像的实际显示信息
        canvas_width = self.ui.canvas.winfo_width()
        canvas_height = self.ui.canvas.winfo_height()

        if self.ui.original_image_size is None:
            print("没有可用的图像尺寸信息")
            return

        img_w, img_h = self.ui.original_image_size

        # 计算图像在画布中的实际显示位置和大小
        if self.ui.display_mode == "fit":
            # 适应模式：计算缩放比例和位置
            scale_x = canvas_width / img_w
            scale_y = canvas_height / img_h
            scale = min(scale_x, scale_y)
            display_width = int(img_w * scale)
            display_height = int(img_h * scale)
            offset_x = (canvas_width - display_width) // 2
            offset_y = (canvas_height - display_height) // 2
        elif self.ui.display_mode == "stretch":
            # 拉伸模式：图像填充整个画布
            scale = 1.0  # 拉伸模式下不需要缩放，但为了代码一致性设置scale
            display_width = canvas_width
            display_height = canvas_height
            offset_x = 0
            offset_y = 0
        else:  # original
            # 原始模式：显示原始大小
            scale = 1.0  # 原始模式下缩放比例为1
            display_width = img_w
            display_height = img_h
            offset_x = (canvas_width - display_width) // 2
            offset_y = (canvas_height - display_height) // 2

        # 将画布坐标转换为图像坐标
        canvas_x = event.x - offset_x
        canvas_y = event.y - offset_y

        # 检查点击是否在图像区域内
        if not (0 <= canvas_x < display_width and 0 <= canvas_y < display_height):
            print(f"点击超出图像范围：({event.x}, {event.y})")
            return

        # 将画布坐标转换为原始图像坐标
        if self.ui.display_mode == "stretch":
            # 拉伸模式：直接映射
            click_x = int(canvas_x * img_w / display_width)
            click_y = int(canvas_y * img_h / display_height)
        else:
            # 其他模式：使用缩放比例
            click_x = int(canvas_x / scale)
            click_y = int(canvas_y / scale)

        # 确保坐标在图像范围内
        click_x = max(0, min(click_x, img_w - 1))
        click_y = max(0, min(click_y, img_h - 1))

        print(f"图像像素坐标: ({click_x}, {click_y})")
        self._show_pixel_response(f"图像像素坐标: ({click_x}, {click_y})", is_roi=1)

        # 获取用户输入的ROI大小，默认为10
        try:
            # 从第四列获取ROI大小设置
            roi_width = int(self.ui.roi_width_entry.get())
            roi_height = int(self.ui.roi_height_entry.get())
            roi_size = max(roi_width, roi_height)  # 使用较大的值作为ROI大小
        except ValueError:
            print("ROI大小输入错误，默认使用10")
            roi_size = 10

        # 获取 ROI 区域的像素值，并确保范围不超过图像边界
        top = max(click_y - roi_size // 2, 0)
        bottom = min(click_y + roi_size // 2, img_h - 1)
        left = max(click_x - roi_size // 2, 0)
        right = min(click_x + roi_size // 2, img_w - 1)
        self.zoom_roi_coords = (top, bottom, left, right)

        roi = self.camera_func.current_frame[top : bottom + 1, left : right + 1]
        print("ROI区域像素值：\n", roi)
        self._show_pixel_response(f"ROI区域像素值：\n {roi}", is_roi=0)
        self._show_pixel_response(
            f"像素区域极大值：{np.max(roi)} \n 像素区域极小值 {np.min(roi)}", is_roi=2
        )

        # 如果放大窗口不存在或已被销毁，则重新创建
        if hasattr(self, "zoom_window") and self.zoom_window is not None:
            if self.zoom_window.winfo_exists():
                self.zoom_window.destroy()
        self.zoom_window = None

        # 创建放大窗口
        self.zoom_window = tk.Toplevel(self.root)
        self.zoom_window.title("放大视图")
        self.zoom_label = tk.Label(self.zoom_window)
        self.zoom_label.pack()

        # 启动定时刷新
        if hasattr(self, "zoom_update_job") and self.zoom_update_job is not None:
            self.root.after_cancel(self.zoom_update_job)
        self.zoom_update_job = None
        self.zoom_update_job = self.root.after(100, self.update_zoom_window)

        # 可视化部分：在图像上绘制ROI标记
        display_frame = self.camera_func.scale_to_8bit(self.camera_func.current_frame)
        # 应用翻转
        flip_mode = self.ui.flip_mode.get()
        display_frame = self.camera_func.apply_flip(display_frame, flip_mode)
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB).copy()

        # 绘制 ROI 边界和中心点
        cv2.rectangle(display_rgb, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.circle(display_rgb, (click_x, click_y), 2, (255, 0, 0), -1)

        # 使用新的显示系统更新图像
        pil_img = Image.fromarray(display_rgb)
        self.ui.set_image(pil_img)

        # 存储当前ROI状态
        self.current_roi_coords = (click_x, click_y, left, top, right, bottom)

    def clear_roi_markers(self):
        """清除ROI标记，显示原图"""
        if self.camera_func.current_frame is None:
            return

        # 显示原图（不绘制ROI标记）
        display_frame = self.camera_func.scale_to_8bit(self.camera_func.current_frame)
        # 应用翻转
        flip_mode = self.ui.flip_mode.get()
        display_frame = self.camera_func.apply_flip(display_frame, flip_mode)
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)

        # 如果启用十字星，绘制十字星
        if self.camera_func.crosshair_enabled:
            display_rgb = self.camera_func.draw_crosshair(display_rgb)

        # 使用新的显示系统更新图像
        pil_img = Image.fromarray(display_rgb)
        self.ui.set_image(pil_img)

        # 清除ROI状态
        self.current_roi_coords = None

        # 清空像素信息显示
        self.ui.pixel_coord_text.delete(1.0, tk.END)
        self.ui.roi_pixel_text.delete(1.0, tk.END)
        self.ui.roi_pixel_value_text.delete(1.0, tk.END)

        print("ROI标记已清除")

    def on_flip_mode_changed(self):
        """翻转模式改变时的回调函数"""
        if self.camera_func.current_frame is not None:
            # 如果有当前帧，重新显示图像
            if self.camera_func.offline_image_mode and self.camera_func.offline_image is not None:
                # 离线图片模式，重新处理图片
                self.process_offline_image()
            else:
                # 实时模式，重新显示当前帧
                self.on_new_frame_event()

    def update_zoom_window(self):
        """更新放大窗口"""
        if (
            not hasattr(self, "zoom_window")
            or not self.zoom_window
            or not self.zoom_window.winfo_exists()
        ):
            return  # 放大窗口已关闭，停止刷新

        if self.camera_func.current_frame is None:
            self.zoom_update_job = self.root.after(100, self.update_zoom_window)
            return

        try:
            top, bottom, left, right = self.zoom_roi_coords
            roi = self.camera_func.current_frame[top : bottom + 1, left : right + 1]

            # 放大 ROI 区域
            roi_8bit = self.camera_func.scale_to_8bit(roi)
            roi_resized = cv2.resize(
                roi_8bit, None, fx=50, fy=50, interpolation=cv2.INTER_NEAREST
            )
            roi_display = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)

            # 绘制灰度值
            for i in range(roi_8bit.shape[0]):
                for j in range(roi_8bit.shape[1]):
                    gray_value = roi[i, j]
                    text_position = (j * 50 + 5, i * 50 + 15)
                    cv2.putText(
                        roi_display,
                        str(int(gray_value)),
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            pil_img = Image.fromarray(roi_display)
            tk_img = ImageTk.PhotoImage(pil_img)

            label = self.zoom_window.children.get("!label")
            if label:
                label.config(image=tk_img)
                label.image = tk_img

            # 下一帧继续刷新
            self.zoom_update_job = self.root.after(100, self.update_zoom_window)

        except Exception as e:
            print(f"更新放大窗口出错: {e}")

    def _show_pixel_response(self, text, is_roi=0):
        """显示像素信息到对应的文本框"""
        if is_roi == 0:
            self.ui.roi_pixel_text.delete(1.0, tk.END)
            self.ui.roi_pixel_text.insert(tk.END, text + "\n")
            self.ui.roi_pixel_text.see(tk.END)
        elif is_roi == 1:
            self.ui.pixel_coord_text.delete(1.0, tk.END)
            self.ui.pixel_coord_text.insert(tk.END, text + "\n")
            self.ui.pixel_coord_text.see(tk.END)
        else:
            self.ui.roi_pixel_value_text.delete(1.0, tk.END)
            self.ui.roi_pixel_value_text.insert(tk.END, text + "\n")
            self.ui.roi_pixel_value_text.see(tk.END)

    def _update_fps(self):
        """每秒在 Tk 界面更新一次帧率"""
        fps = self.camera_func.frame_count
        self.ui.capture_fps_label.config(text=f"采集帧率：{fps} FPS")
        self.camera_func.frame_count = 0
        # 下一次再过 1000 ms 调度自己
        self.ui.capture_fps_label.after(1000, self._update_fps)

    def on_new_frame_event(self):
        """处理新帧事件"""
        try:
            frame_16bit = self.camera_func.play_queue.get_nowait()
            self.camera_func.current_frame = frame_16bit
        except queue.Empty:
            return

        # 1) 手动拉伸到8位
        display_frame = self.camera_func.scale_to_8bit(frame_16bit)
        
        # 2) 应用翻转
        flip_mode = self.ui.flip_mode.get()
        display_frame = self.camera_func.apply_flip(display_frame, flip_mode)

        # 3) 再把灰度转成 RGB 以便 PhotoImage 显示
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)

        # 如果启用，则绘制十字星
        if self.camera_func.crosshair_enabled:
            display_rgb = self.camera_func.draw_crosshair(display_rgb)

        # 如果存在ROI标记，则绘制ROI标记
        if hasattr(self, "current_roi_coords") and self.current_roi_coords is not None:
            click_x, click_y, left, top, right, bottom = self.current_roi_coords
            cv2.rectangle(display_rgb, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.circle(display_rgb, (click_x, click_y), 2, (255, 0, 0), -1)

        # 3) 转为 numpy -> PIL 并使用新的显示系统
        pil_img = Image.fromarray(display_rgb)
        self.ui.set_image(pil_img)

        # 帧率计算：滚动平均
        now = time.time()
        self.camera_func.timestamps.append(now)
        if len(self.camera_func.timestamps) > 1:
            # 在队列中有 >=2 个时刻时才能计算速率
            total_time = (
                self.camera_func.timestamps[-1] - self.camera_func.timestamps[0]
            )
            frame_count = len(self.camera_func.timestamps) - 1
            if total_time > 0:
                fps_calc = frame_count / total_time
                self.ui.display_fps_label.config(text=f"当前帧率：{fps_calc:.2f} FPS")

    def show_histogram(self):
        """显示当前图像的直方图 - 使用matplotlib独立窗口"""
        if self.camera_func.current_frame is None:
            print("没有可用的图像数据")
            return
        
        # 确保matplotlib字体配置正确
        setup_matplotlib_chinese_font()
        
        # 获取当前图像
        image = self.camera_func.current_frame
        
        # 创建matplotlib图形窗口
        plt.ion()  # 开启交互模式
        fig = plt.figure(figsize=(12, 8))
        
        # 创建单个子图用于柱状直方图
        ax = plt.subplot(1, 1, 1)
        
        # 根据图像类型绘制直方图
        if len(image.shape) == 3:
            # 彩色图像 - 绘制彩色直方图
            self._draw_color_histogram_standalone(ax, image)
        else:
            # 灰度图像 - 绘制灰度直方图
            self._draw_gray_histogram_standalone(ax, image)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 为标题留出空间
        
        # 显示matplotlib窗口
        plt.show(block=False)  # 非阻塞显示
        
        print("直方图窗口已打开")

    def _draw_gray_histogram_standalone(self, ax, image,
                                    log_y=True,
                                    clip_tail_for_view=0.005,   # 仅用于显示的两端裁剪比例（0~0.02较常用）
                                    show_kde=False):            # 灰度为离散值时KDE意义有限，默认False
        """
        绘制更符合人眼观察的灰度直方图（独立窗口版）

        Args:
            ax: matplotlib Axes
            image: np.ndarray, 灰度或BGR
            log_y (bool): y轴使用对数刻度，更易看尾部细节
            clip_tail_for_view (float): 仅用于可视化的两端裁剪比例，如0.005表示裁掉上下各0.5%像素的灰度范围
            show_kde (bool): 是否叠加KDE曲线（高分辨率连续灰度时可开）
        """
        # ---- 预处理：转灰度、去NaN/Inf、转为float计算 ----
        if image is None:
            ax.text(0.5, 0.5, "No image", ha='center', va='center', transform=ax.transAxes)
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if (image.ndim == 3) else image
        gray = np.asarray(gray)
        gray = gray[np.isfinite(gray)]
        if gray.size == 0:
            ax.text(0.5, 0.5, "Empty image", ha='center', va='center', transform=ax.transAxes)
            return

        # 尽量保持原始灰度单位（支持10-16bit）
        gray_min, gray_max = float(np.min(gray)), float(np.max(gray))
        dynamic_range = gray_max - gray_min if gray_max > gray_min else 1.0

        # ---- 统计量 ----
        mean_val = float(np.mean(gray))
        std_val  = float(np.std(gray))
        median_val = float(np.median(gray))
        p1, p5, p95, p99 = np.percentile(gray, [1, 5, 95, 99])
        # 众数（用直方图近似）
        # 先用较细分箱估一个峰
        _bins_for_mode = min(1024, max(32, int(np.sqrt(gray.size))))
        hist_tmp, edges_tmp = np.histogram(gray, bins=_bins_for_mode, range=(gray_min, gray_max))
        mode_idx = int(np.argmax(hist_tmp))
        mode_val = 0.5 * (edges_tmp[mode_idx] + edges_tmp[mode_idx + 1])

        # 信息熵（基于归一化直方图）
        p_hist = hist_tmp / np.sum(hist_tmp) if np.sum(hist_tmp) > 0 else hist_tmp
        entropy = float(-np.sum(p_hist[p_hist > 0] * np.log2(p_hist[p_hist > 0])))

        # ---- 仅用于显示的范围裁剪（防止极端值把横轴拉飞）----
        if 0 < clip_tail_for_view < 0.2:
            lo, hi = np.percentile(gray, [clip_tail_for_view * 100, 100 - clip_tail_for_view * 100])
            view_min, view_max = float(lo), float(hi)
            if view_max <= view_min:
                view_min, view_max = gray_min, gray_max
        else:
            view_min, view_max = gray_min, gray_max

        # ---- Freedman–Diaconis 自适应分箱（上限256更易读）----
        q25, q75 = np.percentile(gray, [25, 75])
        iqr = max(q75 - q25, 1e-9)
        bin_width = 2 * iqr * (gray.size ** (-1/3))
        if bin_width <= 0 or not np.isfinite(bin_width):
            # 退化情况：用sqrt(N)策略
            num_bins = int(np.sqrt(gray.size))
        else:
            num_bins = int(np.clip(np.ceil((view_max - view_min) / bin_width), 32, 256))

        hist, edges = np.histogram(gray, bins=num_bins, range=(view_min, view_max))
        centers = 0.5 * (edges[:-1] + edges[1:])

        # ---- 绘图 ----
        ax.clear()

        # 条形图（密度而非绝对计数，更符合比例感；高度再乘以像素总数可换回计数）
        density = hist / (hist.sum() + 1e-12)
        ax.bar(centers, density, align='center', width=(edges[1]-edges[0]), alpha=0.75,
            edgecolor='none')

        # y轴设置
        ax.set_ylabel('Density' + (' (log)' if log_y else ''), fontsize=12, fontweight='bold')
        if log_y:
            ax.set_yscale('log')

        # x轴：使用原始灰度单位
        ax.set_xlabel('Gray level (raw units)', fontsize=12, fontweight='bold')
        ax.set_xlim(view_min, view_max)

        # 叠加 CDF（右侧y轴，百分比）
        cdf = np.cumsum(hist).astype(np.float64)
        cdf /= (cdf[-1] + 1e-12)
        ax2 = ax.twinx()
        ax2.plot(centers, cdf * 100.0, linewidth=2, alpha=0.9)
        ax2.set_ylabel('Cumulative %', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)

        # （可选）KDE，仅在灰度近似连续时有意义
        if show_kde and gray.size >= 5000:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(gray.astype(np.float64))
                xs = np.linspace(view_min, view_max, 512)
                ax.plot(xs, kde(xs), linewidth=1.5, alpha=0.8)
            except Exception:
                pass

        # 垂线标记：1/50/99百分位、均值、中位数、众数
        def _vline(x, color, label):
            ax.axvline(x, color=color, linestyle='--', linewidth=1.8, alpha=0.9)
            ax.text(x, ax.get_ylim()[1], f' {label}', color=color, fontsize=9,
                    va='bottom', ha='left', rotation=90, alpha=0.9)

        _vline(p1,   '#888888', 'P1')
        _vline(median_val, '#2e7d32', 'P50')
        _vline(p99,  '#888888', 'P99')
        _vline(mean_val,   '#d32f2f', 'Mean')
        _vline(mode_val,   '#ef6c00', 'Mode')

        # 网格与标题
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_title('Gray Histogram', fontsize=14, fontweight='bold', pad=16)

        # 统计侧栏（紧凑）
        nonzero_ratio = float(np.count_nonzero(gray)) / float(gray.size)
        stats_text = (
            f"Size: {int(image.shape[1])}×{int(image.shape[0])}\n"
            f"Pixels: {gray.size:,}\n"
            f"Min/Max: {gray_min:.1f}/{gray_max:.1f}\n"
            f"Mean/Std: {mean_val:.2f}/{std_val:.2f}\n"
            f"P1/P50/P99: {p1:.1f}/{median_val:.1f}/{p99:.1f}\n"
            f"Mode≈ {mode_val:.1f}\n"
            f"Entropy: {entropy:.2f} bits\n"
            f"Non-zero: {nonzero_ratio*100:.1f}%\n"
            f"Bins: {num_bins} (view {view_min:.1f}–{view_max:.1f})"
        )
        ax.text(1.02, 0.98, stats_text, transform=ax.transAxes,
                va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.9))


    def _draw_color_histogram_standalone(self, ax, image):
        """绘制彩色直方图 - 独立窗口版本"""
        # 分离BGR通道
        b, g, r = cv2.split(image)
        colors = ['blue', 'green', 'red']
        channels = [b, g, r]
        channel_names = ['蓝色 (B)', '绿色 (G)', '红色 (R)']
        
        # 计算各通道直方图
        hists = []
        for channel in channels:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hists.append(hist.flatten())
        
        # 绘制柱状彩色直方图
        width = 0.8
        x = np.arange(256)
        for i, (hist, color) in enumerate(zip(hists, colors)):
            ax.bar(x + i * width/3, hist, width/3, color=color, alpha=0.7, label=channel_names[i])
        
        ax.set_xlabel('像素值', fontsize=12, fontweight='bold')
        ax.set_ylabel('像素数量', fontsize=12, fontweight='bold')
        ax.set_title('彩色直方图 (柱状)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 255)
        
        # 设置Y轴范围
        max_hist = max([h.max() for h in hists])
        ax.set_ylim(0, max_hist * 1.05)
        
        ax.legend(loc='upper right', fontsize=10)
        
        # 计算彩色图像统计信息
        stats_text = f'''图像统计信息:
        
图像尺寸: {image.shape[1]} × {image.shape[0]} 像素
总像素数: {image.size:,}
通道数: {image.shape[2]}

各通道统计:'''
        
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            min_val = np.min(channel)
            max_val = np.max(channel)
            peak_idx = np.argmax(hists[i])
            
            stats_text += f'''
{name}:
• 范围: {min_val} - {max_val}
• 均值: {mean_val:.2f}
• 标准差: {std_val:.2f}
• 峰值: {peak_idx}'''
        
        ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    def calculate_image_clarity(self):
        """计算图像清晰度指标"""
        if self.camera_func.current_frame is None:
            print("没有可用的图像数据")
            return
        
        try:
            # 获取当前图像
            image = self.camera_func.current_frame
            
            # 确保图像是灰度图
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            
            # 计算拉普拉斯算子（清晰度指标）
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            clarity_score = laplacian.var()
            
            # 计算梯度幅值（另一种清晰度指标）
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_score = gradient_magnitude.mean()
            
            # 更新状态显示（分行显示）
            clarity_text = f"图像清晰度: {gradient_score:.2f}"
            self.ui.image_clarity_label.config(text=clarity_text)
            
            print(f"图像清晰度计算完成: {clarity_text}")
            
        except Exception as e:
            print(f"计算图像清晰度时出错: {e}")
            import traceback
            traceback.print_exc()

    def calculate_image_brightness(self):
        """计算图像亮度指标"""
        if self.camera_func.current_frame is None:
            print("没有可用的图像数据")
            return
        
        try:
            # 获取当前图像和位深参数
            image = self.camera_func.current_frame
            bit_max = getattr(self.camera_func, 'bit_max', 4095)  # 默认12位
            
            # 确保图像是灰度图
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            
            # 计算各种亮度指标（基于实际位深）
            mean_brightness = np.mean(gray_image)
            std_brightness = np.std(gray_image)
            
            # 计算相对亮度（相对于最大可能值）
            relative_mean = (mean_brightness / bit_max) * 100
            relative_std = (std_brightness / bit_max) * 100
            
            # 计算亮度均匀性（基于相对标准差）
            brightness_uniformity = 1.0 / (1.0 + relative_std / relative_mean) if relative_mean > 0 else 0
            
            # 更新状态显示（分行显示）
            brightness_text = f"图像亮度均匀性: {brightness_uniformity:.3f}"
            self.ui.image_brightness_label.config(text=brightness_text)
            
            print(f"图像亮度计算完成: {brightness_text}")
            
        except Exception as e:
            print(f"计算图像亮度时出错: {e}")
            import traceback
            traceback.print_exc()

    def _get_gray_current(self):
        image = self.camera_func.current_frame
        if image is None:
            return None
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    # 已移除时序帧缓存逻辑

    def calculate_image_quality_metrics(self):
        """计算并显示五个图像质量指标"""
        from preprocessor import ImagePreprocessor as IP
        gray = self._get_gray_current()
        if gray is None:
            print("没有可用的图像数据")
            return
        try:
            # 转为float便于频域计算
            g = gray.astype(np.float64)

            half_power = IP.radial_power_half_freq(g)
            stripe = IP.stripe_index(g)
            halo = IP.halo_ratio(g)
            lowfreq = IP.lowfreq_rms_ratio(g)

            # 更新UI显示（滚动文本）
            if hasattr(self.ui, 'quality_metrics_text'):
                lines = [
                    f"频域清晰度(HPF): {half_power:.4f} cyc/px",
                    f"条纹指数: {stripe:.3f}",
                    f"光晕能量比: {halo:.3f}",
                    f"低频起伏比: {lowfreq:.3f}",
                ]
                text = "\n".join(lines)
                self.ui.quality_metrics_text.config(state="normal")
                self.ui.quality_metrics_text.delete(1.0, tk.END)
                self.ui.quality_metrics_text.insert(1.0, text)
                self.ui.quality_metrics_text.config(state="disabled")

            print("图像质量指标计算完成")
        except Exception as e:
            print(f"计算图像质量指标时出错: {e}")
            import traceback
            traceback.print_exc()

    def clear_quality_metrics(self):
        """清除图像质量评价显示"""
        try:
            # 清除清晰度和亮度显示
            self.ui.image_clarity_label.config(text="")
            self.ui.image_brightness_label.config(text="")
            # 清空质量文本面板
            if hasattr(self.ui, 'quality_metrics_text'):
                self.ui.quality_metrics_text.config(state="normal")
                self.ui.quality_metrics_text.delete(1.0, tk.END)
                self.ui.quality_metrics_text.config(state="disabled")
            
            print("图像质量评价显示已清除")
            
        except Exception as e:
            print(f"清除图像质量评价时出错: {e}")

    def on_closing(self):
        """窗口关闭事件"""
        self.camera_func.cleanup()
        try:
            if hasattr(self, "zoom_update_job"):
                self.root.after_cancel(self.zoom_update_job)
        except Exception:
            pass

        self.root.destroy()


def main():
    """主函数"""
    print("相机控制应用程序启动")
    print("所有参数现在通过界面设置:")
    print("- 摄像头序号: 在数据源区域设置")
    print("- 数据源类型: 摄像头/测试视频/离线文件")
    print("- 波段选择: mwir/lwir/swir")
    print("- 位深设置: 默认4095")

    root = tk.Tk()
    app = CameraApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
