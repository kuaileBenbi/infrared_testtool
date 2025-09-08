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

from ui_config import UIConfig
from camera_functions import CameraFunctions


class CameraApp:
    """主相机应用程序类"""

    def __init__(self, root):
        self.root = root

        # 初始化界面配置
        self.ui = UIConfig(root)

        # 初始化相机功能
        self.camera_func = CameraFunctions()

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

        # 设置翻转回调函数
        self.ui.flip_callback = self.on_flip_mode_changed

        # 启动帧率更新
        self.ui.capture_fps_label.after(1000, self._update_fps)

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
        self.ui.correction_bw_btn.config(command=self.camera_func.correction_bw)
        self.ui.toggle_crosshair_btn.config(command=self.camera_func.toggle_crosshair)
        self.ui.clear_roi_btn.config(command=self.clear_roi_markers)

        # 数据源控制
        self.ui.start_stop_btn.config(command=self.on_start_stop_click)
        self.ui.on_start_stop_click = self.on_start_stop_click

    def load_file(self):
        """加载校正文件"""
        filepath = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="选择要加载的文件",
            filetypes=[("所有文件", "*.*")],
        )
        if not filepath:
            return  # 用户取消
        print(f"加载校正文件: {filepath}")
        self.camera_func.whonpz = filepath

    def load_bp_file(self):
        """加载坏点文件"""
        filepath = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="选择坏点文件",
            filetypes=[("NPZ文件", "*.npz"), ("所有文件", "*.*")],
        )
        if not filepath:
            return  # 用户取消

        self.camera_func.bp_npz_path = filepath
        print(f"成功加载坏点文件: {filepath}")

    def load_local_image(self):
        """加载本地图像"""
        filepath = filedialog.askopenfilename(
            initialdir=os.getcwd(),
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
            print(image.mean())
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

            print(f"开始自动保存 {save_count} 张图片，保存间隔 0.03 秒")
            self.camera_func.auto_save_100(save_count)

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
            display_width = canvas_width
            display_height = canvas_height
            offset_x = 0
            offset_y = 0
        else:  # original
            # 原始模式：显示原始大小
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
