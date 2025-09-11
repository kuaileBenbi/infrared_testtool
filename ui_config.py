import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from PIL import Image, ImageTk


class UIConfig:
    """界面配置类，负责创建和管理所有UI组件"""

    def __init__(self, root):
        self.root = root
        self.root.title("Camera Control Panel")

        # 字体配置系统
        self.setup_fonts()

        # 创建主容器
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建各个区域
        self.create_left_panel()
        self.create_center_panel()
        
        # 设置LabelFrame字体
        self.setup_labelframe_fonts()

    def setup_fonts(self):
        """设置字体系统"""
        # 基础字体
        self.default_font = tkfont.nametofont("TkDefaultFont")
        self.status_font = tkfont.nametofont("TkDefaultFont")
        self.label_font = tkfont.nametofont("TkDefaultFont")
        self.button_font = tkfont.nametofont("TkDefaultFont")
        self.text_font = tkfont.nametofont("TkDefaultFont")
        self.small_font = tkfont.nametofont("TkDefaultFont")
        self.labelframe_font = tkfont.nametofont("TkDefaultFont")
        self.title_font = tkfont.nametofont("TkDefaultFont")
        
        # # 获取系统中可用的字体
        # available_fonts = tkfont.families()
        
        # # 选择合适的中文字体
        # chinese_font = self._find_best_chinese_font(available_fonts)
        # monospace_font = self._find_best_monospace_font(available_fonts)
        # chinese_font = "gothic"

        # """
        #             "fangsong ti",  # 系统中发现的字体
        #     "song ti",      # 系统中发现的字体
        #     "mincho",       # 系统中发现的字体
        #     "clearlyu",     # 系统中发现的字体
        #     "gothic",       # 系统中发现的字体
        #     "fixed",        # 默认等宽字体
        # """
        
        # print(f"使用中文字体: {chinese_font}")
        # print(f"使用等宽字体: {monospace_font}")
        
        # # 定义不同用途的字体
        # # 标题字体 - 稍大一些，用于重要标签
        # self.title_font = tkfont.Font(
        #     family=chinese_font,
        #     size=11,
        #     weight="bold"
        # )
        
        # # 按钮字体 - 清晰易读
        # self.button_font = tkfont.Font(
        #     family=chinese_font,
        #     size=9,
        #     weight="normal"
        # )
        
        # # 标签字体 - 中等大小
        # self.label_font = tkfont.Font(
        #     family=chinese_font,
        #     size=9,
        #     weight="normal"
        # )
        
        # # 文本输入/文本显示字体 - 优先中文等宽，否则使用中文字体
        # best_mono_cn = self._find_best_monospace_chinese_font(available_fonts)
        # self.text_font = tkfont.Font(
        #     family=best_mono_cn if best_mono_cn else chinese_font,
        #     size=9,
        #     weight="normal"
        # )
        
        # # 状态字体 - 用于状态显示
        # self.status_font = tkfont.Font(
        #     family=chinese_font,
        #     size=9,
        #     weight="normal"
        # )
        
        # # 小字体 - 用于辅助信息
        # self.small_font = tkfont.Font(
        #     family=chinese_font,
        #     size=8,
        #     weight="normal"
        # )
        
        # # LabelFrame标题字体
        # self.labelframe_font = tkfont.Font(
        #     family=chinese_font,
        #     size=10,
        #     weight="bold"
        # )
    
    def _find_best_chinese_font(self, available_fonts):
        """查找最佳的中文字体"""
        # 按优先级排序的中文字体候选
        chinese_candidates = [
            "fangsong ti",  # 系统中发现的字体
            "song ti",      # 系统中发现的字体
            "mincho",       # 系统中发现的字体
            "clearlyu",     # 系统中发现的字体
            "gothic",       # 系统中发现的字体
            "fixed",        # 默认等宽字体
        ]
        
        for candidate in chinese_candidates:
            if candidate in available_fonts:
                return candidate
        
        # 如果没有找到合适的字体，使用默认字体
        return "fixed"
    
    def _find_best_monospace_font(self, available_fonts):
        """查找最佳的等宽字体"""
        # 按优先级排序的等宽字体候选
        monospace_candidates = [
            "courier 10 pitch",  # 系统中发现的字体
            "fixed",             # 默认等宽字体
            "clearlyu",          # 系统中发现的字体
        ]
        
        for candidate in monospace_candidates:
            if candidate in available_fonts:
                return candidate
        
        # 如果没有找到合适的字体，使用默认字体
        return "fixed"

    def _find_best_monospace_chinese_font(self, available_fonts):
        """查找最佳的支持中文的等宽字体"""
        candidates = [
            "WenQuanYi Micro Hei Mono",
            "Noto Sans Mono CJK SC",
            "Noto Sans Mono CJK",
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "Source Han Sans CN",
            "Source Han Mono SC",
            "SimSun-ExtB",
            "FangSong",
            "fangsong ti",
            "song ti",
            "mincho",
            "clearlyu"
        ]
        for name in candidates:
            if name in available_fonts:
                return name
        return None

    def setup_labelframe_fonts(self):
        """设置所有LabelFrame的字体"""
        # 获取所有LabelFrame并设置字体
        for widget in self.root.winfo_children():
            self._set_labelframe_fonts_recursive(widget)
    
    def _set_labelframe_fonts_recursive(self, widget):
        """递归设置LabelFrame字体"""
        if isinstance(widget, tk.LabelFrame):
            widget.config(font=self.labelframe_font)
        
        # 递归处理子组件
        for child in widget.winfo_children():
            self._set_labelframe_fonts_recursive(child)

    def create_left_panel(self):
        """创建左侧图像显示面板"""
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建图像显示容器
        self.image_container = tk.Frame(self.left_frame)
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 图像显示画布 - 使用自适应大小
        self.canvas = tk.Canvas(self.image_container, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定窗口大小变化事件
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # 图像显示相关变量
        self.current_image = None
        self.original_image_size = None
        self.display_mode = "fit"  # fit, stretch, original

        # 帧率显示
        fps_frame = tk.Frame(self.left_frame)
        fps_frame.pack(fill=tk.X, pady=5)

        self.display_fps_label = tk.Label(fps_frame, text="显示帧率：0 FPS", font=self.status_font)
        self.display_fps_label.pack(side=tk.LEFT, padx=10)

        # 添加显示模式控制按钮
        self.create_display_controls(fps_frame)
        
        # 添加翻转控制按钮
        self.create_flip_controls(fps_frame)

    def create_display_controls(self, parent_frame):
        """创建显示控制按钮"""
        control_frame = tk.Frame(parent_frame)
        control_frame.pack(side=tk.RIGHT, padx=10)

        # 显示模式标签
        tk.Label(control_frame, text="显示模式:", font=self.label_font).pack(side=tk.LEFT)

        # 显示模式按钮
        self.fit_btn = tk.Button(
            control_frame,
            text="适应",
            width=6,
            font=self.button_font,
            command=lambda: self.set_display_mode("fit"),
        )
        self.fit_btn.pack(side=tk.LEFT, padx=2)

        self.stretch_btn = tk.Button(
            control_frame,
            text="拉伸",
            width=6,
            font=self.button_font,
            command=lambda: self.set_display_mode("stretch"),
        )
        self.stretch_btn.pack(side=tk.LEFT, padx=2)

        self.original_btn = tk.Button(
            control_frame,
            text="原始",
            width=6,
            font=self.button_font,
            command=lambda: self.set_display_mode("original"),
        )
        self.original_btn.pack(side=tk.LEFT, padx=2)

        # 默认选中适应模式
        self.fit_btn.config(relief=tk.SUNKEN)

    def set_display_mode(self, mode):
        """设置显示模式"""
        self.display_mode = mode

        # 更新按钮状态
        for btn in [self.fit_btn, self.stretch_btn, self.original_btn]:
            btn.config(relief=tk.RAISED)

        if mode == "fit":
            self.fit_btn.config(relief=tk.SUNKEN)
        elif mode == "stretch":
            self.stretch_btn.config(relief=tk.SUNKEN)
        elif mode == "original":
            self.original_btn.config(relief=tk.SUNKEN)

        # 重新显示当前图像
        if self.current_image is not None:
            self.update_image_display()

    def create_flip_controls(self, parent_frame):
        """创建翻转控制按钮"""
        flip_frame = tk.Frame(parent_frame)
        flip_frame.pack(side=tk.RIGHT, padx=10)
        
        # 翻转模式标签
        tk.Label(flip_frame, text="翻转:", font=self.label_font).pack(side=tk.LEFT)
        
        # 翻转模式变量
        self.flip_mode = tk.StringVar(value="none")
        
        # 翻转按钮
        self.no_flip_btn = tk.Button(
            flip_frame,
            text="无",
            width=4,
            font=self.button_font,
            command=lambda: self.set_flip_mode("none"),
        )
        self.no_flip_btn.pack(side=tk.LEFT, padx=2)
        
        self.horizontal_flip_btn = tk.Button(
            flip_frame,
            text="水平",
            width=4,
            font=self.button_font,
            command=lambda: self.set_flip_mode("horizontal"),
        )
        self.horizontal_flip_btn.pack(side=tk.LEFT, padx=2)
        
        self.vertical_flip_btn = tk.Button(
            flip_frame,
            text="垂直",
            width=4,
            font=self.button_font,
            command=lambda: self.set_flip_mode("vertical"),
        )
        self.vertical_flip_btn.pack(side=tk.LEFT, padx=2)
        
        self.both_flip_btn = tk.Button(
            flip_frame,
            text="双向",
            width=4,
            font=self.button_font,
            command=lambda: self.set_flip_mode("both"),
        )
        self.both_flip_btn.pack(side=tk.LEFT, padx=2)
        
        # 默认选中无翻转
        self.no_flip_btn.config(relief=tk.SUNKEN)

    def set_flip_mode(self, mode):
        """设置翻转模式"""
        self.flip_mode.set(mode)
        
        # 更新按钮状态
        for btn in [self.no_flip_btn, self.horizontal_flip_btn, self.vertical_flip_btn, self.both_flip_btn]:
            btn.config(relief=tk.RAISED)
        
        if mode == "none":
            self.no_flip_btn.config(relief=tk.SUNKEN)
        elif mode == "horizontal":
            self.horizontal_flip_btn.config(relief=tk.SUNKEN)
        elif mode == "vertical":
            self.vertical_flip_btn.config(relief=tk.SUNKEN)
        elif mode == "both":
            self.both_flip_btn.config(relief=tk.SUNKEN)
        
        # 如果有翻转回调函数，则调用它
        if hasattr(self, 'flip_callback') and self.flip_callback:
            self.flip_callback()

    def on_canvas_resize(self, event):
        """画布大小变化时的处理"""
        if self.current_image is not None:
            self.update_image_display()

    def update_image_display(self):
        """更新图像显示"""
        if self.current_image is None:
            return

        # 获取画布当前大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return  # 画布还未完全初始化

        # 获取原始图像大小
        img_width, img_height = self.original_image_size

        # 根据显示模式计算显示尺寸
        if self.display_mode == "fit":
            # 适应模式：保持宽高比，适应画布
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y)
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)

        elif self.display_mode == "stretch":
            # 拉伸模式：填充整个画布
            display_width = canvas_width
            display_height = canvas_height

        else:  # original
            # 原始模式：显示原始大小
            display_width = img_width
            display_height = img_height

        # 调整图像大小
        resized_image = self.current_image.resize(
            (display_width, display_height), Image.LANCZOS
        )

        # 转换为PhotoImage
        self.photo_image = ImageTk.PhotoImage(resized_image)

        # 清除画布并显示新图像
        self.canvas.delete("all")

        # 计算图像在画布中的位置（居中显示）
        x = (canvas_width - display_width) // 2
        y = (canvas_height - display_height) // 2

        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image)

    def set_image(self, image):
        """设置要显示的图像"""
        if image is not None:
            self.current_image = image
            self.original_image_size = image.size
            self.update_image_display()

    def create_center_panel(self):
        """创建中间控制面板"""
        # 创建中间容器
        self.center_container = tk.Frame(self.main_frame)
        self.center_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=10)

        # 第一列：数据源和存储控制
        self.control_col = tk.Frame(self.center_container)
        self.control_col.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # 数据源选择
        self.create_data_source_frame()

        # 存储控制
        self.create_storage_frame()

        # 第二列：校正/盲元和校正计算
        self.correction_col = tk.Frame(self.center_container)
        self.correction_col.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # 校正/盲元控制
        self.create_correction_frame()

        # 第三列：基础处理和图像增强
        self.basic_processing_col = tk.Frame(self.center_container)
        self.basic_processing_col.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # 基础处理控制
        self.create_basic_processing_frame()

        # 第四列：图像像素坐标平移等（固定列宽）
        self.pixel_control_col = tk.Frame(self.center_container, width=250)
        self.pixel_control_col.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.pixel_control_col.pack_propagate(False)  # 固定列宽

        # 像素控制
        self.create_pixel_control_frame()

    def create_data_source_frame(self):
        """创建数据源选择框架"""
        data_source_frame = tk.LabelFrame(
            self.control_col, text="数据源", padx=5, pady=5
        )
        data_source_frame.pack(fill=tk.X, pady=5)

        # 数据源选择变量
        self.data_source_var = tk.StringVar(value="camera")

        # 创建单选按钮
        self.camera_radio = tk.Radiobutton(
            data_source_frame,
            text="摄像头",
            variable=self.data_source_var,
            value="camera",
            font=self.label_font,
            command=self.on_data_source_change,
        )
        self.camera_radio.pack(anchor=tk.W, pady=2)

        self.test_video_radio = tk.Radiobutton(
            data_source_frame,
            text="测试视频",
            variable=self.data_source_var,
            value="test_video",
            font=self.label_font,
            command=self.on_data_source_change,
        )
        self.test_video_radio.pack(anchor=tk.W, pady=2)

        self.offline_file_radio = tk.Radiobutton(
            data_source_frame,
            text="离线文件",
            variable=self.data_source_var,
            value="offline_file",
            font=self.label_font,
            command=self.on_data_source_change,
        )
        self.offline_file_radio.pack(anchor=tk.W, pady=2)

        # 设备号输入框（仅摄像头模式）
        device_frame = tk.Frame(data_source_frame)
        device_frame.pack(fill=tk.X, pady=2)

        tk.Label(device_frame, text="设备号:", font=self.label_font).pack(side=tk.LEFT)
        self.device_num_entry = tk.Entry(device_frame, width=5, font=self.text_font)
        self.device_num_entry.pack(side=tk.LEFT, padx=5)
        self.device_num_entry.insert(0, "0")

        # 波段选择
        band_frame = tk.Frame(data_source_frame)
        band_frame.pack(fill=tk.X, pady=2)

        tk.Label(band_frame, text="波段:", font=self.label_font).pack(side=tk.LEFT)
        self.band_var = tk.StringVar(value="mwir")
        band_combo = tk.OptionMenu(band_frame, self.band_var, "mwir", "lwir", "swir")
        band_combo.config(font=self.label_font)
        band_combo.pack(side=tk.LEFT, padx=5)

        # 位深设置
        bit_frame = tk.Frame(data_source_frame)
        bit_frame.pack(fill=tk.X, pady=2)

        tk.Label(bit_frame, text="位深:", font=self.label_font).pack(side=tk.LEFT)
        self.bit_max_entry = tk.Entry(bit_frame, width=8, font=self.text_font)
        self.bit_max_entry.pack(side=tk.LEFT, padx=5)
        self.bit_max_entry.insert(0, "4095")

        # 开始/停止按钮
        self.start_stop_btn = tk.Button(
            data_source_frame, text="开始", font=self.button_font, command=self.on_start_stop_click
        )
        self.start_stop_btn.pack(fill=tk.X, pady=2)


    def on_data_source_change(self):
        """数据源改变时的处理"""
        source = self.data_source_var.get()
        if source == "camera":
            self.device_num_entry.config(state=tk.NORMAL)
        else:
            self.device_num_entry.config(state=tk.DISABLED)

    def on_start_stop_click(self):
        """开始/停止按钮点击处理"""
        # 这个方法将在主应用程序中重写
        pass

    def create_storage_frame(self):
        """创建存储控制框架"""
        storage_frame = tk.LabelFrame(self.control_col, text="存储", padx=5, pady=5)
        storage_frame.pack(fill=tk.X, pady=5)

        self.save_background_btn = tk.Button(storage_frame, text="保存背景图", font=self.button_font)
        self.save_background_btn.pack(fill=tk.X, pady=2)

        self.save_whiteground_btn = tk.Button(storage_frame, text="保存亮场图", font=self.button_font)
        self.save_whiteground_btn.pack(fill=tk.X, pady=2)

        # 保存张数输入
        save_count_frame = tk.Frame(storage_frame)
        save_count_frame.pack(fill=tk.X, pady=2)

        tk.Label(save_count_frame, text="保存张数:", font=self.label_font).pack(side=tk.LEFT)
        self.save_count_entry = tk.Entry(save_count_frame, width=8, font=self.text_font)
        self.save_count_entry.pack(side=tk.LEFT, padx=5)
        self.save_count_entry.insert(0, "100")  # 默认100张

        self.auto_save_100_btn = tk.Button(storage_frame, text="自动保存指定张数", font=self.button_font)
        self.auto_save_100_btn.pack(fill=tk.X, pady=2)

        self.save_stop_btn = tk.Button(storage_frame, text="连续保存(停止)图像", font=self.button_font)
        self.save_stop_btn.pack(fill=tk.X, pady=2)

    def create_correction_frame(self):
        """创建校正/盲元控制框架"""
        correction_frame = tk.LabelFrame(
            self.correction_col, text="校正/盲元", padx=5, pady=5
        )
        correction_frame.pack(fill=tk.X, pady=5)

        self.non_uniform_0_btn = tk.Button(correction_frame, text="单点校正(b-)", font=self.button_font)
        self.non_uniform_0_btn.pack(fill=tk.X, pady=2)

        self.non_uniform_1_btn = tk.Button(correction_frame, text="单点校正(-b)", font=self.button_font)
        self.non_uniform_1_btn.pack(fill=tk.X, pady=2)

        self.bp_correction_btn = tk.Button(correction_frame, text="坏点检测", font=self.button_font)
        self.bp_correction_btn.pack(fill=tk.X, pady=2)

        self.bp_table_compensation_btn = tk.Button(correction_frame, text="查表补偿", font=self.button_font)
        self.bp_table_compensation_btn.pack(fill=tk.X, pady=2)

        self.load_bp_file_btn = tk.Button(correction_frame, text="加载坏点文件", font=self.button_font)
        self.load_bp_file_btn.pack(fill=tk.X, pady=2)

        self.linear_correction_btn = tk.Button(correction_frame, text="线性校正", font=self.button_font)
        self.linear_correction_btn.pack(fill=tk.X, pady=2)

        self.quadrast_correction_btn = tk.Button(correction_frame, text="非线性校正", font=self.button_font)
        self.quadrast_correction_btn.pack(fill=tk.X, pady=2)

        self.dark_white_correction_btn = tk.Button(correction_frame, text="明暗校正", font=self.button_font)
        self.dark_white_correction_btn.pack(fill=tk.X, pady=2)

        self.load_file_btn = tk.Button(correction_frame, text="加载校正文件", font=self.button_font)
        self.load_file_btn.pack(fill=tk.X, pady=2)

        # 校正计算
        correction_calc_frame = tk.LabelFrame(
            self.correction_col, text="校正计算", padx=5, pady=5
        )
        correction_calc_frame.pack(fill=tk.X, pady=5)

        self.correction_bw_btn = tk.Button(correction_calc_frame, text="计算黑白场校正", font=self.button_font)
        self.correction_bw_btn.pack(fill=tk.X, pady=2)

    def create_basic_processing_frame(self):
        """创建基础处理控制框架"""
        # 基础处理
        basic_frame = tk.LabelFrame(
            self.basic_processing_col, text="基础处理", padx=5, pady=5
        )
        basic_frame.pack(fill=tk.X, pady=5)

        self.show_original_btn = tk.Button(basic_frame, text="显示原图", font=self.button_font)
        self.show_original_btn.pack(fill=tk.X, pady=2)

        self.img_imadjust_btn = tk.Button(basic_frame, text="灰度拉伸", font=self.button_font)
        self.img_imadjust_btn.pack(fill=tk.X, pady=2)

        self.toggle_crosshair_btn = tk.Button(basic_frame, text="开关十字星", font=self.button_font)
        self.toggle_crosshair_btn.pack(fill=tk.X, pady=2)
        
        self.show_histogram_btn = tk.Button(basic_frame, text="显示直方图", font=self.button_font)
        self.show_histogram_btn.pack(fill=tk.X, pady=2)

        # 图像质量评价
        quality_frame = tk.LabelFrame(
            self.basic_processing_col, text="图像质量评价", padx=5, pady=5
        )
        quality_frame.pack(fill=tk.X, pady=5)

        self.calculate_clarity_btn = tk.Button(quality_frame, text="计算清晰度", font=self.button_font)
        self.calculate_clarity_btn.pack(fill=tk.X, pady=2)

        self.calculate_brightness_btn = tk.Button(quality_frame, text="计算亮度", font=self.button_font)
        self.calculate_brightness_btn.pack(fill=tk.X, pady=2)

        self.clear_quality_btn = tk.Button(quality_frame, text="清除评价", font=self.button_font)
        self.clear_quality_btn.pack(fill=tk.X, pady=2)

        # 图像拉伸功能
        stretch_frame = tk.LabelFrame(
            self.basic_processing_col, text="图像拉伸", padx=5, pady=5
        )
        stretch_frame.pack(fill=tk.X, pady=5)

        # 拉伸级别选择
        level_frame = tk.Frame(stretch_frame)
        level_frame.pack(fill=tk.X, pady=2)
        tk.Label(level_frame, text="拉伸级别:", font=self.label_font).pack(side=tk.LEFT)
        self.stretch_level_var = tk.StringVar(value="off")
        level_combo = ttk.Combobox(level_frame, textvariable=self.stretch_level_var, 
                                  values=["off", "light", "medium", "strong"], 
                                  state="readonly", width=10, font=self.text_font)
        level_combo.pack(side=tk.LEFT, padx=5)

        # 下采样参数
        downsample_frame = tk.Frame(stretch_frame)
        downsample_frame.pack(fill=tk.X, pady=2)
        tk.Label(downsample_frame, text="下采样:", font=self.label_font).pack(side=tk.LEFT)
        self.downsample_var = tk.StringVar(value="1")
        downsample_combo = ttk.Combobox(downsample_frame, textvariable=self.downsample_var,
                                       values=["1", "2", "4"], state="readonly", 
                                       width=5, font=self.text_font)
        downsample_combo.pack(side=tk.LEFT, padx=5)

        # 中值滤波核大小
        median_frame = tk.Frame(stretch_frame)
        median_frame.pack(fill=tk.X, pady=2)
        tk.Label(median_frame, text="中值核:", font=self.label_font).pack(side=tk.LEFT)
        self.median_ksize_var = tk.StringVar(value="3")
        median_combo = ttk.Combobox(median_frame, textvariable=self.median_ksize_var,
                                   values=["1", "3", "5", "7"], state="readonly", 
                                   width=5, font=self.text_font)
        median_combo.pack(side=tk.LEFT, padx=5)

        # 应用拉伸按钮
        self.apply_stretch_btn = tk.Button(stretch_frame, text="应用拉伸", font=self.button_font)
        self.apply_stretch_btn.pack(fill=tk.X, pady=2)

        # 图像增强
        enhance_frame = tk.LabelFrame(
            self.basic_processing_col, text="图像增强", padx=5, pady=5
        )
        enhance_frame.pack(fill=tk.X, pady=5)

        self.img_enhance_btn = tk.Button(enhance_frame, text="图像增强", font=self.button_font)
        self.img_enhance_btn.pack(fill=tk.X, pady=2)

        self.image_sharpen_btn = tk.Button(enhance_frame, text="图像锐化", font=self.button_font)
        self.image_sharpen_btn.pack(fill=tk.X, pady=2)

        self.img_denoise_btn = tk.Button(enhance_frame, text="图像去噪", font=self.button_font)
        self.img_denoise_btn.pack(fill=tk.X, pady=2)

    def create_pixel_control_frame(self):
        """创建像素控制框架"""
        # ROI控制
        roi_frame = tk.LabelFrame(
            self.pixel_control_col, text="ROI控制", padx=5, pady=5
        )
        roi_frame.pack(fill=tk.X, pady=5)

        self.clear_roi_btn = tk.Button(roi_frame, text="清除ROI标记", font=self.button_font)
        self.clear_roi_btn.pack(fill=tk.X, pady=2)

        # ROI大小设置
        roi_size_frame = tk.LabelFrame(
            self.pixel_control_col, text="ROI大小设置", padx=5, pady=5
        )
        roi_size_frame.pack(fill=tk.X, pady=5)

        # ROI宽度设置
        width_frame = tk.Frame(roi_size_frame)
        width_frame.pack(fill=tk.X, pady=2)
        tk.Label(width_frame, text="宽度:", font=self.label_font).pack(side=tk.LEFT)
        self.roi_width_entry = tk.Entry(width_frame, width=8, font=self.text_font)
        self.roi_width_entry.pack(side=tk.LEFT, padx=5)
        self.roi_width_entry.insert(0, "10")  # 默认宽度10

        # ROI高度设置
        height_frame = tk.Frame(roi_size_frame)
        height_frame.pack(fill=tk.X, pady=2)
        tk.Label(height_frame, text="高度:", font=self.label_font).pack(side=tk.LEFT)
        self.roi_height_entry = tk.Entry(height_frame, width=8, font=self.text_font)
        self.roi_height_entry.pack(side=tk.LEFT, padx=5)
        self.roi_height_entry.insert(0, "10")  # 默认高度10

        # 像素坐标显示
        pixel_coord_frame = tk.LabelFrame(
            self.pixel_control_col, text="像素坐标", padx=5, pady=5
        )
        pixel_coord_frame.pack(fill=tk.X, pady=5)

        self.pixel_coord_text = tk.Text(
            pixel_coord_frame,
            height=4,
            font=self.text_font,
            wrap="none",
            relief="solid",
            bd=1,
        )
        self.pixel_coord_text.pack(fill=tk.X, padx=5, pady=2)

        # ROI像素4显示
        roi_pixel_frame = tk.LabelFrame(
            self.pixel_control_col, text="ROI像素值", padx=5, pady=5
        )
        roi_pixel_frame.pack(fill=tk.X, pady=5)

        self.roi_pixel_text = tk.Text(
            roi_pixel_frame,
            height=6,
            font=self.text_font,
            wrap="none",
            relief="solid",
            bd=1,
        )
        self.roi_pixel_text.pack(fill=tk.X, padx=5, pady=2)

        # ROI像素极值显示
        roi_value_frame = tk.LabelFrame(
            self.pixel_control_col, text="ROI像素极值", padx=5, pady=5
        )
        roi_value_frame.pack(fill=tk.X, pady=5)

        self.roi_pixel_value_text = tk.Text(
            roi_value_frame,
            height=3,
            font=self.text_font,
            wrap="none",
            relief="solid",
            bd=1,
        )
        self.roi_pixel_value_text.pack(fill=tk.X, padx=5, pady=2)

        # 状态信息显示
        status_frame = tk.LabelFrame(
            self.pixel_control_col, text="状态信息", padx=5, pady=5
        )
        status_frame.pack(fill=tk.X, pady=5)

        # 数据源状态
        self.data_source_status = tk.Label(
            status_frame, text="状态: 未启动", fg="red", font=self.status_font, anchor="w"
        )
        self.data_source_status.pack(fill=tk.X, pady=2)

        # 帧率显示
        self.capture_fps_label = tk.Label(
            status_frame, text="采集帧率: 0 FPS", font=self.status_font, anchor="w"
        )
        self.capture_fps_label.pack(fill=tk.X, pady=2)

        # 图像统计信息显示
        self.image_stats_label = tk.Label(
            status_frame, text="图像统计: 等待数据...", font=self.status_font, anchor="w"
        )
        self.image_stats_label.pack(fill=tk.X, pady=2)

        # 图像清晰度显示
        self.image_clarity_label = tk.Label(
            status_frame, text="", font=self.status_font, fg="blue", anchor="w"
        )
        self.image_clarity_label.pack(fill=tk.X, pady=2)

        # 图像亮度显示
        self.image_brightness_label = tk.Label(
            status_frame, text="", font=self.status_font, fg="green", anchor="w"
        )
        self.image_brightness_label.pack(fill=tk.X, pady=2)