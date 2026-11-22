import sys
import cv2
import numpy as np
import psutil
import pygetwindow as gw
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSlider, QSpinBox, 
                            QDoubleSpinBox, QCheckBox, QFileDialog, QGroupBox,
                            QTabWidget, QScrollArea, QListWidget)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap

class DemosaicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("极致紧凑去马赛克工具 (支持进程选择)")
        self.setMinimumSize(800, 600)
        
        # 主控件
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # 主布局
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        
        # 左侧图像显示区域
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 400)
        self.main_layout.addWidget(self.image_display, stretch=2)
        
        # 右侧控制区域
        self.control_area = QScrollArea()
        self.control_area.setWidgetResizable(True)
        self.control_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout(self.control_widget)
        self.control_layout.setContentsMargins(3, 3, 3, 3)
        self.control_layout.setSpacing(3)
        self.control_area.setWidget(self.control_widget)
        self.main_layout.addWidget(self.control_area, stretch=1)
        
        # 初始化UI
        self.init_ui()
        
        # 当前图像
        self.current_image = None
        self.processed_image = None
        self.target_process = None
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.capture_process_window)
    
    def init_ui(self):
        # 文件操作组
        file_group = QGroupBox("输入源选择")
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(3, 3, 3, 3)
        file_layout.setSpacing(3)
        
        # 创建选项卡
        self.input_tabs = QTabWidget()
        self.input_tabs.setDocumentMode(True)  # 紧凑样式
        
        # 文件输入标签页
        file_tab = QWidget()
        file_tab_layout = QVBoxLayout(file_tab)
        file_tab_layout.setContentsMargins(3, 3, 3, 3)
        file_tab_layout.setSpacing(3)
        
        self.load_image_btn = QPushButton("加载图像")
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_video_btn = QPushButton("加载视频帧")
        self.load_video_btn.clicked.connect(self.load_video_frame)
        
        file_tab_layout.addWidget(self.load_image_btn)
        file_tab_layout.addWidget(self.load_video_btn)
        file_tab_layout.addStretch()
        file_tab.setLayout(file_tab_layout)
        self.input_tabs.addTab(file_tab, "文件")
        
        # 进程输入标签页
        process_tab = QWidget()
        process_tab_layout = QVBoxLayout(process_tab)
        process_tab_layout.setContentsMargins(3, 3, 3, 3)
        process_tab_layout.setSpacing(3)
        
        self.process_list = QListWidget()
        self.process_list.setAlternatingRowColors(True)
        self.process_list.itemDoubleClicked.connect(self.select_process)
        
        self.refresh_process_btn = QPushButton("刷新进程列表")
        self.refresh_process_btn.clicked.connect(self.refresh_process_list)
        
        self.capture_btn = QPushButton("开始捕获")
        self.capture_btn.setCheckable(True)
        self.capture_btn.toggled.connect(self.toggle_capture)
        
        process_tab_layout.addWidget(self.process_list)
        process_tab_layout.addWidget(self.refresh_process_btn)
        process_tab_layout.addWidget(self.capture_btn)
        process_tab_layout.addStretch()
        process_tab.setLayout(process_tab_layout)
        self.input_tabs.addTab(process_tab, "进程")
        
        file_layout.addWidget(self.input_tabs)
        file_group.setLayout(file_layout)
        self.control_layout.addWidget(file_group)
        
        # 保存按钮
        self.save_image_btn = QPushButton("保存结果")
        self.save_image_btn.clicked.connect(self.save_image)
        self.control_layout.addWidget(self.save_image_btn)
        
        # 算法选择组
        algo_group = QGroupBox("去马赛克算法")
        algo_layout = QVBoxLayout()
        algo_layout.setContentsMargins(3, 3, 3, 3)
        algo_layout.setSpacing(3)
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "传统插值法", 
            "边缘导向滤波", 
            "深度学习(ESRGAN)", 
            "深度学习(GFPGAN)", 
            "深度学习(DeOldify)"
        ])
        
        algo_layout.addWidget(self.algo_combo)
        algo_group.setLayout(algo_layout)
        self.control_layout.addWidget(algo_group)
        
        # 参数调节组
        param_group = QGroupBox("算法参数")
        param_layout = QVBoxLayout()
        param_layout.setContentsMargins(3, 3, 3, 3)
        param_layout.setSpacing(3)
        
        # 通用参数
        self.strength_slider = self.create_slider_control("强度", 0, 100, 50, param_layout)
        self.detail_slider = self.create_slider_control("细节保留", 0, 100, 70, param_layout)
        
        # 高级参数
        self.advanced_check = QCheckBox("显示高级参数")
        self.advanced_check.stateChanged.connect(self.toggle_advanced_params)
        param_layout.addWidget(self.advanced_check)
        
        # 高级参数容器
        self.advanced_params = QWidget()
        self.advanced_layout = QVBoxLayout(self.advanced_params)
        self.advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_layout.setSpacing(3)
        
        # 添加一些示例高级参数
        self.noise_reduction = self.create_slider_control("降噪", 0, 100, 30, self.advanced_layout)
        self.edge_enhance = self.create_slider_control("边缘增强", 0, 100, 40, self.advanced_layout)
        self.color_correction = self.create_checkbox_control("自动色彩校正", True, self.advanced_layout)
        
        param_layout.addWidget(self.advanced_params)
        self.advanced_params.hide()
        
        param_group.setLayout(param_layout)
        self.control_layout.addWidget(param_group)
        
        # 处理按钮
        self.process_btn = QPushButton("执行去马赛克")
        self.process_btn.clicked.connect(self.process_image)
        self.control_layout.addWidget(self.process_btn)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 10px; color: #666;")
        self.control_layout.addWidget(self.status_label)
        
        # 填充剩余空间
        self.control_layout.addStretch()
        
        # 初始刷新进程列表
        self.refresh_process_list()
    
    def create_slider_control(self, label, min_val, max_val, default_val, parent_layout):
        """创建紧凑的滑块控件组"""
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(3)
        
        label_widget = QLabel(label)
        label_widget.setFixedWidth(60)
        control_layout.addWidget(label_widget)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        control_layout.addWidget(slider, stretch=1)
        
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setFixedWidth(50)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        control_layout.addWidget(spin)
        
        parent_layout.addLayout(control_layout)
        return slider
    
    def create_checkbox_control(self, label, default_state, parent_layout):
        """创建紧凑的复选框控件"""
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(3)
        
        checkbox = QCheckBox(label)
        checkbox.setChecked(default_state)
        control_layout.addWidget(checkbox)
        
        parent_layout.addLayout(control_layout)
        return checkbox
    
    def toggle_advanced_params(self):
        """切换高级参数的显示状态"""
        if self.advanced_check.isChecked():
            self.advanced_params.show()
        else:
            self.advanced_params.hide()
    
    def refresh_process_list(self):
        """刷新进程列表"""
        self.process_list.clear()
        processes = []
        
        # 获取所有窗口进程
        try:
            windows = gw.getAllTitles()
            for i, title in enumerate(windows):
                if title:  # 过滤掉空标题
                    self.process_list.addItem(f"{i+1}. {title}")
        except Exception as e:
            self.status_label.setText(f"获取窗口列表错误: {str(e)}")
    
    def select_process(self, item):
        """选择目标进程"""
        process_name = item.text().split(". ", 1)[1]
        self.target_process = process_name
        self.status_label.setText(f"已选择: {process_name}")
    
    def toggle_capture(self, checked):
        """切换窗口捕获状态"""
        if checked:
            if self.target_process:
                self.capture_btn.setText("停止捕获")
                self.capture_timer.start(200)  # 每200ms捕获一次
                self.status_label.setText(f"正在捕获: {self.target_process}")
            else:
                self.capture_btn.setChecked(False)
                self.status_label.setText("错误: 请先选择目标进程")
        else:
            self.capture_btn.setText("开始捕获")
            self.capture_timer.stop()
            self.status_label.setText("捕获已停止")
    
    def capture_process_window(self):
        """捕获目标进程窗口"""
        try:
            # 查找目标窗口
            target_window = gw.getWindowsWithTitle(self.target_process)
            if target_window:
                win = target_window[0]
                
                # 激活窗口以确保捕获
                win.activate()
                
                # 获取窗口位置和大小
                x, y, w, h = win.left, win.top, win.width, win.height
                
                # 使用PIL捕获窗口
                import pyautogui
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
                
                # 转换为OpenCV格式
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                self.current_image = img
                self.display_image(img)
            else:
                self.status_label.setText(f"错误: 找不到窗口 {self.target_process}")
                self.capture_btn.setChecked(False)
                self.capture_timer.stop()
        except Exception as e:
            self.status_label.setText(f"捕获错误: {str(e)}")
            self.capture_btn.setChecked(False)
            self.capture_timer.stop()
    
    def load_image(self):
        """加载图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.display_image(self.current_image)
                self.status_label.setText(f"已加载: {file_path}")
            else:
                self.status_label.setText("错误: 无法加载图像")
    
    def load_video_frame(self):
        """从视频中加载帧"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.current_image = frame
                    self.display_image(self.current_image)
                    self.status_label.setText(f"已加载视频帧: {file_path}")
                else:
                    self.status_label.setText("错误: 无法读取视频帧")
                cap.release()
            else:
                self.status_label.setText("错误: 无法打开视频文件")
    
    def save_image(self):
        """保存处理后的图像"""
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", 
                "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg)"
            )
            
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                self.status_label.setText(f"已保存: {file_path}")
    
    def display_image(self, image):
        """在GUI中显示图像"""
        if image is not None:
            # 转换为RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应显示区域
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            
            # 创建QImage并显示
            q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # 缩放以适应显示区域
            display_size = self.image_display.size()
            scaled_pixmap = pixmap.scaled(
                display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.image_display.setPixmap(scaled_pixmap)
    
    def process_image(self):
        """执行去马赛克处理"""
        if self.current_image is None:
            self.status_label.setText("错误: 请先加载图像")
            return
        
        self.status_label.setText("处理中...")
        QApplication.processEvents()  # 更新UI
        
        try:
            # 这里应该调用实际的去马赛克算法
            # 示例: 使用简单的插值方法
            strength = self.strength_slider.value() / 100.0
            detail = self.detail_slider.value() / 100.0
            
            # 模拟处理 - 实际应用中应替换为真正的去马赛克算法
            self.processed_image = self.simulate_demosaic(self.current_image, strength, detail)
            
            # 显示结果
            self.display_image(self.processed_image)
            self.status_label.setText("处理完成")
        except Exception as e:
            self.status_label.setText(f"错误: {str(e)}")
    
    def simulate_demosaic(self, image, strength, detail):
        """模拟去马赛克效果 - 实际应用中应替换为真正的算法"""
        # 示例: 使用高斯模糊和边缘增强来模拟
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=10*(1-strength))
        
        # 保留细节
        edges = cv2.Laplacian(image, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 混合结果
        result = cv2.addWeighted(blurred, 1.0 - detail, edges, detail, 0)
        
        return result
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图像"""
        super().resizeEvent(event)
        if self.current_image is not None:
            self.display_image(self.current_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemosaicApp()
    window.show()
    sys.exit(app.exec_())