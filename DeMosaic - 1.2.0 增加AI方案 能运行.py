import sys
import cv2
import numpy as np
import psutil
import pygetwindow as gw
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSlider, QSpinBox, 
                            QDoubleSpinBox, QCheckBox, QFileDialog, QGroupBox,
                            QTabWidget, QScrollArea, QListWidget, QProgressBar)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import torch
from PIL import Image
import requests
import io

class AIWorker(QThread):
    """AI处理线程"""
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, algorithm, image, params):
        super().__init__()
        self.algorithm = algorithm
        self.image = image
        self.params = params
        self.running = True

    def run(self):
        try:
            result = None
            if self.algorithm == "ESRGAN":
                result = self.process_esrgan()
            elif self.algorithm == "GFPGAN":
                result = self.process_gfpgan()
            elif self.algorithm == "DeOldify":
                result = self.process_deoldify()
            elif self.algorithm == "Waifu2x":
                result = self.process_waifu2x()
            elif self.algorithm == "Real-ESRGAN":
                result = self.process_real_esrgan()
            elif self.algorithm == "BSRGAN":
                result = self.process_bsrgan()
            elif self.algorithm == "SwinIR":
                result = self.process_swinir()
            
            if result is not None and self.running:
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"AI处理错误: {str(e)}")

    def process_esrgan(self):
        """使用ESRGAN处理图像"""
        self.progress.emit(10)
        # 这里应该是实际的ESRGAN模型调用
        # 示例中使用简单的超分辨率代替
        img = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.progress.emit(100)
        return img

    def process_gfpgan(self):
        """使用GFPGAN处理图像"""
        self.progress.emit(10)
        # 这里应该是实际的GFPGAN模型调用
        # 示例中使用简单的面部增强代替
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                face = self.image[y:y+h, x:x+w]
                face = cv2.bilateralFilter(face, 9, 75, 75)
                self.image[y:y+h, x:x+w] = face
        self.progress.emit(100)
        return self.image

    def process_deoldify(self):
        """使用DeOldify处理图像"""
        self.progress.emit(10)
        # 这里应该是实际的DeOldify模型调用
        # 示例中使用简单的色彩增强代替
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.progress.emit(100)
        return result

    def process_waifu2x(self):
        """使用Waifu2x处理图像"""
        self.progress.emit(10)
        # 这里应该是实际的Waifu2x模型调用
        # 示例中使用简单的降噪和超分辨率代替
        img = cv2.fastNlMeansDenoisingColored(self.image, None, 10, 10, 7, 21)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.progress.emit(100)
        return img

    def process_real_esrgan(self):
        """使用Real-ESRGAN处理图像"""
        self.progress.emit(10)
        # 这里应该是实际的Real-ESRGAN模型调用
        img = cv2.detailEnhance(self.image, sigma_s=10, sigma_r=0.15)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        self.progress.emit(100)
        return img

    def process_bsrgan(self):
        """使用BSRGAN处理图像"""
        self.progress.emit(10)
        # 这里应该是实际的BSRGAN模型调用
        img = cv2.edgePreservingFilter(self.image, flags=1, sigma_s=60, sigma_r=0.4)
        self.progress.emit(100)
        return img

    def process_swinir(self):
        """使用SwinIR处理图像"""
        self.progress.emit(10)
        # 这里应该是实际的SwinIR模型调用
        img = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        self.progress.emit(100)
        return img

    def stop(self):
        self.running = False

class DemosaicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI去马赛克工具箱")
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
        self.ai_worker = None
    
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
        
        # AI算法选择组
        ai_group = QGroupBox("AI去马赛克方案")
        ai_layout = QVBoxLayout()
        ai_layout.setContentsMargins(3, 3, 3, 3)
        ai_layout.setSpacing(3)
        
        self.ai_combo = QComboBox()
        self.ai_combo.addItems([
            "ESRGAN (通用超分辨率)",
            "GFPGAN (面部修复)",
            "DeOldify (色彩恢复)",
            "Waifu2x (动漫风格)",
            "Real-ESRGAN (真实图像增强)",
            "BSRGAN (盲超分辨率)",
            "SwinIR (基于Transformer)"
        ])
        
        ai_layout.addWidget(self.ai_combo)
        ai_group.setLayout(ai_layout)
        self.control_layout.addWidget(ai_group)
        
        # 参数调节组
        param_group = QGroupBox("AI参数调节")
        param_layout = QVBoxLayout()
        param_layout.setContentsMargins(3, 3, 3, 3)
        param_layout.setSpacing(3)
        
        # 通用参数
        self.scale_slider = self.create_slider_control("放大倍数", 1, 4, 2, param_layout)
        self.denoise_slider = self.create_slider_control("降噪强度", 0, 100, 50, param_layout)
        
        # 高级参数
        self.advanced_check = QCheckBox("显示高级参数")
        self.advanced_check.stateChanged.connect(self.toggle_advanced_params)
        param_layout.addWidget(self.advanced_check)
        
        # 高级参数容器
        self.advanced_params = QWidget()
        self.advanced_layout = QVBoxLayout(self.advanced_params)
        self.advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_layout.setSpacing(3)
        
        # 添加高级参数
        self.face_enhance = self.create_checkbox_control("面部增强", True, self.advanced_layout)
        self.color_correction = self.create_checkbox_control("色彩校正", True, self.advanced_layout)
        self.edge_enhance = self.create_slider_control("边缘锐化", 0, 100, 40, self.advanced_layout)
        self.texture_preserve = self.create_slider_control("纹理保留", 0, 100, 60, self.advanced_layout)
        
        param_layout.addWidget(self.advanced_params)
        self.advanced_params.hide()
        
        param_group.setLayout(param_layout)
        self.control_layout.addWidget(param_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.control_layout.addWidget(self.progress_bar)
        
        # 处理按钮
        self.process_btn = QPushButton("执行AI去马赛克")
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
        spin.setFixedWidth(40)
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
        """执行AI去马赛克处理"""
        if self.current_image is None:
            self.status_label.setText("错误: 请先加载图像")
            return
        
        if self.ai_worker is not None and self.ai_worker.isRunning():
            self.ai_worker.stop()
            self.ai_worker.wait()
        
        # 获取选择的算法
        algorithm_map = {
            0: "ESRGAN",
            1: "GFPGAN",
            2: "DeOldify",
            3: "Waifu2x",
            4: "Real-ESRGAN",
            5: "BSRGAN",
            6: "SwinIR"
        }
        algorithm = algorithm_map.get(self.ai_combo.currentIndex(), "ESRGAN")
        
        # 准备参数
        params = {
            "scale": self.scale_slider.value(),
            "denoise": self.denoise_slider.value(),
            "face_enhance": self.face_enhance.isChecked(),
            "color_correction": self.color_correction.isChecked(),
            "edge_enhance": self.edge_enhance.value(),
            "texture_preserve": self.texture_preserve.value()
        }
        
        self.status_label.setText(f"正在使用 {algorithm} 处理...")
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        
        # 创建并启动工作线程
        self.ai_worker = AIWorker(algorithm, self.current_image, params)
        self.ai_worker.finished.connect(self.on_ai_finished)
        self.ai_worker.progress.connect(self.progress_bar.setValue)
        self.ai_worker.error.connect(self.status_label.setText)
        self.ai_worker.start()
    
    def on_ai_finished(self, result):
        """AI处理完成"""
        self.processed_image = result
        self.display_image(result)
        self.status_label.setText("AI处理完成")
        self.progress_bar.setValue(100)
    
    def closeEvent(self, event):
        """窗口关闭时停止所有线程"""
        if self.ai_worker is not None and self.ai_worker.isRunning():
            self.ai_worker.stop()
            self.ai_worker.wait()
        if self.capture_timer.isActive():
            self.capture_timer.stop()
        event.accept()
    
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