import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
from torchvision.transforms import ToTensor
import psutil
import threading
from enum import Enum
import hashlib
import shutil
import urllib.request
from pathlib import Path
import traceback
import json
import requests
from packaging import version

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 权重文件和模型配置
WEIGHTS_DIR = "weights"
CONFIG_URL = "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/model_zoo.json"  # 示例配置URL

class AIModels(Enum):
    REAL_ESRGAN = "Real-ESRGAN (通用增强)"
    REAL_ESRGAN_V3 = "Real-ESRGAN-v3 (最新通用)"
    GFPGAN = "GFPGAN (人脸修复)"
    SWINIR = "SwinIR (Transformer增强)"
    AUTO = "自动选择最佳模型"

class ModelManager:
    @staticmethod
    def get_latest_config():
        """获取最新的模型配置"""
        try:
            response = requests.get(CONFIG_URL, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"无法获取最新配置: {str(e)}")
            # 返回本地默认配置
            return {
                "RealESRGAN_x4plus.pth": {
                    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                    "md5": "99ec365d4afad750833258a1a24f44ca",
                    "version": "0.1.0"
                },
                "realesr-general-x4v3.pth": {
                    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                    "md5": "91a7644643c884ee00737db24e478156",
                    "version": "0.2.5.0"
                },
                "GFPGANv1.4.pth": {
                    "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                    "md5": "94d735072630ab734561130a47bc44f8",
                    "version": "1.3.0"
                },
                "SwinIR_4x.pth": {
                    "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
                    "md5": "7dcd77eb5ff67057581203b267ef3c23",
                    "version": "0.0"
                }
            }

    @staticmethod
    def ensure_weights():
        """确保所有权重文件存在且有效，自动更新到最新版本"""
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        failed_weights = []
        
        # 获取最新配置
        latest_config = ModelManager.get_latest_config()
        
        for filename, config in latest_config.items():
            filepath = os.path.join(WEIGHTS_DIR, filename)
            local_version = "0.0.0"
            
            # 检查本地文件是否存在
            if os.path.exists(filepath):
                # 检查本地版本信息
                version_file = filepath + ".version"
                if os.path.exists(version_file):
                    with open(version_file, 'r') as f:
                        local_version = f.read().strip()
                
                # 比较版本
                if version.parse(local_version) >= version.parse(config.get("version", "0.0.0")):
                    # 验证文件MD5
                    if ModelManager.verify_file(filepath, config.get("md5", "")):
                        print(f"权重文件已验证且为最新版: {filename}")
                        continue
                
                # 备份旧文件
                backup_path = filepath + ".old"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                shutil.move(filepath, backup_path)
                print(f"备份旧权重文件: {filename} -> {backup_path}")
            
            # 下载最新权重文件
            print(f"开始下载权重文件: {filename} (版本: {config.get('version', '未知')})")
            try:
                ModelManager.download_file(config["url"], filepath, config.get("md5", ""))
                
                # 保存版本信息
                with open(filepath + ".version", 'w') as f:
                    f.write(config.get("version", "0.0.0"))
                
                print(f"成功下载权重文件: {filename}")
            except Exception as e:
                print(f"下载权重文件失败: {filename}, 错误: {str(e)}")
                failed_weights.append(filename)
                continue
        
        if failed_weights:
            raise RuntimeError(f"以下权重文件下载失败: {', '.join(failed_weights)}")

    @staticmethod
    def verify_file(filepath, expected_md5):
        """验证文件MD5哈希"""
        if not os.path.exists(filepath):
            return False
            
        # 如果MD5为空则跳过验证
        if not expected_md5:
            return True
            
        try:
            md5_hash = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
                    
            return md5_hash.hexdigest() == expected_md5
        except Exception:
            return False

    @staticmethod
    def download_file(url, filepath, expected_md5=None):
        """下载文件并验证"""
        temp_path = filepath + ".download"
        
        # 下载文件
        def update_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r下载进度: {percent}%", end="", flush=True)
        
        try:
            urllib.request.urlretrieve(url, temp_path, reporthook=update_progress)
            print("\n下载完成")
            
            # 验证文件
            if expected_md5 and not ModelManager.verify_file(temp_path, expected_md5):
                os.remove(temp_path)
                raise ValueError("下载文件MD5校验失败")
                
            # 移动临时文件到目标位置
            if os.path.exists(filepath):
                os.remove(filepath)
            shutil.move(temp_path, filepath)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

class ImageProcessor:
    def __init__(self):
        # 确保权重文件存在且为最新
        try:
            print("正在检查权重文件...")
            ModelManager.ensure_weights()
            print("权重文件检查完成")
        except Exception as e:
            print("权重文件错误", f"无法获取权重文件: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("权重文件错误", f"无法获取权重文件: {str(e)}\n{traceback.format_exc()}")
            raise
        
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """加载真实AI模型"""
        try:
            # Real-ESRGAN模型 (v0.1)
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                model_path = os.path.join(WEIGHTS_DIR, 'RealESRGAN_x4plus.pth')
                model = RRDBNet(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    num_feat=64, 
                    num_block=23, 
                    num_grow_ch=32, 
                    scale=4
                )
                
                if torch.cuda.is_available():
                    loadnet = torch.load(model_path)
                else:
                    loadnet = torch.load(model_path, map_location=torch.device('cpu'))
                
                if 'params' in loadnet:
                    model.load_state_dict(loadnet['params'], strict=True)
                elif 'params_ema' in loadnet:
                    model.load_state_dict(loadnet['params_ema'], strict=True)
                else:
                    model.load_state_dict(loadnet, strict=True)
                
                model.eval()
                model = model.to(device)
            
                self.models['REAL_ESRGAN'] = RealESRGANer(
                    scale=4,
                    model_path=model_path,
                    dni_weight=None,
                    device=device,
                    model=model,
                    tile=0
                )
            except Exception as e:
                print(f"加载Real-ESRGAN失败: {str(e)}")
        
            # Real-ESRGAN v3模型 (特殊处理)
            try:
                model_path = os.path.join(WEIGHTS_DIR, 'realesr-general-x4v3.pth')
                
                # 直接使用RealESRGANer自动加载（推荐方式）
                self.models['REAL_ESRGAN_V3'] = RealESRGANer(
                    scale=4,
                    model_path=model_path,
                    dni_weight=None,
                    device=device,
                    tile=0,
                    model=None,  # 让加载器自动处理模型
                    pre_pad=0
                )
                
                # 备用方案：如果自动加载失败，尝试手动加载
                if not hasattr(self.models['REAL_ESRGAN_V3'], 'model'):
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    
                    if torch.cuda.is_available():
                        loadnet = torch.load(model_path)
                    else:
                        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
                    
                    model = RRDBNet(
                        num_in_ch=3, 
                        num_out_ch=3, 
                        num_feat=64, 
                        num_block=23, 
                        num_grow_ch=32, 
                        scale=4
                    )
                    
                    # 宽松加载模式
                    if 'params_ema' in loadnet:
                        model.load_state_dict(loadnet['params_ema'], strict=False)
                    elif 'params' in loadnet:
                        model.load_state_dict(loadnet['params'], strict=False)
                    else:
                        model.load_state_dict(loadnet, strict=False)
                    
                    model.eval()
                    model = model.to(device)
                    
                    self.models['REAL_ESRGAN_V3'] = RealESRGANer(
                        scale=4,
                        model_path=model_path,
                        dni_weight=None,
                        device=device,
                        model=model,
                        tile=0
                    )
                    
            except Exception as e:
                print(f"加载Real-ESRGAN-v3失败: {str(e)}")
            
            # GFPGAN模型
            try:
                from gfpgan import GFPGANer
                self.models['GFPGAN'] = GFPGANer(
                    model_path=os.path.join(WEIGHTS_DIR, 'GFPGANv1.4.pth'),
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=device
                )
            except Exception as e:
                print(f"加载GFPGAN失败: {str(e)}")
        
            # SwinIR模型
            try:
                from swinir import SwinIR
                swinir_model_path = os.path.join(WEIGHTS_DIR, 'SwinIR_4x.pth')
                self.models['SWINIR'] = SwinIR(
                    upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            
                if torch.cuda.is_available():
                    self.models['SWINIR'].load_state_dict(torch.load(swinir_model_path)['params'])
                else:
                    self.models['SWINIR'].load_state_dict(torch.load(swinir_model_path, map_location=torch.device('cpu'))['params'])
            
                self.models['SWINIR'].to(device).eval()
            except Exception as e:
                print(f"加载SwinIR失败: {str(e)}")
            
            print("所有模型加载完成")
        except Exception as e:
            print("模型加载错误", f"无法加载AI模型: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("模型加载错误", f"无法加载AI模型: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def detect_best_model(self, img):
        """自动检测最佳模型"""
        try:
            # 简单检测图像是否包含人脸
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                print("检测到人脸，推荐使用GFPGAN")
                return 'GFPGAN'
            
            # 如果没有检测到人脸，使用最新通用模型
            if 'REAL_ESRGAN_V3' in self.models:
                print("未检测到人脸，推荐使用Real-ESRGAN-v3")
                return 'REAL_ESRGAN_V3'
            elif 'REAL_ESRGAN' in self.models:
                print("未检测到人脸，推荐使用Real-ESRGAN")
                return 'REAL_ESRGAN'
            elif 'SWINIR' in self.models:
                print("未检测到人脸，推荐使用SwinIR")
                return 'SWINIR'
            else:
                print("没有可用模型")
                return None
        except Exception as e:
            print(f"自动检测模型失败: {str(e)}")
            return 'REAL_ESRGAN_V3'  # 默认返回最新通用模型
    
    @staticmethod
    def load_image(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("无法加载图像文件")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    
    @staticmethod
    def save_image(img, path):
        cv2.imwrite(path, img)
    
    def process_image(self, img, model_name, scale=2, fidelity=0.7, progress_callback=None):
        """使用真实AI模型处理图像"""
        try:
            if progress_callback:
                progress_callback(10, "开始处理...")
            
            # 自动选择模型
            if model_name == "AUTO":
                model_name = self.detect_best_model(img)
                if model_name is None:
                    raise RuntimeError("无法自动选择模型")
                progress_callback(20, f"自动选择模型: {model_name}")
            
            if model_name == "REAL_ESRGAN":
                if progress_callback:
                    progress_callback(30, "使用Real-ESRGAN处理中...")
                output, _ = self.models['REAL_ESRGAN'].enhance(img, outscale=scale)
                return output
            
            elif model_name == "REAL_ESRGAN_V3":
                if progress_callback:
                    progress_callback(30, "使用Real-ESRGAN-v3处理中...")
                output, _ = self.models['REAL_ESRGAN_V3'].enhance(img, outscale=scale)
                return output
            
            elif model_name == "GFPGAN":
                if progress_callback:
                    progress_callback(30, "使用GFPGAN处理中...")
                _, _, output = self.models['GFPGAN'].enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True, weight=fidelity)
                return output
            
            elif model_name == "SWINIR":
                if progress_callback:
                    progress_callback(30, "使用SwinIR处理中...")
                img_tensor = ToTensor()(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = self.models['SWINIR'](img_tensor)
                output = output.squeeze().permute(1, 2, 0).cpu().numpy()
                output = np.clip(output * 255, 0, 255).astype(np.uint8)
                if scale > 1:
                    h, w = img.shape[:2]
                    output = cv2.resize(output, (w*scale, h*scale))
                return output
            
            return img
        except Exception as e:
            raise RuntimeError(f"AI处理失败: {str(e)}\n{traceback.format_exc()}")

class AIEnhanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI图像去马赛克工具 (智能版)")
        
        # 初始化变量
        self.image_path = tk.StringVar()
        self.process_var = tk.StringVar()
        self.output_path = tk.StringVar()
        self.ai_model = tk.StringVar(value=AIModels.AUTO.value)
        self.params = {
            "scale": tk.IntVar(value=2),
            "fidelity": tk.DoubleVar(value=0.7),
            "tile": tk.IntVar(value=0)
        }
        
        # 初始化图像处理器
        self.processor = None
        try:
            self.processor = ImageProcessor()
        except Exception as e:
            print("初始化错误", f"无法初始化图像处理器: {str(e)}")
            messagebox.showerror("初始化错误", f"无法初始化图像处理器: {str(e)}")
            self.root.destroy()
            return
        
        # 创建界面
        self.create_ui()
        
    def create_ui(self):
        # 主容器 - 左右分割
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(control_frame, weight=1)
        
        # 右侧图像预览面板
        preview_frame = ttk.Frame(main_paned, padding="10")
        main_paned.add(preview_frame, weight=2)
        
        # 控制面板内容
        self.create_control_panel(control_frame)
        
        # 图像预览面板
        self.create_preview_panel(preview_frame)
        
    def create_control_panel(self, parent):
        # 输入区域
        input_frame = ttk.LabelFrame(parent, text="输入源", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        # 文件选择
        ttk.Label(input_frame, text="图像路径:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.image_path, width=30).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="浏览", command=self.browse_file, width=8).grid(row=0, column=2, padx=5, pady=5)
        
        # 进程选择
        ttk.Label(input_frame, text="目标进程:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.process_combo = ttk.Combobox(input_frame, textvariable=self.process_var, width=28)
        self.process_combo.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="刷新进程", command=self.refresh_processes, width=8).grid(row=1, column=2, padx=5, pady=5)
        self.refresh_processes()
        
        # AI模型选择区域
        model_frame = ttk.LabelFrame(parent, text="AI模型选择", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        # 主模型选择
        ttk.Label(model_frame, text="选择模型:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        model_values = [m.value for m in AIModels]
        ttk.Combobox(model_frame, textvariable=self.ai_model, values=model_values, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        # 参数区域
        param_frame = ttk.LabelFrame(parent, text="处理参数", padding="10")
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="缩放倍数:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(param_frame, from_=1, to=4, textvariable=self.params["scale"], width=5).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(param_frame, text="保真度:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.params["fidelity"], orient="horizontal", length=150).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(param_frame, text="分块大小:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(param_frame, from_=0, to=512, increment=64, textvariable=self.params["tile"], width=5).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # 输出区域
        output_frame = ttk.LabelFrame(parent, text="输出设置", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="输出路径:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_path, width=30).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(output_frame, text="浏览", command=self.browse_output, width=8).grid(row=0, column=2, padx=5, pady=5)
        
        # 进度条
        self.progress_frame = ttk.LabelFrame(parent, text="处理进度", padding="10")
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="准备就绪")
        self.progress_label.pack(pady=5)
        
        # 操作按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="加载图像", command=self.load_image).pack(side="left", padx=5)
        ttk.Button(button_frame, text="处理图像", command=self.process_image).pack(side="left", padx=5)
        ttk.Button(button_frame, text="保存结果", command=self.save_result).pack(side="left", padx=5)
        ttk.Button(button_frame, text="检查更新", command=self.check_updates).pack(side="left", padx=5)
        ttk.Button(button_frame, text="关于", command=self.show_about).pack(side="right", padx=5)
    
    def create_preview_panel(self, parent):
        # 使用PanedWindow垂直分割预览区域
        preview_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        preview_paned.pack(fill=tk.BOTH, expand=True)
        
        # 原图预览 - 使用Frame和Canvas实现可滚动区域
        orig_frame = ttk.LabelFrame(preview_paned, text="原始图像", padding="5")
        preview_paned.add(orig_frame, weight=1)
        
        self.original_canvas = tk.Canvas(orig_frame, bg='gray')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        orig_vscroll = ttk.Scrollbar(orig_frame, orient="vertical", command=self.original_canvas.yview)
        orig_hscroll = ttk.Scrollbar(orig_frame, orient="horizontal", command=self.original_canvas.xview)
        self.original_canvas.configure(yscrollcommand=orig_vscroll.set, xscrollcommand=orig_hscroll.set)
        
        orig_vscroll.pack(side="right", fill="y")
        orig_hscroll.pack(side="bottom", fill="x")
        self.original_canvas.pack(side="left", fill="both", expand=True)
        
        # 处理后图像预览
        proc_frame = ttk.LabelFrame(preview_paned, text="处理后图像", padding="5")
        preview_paned.add(proc_frame, weight=1)
        
        self.processed_canvas = tk.Canvas(proc_frame, bg='gray')
        
        # 添加滚动条
        proc_vscroll = ttk.Scrollbar(proc_frame, orient="vertical", command=self.processed_canvas.yview)
        proc_hscroll = ttk.Scrollbar(proc_frame, orient="horizontal", command=self.processed_canvas.xview)
        self.processed_canvas.configure(yscrollcommand=proc_vscroll.set, xscrollcommand=proc_hscroll.set)
        
        proc_vscroll.pack(side="right", fill="y")
        proc_hscroll.pack(side="bottom", fill="x")
        self.processed_canvas.pack(side="left", fill="both", expand=True)
        
        # 绑定鼠标滚轮缩放
        self.original_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.processed_canvas.bind("<MouseWheel>", self.on_mousewheel)
        
        # 存储图像缩放比例
        self.image_scale = 1.0
    
    def on_mousewheel(self, event):
        """鼠标滚轮缩放图像"""
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.image_scale *= scale_factor
        self.image_scale = max(0.1, min(self.image_scale, 5.0))  # 限制缩放范围
        
        if hasattr(self, 'original_img_tk'):
            self.show_image_on_canvas(self.original_img_tk, self.original_canvas, scale=self.image_scale)
        if hasattr(self, 'processed_img_tk'):
            self.show_image_on_canvas(self.processed_img_tk, self.processed_canvas, scale=self.image_scale)
    
    def update_progress(self, value, message):
        """更新进度条和状态信息"""
        self.progress_bar['value'] = value
        self.progress_label['text'] = message
        self.root.update_idletasks()
    
    def show_image_on_canvas(self, img_tk, canvas, scale=1.0):
        """在Canvas上显示图像，支持缩放和滚动"""
        canvas.delete("all")
        
        # 计算缩放后的尺寸
        img_width = int(img_tk.width() * scale)
        img_height = int(img_tk.height() * scale)
        
        # 创建缩放后的图像
        pil_img = ImageTk.getimage(img_tk)
        resized_img = pil_img.resize((img_width, img_height), Image.LANCZOS)
        resized_tk = ImageTk.PhotoImage(resized_img)
        
        # 显示图像并更新Canvas的滚动区域
        canvas.create_image(0, 0, anchor=tk.NW, image=resized_tk)
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # 保存引用避免被垃圾回收
        if canvas == self.original_canvas:
            self.original_img_tk_scaled = resized_tk
        else:
            self.processed_img_tk_scaled = resized_tk
    
    def refresh_processes(self):
        processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                processes.append(f"{proc.info['name']} (PID: {proc.info['pid']})")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        self.process_combo['values'] = sorted(processes)
        if processes:
            self.process_combo.current(0)
    
    def browse_file(self):
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.askopenfilename(title="选择图像文件", filetypes=filetypes)
        if filename:
            self.image_path.set(filename)
            base, ext = os.path.splitext(filename)
            self.output_path.set(f"{base}_enhanced{ext}")
    
    def browse_output(self):
        initial_file = self.output_path.get() or self.image_path.get()
        if not initial_file:
            initial_file = "enhanced.jpg"
        
        filetypes = [
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("BMP", "*.bmp"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.asksaveasfilename(
            title="保存处理后的文件",
            initialfile=os.path.basename(initial_file),
            defaultextension=".jpg",
            filetypes=filetypes
        )
        if filename:
            self.output_path.set(filename)
    
    def check_updates(self):
        """检查模型更新"""
        try:
            self.update_progress(0, "正在检查更新...")
            ModelManager.ensure_weights()
            self.update_progress(100, "已是最新版本")
            messagebox.showinfo("成功", "所有模型已是最新版本")
        except Exception as e:
            messagebox.showerror("错误", f"检查更新失败: {str(e)}")
            self.update_progress(0, "检查更新失败")
    
    def load_image(self):
        if not self.image_path.get():
            messagebox.showerror("错误", "请先选择图像文件")
            return
        
        try:
            # 加载图像
            self.original_img = ImageProcessor.load_image(self.image_path.get())
            
            # 显示原始图像
            img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.original_img_tk = ImageTk.PhotoImage(img_pil)
            
            # 重置缩放比例
            self.image_scale = 1.0
            self.show_image_on_canvas(self.original_img_tk, self.original_canvas)
            
            # 清除之前的处理结果
            if hasattr(self, 'processed_img_tk'):
                del self.processed_img_tk
            self.processed_canvas.delete("all")
            
            # 自动检测最佳模型
            model_name = self.processor.detect_best_model(self.original_img)
            if model_name:
                for model in AIModels:
                    if model.name == model_name:
                        self.ai_model.set(model.value)
                        break
            
            self.update_progress(0, "图像加载完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")
    
    def process_image(self):
        if not hasattr(self, 'original_img') or self.original_img is None:
            messagebox.showerror("错误", "请先加载图像")
            return
        
        model_name = self.ai_model.get().split()[0]  # 获取模型名称前缀
        
        # 显示进度条
        self.update_progress(0, "准备处理...")
        
        # 在后台线程中处理图像
        def process_in_thread():
            try:
                scale = self.params["scale"].get()
                fidelity = self.params["fidelity"].get()
                
                # 使用真实AI模型处理图像
                def progress_callback(value, message):
                    self.root.after(0, lambda: self.update_progress(value, message))
                
                processed_img = self.processor.process_image(
                    self.original_img, 
                    model_name=model_name,
                    scale=scale,
                    fidelity=fidelity,
                    progress_callback=progress_callback
                )
                
                # 显示处理后的图像
                img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                self.processed_img_tk = ImageTk.PhotoImage(img_pil)
                self.processed_img = processed_img
                
                # 在主线程中更新UI
                self.root.after(0, lambda: self.show_image_on_canvas(
                    self.processed_img_tk, self.processed_canvas))
                
                self.root.after(0, lambda: self.update_progress(100, "处理完成!"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"处理图像失败: {str(e)}"))
                self.root.after(0, lambda: self.update_progress(0, "处理失败"))
        
        # 启动处理线程
        thread = threading.Thread(target=process_in_thread)
        thread.start()
    
    def save_result(self):
        if not hasattr(self, 'processed_img') or self.processed_img is None:
            messagebox.showerror("错误", "没有可保存的处理结果")
            return
        
        output_path = self.output_path.get()
        if not output_path:
            messagebox.showerror("错误", "请指定输出路径")
            return
        
        try:
            ImageProcessor.save_image(self.processed_img, output_path)
            messagebox.showinfo("成功", f"图像已保存到: {output_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图像失败: {str(e)}")
    
    def show_about(self):
        about_text = """AI图像去马赛克工具 (智能版)

主要特点:
- 自动选择最佳AI模型
- 自动下载最新权重文件
- 自动适配最新版AI模型
- 智能检测图像内容选择模型

集成的AI模型:
- Real-ESRGAN: 通用图像增强和超分辨率
- Real-ESRGAN-v3: 最新通用增强模型
- GFPGAN: 专注人脸修复和增强
- SwinIR: 基于Transformer的最新增强模型

系统要求:
- Python 3.8+
- PyTorch with CUDA (推荐)
- 至少4GB显存 (GPU模式)
- 8GB以上内存

操作说明:
1. 加载图像文件
2. 选择AI模型(或使用自动选择)
3. 点击"处理图像"按钮
4. 使用鼠标滚轮可以缩放预览图像
5. 满意后保存结果

权重文件管理:
- 自动检查权重文件完整性
- 自动下载最新权重文件
- 自动备份旧版本文件
- 点击"检查更新"可手动更新"""
        messagebox.showinfo("关于", about_text)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = AIEnhanceApp(root)
        
        # 只有处理器成功初始化时才继续
        if hasattr(app, 'processor') and app.processor is not None:
            root.minsize(1000, 700)
            root.mainloop()
        else:
            print("程序初始化失败，请检查错误信息")
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        traceback.print_exc()
        # 显示错误对话框
        tk.Tk().withdraw()  # 隐藏主窗口
        print("程序启动失败，请检查错误信息")
        messagebox.showerror("启动错误", f"程序启动失败: {str(e)}")