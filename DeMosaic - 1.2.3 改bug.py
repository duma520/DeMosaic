import sys
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

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AIModels(Enum):
    REAL_ESRGAN = "Real-ESRGAN (通用增强)"
    GFPGAN = "GFPGAN (人脸修复)"
    ESRGAN = "ESRGAN (超分辨率)"
    SWINIR = "SwinIR (Transformer增强)"

class ImageProcessor:
    def __init__(self):
        # 初始化模型（实际使用时需要先下载预训练权重）
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """加载真实AI模型"""
        try:
             # 检查基础依赖
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from basicsr.utils.download_util import load_file_from_url
            except ImportError:
                messagebox.showerror("依赖错误", "请先安装依赖库：pip install basicsr")
                sys.exit(1)
            
            # Real-ESRGAN模型
            from realesrgan import RealESRGANer
            self.models['REAL_ESRGAN'] = RealESRGANer(
                scale=4,
                model_path='weights/RealESRGAN_x4plus.pth',
                model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                device=device
            )
            
            # GFPGAN模型
            from gfpgan import GFPGANer
            self.models['GFPGAN'] = GFPGANer(
                model_path='weights/GFPGANv1.4.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=device
            )
            
            # ESRGAN模型
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
        except ImportError:
            messagebox.showerror("依赖错误", "请先安装依赖库：pip install basicsr")
            sys.exit(1)
            esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            esrgan_model_path = load_file_from_url(
                url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
                model_dir='weights', progress=True, file_name=None)
            self.models['ESRGAN'] = esrgan_model
            self.models['ESRGAN'].load_state_dict(torch.load(esrgan_model_path)['params'], strict=True)
            self.models['ESRGAN'].to(device).eval()
            
            # SwinIR模型
            from swinir import SwinIR
            self.models['SWINIR'] = SwinIR(
                upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            swinir_model_path = 'weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
            self.models['SWINIR'].load_state_dict(torch.load(swinir_model_path)['params'])
            self.models['SWINIR'].to(device).eval()
            
        except Exception as e:
            messagebox.showerror("模型加载错误", f"无法加载AI模型: {str(e)}")
    
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
    
    def process_image(self, img, model_name, scale=2, fidelity=0.7):
        """使用真实AI模型处理图像"""
        try:
            if model_name == "REAL_ESRGAN":
                output, _ = self.models['REAL_ESRGAN'].enhance(img, outscale=scale)
                return output
            
            elif model_name == "GFPGAN":
                _, _, output = self.models['GFPGAN'].enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True, weight=fidelity)
                return output
            
            elif model_name == "ESRGAN":
                img_tensor = ToTensor()(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = self.models['ESRGAN'](img_tensor)
                output = output.squeeze().permute(1, 2, 0).cpu().numpy()
                output = np.clip(output * 255, 0, 255).astype(np.uint8)
                if scale > 1:
                    h, w = img.shape[:2]
                    output = cv2.resize(output, (w*scale, h*scale))
                return output
            
            elif model_name == "SWINIR":
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
            raise RuntimeError(f"AI处理失败: {str(e)}")

class AIEnhanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI图像去马赛克工具 (真实模型版)")
        
        # 初始化变量
        self.image_path = tk.StringVar()
        self.process_var = tk.StringVar()
        self.output_path = tk.StringVar()
        self.ai_model = tk.StringVar(value=AIModels.REAL_ESRGAN.value)
        self.params = {
            "scale": tk.IntVar(value=2),
            "fidelity": tk.DoubleVar(value=0.7),
            "tile": tk.IntVar(value=0)
        }
        
        # 初始化图像处理器
        self.processor = ImageProcessor()
        
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
        
        # 操作按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="加载图像", command=self.load_image).pack(side="left", padx=5)
        ttk.Button(button_frame, text="处理图像", command=self.process_image).pack(side="left", padx=5)
        ttk.Button(button_frame, text="保存结果", command=self.save_result).pack(side="left", padx=5)
        ttk.Button(button_frame, text="关于", command=self.show_about).pack(side="right", padx=5)
    
    def create_preview_panel(self, parent):
        # 原图预览
        orig_frame = ttk.LabelFrame(parent, text="原始图像", padding="5")
        orig_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.original_canvas = tk.Canvas(orig_frame, bg='gray')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 处理后图像预览
        proc_frame = ttk.LabelFrame(parent, text="处理后图像", padding="5")
        proc_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.processed_canvas = tk.Canvas(proc_frame, bg='gray')
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定窗口大小变化事件
        parent.bind("<Configure>", self.on_window_resize)
    
    def on_window_resize(self, event):
        if hasattr(self, 'original_img_tk'):
            self.show_image_on_canvas(self.original_img_tk, self.original_canvas)
        if hasattr(self, 'processed_img_tk'):
            self.show_image_on_canvas(self.processed_img_tk, self.processed_canvas)
    
    def show_image_on_canvas(self, img_tk, canvas):
        canvas.delete("all")
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        img_width = img_tk.width()
        img_height = img_tk.height()
        
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
    
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
            self.show_image_on_canvas(self.original_img_tk, self.original_canvas)
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")
    
    def process_image(self):
        if not hasattr(self, 'original_img') or self.original_img is None:
            messagebox.showerror("错误", "请先加载图像")
            return
        
        model_name = self.ai_model.get().split()[0]  # 获取模型名称前缀
        
        try:
            # 在后台线程中处理图像，避免界面冻结
            def process_in_thread():
                try:
                    scale = self.params["scale"].get()
                    fidelity = self.params["fidelity"].get()
                    
                    # 使用真实AI模型处理图像
                    processed_img = self.processor.process_image(
                        self.original_img, 
                        model_name=model_name,
                        scale=scale,
                        fidelity=fidelity
                    )
                    
                    # 显示处理后的图像
                    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    self.processed_img_tk = ImageTk.PhotoImage(img_pil)
                    self.processed_img = processed_img
                    
                    # 在主线程中更新UI
                    self.root.after(0, lambda: self.show_image_on_canvas(
                        self.processed_img_tk, self.processed_canvas))
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "错误", f"处理图像失败: {str(e)}"))
            
            # 显示处理中提示
            processing_msg = messagebox.showinfo("处理中", "正在使用AI处理图像，请稍候...", parent=self.root)
            
            # 启动处理线程
            thread = threading.Thread(target=process_in_thread)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"处理图像失败: {str(e)}")
    
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
        about_text = """AI图像去马赛克工具 (真实模型版)

集成的AI模型:
- Real-ESRGAN: 通用图像增强和超分辨率
- GFPGAN: 专注人脸修复和增强
- ESRGAN: 传统超分辨率模型
- SwinIR: 基于Transformer的最新增强模型

系统要求:
- Python 3.8+
- PyTorch with CUDA (推荐)
- 至少4GB显存 (GPU模式)
- 8GB以上内存

注意: 首次使用需要下载预训练权重文件
权重文件会自动下载到weights目录"""
        messagebox.showinfo("关于", about_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = AIEnhanceApp(root)
    root.minsize(900, 650)
    root.mainloop()