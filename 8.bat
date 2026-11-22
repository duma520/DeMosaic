cls
# 创建 weights 文件夹
mkdir -Force weights

# 下载文件（会自动显示进度条）

Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "weights/RealESRGAN_x4plus.pth"

Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -OutFile "weights/GFPGANv1.4.pth"

Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" -OutFile "weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"

pip install basicsr
# 或者直接从 GitHub 安装最新版
pip install git+https://github.com/xinntao/BasicSR.git
