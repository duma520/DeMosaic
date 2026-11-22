@echo off
echo 正在安装AI去马赛克工具箱...
pip install PyQt5 opencv-python numpy torch Pillow psutil pygetwindow pyautogui requests basicsr facexlib gfpgan realesrgan --extra-index-url https://download.pytorch.org/whl/cu117
if %errorlevel% equ 0 (
    echo 安装成功！请运行: python DeMosaic.py
) else (
    echo 安装失败，请检查网络或手动安装
)
pause
