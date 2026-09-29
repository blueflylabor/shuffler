import os
import sys
import threading
import time
import webbrowser


def run_server():
    """在独立线程中启动 Gradio WebUI 服务"""
    # 确保当前工作目录正确（特别是在 PyInstaller 打包后）
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 导入并启动 WebUI (引用自 web.py)
    from web import demo

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,  # 我们通过下面手动拉起浏览器
        quiet=True,
    )


def open_browser():
    """等待服务就绪后自动拉起浏览器"""
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:7860")


if __name__ == "__main__":
    print("==================================================")
    print("   🎬 正在启动 视频多维对抗重构与评估系统 (Desktop)")
    print("   🌐 访问地址: http://127.0.0.1:7860")
    print("==================================================")

    # 1. 启动浏览器线程
    threading.Thread(target=open_browser, daemon=True).start()

    # 2. 启动 WebUI 服务
    run_server()