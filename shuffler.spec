# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# 收集可能隐式引入的动态库和依赖
# 收集可能隐式引入的动态库和依赖
hiddenimports = [
    'cv2', 'torch', 'torchvision', 'lpips', 'pywt', 'pydub',
    'gradio', 'numpy', 'scipy', 'PIL', 'ultralytics',
    'pkg_resources', 'setuptools'
]
hiddenimports += collect_submodules('lpips')
hiddenimports += collect_submodules('ultralytics')

# 收集数据文件（例如本地模型权重 yolo 等）
datas = []
if os.path.exists('yolov8n.pt'):
    datas.append(('yolov8n.pt', '.'))

a = Analysis(
    ['app_launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ShufflerVideoReconstructor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 设为 True 可以保留终端查看运行日志和硬件加速状态（打包完成后可改为 False 纯窗口模式）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ShufflerVideoReconstructor',
)