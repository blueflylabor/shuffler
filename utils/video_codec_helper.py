import os
import subprocess
import torch


class VideoCodecHelper:
    """视频编解码硬件加速与 CRF 码率控制助手"""

    @staticmethod
    def detect_hardware_encoder():
        """自动检测最佳硬件编码器: Mac VideoToolbox > NVIDIA NVENC > CPU libx264"""
        # 1. 检测 Apple Silicon (Mac MPS/Darwin)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "h264_videotoolbox", "🟢 启用 Apple Silicon VideoToolbox 硬件加速编码"

        # 2. 检测 NVIDIA GPU (CUDA)
        if torch.cuda.is_available():
            return "h264_nvenc", "🟢 启用 NVIDIA NVENC 硬件加速编码"

        # 3. 兜底 CPU
        return "libx264", "🟡 使用 CPU 软件编码 (libx264)"

    @staticmethod
    def get_video_duration(video_path):
        """利用 ffprobe 精准获取音视频时长 (秒)"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(res.stdout.strip())
        except Exception:
            return 0.0