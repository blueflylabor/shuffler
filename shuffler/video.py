import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from .applier import PatchApplier


class VideoAdversarialInjector:
    """视频对抗隐蔽贴图注入器"""

    def __init__(self, applier: PatchApplier, frame_rate_ratio: float = 1.0):
        """
        :param applier: 贴图叠加逻辑器
        :param frame_rate_ratio: 帧插值比例 (如 1.0 表示每帧都插；0.5 表示随机选择 50% 的帧进行植入)
        """
        self.applier = applier
        self.frame_rate_ratio = frame_rate_ratio

    def process_video(self, input_video_path: str, output_video_path: str):
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"视频文件不存在: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法读取视频文件: {input_video_path}")

        # 获取原视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)) or ".", exist_ok=True)

        # 设置编码格式为 mp4v / H264
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        print(f"\n[VideoInjector] 开始对视频执行隐蔽对抗植入:")
        print(f" 🎬 分辨率: {width}x{height} | FPS: {fps:.2f} | 总帧数: {total_frames}")
        print(f" 🛡️  隐蔽模式: {'开启 (肉眼不可见)' if self.applier.imperceptible else '关闭 (显式贴图)'}")

        pbar = tqdm(total=total_frames, desc="处理进度", unit="帧")

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # 控制时间维度插值频率
            should_inject = np.random.rand() < self.frame_rate_ratio

            if should_inject:
                # BGR (OpenCV) -> RGB (PIL)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # 执行隐蔽对抗贴图注入
                injected_pil = self.applier.apply(pil_img)

                # RGB (PIL) -> BGR (OpenCV)
                injected_bgr = cv2.cvtColor(np.array(injected_pil), cv2.COLOR_RGB2BGR)
                writer.write(injected_bgr)
            else:
                writer.write(frame_bgr)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        writer.release()
        pbar.close()
        print(f"[VideoInjector] 视频对抗植入完成！导出路径: {output_video_path}\n")