import math
import random
from typing import List, Tuple, Optional
import cv2
import numpy as np


class TemporalRestructurer:
    """
    时域重构处理器 (Temporal Restructuring)
    1. 非线性正弦帧率变速 (0.98x ~ 1.02x 波动，亚帧平滑插值)
    2. 镜头切点检测 (Scene Cut Detection) 与切点高频过渡帧注入
    """

    def __init__(
        self,
        speed_amplitude: float = 0.02,       # 变速波动幅度 (0.02 即 0.98x ~ 1.02x)
        speed_period_frames: int = 150,      # 变速波动周期 (帧数)
        scene_cut_threshold: float = 0.55,   # 镜头切点检测阈值 (直方图相关性系数，越低说明差异越大)
        noise_intensity: float = 8.0,        # 切点过渡帧注入的高频噪声强度
    ):
        self.speed_amplitude = speed_amplitude
        self.speed_period = speed_period_frames
        self.scene_cut_threshold = scene_cut_threshold
        self.noise_intensity = noise_intensity

        self.prev_hist: Optional[np.ndarray] = None

    def calculate_frame_mapping(self, total_frames: int) -> List[float]:
        """
        预先计算输出帧序列对应的输入帧浮点索引列表。
        利用正弦波 v(t) = 1.0 + A * sin(2π * t / T) 积分累加，得出非线性时间映射矩阵。
        """
        source_indices = []
        curr_src_idx = 0.0

        for out_idx in range(total_frames):
            # 当前时间点的变速因子 (在 1-A 到 1+A 之间正弦波动)
            speed_factor = 1.0 + self.speed_amplitude * math.sin(
                2.0 * math.pi * out_idx / self.speed_period
            )
            source_indices.append(curr_src_idx)
            curr_src_idx += speed_factor

        return source_indices

    def interpolate_frame(
        self, frame1: np.ndarray, frame2: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        亚帧双帧加权融合插值 (Sub-frame Linear Interpolation)
        当浮点索引介于 frame1 与 frame2 之间时，平滑混合两帧，保证变速视觉无卡顿。
        """
        if alpha <= 0.001:
            return frame1
        if alpha >= 0.999:
            return frame2

        return cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)

    def detect_scene_cut(self, frame_bgr: np.ndarray) -> bool:
        """
        基于 HSV 3D 直方图相关性检测镜头切换点 (Scene Cut)
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # 计算 8x8x8 HSV 直方图
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)

        is_cut = False
        if self.prev_hist is not None:
            # 比较前后帧直方图相关性 (HISTCMP_CORREL，值为 1 表示完全相同，0 或负数表示差异极大)
            similarity = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < self.scene_cut_threshold:
                is_cut = True

        self.prev_hist = hist
        return is_cut

    def inject_cut_transition_noise(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        在镜头切点处施加包含高频对抗微扰的过渡帧
        """
        h, w, c = frame_bgr.shape
        # 生成高频高斯噪声
        noise = np.random.normal(0, self.noise_intensity, (h, w, c)).astype(
            np.float32
        )

        # 结合极轻微的高通滤波高频掩码，集中在边缘轮廓区域加噪
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges = np.repeat(edges[:, :, np.newaxis], 3, axis=2)

        # 在高频边缘区加强扰动
        noisy_frame = frame_bgr.astype(np.float32) + noise * (0.5 + 0.5 * edges)
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)

        return noisy_frame