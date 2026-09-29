import math
import random
from typing import Tuple
import cv2
import numpy as np


class SpatialAffineWarper:
    """
    空间与仿射微变处理器 (Spatial & Affine Warper)
    实现呼吸式亚像素微裁切以及帧间平滑的几何仿射扭曲，
    破坏视频帧在坐标空间与哈希映射上的严格对齐。
    """

    def __init__(
        self,
        crop_range: Tuple[float, float] = (0.005, 0.010),  # 裁剪比例 0.5% ~ 1.0%
        crop_period_frames: int = 120,                      # 呼吸缩放周期 (帧数)
        affine_interval_range: Tuple[int, int] = (15, 30), # 仿射变换目标更新间隔 (15~30帧)
        max_rotation_deg: float = 0.3,                      # 旋转角幅度上限 (度)
        max_scale_delta: float = 0.005,                     # 缩放幅度上限 (0.5%)
        max_translation_px: float = 2.0,                    # 位移幅度上限 (像素)
    ):
        self.crop_min, self.crop_max = crop_range
        self.crop_period = crop_period_frames
        self.affine_interval_range = affine_interval_range
        self.max_rotation_deg = max_rotation_deg
        self.max_scale_delta = max_scale_delta
        self.max_translation_px = max_translation_px

        # 当前与目标的仿射状态参数: (rot, scale, tx, ty)
        self._curr_affine = (0.0, 1.0, 0.0, 0.0)
        self._target_affine = (0.0, 1.0, 0.0, 0.0)
        self._prev_affine = (0.0, 1.0, 0.0, 0.0)

        self._step_counter = 0
        self._transition_steps = random.randint(*self.affine_interval_range)

    def _generate_random_affine_target(self) -> Tuple[float, float, float, float]:
        """生成一组微小仿射目标参数 (rot_deg, scale, tx, ty)"""
        rot = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        scale = 1.0 + random.uniform(-self.max_scale_delta, self.max_scale_delta)
        tx = random.uniform(-self.max_translation_px, self.max_translation_px)
        ty = random.uniform(-self.max_translation_px, self.max_translation_px)
        return rot, scale, tx, ty

    def _update_affine_state(self):
        """平滑更新仿射状态，避免帧间参数剧烈跳变引发闪烁"""
        if self._step_counter >= self._transition_steps:
            # 达到目标点，设为下一阶段起点并生成新目标
            self._prev_affine = self._target_affine
            self._target_affine = self._generate_random_affine_target()
            self._step_counter = 0
            self._transition_steps = random.randint(*self.affine_interval_range)

        # 线性插值 compute progress t ∈ [0, 1]
        t = self._step_counter / float(self._transition_steps)
        # 使用 smoothstep 缓动函数让过渡更自然: t * t * (3 - 2 * t)
        t_smooth = t * t * (3.0 - 2.0 * t)

        p_rot, p_scale, p_tx, p_ty = self._prev_affine
        t_rot, t_scale, t_tx, t_ty = self._target_affine

        c_rot = p_rot + (t_rot - p_rot) * t_smooth
        c_scale = p_scale + (t_scale - p_scale) * t_smooth
        c_tx = p_tx + (t_tx - p_tx) * t_smooth
        c_ty = p_ty + (t_ty - p_ty) * t_smooth

        self._curr_affine = (c_rot, c_scale, c_tx, c_ty)
        self._step_counter += 1

    def apply_subpixel_crop(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        1. 亚像素呼吸式微裁切与缩放 (Pan & Zoom)
        利用正弦波曲线计算当前帧的微缩放比例，并使用高质量双三次插值还原分辨率。
        """
        h, w = frame_bgr.shape[:2]

        # 计算平滑正弦波变化的裁切比例 (0.5% ~ 1.0%)
        sin_val = (math.sin(2.0 * math.pi * frame_idx / self.crop_period) + 1.0) / 2.0
        crop_ratio = self.crop_min + (self.crop_max - self.crop_min) * sin_val

        # 计算裁切边界
        crop_x = int(w * crop_ratio)
        crop_y = int(h * crop_ratio)

        x1, y1 = crop_x, crop_y
        x2, y2 = w - crop_x, h - crop_y

        # 防止边界异常
        if x2 <= x1 or y2 <= y1:
            return frame_bgr

        # 执行裁切并重新缩放到原始尺寸
        cropped = frame_bgr[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
        return resized

    def apply_affine_warp(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        2. 微观几何仿射扭曲 (Smooth Micro Affine Warp)
        应用旋转、微缩放和平移变换，边缘采用镜像填充 (BORDER_REFLECT_101) 避免黑边。
        """
        h, w = frame_bgr.shape[:2]
        self._update_affine_state()
        rot, scale, tx, ty = self._curr_affine

        center = (w / 2.0, h / 2.0)
        # 获取 2x3 仿射矩阵
        M = cv2.getRotationMatrix2D(center, rot, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # 执行仿射变换，BORDER_REFLECT_101 可完美隐藏边缘缝隙
        warped = cv2.warpAffine(
            frame_bgr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        对单帧依次施加呼吸裁剪与平滑仿射变换
        """
        # 1. 亚像素呼吸微裁切
        frame = self.apply_subpixel_crop(frame_bgr, frame_idx)
        # 2. 仿射变换扭曲
        frame = self.apply_affine_warp(frame)
        return frame