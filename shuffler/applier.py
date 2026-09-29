import random
from typing import Tuple
import numpy as np
import cv2
from PIL import Image, ImageFilter
from .provider import PatchProvider


class PatchApplier:
    """贴图叠加应用器，支持常规随机叠加与隐蔽式（肉眼不可见）对抗叠加"""

    def __init__(
            self,
            provider: PatchProvider,
            num_patches_range: Tuple[int, int] = (10, 50),
            patch_scale_range: Tuple[float, float] = (0.05, 0.15),
            imperceptible: bool = True,
            alpha_limit: float = 0.03,  # 透明度上限 (3%)
            use_edge_masking: bool = True  # 是否启用边缘纹理遮罩
    ):
        self.provider = provider
        self.num_patches_range = num_patches_range
        self.patch_scale_range = patch_scale_range
        self.imperceptible = imperceptible
        self.alpha_limit = alpha_limit
        self.use_edge_masking = use_edge_masking

    def _compute_edge_mask(self, base_img: Image.Image) -> np.ndarray:
        """基于 Sobel 算子计算图像高频纹理梯度图，让贴图只落在人眼不敏感的复杂纹理区"""
        gray = np.array(base_img.convert("L"))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        # 归一化为 0~1 的权重图
        norm_grad = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        return norm_grad

    def apply(self, base_img: Image.Image) -> Image.Image:
        """对单张图像/视频帧执行对抗贴图植入"""
        W, H = base_img.size
        num_patches = random.randint(*self.num_patches_range)
        canvas = base_img.copy().convert("RGBA")

        # 若开启边缘遮罩，提前提取高频梯度图
        edge_mask = self._compute_edge_mask(base_img) if (self.imperceptible and self.use_edge_masking) else None

        for _ in range(num_patches):
            # 随机尺寸
            scale = random.uniform(*self.patch_scale_range)
            pw, ph = int(W * scale), int(H * scale)
            if pw < 4 or ph < 4:
                continue

            # 从提供器获取随机对抗贴图
            patch = self.provider.get_patch((pw, ph))

            # 随机定位
            x = random.randint(0, max(0, W - pw))
            y = random.randint(0, max(0, H - ph))

            # --- 隐蔽性核心算法处理 ---
            if self.imperceptible:
                # 1. 微弱透明度限制 (Alpha Scaled to 1% ~ 3%)
                alpha_factor = random.uniform(0.01, self.alpha_limit)

                # 2. 如果启用边缘纹理遮罩，根据局部复杂程度进一步调低平滑区域透明度
                if edge_mask is not None:
                    local_edge_density = np.mean(edge_mask[y:y + ph, x:x + pw])
                    alpha_factor *= local_edge_density

                # 调整贴图的 Alpha 蒙版
                r, g, b, a = patch.split()
                a = a.point(lambda p: int(p * alpha_factor))
                patch.putalpha(a)

            # 随机旋转增强多样性
            angle = random.randint(0, 360)
            patch = patch.rotate(angle, expand=True, resample=Image.Resampling.BILINEAR)

            # 重新计算旋转后的贴图尺寸与中心位置
            npw, nph = patch.size
            cx, cy = x + pw // 2, y + ph // 2
            nx, ny = max(0, cx - npw // 2), max(0, cy - nph // 2)

            # 叠加至主画布
            canvas.alpha_composite(patch, (nx, ny))

        return canvas.convert("RGB")