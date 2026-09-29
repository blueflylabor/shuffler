import random
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image


class BaseROIInjector:
    """ROI 对抗植入基类，负责统一的贴图合成与透明度控制"""

    def __init__(
        self,
        provider,
        alpha_limit: float = 0.04,
        expand_ratio: float = 0.2,
    ):
        self.provider = provider
        self.alpha_limit = alpha_limit
        self.expand_ratio = expand_ratio

    def detect_rois(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """子类需实现该方法，返回 ROI 矩形列表 [(x, y, w, h), ...]"""
        raise NotImplementedError("子类必须实现 detect_rois 方法")

    def _expand_roi(
        self, x: int, y: int, w: int, h: int, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int]:
        """按比例扩展 Bounding Box，避免漏掉目标边缘特征"""
        ex_w = int(w * self.expand_ratio)
        ex_h = int(h * self.expand_ratio)

        nx = max(0, x - ex_w)
        ny = max(0, y - ex_h)
        nw = min(img_w - nx, w + 2 * ex_w)
        nh = min(img_h - ny, h + 2 * ex_h)
        return (nx, ny, nw, nh)

    def inject_to_frame(
        self,
        frame_bgr: np.ndarray,
        patches_per_roi: Tuple[int, int] = (3, 8),
    ) -> np.ndarray:
        """对检测到的 ROI 执行微透贴图植入"""
        rois = self.detect_rois(frame_bgr)
        if not rois:
            return frame_bgr

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(frame_rgb).convert("RGBA")

        for rx, ry, rw, rh in rois:
            num_patches = random.randint(*patches_per_roi)
            for _ in range(num_patches):
                pw = random.randint(max(8, rw // 8), max(16, rw // 3))
                ph = random.randint(max(8, rh // 8), max(16, rh // 3))

                patch = self.provider.get_patch((pw, ph))
                if patch.mode != "RGBA":
                    patch = patch.convert("RGBA")

                # 低 Alpha 混合 (1% ~ alpha_limit)
                alpha_scale = random.uniform(0.01, self.alpha_limit)
                r, g, b, a = patch.split()
                a = a.point(lambda p: int(p * alpha_scale))
                patch.putalpha(a)

                max_px = max(rx, rx + rw - pw)
                max_py = max(ry, ry + rh - ph)
                px = random.randint(rx, max_px)
                py = random.randint(ry, max_py)

                canvas.alpha_composite(patch, (px, py))

        result_rgb = np.array(canvas.convert("RGB"))
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)