import numpy as np
import cv2

class RandomROIInjector:
    """随机贴图注入器：在图像范围内随机位置、随机大小注入贴图 Patch"""

    def __init__(
        self,
        provider,
        alpha_limit=0.03,
        min_patch_ratio=0.08,
        max_patch_ratio=0.20,
        patches_per_frame=1,
    ):
        """
        :param provider: PatchProvider 实例
        :param alpha_limit: 透明度上限 (0.01 - 0.15)
        :param min_patch_ratio: 贴图相对短边最小比例
        :param max_patch_ratio: 贴图相对短边最大比例
        :param patches_per_frame: 每帧随机插入的贴图数量
        """
        self.provider = provider
        self.alpha_limit = alpha_limit
        self.min_patch_ratio = min_patch_ratio
        self.max_patch_ratio = max_patch_ratio
        self.patches_per_frame = patches_per_frame

    def inject_to_frame(self, frame):
        if frame is None:
            return frame

        h, w, c = frame.shape
        min_side = min(h, w)

        for _ in range(self.patches_per_frame):
            # 1. 随机计算贴图尺寸
            patch_ratio = np.random.uniform(self.min_patch_ratio, self.max_patch_ratio)
            pw = int(min_side * patch_ratio)
            ph = int(min_side * patch_ratio)

            if pw <= 0 or ph <= 0 or pw >= w or ph >= h:
                continue

            # 2. 从图库中获取对应尺寸的贴图 Patch
            patch = self.provider.get_patch(pw, ph)

            if patch is None or patch.shape[:2] != (ph, pw):
                continue

            # 3. 随机选择贴图注入坐标 (x, y)
            x = np.random.randint(0, w - pw)
            y = np.random.randint(0, h - ph)

            # 4. Alpha 混合融入
            alpha = np.random.uniform(0.01, self.alpha_limit)
            roi = frame[y : y + ph, x : x + pw]

            blended = cv2.addWeighted(roi, 1.0 - alpha, patch, alpha, 0)
            frame[y : y + ph, x : x + pw] = blended

        return frame