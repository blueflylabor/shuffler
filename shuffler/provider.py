import random
from enum import Enum
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
from .manager import StickerManager


class PatchAlgorithm(str, Enum):
    NOISE = "noise"
    GRID_NOISE = "grid_noise"
    VORONOI = "voronoi"
    GEOMETRIC = "geometric"
    FREQUENCY = "frequency"
    MIXED = "mixed"


class PatchSourceType(str, Enum):
    ALGORITHM = "algorithm"
    CUSTOM = "custom"
    BUILTIN = "builtin"
    ALL = "all"  # 混合 Custom + Builtin


class PatchProvider:
    """整合 StickerManager 与 算法生成器的统一接口"""

    def __init__(
            self,
            manager: StickerManager,
            source_type: PatchSourceType = PatchSourceType.ALGORITHM,
            algo_type: PatchAlgorithm = PatchAlgorithm.MIXED,
            shape_mask: str = "random"
    ):
        self.manager = manager
        self.source_type = PatchSourceType(source_type)
        self.algo_type = PatchAlgorithm(algo_type)
        self.shape_mask = shape_mask

        # 从 StickerManager 载入对应图片库
        self.loaded_patches = []
        if self.source_type in (PatchSourceType.CUSTOM, PatchSourceType.BUILTIN, PatchSourceType.ALL):
            self.loaded_patches = self.manager.load_stickers(source_type=self.source_type.value)
            if not self.loaded_patches and self.source_type == PatchSourceType.CUSTOM:
                print("[Shuffler Warning] 自定义贴图库为空，请先运行 `python main.py import --src <文件夹>` 批量导入！")

    def _apply_shape_mask(self, img: Image.Image, shape_type: str) -> Image.Image:
        w, h = img.size
        if shape_type == "random":
            shape_type = random.choice(["square", "circle", "rounded_rect", "polygon"])

        if shape_type == "square":
            return img

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        if shape_type == "circle":
            draw.ellipse([0, 0, w, h], fill=255)
        elif shape_type == "rounded_rect":
            radius = min(w, h) // 4
            draw.rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
        elif shape_type == "polygon":
            num_pts = random.randint(5, 8)
            points = []
            cx, cy = w / 2, h / 2
            for i in range(num_pts):
                angle = (2 * np.pi / num_pts) * i
                r = random.uniform(0.35, 0.5) * min(w, h)
                points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
            draw.polygon(points, fill=255)

        orig_alpha = img.split()[-1]
        final_alpha = Image.composite(orig_alpha, mask, orig_alpha)
        img.putalpha(final_alpha)
        return img

    def _generate_by_algo(self, size: Tuple[int, int], algo: PatchAlgorithm) -> Image.Image:
        w, h = size
        if algo == PatchAlgorithm.MIXED:
            algo = random.choice([
                PatchAlgorithm.NOISE, PatchAlgorithm.GRID_NOISE,
                PatchAlgorithm.VORONOI, PatchAlgorithm.GEOMETRIC, PatchAlgorithm.FREQUENCY
            ])

        if algo == PatchAlgorithm.GRID_NOISE:
            grid_size = random.choice([4, 8, 16])
            gh, gw = max(1, h // grid_size), max(1, w // grid_size)
            small = np.random.randint(0, 256, (gh, gw, 3), dtype=np.uint8)
            res = Image.fromarray(small, mode="RGB").resize((w, h), Image.Resampling.NEAREST).convert("RGBA")

        elif algo == PatchAlgorithm.VORONOI:
            num_pts = random.randint(8, 20)
            pts = np.random.rand(num_pts, 2) * [w, h]
            colors = np.random.randint(0, 256, (num_pts, 3), dtype=np.uint8)
            gy, gx = np.mgrid[0:h, 0:w]
            grid = np.stack([gx, gy], axis=-1)
            dists = np.linalg.norm(grid[:, :, None, :] - pts[None, None, :, :], axis=-1)
            nearest = np.argmin(dists, axis=-1)
            res = Image.fromarray(colors[nearest], mode="RGB").convert("RGBA")

        else:
            noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            res = Image.fromarray(noise, mode="RGB").convert("RGBA")

        return self._apply_shape_mask(res, self.shape_mask)

    def get_patch(self, size: Tuple[int, int]) -> Image.Image:
        """从对应库中提取或由算法生成指定大小的贴图"""
        if self.loaded_patches:
            patch = random.choice(self.loaded_patches)
            patch_resized = patch.resize(size, Image.Resampling.BILINEAR)
            return self._apply_shape_mask(patch_resized, self.shape_mask)
        else:
            return self._generate_by_algo(size, self.algo_type)