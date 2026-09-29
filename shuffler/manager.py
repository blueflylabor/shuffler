import os
import glob
import shutil
import random
from typing import List, Tuple
from PIL import Image, ImageDraw


class StickerManager:
    """贴图库管理器：管理内置库 (builtin) 与 用户自定义库 (custom)"""

    def __init__(self, assets_dir: str = "shuffler_assets"):
        self.assets_dir = os.path.abspath(assets_dir)
        self.builtin_dir = os.path.join(self.assets_dir, "builtin")
        self.custom_dir = os.path.join(self.assets_dir, "custom")

        self._ensure_directories()
        self._check_and_init_builtin_stickers()

    def _ensure_directories(self):
        """确保物理存储目录存在"""
        os.makedirs(self.builtin_dir, exist_ok=True)
        os.makedirs(self.custom_dir, exist_ok=True)

    def _check_and_init_builtin_stickers(self):
        """若内置库为空，自动生成预置的标准对抗贴片/几何干扰图样表件"""
        existing = glob.glob(os.path.join(self.builtin_dir, "*.png"))
        if not existing:
            print("[StickerManager] 正在初始化默认内置贴图库 (Builtin Library)...")
            self._generate_default_builtin_assets()

    def _generate_default_builtin_assets(self):
        """生成一组内置的高对比度几何与正弦纹理贴片存入 builtin 目录"""
        for i in range(5):
            size = 256
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # 生成异形基础
            colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]
            draw.rectangle([0, 0, size, size], fill=random.choice(colors))
            for _ in range(10):
                x1, y1 = random.randint(0, size), random.randint(0, size)
                x2, y2 = random.randint(0, size), random.randint(0, size)
                draw.ellipse([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=random.choice(colors))

            out_path = os.path.join(self.builtin_dir, f"builtin_patch_{i + 1}.png")
            img.save(out_path)

    def import_stickers(self, source_path: str) -> int:
        """批量导入用户自定义贴图（支持单个文件或整个文件夹递归导入）"""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"导入源路径不存在: {source_path}")

        valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        candidates = []

        if os.path.isfile(source_path):
            candidates.append(source_path)
        elif os.path.isdir(source_path):
            for ext in valid_exts:
                candidates.extend(glob.glob(os.path.join(source_path, "**", ext), recursive=True))

        imported_count = 0
        for src_fp in candidates:
            try:
                # 校验图片可读性并统一转存至 custom_dir
                with Image.open(src_fp) as img:
                    fname = os.path.basename(src_fp)
                    dest_fp = os.path.join(self.custom_dir, fname)

                    # 避免同名覆盖，自动重命名
                    counter = 1
                    base, ext = os.path.splitext(fname)
                    while os.path.exists(dest_fp):
                        dest_fp = os.path.join(self.custom_dir, f"{base}_{counter}{ext}")
                        counter += 1

                    shutil.copy2(src_fp, dest_fp)
                    imported_count += 1
            except Exception as e:
                print(f"[StickerManager Warning] 跳过无效文件 {src_fp}: {e}")

        print(f"[StickerManager] 成功批量导入 {imported_count} 张贴图至自定义库: {self.custom_dir}")
        return imported_count

    def load_stickers(self, source_type: str = "custom") -> List[Image.Image]:
        """按库类型（custom / builtin / all）加载所有图片贴图到内存"""
        target_dirs = []
        if source_type in ("custom", "all"):
            target_dirs.append(self.custom_dir)
        if source_type in ("builtin", "all"):
            target_dirs.append(self.builtin_dir)

        valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        filepaths = []
        for d in target_dirs:
            for ext in valid_exts:
                filepaths.extend(glob.glob(os.path.join(d, ext)))

        loaded_patches = []
        for fp in filepaths:
            try:
                img = Image.open(fp).convert("RGBA")
                loaded_patches.append(img)
            except Exception as e:
                print(f"[StickerManager Warning] 无法读取贴图 {fp}: {e}")

        return loaded_patches

    def list_info(self):
        """统计当前库信息"""
        builtin_cnt = len(glob.glob(os.path.join(self.builtin_dir, "*.*")))
        custom_cnt = len(glob.glob(os.path.join(self.custom_dir, "*.*")))
        print("\n=== Shuffler 贴图库统计 ===")
        print(f"📦 内置库 (Builtin):  {builtin_cnt} 张 ({self.builtin_dir})")
        print(f"🎨 自定义库 (Custom): {custom_cnt} 张 ({self.custom_dir})\n")