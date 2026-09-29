import json
import os


class PresetManager:
    """预设模板管理器：提供默认模板并支持自定义预设的保存与加载"""

    DEFAULT_PRESETS = {
        "⚡ 轻度去重 (极高画质保真)": {
            "enable_roi": True,
            "roi_detector": "随机位置注入",
            "roi_alpha": 0.02,
            "enable_spatial": True,
            "spatial_rot": 0.5,
            "spatial_scale": 0.008,
            "enable_color": True,
            "enable_dct": False,
            "uv_noise": 0.8,
            "enable_temporal": False,
            "speed_amp": 0.01,
            "enable_audio": True,
            "audio_speed_ratio": 1.003,
        },
        "🎯 标准对抗 (适合短视频平台)": {
            "enable_roi": True,
            "roi_detector": "YOLO 深度学习",
            "roi_alpha": 0.03,
            "enable_spatial": True,
            "spatial_rot": 1.2,
            "spatial_scale": 0.015,
            "enable_color": True,
            "enable_dct": True,
            "uv_noise": 1.5,
            "enable_temporal": True,
            "speed_amp": 0.02,
            "enable_audio": True,
            "audio_speed_ratio": 1.008,
        },
        "🔥 深度重构 (强特征破坏/高去重率)": {
            "enable_roi": True,
            "roi_detector": "YOLO 深度学习",
            "roi_alpha": 0.06,
            "enable_spatial": True,
            "spatial_rot": 2.5,
            "spatial_scale": 0.025,
            "enable_color": True,
            "enable_dct": True,
            "uv_noise": 3.0,
            "enable_temporal": True,
            "speed_amp": 0.04,
            "enable_audio": True,
            "audio_speed_ratio": 1.015,
        },
    }

    def __init__(self, storage_path=None):
        if storage_path is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            storage_path = os.path.join(base_dir, "presets.json")
        self.storage_path = storage_path
        self.presets = self._load_presets()

    def _load_presets(self):
        """加载本地配置，如不存在则创建默认配置"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    custom = json.load(f)
                    # 融合默认与自定义
                    merged = self.DEFAULT_PRESETS.copy()
                    merged.update(custom)
                    return merged
            except Exception:
                return self.DEFAULT_PRESETS.copy()
        return self.DEFAULT_PRESETS.copy()

    def save_custom_preset(self, name, config_dict):
        """保存自定义模板"""
        self.presets[name] = config_dict
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, ensure_ascii=False, indent=2)
            return True, f"成功保存预设: {name}"
        except Exception as e:
            return False, f"保存预设失败: {str(e)}"

    def get_preset_names(self):
        return list(self.presets.keys())

    def get_preset_config(self, name):
        return self.presets.get(name, self.DEFAULT_PRESETS["🎯 标准对抗 (适合短视频平台)"])