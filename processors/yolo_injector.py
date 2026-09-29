from typing import List, Tuple, Optional, Union
import numpy as np
from .base_injector import BaseROIInjector

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YoloROIInjector(BaseROIInjector):
    """基于 YOLO (Ultralytics) 的深度学习目标/人脸识别植入器"""

    def __init__(
        self,
        provider,
        model_path: str = "yolov8n.pt",  # 可指定官方权重或自定义人脸模型如 yolov8n-face.pt
        classes: Optional[List[int]] = None,  # 指定关注的类别 ID，例如 [0] 代表 Person (人)
        conf_threshold: float = 0.25,
        alpha_limit: float = 0.04,
        expand_ratio: float = 0.2,
        device: Union[str, int] = "",  # CUDA 设备标识，如 "0" 或 "cpu"
    ):
        super().__init__(provider, alpha_limit, expand_ratio)

        if YOLO is None:
            raise ImportError(
                "未安装 ultralytics 依赖包，请先运行: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.device = device

    def detect_rois(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        height, width = frame_bgr.shape[:2]

        # 执行 YOLO 推理
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        rois = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取 [x1, y1, x2, y2]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                roi = self._expand_roi(x1, y1, w, h, width, height)
                rois.append(roi)

        return rois