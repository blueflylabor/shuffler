import os
from typing import List, Tuple, Optional
import cv2
import numpy as np
from .base_injector import BaseROIInjector


class HaarROIInjector(BaseROIInjector):
    """基于 Haar Cascade 的人脸/目标检测植入器"""

    def __init__(
        self,
        provider,
        cascade_path: Optional[str] = None,
        alpha_limit: float = 0.04,
        expand_ratio: float = 0.2,
        min_size: Tuple[int, int] = (30, 30),
    ):
        super().__init__(provider, alpha_limit, expand_ratio)
        self.min_size = min_size

        if cascade_path and os.path.exists(cascade_path):
            self.detector = cv2.CascadeClassifier(cascade_path)
        else:
            default_cascade = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if not os.path.exists(default_cascade):
                raise FileNotFoundError("未找到 Haar 级联分类器模型文件。")
            self.detector = cv2.CascadeClassifier(default_cascade)

    def detect_rois(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_size,
        )

        rois = []
        for x, y, w, h in faces:
            roi = self._expand_roi(x, y, w, h, width, height)
            rois.append(roi)

        return rois