import os
import cv2
import numpy as np


class HaarROIInjector:
    def __init__(self, provider, alpha_limit: float = 0.03):
        self.provider = provider
        self.alpha_limit = alpha_limit

        # 安全获取 haarcascades 路径
        cascade_path = None
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
        else:
            # 兼容旧版本或缺失 cv2.data 的环境
            cv2_dir = os.path.dirname(cv2.__file__)
            possible_path = os.path.join(
                cv2_dir, "data", "haarcascade_frontalface_default.xml"
            )
            if os.path.exists(possible_path):
                cascade_path = possible_path

        if cascade_path and os.path.exists(cascade_path):
            self.detector = cv2.CascadeClassifier(cascade_path)
        else:
            # 如果依然找不到内置 XML，回退到系统级/默认加载
            self.detector = cv2.CascadeClassifier(
                cv2.samples.findFile("haarcascades/haarcascade_frontalface_default.xml")
                if hasattr(cv2, "samples")
                else "haarcascade_frontalface_default.xml"
            )

    def inject_to_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.detector.empty():
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces:
            patch = self.provider.get_patch(w, h)
            if patch is None:
                continue

            # 按 alpha_limit 进行混合融合
            roi = frame[y : y + h, x : x + w]
            blended = cv2.addWeighted(
                roi, 1.0 - self.alpha_limit, patch, self.alpha_limit, 0
            )
            frame[y : y + h, x : x + w] = blended

        return frame