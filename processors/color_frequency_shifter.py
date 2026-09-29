import random
import cv2
import numpy as np


class ColorFrequencyShifter:
    """
    色彩空间与高频通道抖动处理器 (Color Space & Frequency Shift)
    包含了：
    1. YCrCb/YUV 色度通道微抖动 (U/V Channel Noise)
    2. 8x8 块高频 DCT 频域微扰动 (High-Frequency DCT Noise)
    3. 动态 LUT 曲线色彩微调 (Subtle Color LUT Shift)
    """

    def __init__(
        self,
        uv_noise_std: float = 1.5,       # U/V 色度通道高斯噪声标准差
        dct_freq_noise_std: float = 0.5,  # 高频 DCT 系数噪声强度
        lut_jitter_range: float = 2.0,    # LUT 曲线微调幅度
        enable_dct: bool = True,          # 是否开启 DCT 频域扰动
    ):
        self.uv_noise_std = uv_noise_std
        self.dct_freq_noise_std = dct_freq_noise_std
        self.lut_jitter_range = lut_jitter_range
        self.enable_dct = enable_dct

        # 预先生成平滑微调的 1D/3D LUT 表
        self._lut_table = self._generate_subtle_lut()

    def _generate_subtle_lut(self) -> np.ndarray:
        """
        生成极轻微的非线性 RGB 曲线查找表 (256x1x3 uint8)，打破 RGB 直方图分布
        """
        x = np.arange(256, dtype=np.float32)
        lut = np.zeros((256, 1, 3), dtype=np.uint8)

        for c in range(3):  # B, G, R 通道
            # 利用微小 S 型或 Sines 变换微调 RGB 曲线
            shift_amplitude = random.uniform(-self.lut_jitter_range, self.lut_jitter_range)
            curve = x + shift_amplitude * np.sin(np.pi * x / 255.0)
            curve = np.clip(curve, 0, 255).astype(np.uint8)
            lut[:, 0, c] = curve

        return lut

    def apply_uv_channel_jitter(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        1. YCrCb 色度通道（Cr / Cb）加噪
        Y 通道（亮度）保持不变，对 Cr/Cb（色度）通道注入微量高斯噪声
        """
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y, cr, cb = cv2.split(ycrcb)

        # 生成与 Cr/Cb 同尺寸的高斯噪声
        noise_cr = np.random.normal(0, self.uv_noise_std, cr.shape).astype(np.float32)
        noise_cb = np.random.normal(0, self.uv_noise_std, cb.shape).astype(np.float32)

        cr = np.clip(cr + noise_cr, 0, 255)
        cb = np.clip(cb + noise_cb, 0, 255)

        merged_ycrcb = cv2.merge([y, cr, cb]).astype(np.uint8)
        return cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)

    def apply_dct_frequency_shift(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        2. 8x8 块 DCT 高频系数微扰动
        对画面分块执行 2D DCT，仅在大气高频（右下角系数）中添加噪声
        """
        if not self.enable_dct:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        # 裁剪为 8 的倍数尺寸方便 8x8 DCT 分块计算
        h_8, w_8 = (h // 8) * 8, (w // 8) * 8
        if h_8 == 0 or w_8 == 0:
            return frame_bgr

        img_crop = frame_bgr[:h_8, :w_8].astype(np.float32)
        output = img_crop.copy()

        # 构造高频掩码 Mask：8x8 矩阵中，右下角（u+v >= 6）代表高频成分
        hf_mask = np.zeros((8, 8), dtype=np.float32)
        for u in range(8):
            for v in range(8):
                if u + v >= 6:  # 高频区域
                    hf_mask[u, v] = 1.0

        for ch in range(3):
            channel_data = img_crop[:, :, ch]
            # 分块 DCT
            blocks = channel_data.reshape(h_8 // 8, 8, w_8 // 8, 8).swapaxes(1, 2)

            # 随机生成高频噪声
            noise = np.random.normal(0, self.dct_freq_noise_std, blocks.shape).astype(np.float32)
            noise *= hf_mask  # 仅保留高频部分

            # 对每个 8x8 块执行 DCT -> 叠加高频噪声 -> IDCT
            for i in range(h_8 // 8):
                for j in range(w_8 // 8):
                    dct_block = cv2.dct(blocks[i, j])
                    dct_block += noise[i, j]
                    idct_block = cv2.idct(dct_block)
                    output[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8, ch] = idct_block

        output = np.clip(output, 0, 255).astype(np.uint8)

        # 将边缘未切分部分拼回
        res = frame_bgr.copy()
        res[:h_8, :w_8] = output
        return res

    def apply_lut_mapping(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        3. 应用微调 LUT 表
        """
        return cv2.LUT(frame_bgr, self._lut_table)

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        对单帧依次施加：LUT 色彩映射 -> U/V 色度抖动 -> DCT 高频扰动
        """
        # 1. LUT 曲线微调
        frame = self.apply_lut_mapping(frame_bgr)
        # 2. U/V 通道微抖动
        frame = self.apply_uv_channel_jitter(frame)
        # 3. 高频 DCT 扰动
        frame = self.apply_dct_frequency_shift(frame)
        return frame