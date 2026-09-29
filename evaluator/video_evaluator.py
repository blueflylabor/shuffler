import cv2
import numpy as np
import torch
import pywt
import lpips
from torchvision import transforms


class VideoEvaluator:
    """综合视频重构评估器：支持 LPIPS 深度特征保真度与 Multi-pHash 防重破坏评估"""

    def __init__(self, sample_step=5, device=None):
        """
        :param sample_step: 采帧步长（每隔 step 帧采样一帧）
        :param device: 推理设备 (cuda, mps, cpu)
        """
        self.sample_step = sample_step

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 1. 初始化 LPIPS 深度感知损失模型 (AlexNet)
        try:
            self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device)
            self.lpips_enabled = True
        except Exception:
            self.lpips_enabled = False

        # 图像预处理 Tensor 转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # ------------------ 感知哈希矩阵 (Multi-pHash) ------------------
    @staticmethod
    def calc_ahash(img, hash_size=8):
        """均值哈希 (aHash)"""
        resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        return (gray > avg).flatten()

    @staticmethod
    def calc_dhash(img, hash_size=8):
        """差值哈希 (dHash)"""
        resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        diff = gray[:, 1:] > gray[:, :-1]
        return diff.flatten()

    @staticmethod
    def calc_phash(img, hash_size=8, highfreq_factor=4):
        """感知 DCT 哈希 (pHash)"""
        img_size = hash_size * highfreq_factor
        resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dct = cv2.dct(gray)
        dct_low = dct[:hash_size, :hash_size]
        avg = (dct_low.sum() - dct_low[0, 0]) / (hash_size * hash_size - 1)
        return (dct_low > avg).flatten()

    @staticmethod
    def calc_whash(img, hash_size=8):
        """小波变换哈希 (wHash)"""
        resized = cv2.resize(img, (hash_size * 2, hash_size * 2), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        coeffs = pywt.dwt2(gray, 'haar')
        LL, _ = coeffs
        LL_resized = cv2.resize(LL, (hash_size, hash_size))
        avg = LL_resized.mean()
        return (LL_resized > avg).flatten()

    @staticmethod
    def hamming_distance(h1, h2):
        """计算汉明距离"""
        return int(np.count_nonzero(h1 != h2))

    # ------------------ 保真度指标 (SSIM / PSNR / LPIPS) ------------------
    @staticmethod
    def calc_psnr(img1, img2):
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return 100.0
        return 20.0 * np.log10(255.0 / np.sqrt(mse))

    @staticmethod
    def calc_ssim(img1, img2):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(g1, -1, window)
        mu2 = cv2.filter2D(g2, -1, window)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = cv2.filter2D(g1 ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(g2 ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(g1 * g2, -1, window) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(ssim_map.mean())

    def calc_lpips(self, img1, img2):
        if not self.lpips_enabled:
            return 0.0

        # RGB 转换 & Tensor 标准化
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        t1 = self.transform(rgb1).unsqueeze(0).to(self.device)
        t2 = self.transform(rgb2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.lpips_fn(t1, t2)
        return float(dist.item())

    # ------------------ 主评估函数 ------------------
    def evaluate(self, video_a_path, video_b_path, progress_cb=None):
        cap_a = cv2.VideoCapture(video_a_path)
        cap_b = cv2.VideoCapture(video_b_path)

        total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
        total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = min(total_a, total_b)

        if total_frames <= 0:
            cap_a.release()
            cap_b.release()
            return {"error": "无法读取视频帧，请检查文件权限或格式！"}

        ssim_list, psnr_list, lpips_list = [], [], []
        ahash_dist_list, dhash_dist_list, phash_dist_list, whash_dist_list = [], [], [], []

        frame_idx = 0
        sampled_count = 0

        while True:
            ret_a, frame_a = cap_a.read()
            ret_b, frame_b = cap_b.read()

            if not ret_a or not ret_b:
                break

            if frame_idx % self.sample_step == 0:
                if frame_a.shape != frame_b.shape:
                    frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))

                # 1. 结构与感官画质计算
                ssim_list.append(self.calc_ssim(frame_a, frame_b))
                psnr_list.append(self.calc_psnr(frame_a, frame_b))
                if self.lpips_enabled:
                    lpips_list.append(self.calc_lpips(frame_a, frame_b))

                # 2. 多算法感知哈希矩阵计算
                ha_a, ha_b = self.calc_ahash(frame_a), self.calc_ahash(frame_b)
                hd_a, hd_b = self.calc_dhash(frame_a), self.calc_dhash(frame_b)
                hp_a, hp_b = self.calc_phash(frame_a), self.calc_phash(frame_b)
                hw_a, hw_b = self.calc_whash(frame_a), self.calc_whash(frame_b)

                ahash_dist_list.append(self.hamming_distance(ha_a, ha_b))
                dhash_dist_list.append(self.hamming_distance(hd_a, hd_b))
                phash_dist_list.append(self.hamming_distance(hp_a, hp_b))
                whash_dist_list.append(self.hamming_distance(hw_a, hw_b))

                sampled_count += 1
                if progress_cb:
                    progress_cb(frame_idx, total_frames)

            frame_idx += 1

        cap_a.release()
        cap_b.release()

        avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0

        avg_ahash = float(np.mean(ahash_dist_list)) if ahash_dist_list else 0.0
        avg_dhash = float(np.mean(dhash_dist_list)) if dhash_dist_list else 0.0
        avg_phash = float(np.mean(phash_dist_list)) if phash_dist_list else 0.0
        avg_whash = float(np.mean(whash_dist_list)) if whash_dist_list else 0.0

        # 画质得分 (SSIM 占比 60%, PSNR 占比 40%)
        quality_score = min(100.0, max(0.0, (avg_ssim * 60) + (min(avg_psnr, 45.0) / 45.0 * 40)))

        # 综合哈希破坏率（多维哈希距离综合评估）
        avg_dist_all = (avg_phash * 1.2 + avg_whash * 1.0 + avg_dhash * 0.8 + avg_ahash * 0.6) / 3.6
        defense_score = min(100.0, (avg_dist_all / 12.0) * 100)

        # 诊断报告结论
        if quality_score >= 80.0 and defense_score >= 50.0:
            verdict = "✅ **极佳平衡**: 视频在保持高视觉质感的同时，完成了深度的底层感知哈希与特征破坏！"
        elif quality_score < 70.0:
            verdict = "⚠️ **画质过低**: 重构强度过高，画质损伤较为明显，建议调低 Alpha 或旋转缩放参数。"
        elif defense_score < 30.0:
            verdict = "⚠️ **防重强度不足**: 底层特征变动较小，可能面临平台查重风险，建议开启 DCT 频域抖动与音频重构。"
        else:
            verdict = "🟢 **中规中矩**: 满足基础去重需求，画质与去重效果达到良好平衡。"

        return {
            "avg_ssim": round(avg_ssim, 4),
            "avg_psnr": round(avg_psnr, 2),
            "avg_lpips": round(avg_lpips, 4),
            "avg_ahash": round(avg_ahash, 2),
            "avg_dhash": round(avg_dhash, 2),
            "avg_phash": round(avg_phash, 2),
            "avg_whash": round(avg_whash, 2),
            "quality_score": round(quality_score, 1),
            "defense_score": round(defense_score, 1),
            "verdict": verdict,
        }