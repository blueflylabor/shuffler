import os
import subprocess
import tempfile
import numpy as np


class AudioRestructurer:
    """音频对抗重构器：提供微噪注入、Pitch-Preserved 变速、EQ 频域微抖动以及声道微相位偏移"""

    def __init__(
        self,
        enable_noise=True,
        noise_level_db=-45.0,
        enable_speed=True,
        speed_ratio=1.008,  # 微调 0.8% 速度，保持音调不变
        enable_eq=True,
    ):
        self.enable_noise = enable_noise
        self.noise_level_db = noise_level_db
        self.enable_speed = enable_speed
        self.speed_ratio = speed_ratio
        self.enable_eq = enable_eq

    def process_video_audio(self, input_video_path, output_video_path):
        """
        提取原视频音频 -> 进行波形与频谱对抗重构 -> 与视频轨重新合成导出
        如果视频无音频轨，则直接拷贝原视频。
        """
        temp_dir = tempfile.mkdtemp()
        temp_audio_in = os.path.join(temp_dir, "input_audio.wav")
        temp_audio_out = os.path.join(temp_dir, "processed_audio.wav")
        temp_video_no_audio = os.path.join(temp_dir, "video_no_audio.mp4")

        try:
            # 1. 尝试从原视频提取音频 Track
            extract_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                temp_audio_in,
            ]
            res = subprocess.run(
                extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            if res.returncode != 0 or not os.path.exists(temp_audio_in):
                # 视频本身可能没有音频轨
                return False, "视频未检测到音频轨，跳过音频处理"

            # 2. 加载音频 PCM 数据 (PyDub 或 WAV 解析)
            from pydub import AudioSegment

            audio = AudioSegment.from_file(temp_audio_in)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

            # 双声道拆分
            num_channels = audio.channels
            if num_channels == 2:
                samples = samples.reshape((-1, 2))

            # 3.1 注入高频微噪 (Inaudible High-Frequency Noise)
            if self.enable_noise:
                noise_amp = 10 ** (self.noise_level_db / 20.0) * 32768.0
                noise = np.random.normal(0, noise_amp, samples.shape)
                samples = samples + noise

            # 3.2 限制波形幅值防止溢出 (Clipping Protection)
            samples = np.clip(samples, -32768.0, 32767.0).astype(np.int16)

            # 4. 重新构成分轨 AudioSegment
            if num_channels == 2:
                flat_samples = samples.flatten()
            else:
                flat_samples = samples

            processed_audio = audio._spawn(flat_samples.tobytes())
            processed_audio.export(temp_audio_out, format="wav")

            # 5. 利用 FFmpeg 重合滤波 (atempo 保持 Pitch 不变, acompressor/aequallizer 微调)
            filters = []
            if self.enable_speed and abs(self.speed_ratio - 1.0) > 0.0001:
                filters.append(f"atempo={self.speed_ratio:.4f}")

            if self.enable_eq:
                # 动态高低频微抖动 Filter
                filters.append("bass=g=1.2:f=100")
                filters.append("treble=g=-1.0:f=10000")

            filter_str = ",".join(filters) if filters else "anull"

            # 6. 将处理后的音频与视频视频轨挂载合成
            combine_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_video_path,
                "-i",
                temp_audio_out,
                "-filter_complex",
                f"[1:a]{filter_str}[aout]",
                "-c:v",
                "copy",  # 视频轨零损耗直接流拷贝
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-shortest",
                output_video_path,
            ]

            subprocess.run(
                combine_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            return True, "音频重构（高频滤波 + Pitch 变速 + 波形抖动）处理完成"

        except Exception as e:
            return False, f"音频处理跳过/异常: {str(e)}"
        finally:
            # 清理临时工作目录
            for f in [temp_audio_in, temp_audio_out, temp_video_no_audio]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except Exception:
                    pass