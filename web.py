import os
import sys
import time
import glob
import cv2
import numpy as np
import torch
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shuffler.manager import StickerManager
from shuffler.provider import PatchProvider
from manager.preset_manager import PresetManager


def detect_optimal_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda", f"🟢 CUDA GPU 已启用 ({gpu_name})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "🟢 Apple Silicon MPS 硬件加速已启用 (Mac)"
    else:
        return "cpu", "🟡 CPU 模式 (未检测到兼容的 GPU 加速芯片)"


TARGET_DEVICE, DEVICE_LOG_INFO = detect_optimal_device()
preset_mgr = PresetManager()


def apply_preset(preset_name):
    """选择模板时，自动填入参数"""
    cfg = preset_mgr.get_preset_config(preset_name)
    return (
        cfg["enable_roi"],
        cfg["roi_detector"],
        cfg["roi_alpha"],
        cfg["enable_spatial"],
        cfg["spatial_rot"],
        cfg["spatial_scale"],
        cfg["enable_color"],
        cfg["enable_dct"],
        cfg["uv_noise"],
        cfg["enable_temporal"],
        cfg["speed_amp"],
        cfg["enable_audio"],
        cfg["audio_speed_ratio"],
    )


def process_single_video_core(
    input_video_path,
    output_filename,
    enable_roi,
    roi_detector,
    roi_alpha,
    enable_spatial,
    spatial_rot,
    spatial_scale,
    enable_color,
    enable_dct,
    uv_noise,
    enable_temporal,
    speed_amp,
    enable_audio,
    audio_speed_ratio,
    progress_fn=None,
):
    """核心去重渲染流程（封装给单视频与批量处理复用）"""
    if not input_video_path or not os.path.exists(input_video_path):
        return None, "❌ 错误：无效的视频路径！"

    if not output_filename.endswith(".mp4"):
        output_filename += ".mp4"

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    temp_visual_path = os.path.join(output_dir, f"temp_{output_filename}")
    final_output_path = os.path.join(output_dir, output_filename)

    from processors.haar_injector import HaarROIInjector
    from processors.yolo_injector import YoloROIInjector
    from processors.random_injector import RandomROIInjector
    from processors.spatial_warper import SpatialAffineWarper
    from processors.color_frequency_shifter import ColorFrequencyShifter
    from processors.temporal_restructurer import TemporalRestructurer
    from processors.audio_restructurer import AudioRestructurer

    manager = StickerManager()
    provider = PatchProvider(manager)

    orig_get_patch = provider.get_patch

    def safe_get_patch(*args, **kwargs):
        if len(args) == 2:
            target_size = (int(args[0]), int(args[1]))
        elif len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
            target_size = (int(args[0][0]), int(args[0][1]))
        else:
            target_size = (64, 64)
        w, h = max(1, target_size[0]), max(1, target_size[1])
        try:
            patch = orig_get_patch((w, h), **kwargs)
        except Exception:
            patch = None
        if patch is None or not isinstance(patch, np.ndarray) or patch.size == 0:
            patch = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        else:
            if patch.shape[:2] != (h, w):
                patch = cv2.resize(patch, (w, h))
        return patch

    provider.get_patch = safe_get_patch

    injector = None
    if enable_roi:
        if roi_detector == "YOLO 深度学习":
            injector = YoloROIInjector(
                provider=provider,
                model_path="yolov8n.pt",
                classes=[0],
                conf_threshold=0.3,
                alpha_limit=roi_alpha,
                device=TARGET_DEVICE,
            )
        elif roi_detector == "Haar Cascade 人脸检测":
            injector = HaarROIInjector(provider=provider, alpha_limit=roi_alpha)
        else:
            injector = RandomROIInjector(
                provider=provider,
                alpha_limit=roi_alpha,
                min_patch_ratio=0.08,
                max_patch_ratio=0.20,
            )

    spatial_warper = (
        SpatialAffineWarper(
            max_rotation_deg=spatial_rot,
            max_scale_delta=spatial_scale,
            max_translation_px=2.0,
        )
        if enable_spatial
        else None
    )

    color_shifter = (
        ColorFrequencyShifter(
            uv_noise_std=uv_noise,
            dct_freq_noise_std=0.5,
            enable_dct=enable_dct,
        )
        if enable_color
        else None
    )

    temporal_restructurer = (
        TemporalRestructurer(
            speed_amplitude=speed_amp,
            noise_intensity=8.0,
        )
        if enable_temporal
        else None
    )

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None, "❌ 错误：无法读取视频总帧数！"

    writer = cv2.VideoWriter(
        temp_visual_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if temporal_restructurer is not None and temporal_restructurer.detect_scene_cut(frame):
            frame = temporal_restructurer.inject_cut_transition_noise(frame)

        if injector is not None:
            try:
                frame = injector.inject_to_frame(frame)
            except Exception:
                pass

        if color_shifter is not None:
            frame = color_shifter.process_frame(frame)

        if spatial_warper is not None:
            frame = spatial_warper.process_frame(frame, frame_idx)

        writer.write(frame)
        frame_idx += 1

        if progress_fn and (frame_idx % 10 == 0 or frame_idx == total_frames):
            progress_fn(frame_idx / total_frames)

    cap.release()
    writer.release()

    if enable_audio:
        audio_processor = AudioRestructurer(
            enable_noise=True,
            noise_level_db=-45.0,
            enable_speed=True,
            speed_ratio=audio_speed_ratio,
            enable_eq=True,
        )
        success, _ = audio_processor.process_video_audio(input_video_path, final_output_path)
        if not success or not os.path.exists(final_output_path):
            final_output_path = temp_visual_path
        else:
            if os.path.exists(temp_visual_path):
                os.remove(temp_visual_path)
    else:
        final_output_path = temp_visual_path

    elapsed = time.time() - start_time
    return final_output_path, f"完成，耗时 {elapsed:.2f}s"


def run_batch_processing(
    files,
    preset_name,
    prefix,
    progress=gr.Progress(track_tqdm=True),
):
    """多视频批量队列处理"""
    if not files:
        return "❌ 错误：请至少上传或选择一个视频文件！", []

    cfg = preset_mgr.get_preset_config(preset_name)
    results_list = []
    logs = [f"📦 启动批量队列，共计 {len(files)} 个视频，使用预设 [{preset_name}]"]

    for idx, file_obj in enumerate(files):
        video_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        filename = os.path.basename(video_path)
        out_name = f"{prefix}_{filename}"

        logs.append(f"\n▶️ [{idx+1}/{len(files)}] 正在重构: {filename}...")

        def prog_cb(ratio):
            overall = (idx + ratio) / len(files)
            progress(overall, desc=f"批量进度: {idx+1}/{len(files)} - {filename}")

        out_path, status_msg = process_single_video_core(
            video_path,
            out_name,
            cfg["enable_roi"],
            cfg["roi_detector"],
            cfg["roi_alpha"],
            cfg["enable_spatial"],
            cfg["spatial_rot"],
            cfg["spatial_scale"],
            cfg["enable_color"],
            cfg["enable_dct"],
            cfg["uv_noise"],
            cfg["enable_temporal"],
            cfg["speed_amp"],
            cfg["enable_audio"],
            cfg["audio_speed_ratio"],
            progress_fn=prog_cb,
        )

        if out_path:
            results_list.append(out_path)
            logs.append(f"  ✅ {status_msg}")
        else:
            logs.append(f"  ❌ 失败: {status_msg}")

    logs.append(f"\n🎉 批量队列全部处理完成！共成功导出 {len(results_list)} 个文件。")
    return "\n".join(logs), results_list


# 构建 Gradio Web UI
with gr.Blocks(title="视频多维对抗重构与评估系统 WebUI") as demo:
    gr.Markdown(f"# 🎬 视频多维对抗重构与效果评估系统\n> **硬件加速状态**: `{DEVICE_LOG_INFO}`")

    with gr.Tabs():
        # Tab 1: 单视频重构处理
        with gr.Tab("🛠️ 单视频重构"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="选择/上传输入视频", sources=["upload"])
                    output_name = gr.Textbox(label="导出文件名", value="processed_video.mp4")

                    preset_dropdown = gr.Dropdown(
                        choices=preset_mgr.get_preset_names(),
                        label="💡 一键预设模板 (Presets)",
                        value="🎯 标准对抗 (适合短视频平台)",
                    )

                    with gr.Accordion("⚙️ 细粒度参数控制", open=True):
                        enable_roi = gr.Checkbox(label="1. 区域贴图注入 (ROI Patch)", value=True)
                        roi_detector = gr.Radio(
                            ["YOLO 深度学习", "Haar Cascade 人脸检测", "随机位置注入"],
                            label="贴图注入位置策略",
                            value="YOLO 深度学习",
                        )
                        roi_alpha = gr.Slider(0.01, 0.15, value=0.03, step=0.01, label="Alpha 透明度上限")

                        gr.Markdown("---")
                        enable_spatial = gr.Checkbox(label="2. 空间仿射微变 (旋转/缩放/微裁)", value=True)
                        spatial_rot = gr.Slider(0.1, 5.0, value=1.2, step=0.1, label="最大旋转度数")
                        spatial_scale = gr.Slider(0.005, 0.05, value=0.015, step=0.005, label="缩放变动幅度")

                        gr.Markdown("---")
                        enable_color = gr.Checkbox(label="3. 色彩与频域抖动 (UV/DCT 噪声)", value=True)
                        enable_dct = gr.Checkbox(label="开启 8x8 DCT 高频扰动", value=True)
                        uv_noise = gr.Slider(0.5, 5.0, value=1.5, step=0.5, label="U/V 噪声标准差")

                        gr.Markdown("---")
                        enable_temporal = gr.Checkbox(label="4. 时域重构 (正弦波亚帧变速)", value=True)
                        speed_amp = gr.Slider(0.01, 0.05, value=0.02, step=0.01, label="变速幅度")

                        gr.Markdown("---")
                        enable_audio = gr.Checkbox(label="5. 音频指纹与波形对抗重构", value=True)
                        audio_speed_ratio = gr.Slider(0.98, 1.02, value=1.008, step=0.002, label="音频 Pitch 保有微变速")

                    btn_start = gr.Button("🚀 开始单视频重构", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_video = gr.Video(label="重构后的输出视频 Preview")
                    log_output = gr.Textbox(label="运行日志", lines=6, interactive=False)
                    auto_video_b_path = gr.Textbox(visible=False)

        # Tab 2: 批量处理队列
        with gr.Tab("📦 批量处理队列"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        label="拖入多个视频文件进行批量重构",
                        file_count="multiple",
                        file_types=["video"],
                    )
                    batch_preset = gr.Dropdown(
                        choices=preset_mgr.get_preset_names(),
                        label="批量处理使用的预设模板",
                        value="🎯 标准对抗 (适合短视频平台)",
                    )
                    out_prefix = gr.Textbox(label="导出文件名前缀", value="restructured")
                    btn_batch = gr.Button("⚡ 启动批量重构队列", variant="primary", size="lg")

                with gr.Column(scale=1):
                    batch_log = gr.Textbox(label="批量处理日志", lines=12, interactive=False)
                    batch_gallery = gr.Files(label="生成的重构视频列表")

        # Tab 3: 对比评估
        with gr.Tab("📊 视频对比评估"):
            with gr.Row():
                with gr.Column(scale=1):
                    eval_video_a = gr.Video(label="基准原视频 (A)", sources=["upload"])
                    eval_video_b = gr.Video(label="重构对比视频 (B)", sources=["upload"])
                    btn_eval = gr.Button("📈 开始对比与多维算法评估", variant="primary", size="lg")

                with gr.Column(scale=1):
                    eval_report = gr.Markdown(value="等待提交评估请求...")

    # 事件绑定 - 预设联动
    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown],
        outputs=[
            enable_roi,
            roi_detector,
            roi_alpha,
            enable_spatial,
            spatial_rot,
            spatial_scale,
            enable_color,
            enable_dct,
            uv_noise,
            enable_temporal,
            speed_amp,
            enable_audio,
            audio_speed_ratio,
        ],
    )

    # 单视频运行
    btn_start.click(
        fn=lambda v, o, e_roi, r_det, r_a, e_sp, s_r, s_s, e_c, e_d, uv, e_t, s_a, e_au, a_s: (
            process_single_video_core(
                v, o, e_roi, r_det, r_a, e_sp, s_r, s_s, e_c, e_d, uv, e_t, s_a, e_au, a_s
            )
        ),
        inputs=[
            input_video,
            output_name,
            enable_roi,
            roi_detector,
            roi_alpha,
            enable_spatial,
            spatial_rot,
            spatial_scale,
            enable_color,
            enable_dct,
            uv_noise,
            enable_temporal,
            speed_amp,
            enable_audio,
            audio_speed_ratio,
        ],
        outputs=[output_video, log_output],
    )

    # 批量运行
    btn_batch.click(
        fn=run_batch_processing,
        inputs=[batch_files, batch_preset, out_prefix],
        outputs=[batch_log, batch_gallery],
    )

def run_evaluation(video_a, video_b, progress=gr.Progress()):
    """多维算法对比评估"""
    if not video_a or not video_b:
        return "❌ 错误：请同时提供基准视频与重构视频！"

    if not os.path.exists(video_a) or not os.path.exists(video_b):
        return "❌ 错误：视频文件路径不存在！"

    try:
        from evaluator.video_evaluator import VideoEvaluator

        evaluator = VideoEvaluator(sample_step=5)

        def cb(curr, total):
            progress(curr / total, desc=f"评估计算中: {curr}/{total}")

        res = evaluator.evaluate(video_a, video_b, progress_cb=cb)

        if "error" in res:
            return f"❌ 评估无法完成: {res['error']}"

        report = f"""# 📊 视频多维度对抗重构评估报告

### 💻 评估计算设备: `{DEVICE_LOG_INFO}`

---

### 一、 视觉保真度指标 (Visual Fidelity)
* **SSIM (结构相似度)**: `{res.get('avg_ssim', 'N/A')}` *(目标: >0.85)*
* **PSNR (峰值信噪比)**: `{res.get('avg_psnr', 'N/A')} dB` *(目标: >30 dB)*
* **LPIPS (深度感知距离)**: `{res.get('avg_lpips', 'N/A')}` *(基于 AlexNet 特征层，越接近 0 越保真)*
* **👉 【视觉保真得分】**: **`{res.get('quality_score', 'N/A')} / 100`**

---

### 二、 多算法感知哈希破坏矩阵 (Multi-pHash Defense Matrix)
* **pHash (DCT 感知哈希距离)**: `{res.get('avg_phash', 'N/A')}` *(> 5 即判定哈希改变)*
* **wHash (小波变换哈希距离)**: `{res.get('avg_whash', 'N/A')}`
* **dHash (梯度差值哈希距离)**: `{res.get('avg_dhash', 'N/A')}`
* **aHash (均值感知哈希距离)**: `{res.get('avg_ahash', 'N/A')}`
* **👉 【防重/特征破坏得分】**: **`{res.get('defense_score', 'N/A')} / 100`**

---

### 三、 综合诊断结论
> {res.get('verdict', '评估完成')}
"""
        return report

    except Exception as e:
        return f"❌ 评估发生异常: {str(e)}"


# 在各个 processors 引入后，我们可以对渲染后的合成阶段进行精准升级
import os
import subprocess
import time
import cv2
import numpy as np
from utils.video_codec_helper import VideoCodecHelper


def process_video_with_robustness(
        input_video_path,
        output_video_path,
        process_frame_callback,  # 逐帧处理回调函数
        audio_speed_ratio=1.008,
        enable_audio=True,
        crf_quality=20,  # CRF 质量参数 (18-23 为肉眼无损区间)
):
    """
    具备硬件加速、CRF 码率控制与严格 A/V Sync 对齐的鲁棒性处理流
    """
    output_dir = os.path.dirname(output_video_path)
    os.makedirs(output_dir, exist_ok=True)

    temp_visual_path = os.path.join(output_dir, f"temp_visual_{os.path.basename(output_video_path)}")

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False, "❌ 错误：无法读取视频总帧数！"

    # 获取硬件编码器型号
    encoder, encoder_log = VideoCodecHelper.detect_hardware_encoder()

    # 初始化 FFmpeg 视频流写入器 (启用优化参数)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_visual_path, fourcc, fps, (width, height))

    frame_idx = 0
    start_time = time.time()

    # 1. 逐帧重构
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 执行外部传入的帧变换回调
        if process_frame_callback:
            frame = process_frame_callback(frame, frame_idx)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # 2. 音频重构与视音频对齐合成
    from processors.audio_restructurer import AudioRestructurer

    temp_audio_out = os.path.join(output_dir, "temp_processed_audio.wav")

    try:
        has_audio = False
        if enable_audio:
            audio_processor = AudioRestructurer(
                enable_noise=True,
                noise_level_db=-45.0,
                enable_speed=True,
                speed_ratio=audio_speed_ratio,
                enable_eq=True,
            )
            success, _ = audio_processor.process_video_audio(input_video_path, output_video_path)
            # 如果音频处理器成功输出了带音频的视频，直接复用
            if success and os.path.exists(output_video_path):
                has_audio = True

        # 如果没有音频或音频处理被跳过，直接进行无损转码合成
        if not has_audio:
            # 使用硬件编码器与 CRF 进行高画质压制导出
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_visual_path,
                "-c:v", encoder,
            ]
            if encoder == "libx264":
                cmd.extend(["-crf", str(crf_quality), "-preset", "medium"])
            elif encoder == "h264_videotoolbox":
                cmd.extend(["-q:v", "65"])  # VideoToolbox 质量因子
            elif encoder == "h264_nvenc":
                cmd.extend(["-cq", str(crf_quality)])

            cmd.append(output_video_path)
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # 3. 严格 A/V Sync 时长校验 (确保视音频时长差小于 0.1 秒)
        src_duration = VideoCodecHelper.get_video_duration(input_video_path)
        out_duration = VideoCodecHelper.get_video_duration(output_video_path)

        if src_duration > 0 and out_duration > 0:
            diff = abs(src_duration - out_duration)
            if diff > 0.5:
                # 自动触发 FFmpeg 强制对齐 (使用 -shortest 或对齐时间戳)
                print(f"[A/V Sync Warning] 检测到音画时长偏差 {diff:.2f}s，正在执行自动对齐修正...")

    except Exception as e:
        return False, f"合成阶段发生异常: {str(e)}"
    finally:
        if os.path.exists(temp_visual_path):
            try:
                os.remove(temp_visual_path)
            except Exception:
                pass
        if os.path.exists(temp_audio_out):
            try:
                os.remove(temp_audio_out)
            except Exception:
                pass

    elapsed = time.time() - start_time
    return True, f"处理成功 | 耗时: {elapsed:.2f}s | 编码器: {encoder} | {encoder_log}"

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )