import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from shuffler.provider import PatchProvider
from processors.haar_injector import HaarROIInjector
from processors.yolo_injector import YoloROIInjector
from processors.spatial_warper import SpatialAffineWarper
from processors.color_frequency_shifter import ColorFrequencyShifter
from processors.temporal_restructurer import TemporalRestructurer  # 👈 导入时域重构处理器


def process_video(
        input_path: str,
        output_path: str,
        detector_type: str = "yolo",
        yolo_model: str = "yolov8n.pt",
        enable_spatial: bool = True,
        enable_color_shift: bool = True,
        enable_temporal: bool = True,  # 👈 时域重构开关
):
    if not os.path.exists(input_path):
        print(f"错误: 输入视频路径不存在 -> {input_path}")
        return

    provider = PatchProvider()

    # 1. 初始化各模块处理器
    if detector_type == "haar":
        injector = HaarROIInjector(provider=provider, alpha_limit=0.03)
    elif detector_type == "yolo":
        injector = YoloROIInjector(
            provider=provider,
            model_path=yolo_model,
            classes=[0],
            conf_threshold=0.3,
            alpha_limit=0.03,
        )

    spatial_warper = SpatialAffineWarper() if enable_spatial else None
    color_shifter = ColorFrequencyShifter(enable_dct=True) if enable_color_shift else None
    temporal_restructurer = TemporalRestructurer() if enable_temporal else None

    # 2. 读取视频元信息
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print("🎥 开始读取并处理视频帧...")

    # 3. 计算时域采样映射矩阵
    if temporal_restructurer is not None:
        source_indices = temporal_restructurer.calculate_frame_mapping(total_frames)
    else:
        source_indices = [float(i) for i in range(total_frames)]

    # 预加载所有输入帧（若视频极大，可按需分块流式加载）
    raw_frames = []
    pbar_read = tqdm(total=total_frames, desc="预读帧数据")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
        pbar_read.update(1)

    pbar_read.close()
    cap.release()

    max_src_len = len(raw_frames)
    if max_src_len == 0:
        print("错误: 未从输入视频中读取到任何有效帧！")
        return

    pbar_proc = tqdm(total=len(source_indices), desc="多维重构渲染进度")

    # 4. 逐帧执行多维混淆处理流水线
    for out_idx, src_float_idx in enumerate(source_indices):
        idx_floor = int(np.floor(src_float_idx))
        idx_ceil = min(idx_floor + 1, max_src_len - 1)
        alpha = src_float_idx - idx_floor

        idx_floor = min(idx_floor, max_src_len - 1)

        # A. 时域重构：非线性变速的亚帧双帧融合
        if temporal_restructurer is not None and idx_floor != idx_ceil:
            frame = temporal_restructurer.interpolate_frame(
                raw_frames[idx_floor], raw_frames[idx_ceil], alpha
            )
        else:
            frame = raw_frames[idx_floor].copy()

        # B. 时域重构：镜头切点检测与过渡高频噪声注入
        if temporal_restructurer is not None:
            if temporal_restructurer.detect_scene_cut(frame):
                frame = temporal_restructurer.inject_cut_transition_noise(frame)

        # C. 空间微透贴图植入 (YOLO / Haar)
        frame = injector.inject_to_frame(frame)

        # D. 色彩空间与 DCT 高频通道抖动
        if color_shifter is not None:
            frame = color_shifter.process_frame(frame)

        # E. 空间与仿射微变 (呼吸裁切 + 仿射扭曲)
        if spatial_warper is not None:
            frame = spatial_warper.process_frame(frame, out_idx)

        writer.write(frame)
        pbar_proc.update(1)

    writer.release()
    pbar_proc.close()
    print(f"\n✨ 全部重构处理完成！文件已输出至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="视频多维对抗去重重构全套工具")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", default="output/processed.mp4", help="输出视频路径")
    parser.add_argument("--detector", "-d", choices=["haar", "yolo"], default="yolo")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--no-spatial", action="store_true", help="禁用空间与仿射微变")
    parser.add_argument("--no-color", action="store_true", help="禁用色彩/频域通道抖动")
    parser.add_argument("--no-temporal", action="store_true", help="禁用时域变速与切点重构")

    args = parser.parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        detector_type=args.detector,
        yolo_model=args.yolo_model,
        enable_spatial=not args.no_spatial,
        enable_color_shift=not args.no_color,
        enable_temporal=not args.no_temporal,
    )


if __name__ == "__main__":
    main()