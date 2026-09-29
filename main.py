import argparse
import os
import cv2
from tqdm import tqdm

from shuffler.provider import PatchProvider
from processors.haar_injector import HaarROIInjector
from processors.yolo_injector import YoloROIInjector
from processors.spatial_warper import SpatialAffineWarper
from processors.color_frequency_shifter import ColorFrequencyShifter  # 👈【新增导入】


def process_video(
    input_path: str,
    output_path: str,
    detector_type: str = "yolo",
    yolo_model: str = "yolov8n.pt",
    enable_spatial: bool = True,
    enable_color_shift: bool = True,  # 👈 控制色彩/频域模块
):
    if not os.path.exists(input_path):
        print(f"错误: 输入视频路径不存在 -> {input_path}")
        return

    provider = PatchProvider()

    # 1. ROI 贴图注入器
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

    # 2. 空间仿射微变
    spatial_warper = SpatialAffineWarper() if enable_spatial else None

    # 3. 色彩与高频通道抖动
    color_shifter = ColorFrequencyShifter(enable_dct=True) if enable_color_shift else None

    # 4. 视频流读写
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    pbar = tqdm(total=total_frames, desc="视频处理进度")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理流水线 1：ROI 对抗微透贴图
        frame = injector.inject_to_frame(frame)

        # 处理流水线 2：色彩空间与 DCT 高频抖动
        if color_shifter is not None:
            frame = color_shifter.process_frame(frame)

        # 处理流水线 3：空间与仿射微变 (呼吸裁剪 + 仿射扭曲)
        if spatial_warper is not None:
            frame = spatial_warper.process_frame(frame, frame_idx)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    print(f"处理完成，文件已输出至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="视频多维重构与去重处理工具")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", default="output/processed.mp4", help="输出视频路径")
    parser.add_argument("--detector", "-d", choices=["haar", "yolo"], default="yolo")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--no-spatial", action="store_true", help="禁用空间与仿射微变")
    parser.add_argument("--no-color", action="store_true", help="禁用色彩/频域通道抖动")

    args = parser.parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        detector_type=args.detector,
        yolo_model=args.yolo_model,
        enable_spatial=not args.no_spatial,
        enable_color_shift=not args.no_color,
    )


if __name__ == "__main__":
    main()