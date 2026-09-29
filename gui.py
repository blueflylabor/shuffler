import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QCheckBox, QRadioButton, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QDoubleSpinBox, QFormLayout,
    QMessageBox, QTabWidget, QScrollArea
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt


# ==========================================
# 后台评估线程 (延迟加载评估模块)
# ==========================================
class EvaluationThread(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, dict, str)

    def __init__(self, video_a: str, video_b: str):
        super().__init__()
        self.video_a = video_a
        self.video_b = video_b

    def run(self):
        try:
            from evaluator.video_evaluator import VideoEvaluator
            evaluator = VideoEvaluator(sample_step=5)
            res = evaluator.evaluate(
                self.video_a,
                self.video_b,
                progress_cb=lambda curr, total: self.progress_signal.emit(curr, total)
            )
            self.finished_signal.emit(True, res, "")
        except Exception as e:
            self.finished_signal.emit(False, {}, str(e))


# ==========================================
# 后台视频处理线程 (低内存占用 + 兼容包装)
# ==========================================
class VideoProcessorThread(QThread):
    progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        cap = None
        writer = None
        try:
            import cv2
            import numpy as np

            from shuffler.manager import StickerManager
            from shuffler.provider import PatchProvider

            from processors.haar_injector import HaarROIInjector
            from processors.yolo_injector import YoloROIInjector
            from processors.spatial_warper import SpatialAffineWarper
            from processors.color_frequency_shifter import ColorFrequencyShifter
            from processors.temporal_restructurer import TemporalRestructurer

            input_path = self.config["input_path"]
            output_path = self.config["output_path"]

            if not os.path.exists(input_path):
                self.finished_signal.emit(False, f"输入文件不存在: {input_path}")
                return

            self.log_signal.emit("🚀 正在初始化多维重构处理器模块...")

            # 1. 实例化核心管理与提供器
            manager = StickerManager()
            provider = PatchProvider(manager)

            # 2. 动态兼容 PatchProvider.get_patch 参数签名 (兼容 (w, h) 元组与 w, h 双参数)
            orig_get_patch = provider.get_patch
            def safe_get_patch(*args, **kwargs):
                if len(args) == 2 and isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
                    w, h = int(args[0]), int(args[1])
                    try:
                        return orig_get_patch((w, h), **kwargs)
                    except Exception:
                        return orig_get_patch(w, h, **kwargs)
                return orig_get_patch(*args, **kwargs)

            provider.get_patch = safe_get_patch

            # 3. ROI 贴图注入器配置
            injector = None
            if self.config["enable_roi"]:
                det_type = self.config["roi_detector"]
                alpha_limit = self.config["roi_alpha"]
                if det_type == "yolo":
                    self.log_signal.emit(f"加载 YOLO 模型: {self.config['yolo_model']}...")
                    injector = YoloROIInjector(
                        provider=provider,
                        model_path=self.config["yolo_model"],
                        classes=[0],
                        conf_threshold=0.3,
                        alpha_limit=alpha_limit,
                    )
                else:
                    self.log_signal.emit("加载 Haar Cascade 检测器...")
                    injector = HaarROIInjector(provider=provider, alpha_limit=alpha_limit)

            # 4. 空间仿射微变
            spatial_warper = SpatialAffineWarper(
                max_rotation_deg=self.config["spatial_rot"],
                max_scale_delta=self.config["spatial_scale"],
                max_translation_px=self.config["spatial_trans"],
            ) if self.config["enable_spatial"] else None

            # 5. 色彩与频域抖动
            color_shifter = ColorFrequencyShifter(
                uv_noise_std=self.config["color_uv_noise"],
                dct_freq_noise_std=self.config["color_dct_noise"],
                enable_dct=self.config["color_enable_dct"],
            ) if self.config["enable_color"] else None

            # 6. 时域重构
            temporal_restructurer = TemporalRestructurer(
                speed_amplitude=self.config["temporal_speed_amp"],
                noise_intensity=self.config["temporal_noise_intensity"],
            ) if self.config["enable_temporal"] else None

            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                self.finished_signal.emit(False, "无法读取视频总帧数或视频为空！")
                return

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

            self.log_signal.emit(f"🎥 开始流式重构处理 ({total_frames} 帧)...")

            frame_idx = 0
            while True:
                if self._is_cancelled:
                    self.finished_signal.emit(False, "任务已被用户手动取消！")
                    return

                ret, frame = cap.read()
                if not ret:
                    break

                # 逐帧处理，保持极低内存占用
                if temporal_restructurer is not None and temporal_restructurer.detect_scene_cut(frame):
                    frame = temporal_restructurer.inject_cut_transition_noise(frame)

                if injector is not None:
                    frame = injector.inject_to_frame(frame)

                if color_shifter is not None:
                    frame = color_shifter.process_frame(frame)

                if spatial_warper is not None:
                    frame = spatial_warper.process_frame(frame, frame_idx)

                writer.write(frame)
                frame_idx += 1

                if frame_idx % 10 == 0 or frame_idx == total_frames:
                    self.progress_signal.emit(frame_idx, total_frames)

            self.finished_signal.emit(True, f"🎉 重构完成！导出至: {output_path}")

        except Exception as e:
            self.finished_signal.emit(False, f"处理异常崩溃: {str(e)}")
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if writer is not None and writer.isOpened():
                writer.release()


# ==========================================
# GUI 主窗口界面
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频多维对抗重构与效果评估系统 v1.3")
        self.resize(950, 850)
        self.setMinimumSize(800, 600)

        self.worker_thread = None
        self.eval_thread = None

        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F7;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #D1D1D6;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #1D1D1F;
            }
            QLabel, QCheckBox, QRadioButton {
                color: #1D1D1F;
                font-size: 13px;
            }
            QLineEdit {
                border: 1px solid #C7C7CC;
                border-radius: 6px;
                padding: 5px;
                background-color: #FFFFFF;
                color: #000000;
            }
            QTextEdit {
                border: 1px solid #C7C7CC;
                border-radius: 6px;
                background-color: #1E1E1E;
                color: #00FF66;
                font-family: Menlo, Monaco, "Courier New", Courier;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: 重构处理
        scroll_process = QScrollArea()
        scroll_process.setWidgetResizable(True)
        widget_process = QWidget()
        self._build_process_tab(widget_process)
        scroll_process.setWidget(widget_process)
        self.tab_widget.addTab(scroll_process, "🛠️ 重构处理")

        # Tab 2: 对比评估
        scroll_eval = QScrollArea()
        scroll_eval.setWidgetResizable(True)
        widget_eval = QWidget()
        self._build_eval_tab(widget_eval)
        scroll_eval.setWidget(widget_eval)
        self.tab_widget.addTab(scroll_eval, "📊 视频对比评估")

    def _build_process_tab(self, parent_widget: QWidget):
        layout = QVBoxLayout(parent_widget)
        layout.setSpacing(12)

        file_group = QGroupBox("📁 文件路径设置")
        file_layout = QFormLayout()

        self.input_file_edit = QLineEdit()
        btn_browse_input = QPushButton("选择输入视频")
        btn_browse_input.clicked.connect(lambda: self._select_file(self.input_file_edit, False))
        h_input = QHBoxLayout()
        h_input.addWidget(self.input_file_edit)
        h_input.addWidget(btn_browse_input)

        self.output_file_edit = QLineEdit("output/processed.mp4")
        btn_browse_output = QPushButton("选择保存路径")
        btn_browse_output.clicked.connect(lambda: self._select_file(self.output_file_edit, True))
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_file_edit)
        h_output.addWidget(btn_browse_output)

        file_layout.addRow("输入视频:", h_input)
        file_layout.addRow("输出路径:", h_output)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        feature_group = QGroupBox("⚙️ 重构功能叠加控制 (勾选即可启用该模块)")
        feature_layout = QVBoxLayout()

        self.chk_roi = QCheckBox("1. 目标/人脸区域微透贴图注入 (ROI Patch Injector)")
        self.chk_roi.setChecked(True)
        roi_sub = QWidget()
        roi_sub_layout = QHBoxLayout(roi_sub)
        roi_sub_layout.setContentsMargins(20, 0, 0, 0)
        self.rad_yolo = QRadioButton("YOLO 深度学习")
        self.rad_haar = QRadioButton("Haar Cascade 快速检测")
        self.rad_yolo.setChecked(True)
        self.spin_roi_alpha = QDoubleSpinBox()
        self.spin_roi_alpha.setRange(0.01, 0.15)
        self.spin_roi_alpha.setValue(0.03)
        self.spin_roi_alpha.setSingleStep(0.01)
        roi_sub_layout.addWidget(self.rad_yolo)
        roi_sub_layout.addWidget(self.rad_haar)
        roi_sub_layout.addWidget(QLabel("Alpha 上限:"))
        roi_sub_layout.addWidget(self.spin_roi_alpha)
        roi_sub_layout.addStretch()
        feature_layout.addWidget(self.chk_roi)
        feature_layout.addWidget(roi_sub)

        self.chk_spatial = QCheckBox("2. 空间与仿射微变 (呼吸微裁剪 + 动态旋转/缩放/平移)")
        self.chk_spatial.setChecked(True)
        spatial_sub = QWidget()
        spatial_sub_layout = QHBoxLayout(spatial_sub)
        spatial_sub_layout.setContentsMargins(20, 0, 0, 0)
        self.spin_rot = QDoubleSpinBox()
        self.spin_rot.setRange(0.1, 5.0)
        self.spin_rot.setValue(1.0)
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.005, 0.05)
        self.spin_scale.setValue(0.015)
        self.spin_scale.setSingleStep(0.005)
        spatial_sub_layout.addWidget(QLabel("最大旋转角:"))
        spatial_sub_layout.addWidget(self.spin_rot)
        spatial_sub_layout.addWidget(QLabel("缩放幅度:"))
        spatial_sub_layout.addWidget(self.spin_scale)
        spatial_sub_layout.addStretch()
        feature_layout.addWidget(self.chk_spatial)
        feature_layout.addWidget(spatial_sub)

        self.chk_color = QCheckBox("3. 色彩空间与高频通道抖动 (U/V 色度噪声 + DCT 高频扰动)")
        self.chk_color.setChecked(True)
        color_sub = QWidget()
        color_sub_layout = QHBoxLayout(color_sub)
        color_sub_layout.setContentsMargins(20, 0, 0, 0)
        self.chk_dct = QCheckBox("开启 8x8 DCT 高频扰动")
        self.chk_dct.setChecked(True)
        self.spin_uv_noise = QDoubleSpinBox()
        self.spin_uv_noise.setRange(0.5, 5.0)
        self.spin_uv_noise.setValue(1.5)
        color_sub_layout.addWidget(self.chk_dct)
        color_sub_layout.addWidget(QLabel("U/V 噪声标准差:"))
        color_sub_layout.addWidget(self.spin_uv_noise)
        color_sub_layout.addStretch()
        feature_layout.addWidget(self.chk_color)
        feature_layout.addWidget(color_sub)

        self.chk_temporal = QCheckBox("4. 时域重构 (正弦波 0.98x~1.02x 亚帧变速 + 镜头切点过渡)")
        self.chk_temporal.setChecked(True)
        temporal_sub = QWidget()
        temporal_sub_layout = QHBoxLayout(temporal_sub)
        temporal_sub_layout.setContentsMargins(20, 0, 0, 0)
        self.spin_speed_amp = QDoubleSpinBox()
        self.spin_speed_amp.setRange(0.01, 0.05)
        self.spin_speed_amp.setValue(0.02)
        self.spin_speed_amp.setSingleStep(0.01)
        temporal_sub_layout.addWidget(QLabel("变速幅度:"))
        temporal_sub_layout.addWidget(self.spin_speed_amp)
        temporal_sub_layout.addStretch()
        feature_layout.addWidget(self.chk_temporal)
        feature_layout.addWidget(temporal_sub)

        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)

        self.chk_auto_eval = QCheckBox("⚡ 处理完成后自动进行多维度效果对比评估")
        self.chk_auto_eval.setChecked(True)
        layout.addWidget(self.chk_auto_eval)

        progress_group = QGroupBox("📊 渲染进度与日志")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.log_area = QTextEdit()
        self.log_area.setMinimumHeight(150)
        self.log_area.setReadOnly(True)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_area)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        h_btns = QHBoxLayout()
        self.btn_start = QPushButton("🚀 开始重构处理")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #007AFF; color: white; border-radius: 6px;")
        self.btn_start.clicked.connect(self._start_processing)

        self.btn_cancel = QPushButton("⏹️ 取消处理")
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #FF3B30; color: white; border-radius: 6px;")
        self.btn_cancel.clicked.connect(self._cancel_processing)

        h_btns.addWidget(self.btn_start)
        h_btns.addWidget(self.btn_cancel)
        layout.addLayout(h_btns)

    def _build_eval_tab(self, parent_widget: QWidget):
        layout = QVBoxLayout(parent_widget)
        layout.setSpacing(12)

        group_inputs = QGroupBox("🔍 对比视频路径")
        form_layout = QFormLayout()

        self.eval_input_a = QLineEdit()
        btn_a = QPushButton("选择基准视频 (原视频)")
        btn_a.clicked.connect(lambda: self._select_file(self.eval_input_a, False))
        h_a = QHBoxLayout()
        h_a.addWidget(self.eval_input_a)
        h_a.addWidget(btn_a)

        self.eval_input_b = QLineEdit()
        btn_b = QPushButton("选择对比视频 (重构后)")
        btn_b.clicked.connect(lambda: self._select_file(self.eval_input_b, False))
        h_b = QHBoxLayout()
        h_b.addWidget(self.eval_input_b)
        h_b.addWidget(btn_b)

        form_layout.addRow("基准视频 A:", h_a)
        form_layout.addRow("对比视频 B:", h_b)
        group_inputs.setLayout(form_layout)
        layout.addWidget(group_inputs)

        self.btn_run_eval = QPushButton("📈 开始对比与多维算法评估")
        self.btn_run_eval.setFixedHeight(40)
        self.btn_run_eval.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #34C759; color: white; border-radius: 6px;")
        self.btn_run_eval.clicked.connect(self._run_manual_evaluation)
        layout.addWidget(self.btn_run_eval)

        self.eval_progress = QProgressBar()
        self.eval_progress.setRange(0, 100)
        self.eval_progress.setValue(0)
        layout.addWidget(self.eval_progress)

        group_report = QGroupBox("📋 评估报告与指标诊断")
        report_layout = QVBoxLayout()
        self.report_area = QTextEdit()
        self.report_area.setMinimumHeight(250)
        self.report_area.setReadOnly(True)
        report_layout.addWidget(self.report_area)
        group_report.setLayout(report_layout)
        layout.addWidget(group_report)

    def _select_file(self, line_edit: QLineEdit, is_save: bool):
        if is_save:
            fn, _ = QFileDialog.getSaveFileName(self, "选择保存位置", line_edit.text(), "MP4 (*.mp4)")
        else:
            fn, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        if fn:
            line_edit.setText(fn)

    def _start_processing(self):
        input_path = self.input_file_edit.text().strip()
        output_path = self.output_file_edit.text().strip()

        if not input_path:
            QMessageBox.warning(self, "警告", "请选择输入视频！")
            return

        config = {
            "input_path": input_path,
            "output_path": output_path,
            "enable_roi": self.chk_roi.isChecked(),
            "roi_detector": "yolo" if self.rad_yolo.isChecked() else "haar",
            "roi_alpha": self.spin_roi_alpha.value(),
            "yolo_model": "yolov8n.pt",
            "enable_spatial": self.chk_spatial.isChecked(),
            "spatial_rot": self.spin_rot.value(),
            "spatial_scale": self.spin_scale.value(),
            "spatial_trans": 2.0,
            "enable_color": self.chk_color.isChecked(),
            "color_enable_dct": self.chk_dct.isChecked(),
            "color_uv_noise": self.spin_uv_noise.value(),
            "color_dct_noise": 0.5,
            "enable_temporal": self.chk_temporal.isChecked(),
            "temporal_speed_amp": self.spin_speed_amp.value(),
            "temporal_noise_intensity": 8.0,
        }

        self.log_area.clear()
        self.progress_bar.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker_thread = VideoProcessorThread(config)
        self.worker_thread.progress_signal.connect(lambda cur, tot: self.progress_bar.setValue(int(cur / tot * 100)))
        self.worker_thread.log_signal.connect(self.log_area.append)
        self.worker_thread.finished_signal.connect(self._on_process_finished)
        self.worker_thread.start()

    def _cancel_processing(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_area.append("\n⚠️ 取消信号已发送...")
            self.worker_thread.cancel()

    def _on_process_finished(self, success, message):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        if success:
            self.log_area.append(f"\n✅ {message}")
            if self.chk_auto_eval.isChecked():
                self.log_area.append("⚡ 正在自动开启效果评估算法...")
                video_a = self.input_file_edit.text().strip()
                video_b = self.output_file_edit.text().strip()
                self._trigger_evaluation(video_a, video_b)
            else:
                QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.critical(self, "失败", message)

    def _run_manual_evaluation(self):
        va = self.eval_input_a.text().strip()
        vb = self.eval_input_b.text().strip()
        if not va or not vb:
            QMessageBox.warning(self, "提示", "请先选择需要对比的基准视频与重构视频！")
            return
        self._trigger_evaluation(va, vb)

    def _trigger_evaluation(self, video_a: str, video_b: str):
        self.eval_input_a.setText(video_a)
        self.eval_input_b.setText(video_b)

        self.report_area.clear()
        self.report_area.append("⏳ 正在抽取帧矩阵计算多维评估指标中...")
        self.eval_progress.setValue(0)
        self.btn_run_eval.setEnabled(False)

        self.eval_thread = EvaluationThread(video_a, video_b)
        self.eval_thread.progress_signal.connect(lambda cur, tot: self.eval_progress.setValue(int(cur / tot * 100)))
        self.eval_thread.finished_signal.connect(self._on_eval_finished)
        self.eval_thread.start()

    def _on_eval_finished(self, success, res, err):
        self.btn_run_eval.setEnabled(True)
        if not success:
            self.report_area.append(f"❌ 评估出错: {err}")
            return

        report_md = f"""
==================================================
              📊 视频多维度重构评估报告             
==================================================

一、 画质保真度指标 (Visual Fidelity):
   • SSIM (结构相似度) : {res.get('avg_ssim', 'N/A')}  (目标: >0.85)
   • PSNR (峰值信噪比) : {res.get('avg_psnr', 'N/A')} dB  (目标: >30 dB)
   👉 【画质保真得分】  : {res.get('quality_score', 'N/A')} / 100 分

二、 特征防重/破坏指标 (Anti-Deduplication):
   • pHash 汉明距离   : {res.get('avg_phash_dist', 'N/A')}  (距离>5即判定哈希被破坏)
   • 色彩直方图偏移率 : {res.get('avg_hist_shift', 'N/A')}%
   👉 【防重破坏得分】  : {res.get('defense_score', 'N/A')} / 100 分

==================================================
三、 综合判定与诊断结论:
   {res.get('verdict', '评估完成')}
==================================================
"""
        self.report_area.setText(report_md)
        self.log_area.append(f"📊 评估完成！")
        self.tab_widget.setCurrentIndex(1)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()