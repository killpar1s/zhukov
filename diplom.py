import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QBrush, QColor, QFont
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO


IMAGE_FOLDER = "train/images"
MODEL_PATH = "result/dva2/weights/best.pt"


model = YOLO(MODEL_PATH)


danger_levels = {
     "Gun": "high",
    "Knife": "high",
    "Zippooil": "high",
    "Powerbank": "medium",
    "Lighter": "medium",
    "Laptop": "medium",
    "Scissors": "medium",
    "Pressure": "medium",
    "Tongs": "medium",
    "Wrench": "low",
    "Glassbottle": "low",
    "Pliers": "low",
    "Metalcup": "low",
    "Umbrella": "low"
}

danger_colors = {
    "high": "red",
    "medium": "orange",
    "low": "green"
}

class AnalysisPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel()
        self.image_label.setFixedSize(512, 512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white; border: 2px solid gray;")

        self.result_label = QLabel("Ожидание...")
        self.result_label.setFont(QFont("Arial", 16))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("padding: 10px;")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def analyze(self, image_path):
        results = model(image_path)[0]
        annotated_img = results.plot()
        h, w, ch = annotated_img.shape
        q_img = QImage(annotated_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

        detected_objects = [model.names[int(cls)] for cls in results.boxes.cls]

        if detected_objects:
            labels = []
            for obj in detected_objects:
                level = danger_levels.get(obj, "low")
                color = danger_colors[level]
                labels.append(f"<font color='{color}'><b>{obj}</b></font>")
            self.result_label.setText("Обнаружено: " + ", ".join(labels))
        else:
            self.result_label.setText("<font color='gray'>Опасных предметов не обнаружено</font>")


class XRaySimulator(QWidget):
    def __init__(self, analysis_panel):
        super().__init__()
        self.analysis_panel = analysis_panel

        self.frame_container = QFrame()
        self.frame_container.setFixedSize(800, 512)
        self.frame_container.setStyleSheet("background-color: gray; border: 2px solid black;")

        self.image_label = QLabel(self.frame_container)
        self.image_label.setFixedSize(800, 512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.move(0, 0)  # Обязательно, иначе не покажется

        layout = QVBoxLayout()
        layout.addWidget(self.frame_container)
        self.setLayout(layout)

        self.image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                            if f.lower().endswith(('.jpg', '.png'))]
        self.current_index = 0
        self.x_position = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.moving_image = None

    def start(self):
        self.timer.start(30)

    def stop(self):
        self.timer.stop()

    def update_frame(self):
        if self.moving_image is None and self.current_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_index]
            self.moving_image = cv2.imread(self.current_image_path)
            self.x_position = 0
            self.analysis_panel.analyze(self.current_image_path)

        if self.moving_image is not None:
            frame = self.render_moving_image()
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(q_img))

            self.x_position += 5
            if self.x_position > self.image_label.width():
                self.moving_image = None
                self.current_index += 1

    def render_moving_image(self):
        canvas = 255 * np.ones((370, 800, 3), dtype=np.uint8)
        if self.moving_image is not None:
            img_resized = cv2.resize(self.moving_image, (200, 200))
            x_offset = self.x_position
            y_offset = 90
            x_end = min(x_offset + img_resized.shape[1], canvas.shape[1])
            img_width = x_end - x_offset
            if img_width > 0:
                canvas[y_offset:y_offset+img_resized.shape[0], x_offset:x_end] = img_resized[:, :img_width]
        return canvas



class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интерфейс контроля багажа")
        self.showMaximized()
        self.setStyleSheet("""background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #848482, stop:1 #b0b0b0);""")

        self.analysis_panel = AnalysisPanel()
        self.xray_simulator = XRaySimulator(self.analysis_panel)

        self.start_btn = QPushButton("▶ Старт")
        self.stop_btn = QPushButton("⏸ Стоп")
        self.start_btn.setStyleSheet("padding: 10px; font-size: 16px;")
        self.stop_btn.setStyleSheet("padding: 10px; font-size: 16px;")
        self.start_btn.clicked.connect(self.xray_simulator.start)
        self.stop_btn.clicked.connect(self.xray_simulator.stop)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.xray_simulator)
        content_layout.addWidget(self.analysis_panel)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec_())
