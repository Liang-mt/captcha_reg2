# -*- coding: utf-8 -*-
"""
PyQt5 桌面 GUI 应用：验证码识别系统。
"""

import os
import sys
import shutil

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox,
)

import common
import one_hot
from model import CaptchaModel, mymodel, CRNN, CaptchaNet


class MainWindow(QTabWidget):
    """验证码识别系统主窗口。

    Args:
        model_class: 网络模型类（nn.Module 子类）
        weight_path: 模型权重文件路径
    """

    def __init__(self, model_class: type = CaptchaModel, weight_path: str = common.PRETRAINED_WEIGHT):
        super().__init__()
        self.setWindowTitle('数字验证码识别系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/xf.jpg"))

        self.output_size = 480
        self.img2predict = ""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = self._load_model(model_class, weight_path)
        self._init_ui()

    def _load_model(self, model_class, weight_path: str):
        """加载模型权重。"""
        model = model_class().to(self.device)

        if not os.path.exists(weight_path):
            QMessageBox.critical(self, "错误", f"找不到模型权重文件:\n{weight_path}")
            return model

        try:
            state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"模型加载成功: {weight_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")

        return model

    def _init_ui(self):
        """初始化界面布局。"""
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)

        # 按钮样式
        btn_style = (
            "QPushButton{color:white; background-color:rgb(48,124,208);"
            "border:2px; border-radius:5px; padding:5px; margin:5px}"
            "QPushButton:hover{background-color:rgb(2,110,180)}"
        )

        # 图片识别界面
        img_widget = QWidget()
        img_layout = QVBoxLayout()

        title = QLabel("数字验证码识别")
        title.setFont(font_title)
        title.setAlignment(Qt.AlignCenter)

        # 左右图片区域
        mid_widget = QWidget()
        mid_layout = QHBoxLayout()

        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)

        mid_layout.addWidget(self.left_img)
        mid_layout.addStretch(0)
        mid_layout.addWidget(self.right_img)
        mid_widget.setLayout(mid_layout)

        # 按钮
        btn_upload = QPushButton("上传图片")
        btn_detect = QPushButton("开始检测")
        btn_upload.setFont(font_main)
        btn_detect.setFont(font_main)
        btn_upload.setStyleSheet(btn_style)
        btn_detect.setStyleSheet(btn_style)
        btn_upload.clicked.connect(self.upload_img)  # type: ignore[attr-defined]
        btn_detect.clicked.connect(self.detect_img)  # type: ignore[attr-defined]

        img_layout.addWidget(title)
        img_layout.addWidget(mid_widget, alignment=Qt.AlignCenter)
        img_layout.addWidget(btn_upload)
        img_layout.addWidget(btn_detect)
        img_widget.setLayout(img_layout)

        self.addTab(img_widget, '图片检测')

    def upload_img(self):
        """上传图片并预览。"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '', '*.jpg *.png *.tif *.jpeg'
        )
        if not file_name:
            return

        suffix = file_name.split(".")[-1]
        save_path = os.path.join("images/tmp", f"tmp_upload.{suffix}")
        os.makedirs("images/tmp", exist_ok=True)
        shutil.copy(file_name, save_path)

        # 读取并缩放图片用于预览
        img = cv2.imread(save_path)
        if img is None:
            QMessageBox.warning(self, "错误", "无法读取图片文件")
            return

        height, width = img.shape[:2]
        scale = min(60 / height, 160 / width)
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size)

        preview_path = "images/tmp/upload_show_result.jpg"
        cv2.imwrite(preview_path, img)

        self.img2predict = file_name
        pixmap = QPixmap(preview_path)
        scaled = pixmap.scaled(
            self.left_img.size(),
            aspectRatioMode=Qt.KeepAspectRatio,
            transformMode=Qt.SmoothTransformation,
        )
        self.left_img.setPixmap(scaled)

        # 重置右侧图片
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))

    def detect_img(self):
        """检测已上传的图片。"""
        if not self.img2predict:
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
            return

        try:
            img = Image.open(self.img2predict)

            transform = transforms.Compose([
                transforms.Resize((60, 160)),
                transforms.ToTensor(),
                transforms.Grayscale(),
            ])

            img_tensor = transform(img).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)
                output = output.view(-1, len(common.captcha_array))
                predicted_text = one_hot.vectotext(output)

            print(f"预测结果: {predicted_text}")

            # 在白色背景上绘制识别结果
            img_np = img_tensor.squeeze(0).cpu().numpy()[0]  # [60, 160]
            scale = self.output_size / 60
            im0 = cv2.resize(img_np, (0, 0), fx=scale, fy=scale)
            height, width = im0.shape

            background = np.ones((height, width, 3), dtype=np.uint8) * 255

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4.5
            thickness = 3
            text_size = cv2.getTextSize(predicted_text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2

            cv2.putText(
                background, predicted_text, (text_x, text_y),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
            )

            result_path = "images/tmp/single_result.jpg"
            cv2.imwrite(result_path, background)
            self.right_img.setPixmap(QPixmap(result_path))

        except Exception as e:
            QMessageBox.critical(self, "检测失败", f"图片检测出错:\n{e}")

    def closeEvent(self, event):
        """关闭窗口确认对话框。"""
        reply = QMessageBox.question(
            self, '退出', '确定关闭该系统?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(model_class=mymodel)
    window.show()
    sys.exit(app.exec_())
