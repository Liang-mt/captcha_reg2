# -*- coding: utf-8 -*-
"""
公共配置模块：定义验证码字符集、尺寸及统一的权重路径配置。
"""

# 验证码字符集：0-9 + a-z，共36个字符
captcha_array = list("0123456789abcdefghijklmnopqrstuvwxyz")
# 验证码字符数量
captcha_size = 4

# ============ 统一权重路径配置 ============
# 训练权重保存目录
WEIGHT_DIR = "./weights"
# 预训练权重路径（用于继续训练或预测）
PRETRAINED_WEIGHT = "./weights2/ocr_15.pth"
# 测试集路径
TEST_DATASET_DIR = "./datasets/test"
# 训练集路径
TRAIN_DATASET_DIR = "./datasets/train"
