# -*- coding: utf-8 -*-
"""
自定义 PyTorch Dataset：加载验证码图片并转换为张量。
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import common
import one_hot


class CaptchaDataset(Dataset):
    """验证码数据集。

    图片文件名格式：{标签}_{时间戳}.png
    自动进行 Resize(60,160) + Grayscale + ToTensor 转换。
    """

    def __init__(self, root_dir: str):
        super().__init__()
        self.list_image_path = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
        ]

        self.transform = transforms.Compose([
            transforms.Resize((60, 160)),
            transforms.ToTensor(),
            transforms.Grayscale(),
        ])

    def __getitem__(self, index):
        image_path = self.list_image_path[index]
        image_name = os.path.basename(image_path)

        img = Image.open(image_path)
        img_tensor = self.transform(img)

        # 从文件名解析标签（下划线前的部分）
        label_text = image_name.split("_")[0]

        if all(char in common.captcha_array for char in label_text):
            label_tensor = one_hot.text2vec(label_text)
        else:
            print(f"警告: 标签 '{label_text}' 包含无效字符，已被忽略。")
            label_tensor = torch.zeros(common.captcha_size, len(common.captcha_array))

        # 展平为一维向量 [captcha_size * len(captcha_array)]
        label_tensor = label_tensor.view(-1)
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.list_image_path)


# 保持旧类名的兼容别名
mydatasets = CaptchaDataset


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    d = CaptchaDataset(common.TRAIN_DATASET_DIR)
    img, label = d[0]
    print(f"图片形状: {img.shape}, 标签形状: {label.shape}")

    writer = SummaryWriter("logs")
    writer.add_image("img", img, 1)
    writer.close()
