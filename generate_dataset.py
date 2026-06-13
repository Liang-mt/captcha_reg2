# -*- coding: utf-8 -*-
"""
验证码数据集生成脚本。
用法：
    python generate_dataset.py --output ./datasets/train --num 5000
"""

import os
import random
import time
import argparse

from captcha.image import ImageCaptcha

import common


def generate_dataset(output_dir: str, num_samples: int):
    """生成验证码数据集。

    Args:
        output_dir: 输出目录路径
        num_samples: 生成样本数量
    """
    os.makedirs(output_dir, exist_ok=True)
    image = ImageCaptcha()

    for i in range(num_samples):
        image_val = "".join(random.sample(common.captcha_array, common.captcha_size))
        image_name = os.path.join(output_dir, f"{image_val}_{int(time.time())}.png")
        image.write(image_val, image_name)

        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1}/{num_samples} 张图片")

    print(f"数据集生成完成，共 {num_samples} 张图片，保存至 {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="验证码数据集生成工具")
    parser.add_argument("--output", type=str, default="./datasets/test", help="输出目录路径")
    parser.add_argument("--num", type=int, default=5000, help="生成样本数量")
    args = parser.parse_args()

    generate_dataset(args.output, args.num)
