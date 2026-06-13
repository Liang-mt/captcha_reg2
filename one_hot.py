# -*- coding: utf-8 -*-
"""
One-Hot 编码/解码模块：在文本标签和张量之间转换。
"""

import torch

import common


def text2vec(text: str) -> torch.Tensor:
    """将文本标签转换为 one-hot 张量。

    Args:
        text: 验证码文本，长度应为 captcha_size

    Returns:
        shape [captcha_size, len(captcha_array)] 的 one-hot 张量
    """
    vectors = torch.zeros(common.captcha_size, len(common.captcha_array))
    for i, char in enumerate(text):
        vectors[i, common.captcha_array.index(char)] = 1
    return vectors


def vectotext(vec: torch.Tensor) -> str:
    """将 one-hot 张量转换为文本标签。

    Args:
        shape [captcha_size, len(captcha_array)] 的张量（可为 logits 或 one-hot）

    Returns:
        解码后的文本字符串
    """
    indices = torch.argmax(vec, dim=1)
    return "".join(common.captcha_array[i] for i in indices)


if __name__ == '__main__':
    vec = text2vec("aaab")
    print(f"编码形状: {vec.shape}")
    print(f"解码结果: {vectotext(vec)}")
