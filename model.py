# -*- coding: utf-8 -*-
"""
CNN 模型定义：用于验证码识别的卷积神经网络。
包含新旧两个网络结构，权重可互相加载。
输入：1通道灰度图 60x160
输出：[batch, captcha_size * len(captcha_array)] 即 [batch, 144]
"""

import torch
from torch import nn

import common


# ============ 原始网络（修改前） ============
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # [6, 64, 30, 80]
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # [6, 128, 15, 40]
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # [6, 256, 7, 20]
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # [6, 512, 3, 10]
        )
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)  # [6, 512, 1, 5]
        # )
        self.layer6 = nn.Sequential(
            nn.Flatten(),  # [64, 15360]
            nn.Linear(in_features=15360, out_features=4096),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=common.captcha_size * common.captcha_array.__len__())
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer6(x)
        return x


# ============ 优化网络（新版本） ============
class CaptchaModel(nn.Module):
    """验证码识别 CNN 模型（优化版）。

    网络结构与 mymodel 完全一致，仅代码组织更规范：
        将 layer1~layer4 合并为 features，layer6 改为 classifier。
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # [B, 64, 30, 80]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # [B, 128, 15, 40]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # [B, 256, 7, 20]

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # [B, 512, 3, 10]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                        # [B, 15360]
            nn.Linear(15360, 4096),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(4096, common.captcha_size * len(common.captcha_array)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============ CRNN 网络（CNN + BiLSTM，推荐） ============
class CRNN(nn.Module):
    """CRNN 验证码识别模型（CNN + BiLSTM）。

    参考 meijieru/crnn.pytorch 等开源实现设计。
    相比纯 CNN 模型，BiLSTM 能捕捉字符间的序列依赖关系，
    对有噪声、粘连、变形的验证码识别效果更好。

    网络结构：
        CNN 特征提取器（Conv + BN + ReLU + MaxPool，5层）
        -> 自适应池化压缩高度为1，宽度固定为 captcha_size
        -> BiLSTM 序列建模（2层，双向）
        -> 全连接解码器（每个时间步独立分类）

    输入：[B, 1, 60, 160] 灰度图
    输出：[B, 144] = [B, captcha_size * num_classes]
    """

    def __init__(self, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        num_classes = len(common.captcha_array)  # 36

        # ----- CNN 特征提取器 -----
        self.cnn = nn.Sequential(
            # Block 1: [B, 1, 60, 160] -> [B, 64, 30, 80]
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: [B, 64, 30, 80] -> [B, 128, 15, 40]
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: [B, 128, 15, 40] -> [B, 256, 15, 40]
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: [B, 256, 15, 40] -> [B, 256, 7, 40]
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 5: [B, 256, 7, 40] -> [B, 512, 3, 40]
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        # 自适应池化：高度压缩为1，宽度固定为 captcha_size（4个时间步）
        # [B, 512, 3, 40] -> [B, 512, 1, captcha_size]
        self.pool = nn.AdaptiveAvgPool2d((1, common.captcha_size))

        # ----- BiLSTM 序列建模 -----
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,
        )

        # ----- 全连接解码器 -----
        rnn_output_size = hidden_size * 2  # 双向，输出维度翻倍
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, num_classes),
        )

    def forward(self, x):
        # CNN 提取特征
        conv = self.cnn(x)                           # [B, 512, H', W']
        conv = self.pool(conv)                        # [B, 512, 1, 4]
        conv = conv.squeeze(2)                        # [B, 512, 4]
        conv = conv.permute(2, 0, 1)                  # [4, B, 512]  时间步优先

        # BiLSTM 序列建模
        rnn_out, _ = self.rnn(conv)                   # [4, B, hidden*2]

        # 每个时间步独立解码
        B = x.size(0)
        output = self.decoder(rnn_out)                # [4, B, 36]
        output = output.permute(1, 0, 2)              # [B, 4, 36]
        output = output.reshape(B, -1)                # [B, 144]

        return output


# ============ 增强网络（SE-ResNet 风格，推荐） ============
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    通过全局池化压缩空间信息，再用两层全连接学习通道权重，
    让模型自动聚焦于最有用的特征通道。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


class ResBlock(nn.Module):
    """残差卷积块：Conv -> BN -> ReLU -> Conv -> BN -> SE -> + 残差 -> ReLU。

    相比普通 Sequential，残差连接让梯度能直接回传，
    训练更深的网络时不会退化。
    """

    def __init__(self, in_channels: int, out_channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # 当输入输出通道数不同时，用 1x1 卷积对齐残差
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class CaptchaNet(nn.Module):
    """增强版验证码识别网络（SE-ResNet 风格）。

    相比 CaptchaModel 的改进：
        1. BatchNorm — 训练更稳定，收敛更快
        2. SE 注意力 — 自动聚焦重要特征通道
        3. 残差连接 — 深层网络训练不退化
        4. 自适应池化 — 不硬编码 15360，输入尺寸灵活
        5. Kaiming 初始化 — 更好的权重初始化

    输入：[B, 1, 60, 160] 灰度图
    输出：[B, 144] = [B, captcha_size * num_classes]
    """

    def __init__(self):
        super().__init__()
        num_classes = len(common.captcha_array)

        # 特征提取器：4 个阶段，每阶段用残差块 + 最大池化
        self.stage1 = nn.Sequential(
            ResBlock(1, 64),
            nn.MaxPool2d(2),                     # [B, 64, 30, 80]
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128),
            nn.MaxPool2d(2),                     # [B, 128, 15, 40]
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256),
            nn.MaxPool2d(2),                     # [B, 256, 7, 20]
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, 512),
            nn.MaxPool2d(2),                     # [B, 512, 3, 10]
        )

        # 自适应池化：输出固定 1x1，不再依赖输入尺寸
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, common.captcha_size * num_classes),
        )

        # Kaiming 初始化
        self._init_weights()

    def _init_weights(self):
        """使用 Kaiming 初始化卷积层和线性层权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    data = torch.ones(64, 1, 60, 160)

    # 测试原始网络
    old_model = mymodel()
    out1 = old_model(data)
    print(f"原始网络 mymodel:     输入 {data.shape} -> 输出 {out1.shape}")

    # 测试优化网络
    new_model = CaptchaModel()
    out2 = new_model(data)
    print(f"优化网络 CaptchaModel: 输入 {data.shape} -> 输出 {out2.shape}")

    # 测试增强网络
    net = CaptchaNet()
    out3 = net(data)
    print(f"增强网络 CaptchaNet:   输入 {data.shape} -> 输出 {out3.shape}")

    # 统计参数量
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"\n参数量对比:")
    print(f"  mymodel:      {count_params(old_model):>12,}")
    print(f"  CaptchaModel: {count_params(new_model):>12,}")
    print(f"  CaptchaNet:   {count_params(net):>12,}")

    # 测试 CRNN 网络
    crnn_model = CRNN()
    out3 = crnn_model(data)
    print(f"CRNN 网络:            输入 {data.shape} -> 输出 {out3.shape}")
