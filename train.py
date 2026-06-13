# -*- coding: utf-8 -*-
"""
训练脚本（基础版）：使用 PyTorch 训练验证码识别模型。
用法：
    trainer = Trainer(CaptchaModel)
    trainer.train()
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import common
from my_datasets import CaptchaDataset


class Trainer:
    """基础训练器。

    Args:
        model_class: 网络类，如 mymodel / CaptchaModel / CRNN
        epoch_num: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_interval: 每隔多少轮保存一次权重
    """

    def __init__(self, model_class, epoch_num=10, batch_size=64,
                 learning_rate=0.001, save_interval=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_interval = save_interval

        # 初始化模型
        self.model = model_class().to(self.device)
        self.loss_fn = nn.MultiLabelSoftMarginLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print(f"使用设备: {self.device}")
        print(f"使用模型: {model_class.__name__}")

    def train(self, train_dir=common.TRAIN_DATASET_DIR, test_dir=common.TEST_DATASET_DIR):
        """开始训练。"""
        train_dataset = CaptchaDataset(train_dir)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        writer = SummaryWriter("logs")
        total_step = 0

        for epoch in range(self.epoch_num):
            print(f"------------ 第 {epoch + 1} 轮训练开始 --------------")
            self.model.train()

            for i, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    total_step += 1
                    print(f"  [{epoch + 1}/{self.epoch_num}] step {i} - loss: {loss.item():.4f}")
                    writer.add_scalar("loss", loss.item(), total_step)

            # 定期保存权重
            if (epoch + 1) % self.save_interval == 0:
                os.makedirs(common.WEIGHT_DIR, exist_ok=True)
                weight_path = os.path.join(common.WEIGHT_DIR, f"ocr_{epoch + 1}.pth")
                torch.save(self.model.state_dict(), weight_path)
                print(f"模型已保存至 {weight_path}")

        writer.close()
        print("训练完成！")


if __name__ == '__main__':
    from model import CaptchaModel, mymodel, CRNN, CaptchaNet

    trainer = Trainer(CaptchaModel)
    trainer.train()
