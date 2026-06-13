# -*- coding: utf-8 -*-
"""
训练脚本（改进版）：支持加载预训练权重继续训练，并在训练结束后评估准确率。
用法：
    trainer = Trainer2(CaptchaModel)
    trainer.train()
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import common
import one_hot
from my_datasets import CaptchaDataset


class Trainer2:
    """改进版训练器（支持预训练权重加载 + 准确率评估）。

    Args:
        model_class: 网络类，如 mymodel / CaptchaModel / CRNN
        epoch_num: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_interval: 每隔多少轮保存一次权重
        pretrained_weight: 预训练权重路径，留空则从零开始训练
        use_onehot: True 用 one_hot.vectotext 解码比较，False 用 argmax 直接比较
    """

    def __init__(self, model_class, epoch_num=5, batch_size=64,
                 learning_rate=0.001, save_interval=5,
                 pretrained_weight=common.PRETRAINED_WEIGHT, use_onehot=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_interval = save_interval
        self.use_onehot = use_onehot

        # 初始化模型
        self.model = model_class().to(self.device)

        # 加载预训练权重
        if pretrained_weight and os.path.exists(pretrained_weight):
            self.model.load_state_dict(
                torch.load(pretrained_weight, map_location=self.device, weights_only=True)
            )
            print(f"已加载预训练权重: {pretrained_weight}")
        else:
            print("未找到预训练权重，从零开始训练")

        self.loss_fn = nn.MultiLabelSoftMarginLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print(f"使用设备: {self.device}")
        print(f"使用模型: {model_class.__name__}")

    def calculate_accuracy(self, dataloader):
        """计算模型在给定数据集上的准确率。"""
        self.model.eval()
        correct = 0
        total = len(dataloader.dataset)

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(imgs)
                batch_size = imgs.size(0)
                num_classes = len(common.captcha_array)

                outputs = outputs.view(batch_size, common.captcha_size, num_classes)
                labels = labels.view(batch_size, common.captcha_size, num_classes)

                if self.use_onehot:
                    for j in range(batch_size):
                        pred_text = one_hot.vectotext(outputs[j])
                        true_text = one_hot.vectotext(labels[j])
                        if pred_text == true_text:
                            correct += 1
                else:
                    pred_indices = torch.argmax(outputs, dim=2)
                    true_indices = torch.argmax(labels, dim=2)
                    match_matrix = (pred_indices == true_indices)
                    all_correct = match_matrix.all(dim=1)
                    correct += all_correct.sum().item()

        accuracy = correct / total * 100
        print(f"测试集准确率: {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def train(self, train_dir=common.TRAIN_DATASET_DIR, test_dir=common.TEST_DATASET_DIR):
        """开始训练，最后一轮自动评估准确率。"""
        train_dataset = CaptchaDataset(train_dir)
        test_dataset = CaptchaDataset(test_dir)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

            if (epoch + 1) % self.save_interval == 0:
                os.makedirs(common.WEIGHT_DIR, exist_ok=True)
                weight_path = os.path.join(common.WEIGHT_DIR, f"ocr_{epoch + 1}.pth")
                torch.save(self.model.state_dict(), weight_path)
                print(f"模型已保存至 {weight_path}")

            if epoch + 1 == self.epoch_num:
                self.calculate_accuracy(test_loader)

        writer.close()
        print("训练完成！")


if __name__ == '__main__':
    from model import CaptchaModel, mymodel, CRNN, CaptchaNet

    trainer = Trainer2(CaptchaModel)
    trainer.train()
