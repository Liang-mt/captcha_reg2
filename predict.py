# -*- coding: utf-8 -*-
"""
预测模块：批量测试模型准确率，或对单张图片进行预测。
用法：
    predictor = Predictor(CaptchaModel)
    predictor.test_pred()
    predictor.pred_pic("xxx.png")
"""

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import common
import one_hot
from my_datasets import CaptchaDataset


class Predictor:
    """预测器。

    Args:
        model_class: 网络类，如 mymodel / CaptchaModel / CRNN
        weight_path: 模型权重文件路径
    """

    def __init__(self, model_class, weight_path=common.PRETRAINED_WEIGHT):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_class().to(self.device)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"使用设备: {self.device}")
        print(f"使用模型: {model_class.__name__}")
        print(f"权重加载成功: {weight_path}")

    def test_pred(self, test_dir=common.TEST_DATASET_DIR):
        """在测试集上评估模型准确率。"""
        test_dataset = CaptchaDataset(test_dir)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        total = len(test_dataset)

        correct = 0
        num_classes = len(common.captcha_array)

        for i, (img, label) in enumerate(test_loader):
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                output = self.model(img)

            output = output.view(1, common.captcha_size, num_classes)
            label = label.view(1, common.captcha_size, num_classes)

            pred_text = one_hot.vectotext(output.squeeze(0))
            true_text = one_hot.vectotext(label.squeeze(0))

            if pred_text == true_text:
                correct += 1
                print(f"[{i+1}/{total}] ✓ 正确: {true_text} == {pred_text}")
            else:
                print(f"[{i+1}/{total}] ✗ 错误: 正确={true_text}, 预测={pred_text}")

        accuracy = correct / total * 100
        print(f"\n准确率: {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def pred_pic(self, pic_path):
        """对单张图片进行预测。"""
        transform = transforms.Compose([
            transforms.Resize((60, 160)),
            transforms.ToTensor(),
            transforms.Grayscale(),
        ])

        img = Image.open(pic_path)
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            output = output.view(-1, len(common.captcha_array))
            pred_text = one_hot.vectotext(output)

        print(f"预测结果: {pred_text}")
        return pred_text


if __name__ == '__main__':
    from model import CaptchaModel, mymodel, CRNN, CaptchaNet

    predictor = Predictor(mymodel)
    predictor.test_pred()
    # predictor.pred_pic("./datasets/test/0kwf_1736583132.png")
