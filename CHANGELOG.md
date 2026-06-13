# 修改记录

本文档记录从原始代码到现行代码的所有修改。

## 一、整体架构调整

### 1.1 职责分离

| 原始状态 | 修改后 |
|----------|--------|
| common.py 既包含配置又包含数据集生成逻辑 | 配置与生成分离：common.py 只保留配置，新增 generate_dataset.py 独立生成脚本 |
| 各文件各自硬编码权重路径 | 所有路径集中在 common.py 统一管理 |

### 1.2 新增文件

| 文件 | 用途 |
|------|------|
| generate_dataset.py | 独立的数据集生成脚本，支持 `--output` 和 `--num` 命令行参数 |
| requirements.txt | 记录项目依赖 |
| .gitignore | 排除 IDE 配置、缓存、权重、数据集等 |

---

## 二、common.py

**原始代码问题：**
- 数据集生成逻辑混在配置文件中
- 各文件引用的权重路径不统一

**修改内容：**
- 移除数据集生成代码（迁移至 generate_dataset.py）
- 新增统一路径常量：`WEIGHT_DIR`、`PRETRAINED_WEIGHT`、`TEST_DATASET_DIR`、`TRAIN_DATASET_DIR`

---

## 三、model.py

**原始代码问题：**
- 只有一个网络类 `mymodel`，命名不规范（小写开头）
- `super(mymodel, self).__init__()` 写法冗余
- `common.captcha_array.__len__()` 应改为 `len()`
- 注释的 layer5 代码残留
- forward 中有注释掉的调试代码

**修改内容：**
- 新增 `CaptchaModel` 类：网络结构与 mymodel 一致，代码组织更规范（`features` + `classifier`），保留 `mymodel` 原始版本共存
- 新增 `CRNN` 类：CNN + BiLSTM 架构，参考 meijieru/crnn.pytorch，能捕捉字符间序列依赖
- 新增 `CaptchaNet` 类：SE-ResNet 风格增强网络，包含：
  - `SEBlock`：Squeeze-and-Excitation 通道注意力模块
  - `ResBlock`：残差卷积块（Conv→BN→ReLU→Conv→BN→SE→+残差→ReLU）
  - BatchNorm 稳定训练
  - 自适应池化替代硬编码 15360
  - Kaiming 权重初始化
- `__main__` 测试代码增加 4 个网络的输出形状验证和参数量对比

---

## 四、one_hot.py

**原始代码问题：**
- 包含调试用的 print 语句和测试代码

**修改内容：**
- 清理调试代码
- 添加类型注解和 docstring
- 保留简洁的 `__main__` 测试代码

---

## 五、my_datasets.py

**原始代码问题：**
- 硬编码 Linux 路径 `"/root/captcha_reg/datasets/train"` 作为默认值
- 类名 `mydatasets` 不规范

**修改内容：**
- 改为从 common 模块读取路径：`common.TRAIN_DATASET_DIR`
- 新增规范类名 `CaptchaDataset`，保留 `mydatasets = CaptchaDataset` 兼容别名
- 添加标签字符合法性校验

---

## 六、train.py

**原始代码问题：**
- 用 `cpu = 1` / `cpu = 0` 标志手动选择设备
- 超参数散落在代码各处
- 训练逻辑与脚本耦合

**修改内容：**
- 改为 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 自动选择设备
- 封装为 `Trainer` 类，超参数通过构造函数传入
- 超参数提取为模块级常量（`EPOCH_NUM`、`BATCH_SIZE` 等）
- 权重保存时自动创建目录：`os.makedirs(common.WEIGHT_DIR, exist_ok=True)`
- 统一使用 `weights_only=True` 加载权重

---

## 七、train2.py

**原始代码问题：**
- 准确率计算 bug：`labels.view(-1, 36)` 把整个 batch 扁平化成一个长向量，batch_size > 1 时结果错误
- 权重保存到 `weights3/` 目录（不存在会崩溃）
- 加载权重时未指定 `weights_only=True`

**修改内容：**
- 修复准确率计算：改为 `view(batch, 4, 36)` 逐样本比较，支持两种模式：
  - `use_onehot=True`：用 `one_hot.vectotext()` 循环逐样本解码比较（可打印预测文本）
  - `use_onehot=False`：用 `argmax` 直接张量比较（更快）
- 封装为 `Trainer2` 类，支持预训练权重加载
- 权重保存统一到 `common.WEIGHT_DIR`，自动创建目录
- 最后一轮自动评估准确率

---

## 八、predict.py

**原始代码问题：**
- `pred_pic()` 引用 `weights2/ocr_10.pth`（文件不存在）
- 准确率计算同样存在 batch 扁平化 bug
- 加载权重时 `weights_only` 不一致

**修改内容：**
- 封装为 `Predictor` 类
- 统一使用 `common.PRETRAINED_WEIGHT` 作为默认权重路径
- 修复准确率计算：逐样本重塑为 `[1, 4, 36]` 后用 `one_hot.vectotext()` 比较
- 统一使用 `weights_only=True`
- 测试集逐样本打印 ✓ 正确 / ✗ 错误

---

## 九、window.py

**原始代码问题：**
- 导入了未使用的 `CRNN`
- 有 `about` 页面相关注释代码
- 模型加载无异常处理，加载失败会崩溃
- 图片检测无异常处理
- 变量名 `pred_text` 拼写检查警告
- 类型注解不准确
- PEP 8 不规范（关键字参数空格）

**修改内容：**
- 移除未使用的导入
- 添加模型加载异常处理：`try/except` + `QMessageBox.critical` 提示
- 添加图片检测异常处理，防止 GUI 崩溃
- 变量名 `pred_text` → `predicted_text`
- 类型注解改为 `type` 通用类型
- 修复 PEP 8：`model_class = mymodel` → `model_class=mymodel`
- PyQt5 `.connect()` 添加 `# type: ignore[attr-defined]` 抑制误报
- 支持通过构造参数切换网络和权重路径

---

## 十、Bug 汇总

| Bug | 文件 | 原因 | 修复 |
|-----|------|------|------|
| 准确率计算错误 | train2.py, predict.py | `labels.view(-1, 36)` 把 batch 扁平化，batch_size > 1 时结果全部错误 | 改为逐样本 `[batch, 4, 36]` 比较 |
| 权重路径不存在 | predict.py | `pred_pic()` 引用 `weights2/ocr_10.pth` | 统一使用 `common.PRETRAINED_WEIGHT` |
| 保存目录崩溃 | train2.py | `weights3/` 目录不存在 | 改为 `os.makedirs(common.WEIGHT_DIR, exist_ok=True)` |
| 硬编码路径 | my_datasets.py | Linux 路径 `/root/captcha_reg/datasets/train` | 改为 `common.TRAIN_DATASET_DIR` |
| weights_only 不一致 | 全部文件 | 部分文件加载权重未指定 `weights_only=True` | 统一添加 |
| GUI 崩溃 | window.py | 模型加载或图片检测异常无捕获 | 添加 try/except + 弹窗提示 |
