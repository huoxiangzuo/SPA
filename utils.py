import os
import sys
import json
import pickle
import random
import math
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from sklearn import metrics
import seaborn as sns

# reference from https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_train_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    category = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)

    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(root, cls, i) for i in os.listdir(cls_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cls]

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images for training.".format(len(train_images_path)))

    return train_images_path, train_images_label


def read_val_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    category = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))

    val_images_path = []
    val_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(root, cls, i) for i in os.listdir(cls_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cls]

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images for validation.".format(len(val_images_path)))

    return val_images_path, val_images_label


def read_test_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    category = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))

    test_images_path = []
    test_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(root, cls, i) for i in os.listdir(cls_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cls]

        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images for test.".format(len(test_images_path)))

    return test_images_path, test_images_label


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    labels = []
    pred_classes = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        imgs, label = data
        sample_num += imgs.shape[0]

        pred = model(imgs.to(device))
        pred_class = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_class, label.to(device)).sum()

        loss = loss_function(pred, label.to(device))
        accu_loss += loss

        label = label.cpu().detach().numpy()
        pred_class = pred.cpu().detach().numpy()
        labels.extend(label)
        pred_classes.extend(pred_class)

        data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
    labels = np.array(labels)
    pred_classes = np.array(pred_classes)

    n_classes = pred_classes.shape[1]
    aucs = []
    for i in range(n_classes):
        fpr, tpr, thresholds = metrics.roc_curve(labels, pred_classes[:, i], pos_label=i)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
    macro_auc = np.mean(aucs)

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, pred_classes.argmax(axis=1),
                                                                       average='macro', zero_division=False)
    print("precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, AUC: {:.4f}".format(precision, recall, f1, macro_auc))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-2):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
