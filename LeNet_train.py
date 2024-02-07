import copy
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from matplotlib import pyplot as plt
from torch import optim, nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from LeNet import Net


def train_val_dataloader():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))

    train_data, val_data = Data.random_split(train_data,
                                             [round(len(train_data) * 0.8), round(len(train_data) * 0.2)])
    train_Loader = Data.DataLoader(dataset=train_data,
                                   batch_size=256,
                                   shuffle=True,
                                   num_workers=2)
    val_Loader = Data.DataLoader(dataset=val_data,
                                 batch_size=256,
                                 shuffle=True,
                                 num_workers=2)
    print("Train and val dataloader loaded.")
    return train_Loader, val_Loader


def train_net(net, train_loaders, val_loaders, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判定运行在GPU 还是CPU
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # Adam 优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    model = net.to(device)  # 将模型实例化部署在设备
    best_model_wts = copy.deepcopy(net.state_dict())  # 获得模型参数
    best_val_acc = 0.0  # 预置最佳精准度

    train_loss_history, val_loss_history, train_acc_history, val_acc_history = [], [], [], []  # 预置训练、验证损失和精度
    train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
    train_num, val_num = 0, 0
    since = time.time()  # 训练时间

    # Train Part
    for epoch in range(num_epochs):
        print(f'Epoch:{epoch}/{num_epochs - 1}')
        print('-' * 10)
        for step, (images, labels) in enumerate(train_loaders):
            images, labels = images.to(device), labels.to(device)  # 数据部署在设备
            model.train()  # 模型开始训练模式
            output = model(images)  # 这个感觉不太对
            pre_lab = torch.argmax(output, dim=1)  # 根据输出得到对应预测标签
            loss = criterion(output, labels)  # 用输出算出误差
            optimizer.zero_grad()  # 梯度置零
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            train_loss += loss.item() * images.size(0)
            train_acc += torch.sum(pre_lab == labels).item()
            train_num += images.size(0)
        for step, (images, labels) in enumerate(val_loaders):
            images, labels = images.to(device), labels.to(device)  # 数据部署在设备
            model.eval()  # 评估模式
            output = model(images)  # 得到结果
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            val_loss += loss.item() * images.size(0)
            val_acc += torch.sum(pre_lab == labels).item()
            val_num += images.size(0)
        train_acc_history.append(train_acc / train_num)
        train_loss_history.append(train_loss / train_num)
        val_acc_history.append(val_acc / val_num)
        val_loss_history.append(val_loss / val_num)  # 记录loss 和 acc, 其中 loss 是基于交叉熵计算得到, acc 是统计的正确概率
        print('Train Loss : \t\tTrain acc \t-----> {:.5f}:{:.5f}'
              .format(train_loss_history[-1], train_acc_history[-1]))
        print('Val Loss : \t\t\tVal acc \t-----> {:.5f}:{:.5f}'
              .format(val_loss_history[-1], val_acc_history[-1]))

        if val_acc_history[-1] > best_val_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc_history[-1]
            # 保存模型的操作移到这里，减少文件I/O
            model_dir = './model'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            torch.save(best_model_wts, model_path)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.1f}s'.format(time_elapsed // 60, time_elapsed % 60))
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss": train_loss_history,
                                       "train_acc": train_acc_history,
                                       "val_loss": val_loss_history,
                                       "val_acc": val_acc_history,
                                       })
    return train_process


def plot_loss_and_acc(dataframe):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(dataframe['epoch'], dataframe['train_loss'], 'ro-', label="Training Loss")
    plt.plot(dataframe['epoch'], dataframe['val_loss'], 'bs-', label="Validation Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(dataframe['epoch'], dataframe['train_acc'], 'ro-', label="Training Accuracy")
    plt.plot(dataframe['epoch'], dataframe['val_acc'], 'bs-', label="Validation Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model = Net()
    train_loader, val_loader = train_val_dataloader()
    train_process = train_net(net=model,
                              train_loaders=train_loader,
                              val_loaders=val_loader,
                              num_epochs=1,
                              learning_rate=0.01)
    plot_loss_and_acc(train_process)
    counter = 1
    path = './result/'
    filename = "train_process.csv"  # 初始文件名应该包括.csv
    full_path = os.path.join(path, filename)  # 使用os.path.join来构建完整路径

    # 检查目录是否存在，如果不存在，则创建
    if not os.path.exists(path):
        os.makedirs(path)

    # 检查文件是否存在，如果存在，则更新文件名
    while os.path.exists(full_path):
        new_filename = f"train_process({counter}).csv"  # 更新文件名，保证.csv在末尾
        full_path = os.path.join(path, new_filename)  # 更新完整路径
        counter += 1
    train_process.to_csv(full_path,index=False)
