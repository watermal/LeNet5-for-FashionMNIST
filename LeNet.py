# LeNet-5
import torch
import torch.nn as nn
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 x 28 x 28
        # kernel:5 x 5 x 1 x 6
        # next in_channel = 6
        # (28 + 2 x 2 - 5) / 2 + 1
        # in_channels out_channels kernel_size stride padding
        # 输入通道 输出通道 卷积核大小 步幅 填充
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,kernel_size=5)
        self.flatten = nn.Flatten()
        self.fullyConnected1 = nn.Linear(in_features=400, out_features=120)
        self.fullyConnected2 = nn.Linear(in_features=120, out_features=84)
        self.fullyConnected3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.pool1(x)
        x = self.sig(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fullyConnected1(x)
        x = self.sig(self.conv3(x))
        x = self.fullyConnected2(x)
        x = self.fullyConnected3(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    print('Parameter', summary(net, input_size=(1, 28, 28)))
    print(f'Pytorch Version: {torch.__version__} ,Cuda available is {torch.cuda.is_available()}')
