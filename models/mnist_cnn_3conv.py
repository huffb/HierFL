import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNNWithThreeConvLayers(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MNISTCNNWithThreeConvLayers, self).__init__()

        # 第一个卷积层块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层块
        self.conv2 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 第三个卷积层块
        self.conv3 = nn.Conv2d(1, 32,kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 计算全连接层输入特征数，这里假设经过三次卷积和池化后，特征图大小变为 (28/2/2/2) = 7
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        print("Input shape:", x.shape)
        # 第一个卷积层块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 第二个卷积层块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 第三个卷积层块
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层和Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 根据训练/评估模式自动调整

        # 输出层
        x = self.fc2(x)
        return x

#
# # 实例化模型
# model = MNISTCNNWithThreeConvLayers()
#
# # 打印模型结构
# print(model)