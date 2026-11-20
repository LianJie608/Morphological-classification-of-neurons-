import pandas as pd
import numpy as np
import os
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torchsummary import summary


import torch.nn.init as init
import torch.nn as nn
from collections import OrderedDict
from functools import partial
import math



#通道注意力机制
#在基础网络上增加膨胀多尺度卷积
class DilatedMultiScaleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedMultiScaleConvolution, self).__init__()
        self.dim_reduction = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.dim_increase = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)
        self.channel_adjust = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_down = self.dim_reduction(x)
        out1 = self.conv1(x_down)
        out2 = self.conv2(x_down)
        out3 = self.conv3(x_down)
        fused_features = torch.cat([out1, out2, out3], dim=1)
        fused_features = self.dim_increase(fused_features)

        # 使用残差连接将输入和输出叠加
        x_adjusted = self.channel_adjust(x)
        residual = x_adjusted + fused_features
        return residual

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes=64, ratio=8):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * residual

#空间注意力机制
class SpatialAttention3D(nn.Module):
    def __init__(self, out_channel=64, kernel_size=3, stride=1, padding=1):
        super(SpatialAttention3D, self).__init__()

        self.conv = nn.Conv3d(2, out_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        out = x * residual
        return out

#双重注意力机制
class DAM(nn.Module):
    def __init__(self, channels=64):
        super(DAM, self).__init__()

        self.sa = SpatialAttention3D(out_channel=channels)
        self.ca = ChannelAttention3D(in_planes=channels)

    def forward(self, x):
        residual = x
        out = self.ca(x)
        out = self.sa(out)
        out = out + residual
        return out



#改进后的深度残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 第一个1x1x1卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1=nn.ReLU()

        # 3x3x3卷积层
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()

        # 第二个1x1x1卷积层
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)


        # 快捷连接中的1x1x1卷积层（用于改变通道数或步长）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 当步长不为1或通道数不同时，使用1x1x1卷积来匹配主分支的输出形状
            self.shortcut = nn.Sequential(
                nn.AvgPool3d(kernel_size=1, stride=stride),  # 使用步长为2的池化来减小空间尺寸
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out =self.bn1(self.conv1(x))
        out =self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        # print(out.shape)

        # 应用shortcut连接
        shortcut = self.shortcut(x)
        # print(shortcut.shape)

        out += shortcut
        out = F.relu(out)
        return out



#DSC卷积-3D
class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv3D, self).__init__()

        # 深度卷积
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.bn_depth = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 逐点卷积
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.bn_point = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn_depth(self.depthwise(x)))
        x = self.bn_point(self.pointwise(x))
        return x




class cnn3d_multiscale(nn.Module):
    def __init__(self):
        super(cnn3d_multiscale, self).__init__()
        self.conv1 = self._conv_layer_set(3, 16)
        self.conv2 = self._conv_layer_set_1(16, 32)
        self.residual1 = ResidualBlock(32, 64)
        # self.DropBlock1 = DropBlock3D(drop_prob=0.1, block_size=3)  # 在这里添加DropBlock3D
        self.residual2 = ResidualBlock(64, 128)

        self.conv3 = self._conv_layer_set(128, 128)
        self.residual3 = ResidualBlock(128, 128)

        # self.conv4 = self._conv_layer_set_1(128, 256)
        self.conv4 = DepthwiseSeparableConv3D(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv5 = DepthwiseSeparableConv3D(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.DAM = DAM(512)

        self.conv6 = self._conv_layer_set(512,512)
        self.conv7 = DepthwiseSeparableConv3D(512, 1024, kernel_size=(3, 3, 3), padding=1)

        # 将conv5替换为三维深度可分离卷积层
        # self.conv5 = DepthwiseSeparableConv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        # self.conv6 = DepthwiseSeparableConv3d(512, 1024, kernel_size=(3, 3, 3), padding=1)
        # self.conv7 = DepthwiseSeparableConv3d(1024, 2048, kernel_size=(3, 3, 3), padding=1)
        # self.DropBlock2 = DropBlock3D(drop_prob=0.1, block_size=3)  # 在这里添加DropBlock3D

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512,6)

        self.relu = nn.ReLU(inplace=True)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(p=0.3)

    def _conv_layer_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm3d(out_channels),  # BatchNorm layer
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )
        return conv_layer

    def _conv_layer_set_1(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(out_channels),  # BatchNorm layer
            nn.LeakyReLU(),
            DAM(out_channels),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )
        return conv_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # 将1x1x1卷积核应用在适当的位置
        x = self.residual1(x)
        # x = self.DropBlock1(x)
        x = self.residual2(x)

        x = self.conv3(x)
        x = self.residual3(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.DAM(x)

        x = self.conv6(x)
        x = self.conv7(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x





if __name__ == '__main__':
    # 定义输入数据的维度和类别数
    batch_size = 16
    channels =3
    depth = 112
    height =112
    width =112

    # 实例化cnn3d模型
    model =cnn3d()
    # model=cnn3d_multiscale()
    # model =ModifiedCNN3DMultiscale()
    # model=ShuffleNet(groups=3, num_classes=4, width_mult=1, in_channels=3)
    # model = MobileNet(num_classes=12, sample_size=112, width_mult=1., in_channels=3)
    # model_depth = 50
    # model = generate_model(model_depth=50, n_classes=4,in_channels=3,sample_size=112,sample_duration=16)


    # 将模型放到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    summary(model,input_size=(3,112,112,112))


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    input_data = torch.randn(batch_size, channels, depth, height, width)
    labels = torch.randint(0, 12, (batch_size,))


    # 模拟训练过程
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        inputs, labels = input_data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
