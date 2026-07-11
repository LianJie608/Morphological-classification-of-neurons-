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


# class ResidualMultiScaleConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualMultiScaleConvBlock, self).__init__()
#         self.conv3x3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn3x3x3 = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv3x3x5 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 5), padding=(1, 1, 2))
#         self.bn3x3x5 = nn.BatchNorm3d(out_channels)
#
#         self.conv3x3x7 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 7), padding=(1, 1, 3))
#         self.bn3x3x7 = nn.BatchNorm3d(out_channels)
#
#         self.conv1x1x1 = nn.Conv3d(3 * out_channels, out_channels, kernel_size=1)
#         self.bn1x1x1 = nn.BatchNorm3d(out_channels)
#
#
#         self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#         self.bn_residual = nn.BatchNorm3d(out_channels)
#         self.bn = nn.BatchNorm3d(out_channels)
#
#     def forward(self, x):
#         out3x3x3 = self.conv3x3x3(x)
#         out3x3x3 = self.bn3x3x3(out3x3x3)
#         out3x3x3 = self.relu(out3x3x3)
#
#         out3x3x5 = self.conv3x3x5(x)
#         out3x3x5 = self.bn3x3x5(out3x3x5)
#         out3x3x5 = self.relu(out3x3x5)
#
#         out3x3x7 = self.conv3x3x7(x)
#         out3x3x7 = self.bn3x3x7(out3x3x7)
#         out3x3x7 = self.relu(out3x3x7)
#
#         # 特征拼接
#         concatenated = torch.cat((out3x3x3, out3x3x5, out3x3x7), dim=1)
#         # 1x1x1卷积降维
#         reduced = self.conv1x1x1(concatenated)
#         reduced = self.bn1x1x1(reduced)
#         reduced = self.relu(reduced)
#
#         # # 应用通道注意力机制
#         # reduced_attention = self.channel_attention(reduced)
#         # reduced_attention = self.self_attention(reduced)
#
#         # 残差连接
#         residual = self.residual_conv(x)
#         residual = self.bn_residual(residual)
#
#         # 输出是降维后的特征（带通道注意力）和残差连接的和
#         out = self.relu(self.bn(reduced + residual))
#         return out

# 残差多尺度卷积模块插入第一个卷积核后面
# class cnn3d_multiscale(nn.Module):
#     def __init__(self):
#         super(cnn3d_multiscale, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 16)
#         self.residual_block1 = ResidualMultiScaleConvBlock(16, 32)
#         self.conv2 = self._conv_layer_set(32, 64)
#         self.conv3 = self._conv_layer_set(64, 64)
#
#         self.fc1 = nn.Linear(110592,4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, 12)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1_bn = nn.BatchNorm3d(16)
#         self.conv2_bn = nn.BatchNorm3d(64)
#         self.conv3_bn = nn.BatchNorm3d(64)  # 更新BatchNorm层
#         # self.conv1x1x1_bn = nn.BatchNorm3d(16)
#
#         self.fc1_bn = nn.BatchNorm1d(4096)
#         self.fc2_bn = nn.BatchNorm1d(4096)
#         self.drop = nn.Dropout(p=0.3)
#
#     def _conv_layer_set(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=0,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         x = self.residual_block1(x)
#
#         # x = self.conv1x1x1(x)
#         # x = self.conv1x1x1_bn(x)  # 更新BatchNorm
#         # x = self.relu(x)
#
#         x = self.conv2(x)# 将1x1x1卷积核应用在适当的位置
#         x = self.conv2_bn(x)
#         x = self.conv3(x)
#         x = self.conv3_bn(x)  # BatchNorm
#
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc1_bn(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc2_bn(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         return x

#嵌入两个残差多尺度卷积模块
# class cnn3d_multiscale(nn.Module):
#     def __init__(self):
#         super(cnn3d_multiscale, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 16)
#         self.residual_block1 = ResidualMultiScaleConvBlock(16, 32)
#         self.residual_block2 = ResidualMultiScaleConvBlock(32, 32)
#
#         # self.conv1x1x1 = nn.Conv3d(16, 16, kernel_size=1)  # 添加1x1x1的卷积核
#         self.conv2 = self._conv_layer_set(32, 64)
#         self.conv3 = self._conv_layer_set(64, 128)
#         self.fc1 = nn.Linear(110592, 2048)
#         self.fc2 = nn.Linear(2048, 2048)
#         self.fc3 = nn.Linear(2048, 12)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1_bn = nn.BatchNorm3d(16)
#         self.conv2_bn = nn.BatchNorm3d(64)
#         self.conv3_bn = nn.BatchNorm3d(128)  # 更新BatchNorm层
#         # self.conv1x1x1_bn = nn.BatchNorm3d(16)
#
#         self.fc1_bn = nn.BatchNorm1d(2048)
#         self.fc2_bn = nn.BatchNorm1d(2048)
#         self.drop = nn.Dropout(p=0.3)
#
#     def _conv_layer_set(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=0,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         x = self.residual_block1(x)
#         x = self.residual_block2(x)
#
#         # x = self.conv1x1x1(x)
#         # # x = self.conv1x1x1_bn(x)  # 更新BatchNorm
#         # x = self.relu(x)
#
#         x = self.conv2(x)  # 将1x1x1卷积核应用在适当的位置
#         x = self.conv2_bn(x)
#         x = self.conv3(x)
#         x = self.conv3_bn(x)  # BatchNorm
#
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc1_bn(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc2_bn(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         return x


##DAM+RMCB
# class cnn3d_multiscale(nn.Module):
#     def __init__(self):
#         super(cnn3d_multiscale, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 16)
#         self.residual_block1 = ResidualMultiScaleConvBlock(16, 32)
#         self.residual_block2 = ResidualMultiScaleConvBlock(32, 32)
#
#         # self.conv1x1x1 = nn.Conv3d(16, 16, kernel_size=1)  # 添加1x1x1的卷积核
#         self.conv2 = self._conv_layer_set(32, 64)
#         self.conv3 = self._conv_layer_set(64, 128)
#         self.conv4 = self._conv_layer_set_1(128, 256)
#         self.conv5 = self._conv_layer_set(256,256)
#
#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.fc3 = nn.Linear(1024, 12)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1_bn = nn.BatchNorm3d(16)
#         self.conv2_bn = nn.BatchNorm3d(64)
#         self.conv3_bn = nn.BatchNorm3d(128)  # 更新BatchNorm层
#         self.conv4_bn = nn.BatchNorm3d(256)
#         self.conv5_bn = nn.BatchNorm3d(256)
#
#         # self.conv1x1x1_bn = nn.BatchNorm3d(16)
#         self.fc1_bn = nn.BatchNorm1d(1024)
#         self.fc2_bn = nn.BatchNorm1d(1024)
#         self.drop = nn.Dropout(p=0.3)
#
#     def _conv_layer_set(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=0,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#     def _conv_layer_set_1(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=1,# 注意这里增加了padding以保证输出大小不变
#             ),
#             nn.LeakyReLU(),
#             DAM(out_channels),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         x = self.residual_block1(x)
#         x = self.residual_block2(x)
#
#         # x = self.conv1x1x1(x)
#         # x = self.conv1x1x1_bn(x)  # 更新BatchNorm
#         # x = self.relu(x)
#         x = self.conv2(x)  # 将1x1x1卷积核应用在适当的位置
#         x = self.conv2_bn(x)
#         x = self.conv3(x)
#         x = self.conv3_bn(x)
#         x = self.conv4(x)
#         x = self.conv4_bn(x)
#         x = self.conv5(x)
#         x = self.conv5_bn(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc1_bn(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc2_bn(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         return x



##增加卷积层个数+DAM
# class cnn3d_multiscale(nn.Module):
#     def __init__(self):
#         super(cnn3d_multiscale, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 16)
#         # self.residual_block1 = ResidualMultiScaleConvBlock(16, 32)
#         # self.residual_block2 = ResidualMultiScaleConvBlock(32, 32)
#
#         # self.conv1x1x1 = nn.Conv3d(16, 16, kernel_size=1)  # 添加1x1x1的卷积核
#         self.conv2 = self._conv_layer_set(16, 32)
#         self.conv3 = self._conv_layer_set(32, 64)
#         self.conv4 = self._conv_layer_set_1(64, 128)
#         self.conv5 = self._conv_layer_set(128,256)
#
#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.fc3 = nn.Linear(1024, 12)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1_bn = nn.BatchNorm3d(16)
#         self.conv2_bn = nn.BatchNorm3d(32)
#         self.conv3_bn = nn.BatchNorm3d(64)  # 更新BatchNorm层
#         self.conv4_bn = nn.BatchNorm3d(128)
#         self.conv5_bn = nn.BatchNorm3d(256)
#
#         # self.conv1x1x1_bn = nn.BatchNorm3d(16)
#         self.fc1_bn = nn.BatchNorm1d(1024)
#         self.fc2_bn = nn.BatchNorm1d(1024)
#         self.drop = nn.Dropout(p=0.3)
#
#     def _conv_layer_set(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=0,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#     def _conv_layer_set_1(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=1,# 注意这里增加了padding以保证输出大小不变
#             ),
#             nn.LeakyReLU(),
#             DAM(out_channels),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         # x = self.residual_block1(x)
#         # x = self.residual_block2(x)
#
#         # x = self.conv1x1x1(x)
#         # x = self.conv1x1x1_bn(x)  # 更新BatchNorm
#         # x = self.relu(x)
#         x = self.conv2(x)  # 将1x1x1卷积核应用在适当的位置
#         x = self.conv2_bn(x)
#         x = self.conv3(x)
#         x = self.conv3_bn(x)
#         x = self.conv4(x)
#         x = self.conv4_bn(x)
#         x = self.conv5(x)
#         x = self.conv5_bn(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc1_bn(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc2_bn(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         return x


#Residual block+DAM
# class cnn3d_multiscale(nn.Module):
#     def __init__(self):
#         super(cnn3d_multiscale, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 16)
#         # self.residual_block1 = ResidualMultiScaleConvBlock(16, 32)
#         # self.residual_block2 = ResidualMultiScaleConvBlock(32, 32)
#
#         # self.conv1x1x1 = nn.Conv3d(16, 16, kernel_size=1)  # 添加1x1x1的卷积核
#         self.conv2 = self._conv_layer_set(16, 32)
#         self.residual1 = ResidualBlock(32, 64)
#         self.conv3 = self._conv_layer_set(64, 128)
#         self.conv4 = self._conv_layer_set_1(128, 256)
#         self.conv5 = self._conv_layer_set(256,256)
#         self.residual2 = ResidualBlock(256, 256)
#
#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.fc3 = nn.Linear(1024, 12)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1_bn = nn.BatchNorm3d(16)
#         self.conv2_bn = nn.BatchNorm3d(32)
#         self.conv3_bn = nn.BatchNorm3d(128)  # 更新BatchNorm层
#         self.conv4_bn = nn.BatchNorm3d(256)
#         self.conv5_bn = nn.BatchNorm3d(256)
#
#         # self.conv1x1x1_bn = nn.BatchNorm3d(16)
#         self.fc1_bn = nn.BatchNorm1d(1024)
#         self.fc2_bn = nn.BatchNorm1d(1024)
#         self.drop = nn.Dropout(p=0.3)
#
#     def _conv_layer_set(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=0,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#     def _conv_layer_set_1(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=1,# 注意这里增加了padding以保证输出大小不变
#             ),
#             nn.LeakyReLU(),
#             DAM(out_channels),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         # x = self.residual_block1(x)
#         # x = self.residual_block2(x)
#
#         # x = self.conv1x1x1(x)
#         # x = self.conv1x1x1_bn(x)  # 更新BatchNorm
#         # x = self.relu(x)
#
#         x = self.conv2(x)  # 将1x1x1卷积核应用在适当的位置
#         x = self.conv2_bn(x)
#         x = self.residual1(x)
#         x = self.conv3(x)
#         x = self.conv3_bn(x)
#         x = self.conv4(x)
#         x = self.conv4_bn(x)
#         x = self.conv5(x)
#         x = self.conv5_bn(x)
#         x = self.residual2(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc1_bn(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc2_bn(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         return x

#DAM+RMCB+Residual Block

# class cnn3d_multiscale(nn.Module):
#     def __init__(self):
#         super(cnn3d_multiscale, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 16)
#         self.residual_block1 = ResidualMultiScaleConvBlock(16, 32)
#         self.residual_block2 = ResidualMultiScaleConvBlock(32, 64)
#
#         # self.conv1x1x1 = nn.Conv3d(16, 16, kernel_size=1)  # 添加1x1x1的卷积核
#         self.conv2 = self._conv_layer_set(64, 128)
#         self.residual1 = ResidualBlock(128, 128)
#         self.conv3 = self._conv_layer_set(128, 256)
#         self.conv4 = self._conv_layer_set_1(256,512)
#         self.conv5 = self._conv_layer_set(512,512)
#         self.residual2 = ResidualBlock(512, 512)
#
#         self.fc1 = nn.Linear(4096, 2048)
#         self.fc2 = nn.Linear(2048, 2048)
#         self.fc3 = nn.Linear(2048, 12)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1_bn = nn.BatchNorm3d(16)
#         self.conv2_bn = nn.BatchNorm3d(128)
#         self.conv3_bn = nn.BatchNorm3d(256)  # 更新BatchNorm层
#         self.conv4_bn = nn.BatchNorm3d(512)
#         self.conv5_bn = nn.BatchNorm3d(512)
#
#         # self.conv1x1x1_bn = nn.BatchNorm3d(16)
#         self.fc1_bn = nn.BatchNorm1d(2048)
#         self.fc2_bn = nn.BatchNorm1d(2048)
#         self.drop = nn.Dropout(p=0.3)
#
#     def _conv_layer_set(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=0,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#     def _conv_layer_set_1(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=1,# 注意这里增加了padding以保证输出大小不变
#             ),
#             nn.LeakyReLU(),
#             DAM(out_channels),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         x = self.residual_block1(x)
#         x = self.residual_block2(x)
#
#         # x = self.conv1x1x1(x)
#         # x = self.conv1x1x1_bn(x)  # 更新BatchNorm
#         # x = self.relu(x)
#
#         x = self.conv2(x)  # 将1x1x1卷积核应用在适当的位置
#         x = self.conv2_bn(x)
#         x = self.residual1(x)
#         x = self.conv3(x)
#         x = self.conv3_bn(x)
#         x = self.conv4(x)
#         x = self.conv4_bn(x)
#         x = self.conv5(x)
#         x = self.conv5_bn(x)
#         x = self.residual2(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc1_bn(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc2_bn(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         return x


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

##DropBlock3D正则化
class DropBlock3D(torch.nn.Module):
    def __init__(self, drop_prob, block_size, inplace=False):
        super(DropBlock3D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
        self.inplace = inplace

        # 初始化时计算gamma
        self.gamma = None

    def _compute_gamma(self, input_size):
        # 计算gamma（输入单元被丢弃的比例）
        depth, height, width = input_size
        if self.block_size > depth or self.block_size > height or self.block_size > width:
            return 0

        gamma = (self.drop_prob * depth * height * width) / ((depth - self.block_size + 1) *
                                                             (height - self.block_size + 1) *
                                                             (width - self.block_size + 1))
        gamma = min(gamma, 1 - 1e-6)  # 添加小常数以确保数值稳定性
        return gamma

    def forward(self, x):
        # 获取输入张量的形状
        batch_size, num_channels, depth, height, width = x.size()

        # 计算gamma
        if self.gamma is None:
            self.gamma = self._compute_gamma((depth, height, width))

            # 如果gamma为0或block_size过大，则不应用DropBlock
        if self.gamma == 0:
            return x

            # 生成均匀分布的随机数
        uniform_num = torch.rand(batch_size, 1, 1, 1, 1).to(x.device)

        # 生成dropout决策掩码
        drop_mask = (uniform_num < self.gamma).float()

        # 创建实际的掩码
        keep_mask = torch.ones_like(drop_mask)
        block_mask = F.max_pool3d(
            input=drop_mask,
            kernel_size=(self.block_size, self.block_size, self.block_size),
            stride=1,
            padding=(self.block_size // 2, self.block_size // 2, self.block_size // 2),
        )
        keep_mask.mul_(1 - block_mask)

        # 应用掩码到输入张量
        if self.inplace:
            x.mul_(keep_mask.expand_as(x))
            x.div_(1 - self.gamma + 1e-6)  # 归一化张量
        else:
            x = x.mul(keep_mask.expand_as(x))
            x = x / (1 - self.gamma + 1e-6)  # 归一化张量

        return x

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





# class cnn3d_multiscale(nn.Module):
#     def __init__(self):
#         super(cnn3d_multiscale, self).__init__()
#         self.conv1 = self._conv_layer_set(3, 16)
#         self.conv2 = self._conv_layer_set_1(16, 32)
#         self.residual1 = ResidualBlock(32, 64)
#         # self.DropBlock1 = DropBlock3D(drop_prob=0.1, block_size=3)  # 在这里添加DropBlock3D
#         self.residual2 = ResidualBlock(64, 128)
#
#         self.conv3 = self._conv_layer_set(128, 128)
#         self.residual3 = ResidualBlock(128, 128)
#
#         # self.conv4 = self._conv_layer_set_1(128, 256)
#         self.conv4 = DepthwiseSeparableConv3D(128, 256, kernel_size=(3, 3, 3), padding=1)
#         self.conv5 = DepthwiseSeparableConv3D(256, 512, kernel_size=(3, 3, 3), padding=1)
#         self.DAM = DAM(512)
#
#         self.conv6 = self._conv_layer_set(512,512)
#         self.conv7 = DepthwiseSeparableConv3D(512, 1024, kernel_size=(3, 3, 3), padding=1)
#
#         # 将conv5替换为三维深度可分离卷积层
#         # self.conv5 = DepthwiseSeparableConv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
#         # self.conv6 = DepthwiseSeparableConv3d(512, 1024, kernel_size=(3, 3, 3), padding=1)
#         # self.conv7 = DepthwiseSeparableConv3d(1024, 2048, kernel_size=(3, 3, 3), padding=1)
#         # self.DropBlock2 = DropBlock3D(drop_prob=0.1, block_size=3)  # 在这里添加DropBlock3D
#
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512,6)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.fc1_bn = nn.BatchNorm1d(512)
#         self.fc2_bn = nn.BatchNorm1d(512)
#         self.drop = nn.Dropout(p=0.3)
#
#     def _conv_layer_set(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=0,
#             ),
#             nn.BatchNorm3d(out_channels),  # BatchNorm layer
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#
#     def _conv_layer_set_1(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(3, 3, 3),
#                 stride=1,
#                 padding=1,
#             ),
#             nn.BatchNorm3d(out_channels),  # BatchNorm layer
#             nn.LeakyReLU(),
#             DAM(out_channels),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#         )
#         return conv_layer
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)  # 将1x1x1卷积核应用在适当的位置
#         x = self.residual1(x)
#         # x = self.DropBlock1(x)
#         x = self.residual2(x)
#
#         x = self.conv3(x)
#         x = self.residual3(x)
#
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.DAM(x)
#
#         x = self.conv6(x)
#         x = self.conv7(x)
#
#         x = F.adaptive_avg_pool3d(x, (1, 1, 1))
#
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc1_bn(x)
#         x = self.relu(x)
#         x = self.drop(x)
#
#         x = self.fc2(x)
#         x = self.fc2_bn(x)
#         x = self.relu(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#
#         return x




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
