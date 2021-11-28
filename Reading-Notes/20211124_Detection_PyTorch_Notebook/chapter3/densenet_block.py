"""使用PyTorch来实现DenseNet的一个Block

File name: densenet_block.py
Author: @dongdonghy
Date: 2021-11-28

"""

import torch
from torch import nn
import torch.nn.functional as F

#实现一个Bottleneck类，初始化需要输入通道和GrowthRate这两个参数
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        #通常1×1卷积的通道数为GrowthRate的4倍
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        #将输入x和计算的结果out进行通道拼接
        out = torch.cat((x, out), 1)
        return out

class Denseblock(nn.Module):
    def __init__(self, nChannels, growthRate, nDenseBlocks):
        super(Denseblock, self).__init__()
        layers = []
        #将每一个Bottleneck利用nn.Sequential()整合起来，输入通道数需要线性增长
        for i in range(int(nDenseBlocks)):
            layers.append(Bottleneck(nChannels, growthRate))
            nChannels += growthRate
        self.denseblock = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.denseblock(x)