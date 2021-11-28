"""使用PyTorch来实现DetNet的两个Bottleneck结构A和B

File name: detnet_bottleneck.py
Author: @dongdonghy
Date: 2021-11-28

"""

from torch import nn
class DetBottleneck(nn.Module):
    #初始化时，extra为False时为Bottleneck A，为True时则为Bottleneck B
    def __init__(self, inplanes, planes, stride=1, extra=False):
        super(DetBottleneck, self).__init__()
        #构建连续3个卷积层的Bottleneck
        self.bottleneck = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False), 
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, 
                               dilation=2, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, 1, bias=False),
                nn.BatchNorm2d(planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.extra = extra
        if self.extra:
            self.extra_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        #对于Bottleneck B来讲，需要对恒等映射增加卷积处理，与ResNet类似
        if self.extra:
            identity = self.extra_conv(x)
        else:
            identity = x
        out = self.bottleneck(x)
        out += identity
        out = self.relu(out)
        return out