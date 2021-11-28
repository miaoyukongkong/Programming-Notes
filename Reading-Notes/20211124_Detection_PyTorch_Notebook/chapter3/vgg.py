"""使用PyTorch来搭建VGG16经典网络结构

File name: vgg.py
Author: @dongdonghy
Date: 2021-11-28

"""

from torch import nn

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64

        #循环构造卷积层，一共有13个卷积层
        for i in range(13):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            #在第2、4、7、10、13个卷积层后增加池化层
            if i==1 or i==3 or i==6 or i==9 or i==12:
                layers += [nn.MaxPool2d(2, 2)]
                #在第10个卷积后保持和前边的通道数一致，都为512，其余加倍
                if i!=9:
                    out_dim*=2
        
        self.features = nn.Sequential(*layers)
        
        #VGGNet的3个全连接层，中间有ReLU与Dropout层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        #这里是将特征图的维度从[1, 512, 7, 7]变到[1, 512*7*7]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x